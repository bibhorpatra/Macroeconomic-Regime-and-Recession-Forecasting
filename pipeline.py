"""
pipeline.py
===========
End-to-end orchestration: raw data -> cleaning -> feature engineering ->
labeling -> horizon alignment -> CV hyperparameter search -> rolling OOS
backtest -> evaluation -> persisted results. This is the single source of
truth both the notebooks and this module's __main__ block call into, so the
notebooks and the "real" run can never silently diverge.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from data_cleaning import DataCleaning, load_fredmd_raw
from feature_engineering import build_feature_panel, align_features_to_horizon
from labeling import build_recession_label, build_market_stress_label
from modeling import time_series_cv_search, select_decision_threshold
from backtest import rolling_backtest
from evaluation import (
    aggregate_feature_importance, compute_metrics, metrics_table_with_baseline,
    plot_confusion_matrix, plot_importance_rollup, plot_metrics_bar, plot_regime_probabilities,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_raw_inputs(cfg: dict):
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    data, tcodes = load_fredmd_raw(str(raw_dir / "fredmd_current.csv"))
    usrec = pd.read_csv(raw_dir / "nber_usrec.csv", index_col="Date", parse_dates=True)
    vix_monthly = pd.read_csv(raw_dir / "vix_monthly.csv", index_col="Date", parse_dates=True)
    series_spec = pd.read_csv(raw_dir / "fredmd_series_spec.csv")
    return data, tcodes, usrec, vix_monthly, series_spec


def clean_panel(data: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, str]:
    dc = DataCleaning(data=data.copy())
    dc.remove_null_rows(max_null=cfg["cleaning"]["max_null_rows"])
    dc.remove_null_features(max_null=cfg["cleaning"]["max_null_cols"])
    dc.fill_null_obs()
    return dc.data, dc.summary()


def build_features(cleaned: pd.DataFrame, tcodes: dict, series_spec: pd.DataFrame, vix_monthly: pd.DataFrame, cfg: dict):
    fcfg = cfg["features"]
    result = build_feature_panel(
        cleaned_levels=cleaned,
        transform_codes=tcodes,
        series_spec=series_spec,
        sp500_close=cleaned["SP500"],
        vix_monthly=vix_monthly,
        lag_windows=fcfg["lag_windows"],
        momentum_windows=fcfg["momentum_windows"],
        rolling_windows=fcfg["rolling_windows"],
        drawdown_windows=fcfg["drawdown_windows"],
        volatility_windows=fcfg["volatility_windows"],
        spread_specs=fcfg["spreads"],
        drawdown_source_series=fcfg["drawdown_source_series"],
    )
    return result


def build_label(target: str, cleaned: pd.DataFrame, usrec: pd.DataFrame, vix_monthly: pd.DataFrame, cfg: dict) -> pd.Series:
    if target == "recession":
        return build_recession_label(usrec, cleaned.index).rename("label")
    elif target == "market_stress":
        mcfg = cfg["labeling"]["market_stress"]
        out = build_market_stress_label(
            sp500_close=cleaned["SP500"], vix_monthly=vix_monthly,
            drawdown_threshold=mcfg["drawdown_threshold"], drawdown_window=mcfg["drawdown_window"],
            vix_threshold_percentile=mcfg["vix_threshold_percentile"], vix_min_periods=mcfg["vix_min_periods"],
            index=cleaned.index,
        )
        return out["MktRegime"].rename("label")
    else:
        raise ValueError(target)


def run_target(
    target: str, cfg: dict, save_outputs: bool = True,
    add_autoregressive_feature: bool = False, tune_decision_threshold: bool = False,
):
    """target: 'recession' or 'market_stress'.

    add_autoregressive_feature / tune_decision_threshold: both default OFF.
    Both were tried as ways to close the gap with the persistence baseline
    (see README "What I tried that didn't work") and both empirically made
    the rolling-OOS metrics worse, not better, for both targets. Left as
    opt-in flags so that negative result is reproducible rather than
    silently deleted."""
    assert target in ("recession", "market_stress")
    logger.info("=" * 80)
    logger.info("RUNNING PIPELINE for target=%s", target)
    logger.info("=" * 80)

    data, tcodes, usrec, vix_monthly, series_spec = load_raw_inputs(cfg)
    cleaned, clean_log = clean_panel(data, cfg)
    logger.info("Cleaned panel: %s, %s to %s", cleaned.shape, cleaned.index.min().date(), cleaned.index.max().date())

    feat_result = build_features(cleaned, tcodes, series_spec, vix_monthly, cfg)
    label = build_label(target, cleaned, usrec, vix_monthly, cfg)

    if target == "recession":
        # VIX only exists from 1990-01 onward. Including VIX-derived
        # features in the recession model would force-truncate its entire
        # usable sample (and the ~30 years of pre-1990 recessions in it,
        # including 1960, 1969-70, 1973-75, 1980, 1981-82) down to the
        # VIX era. The recession model therefore uses the full macro panel
        # WITHOUT VIX features; the market-stress model (whose own label is
        # partly defined by VIX and whose usable sample already starts in
        # the 1990s) keeps them.
        vix_cols = [c for c in feat_result.features.columns if "VIX" in c]
        features_for_target = feat_result.features.drop(columns=vix_cols)
        logger.info("Recession target: dropped %d VIX-derived feature columns to preserve pre-1990 history", len(vix_cols))
    else:
        features_for_target = feat_result.features

    metadata_for_target = feat_result.metadata.copy()
    if add_autoregressive_feature:
        # EXPERIMENTAL, off by default. See README "What I tried that
        # didn't work". Adds the target's own actual regime value as a
        # feature; after align_features_to_horizon shifts every feature
        # (this one included) forward by the 1-month horizon below, it
        # becomes "the actual regime as of month t-1", i.e. exactly the
        # persistence-baseline signal, offered to the model as one input
        # among 1,880+ rather than compared against only from the outside.
        # This is leakage-safe by construction (same forward-shift as every
        # other feature) and is standard practice in macro nowcasting
        # (an ARX-style lagged-dependent-variable term). Tested empirically
        # here: it made every rolling-OOS metric worse for both targets
        # rather than better (see README for the actual numbers and the
        # working theory of why). Kept available via this flag for
        # transparency and reproducibility of that experiment, not because
        # it helped.
        features_for_target = features_for_target.copy()
        ar_col = f"{target}_prior_regime"
        features_for_target[ar_col] = label.astype(float)
        metadata_for_target.loc[ar_col] = {
            "source_series": target, "group": "Autoregressive (prior regime)", "feat_type": "autoregressive",
        }

    horizon = cfg["split"]["horizon"]
    X_aligned = align_features_to_horizon(features_for_target, horizon)

    dataset = X_aligned.copy()
    dataset["label"] = label
    dataset = dataset.dropna()
    logger.info("Final aligned dataset: %s rows, %s to %s, positive rate=%.3f",
                dataset.shape, dataset.index.min().date(), dataset.index.max().date(), dataset["label"].mean())

    feature_cols = [c for c in dataset.columns if c != "label"]
    X, y = dataset[feature_cols], dataset["label"].astype(int)

    split_cfg = cfg["split"]["recession"] if target == "recession" else cfg["split"]["market_stress"]
    split_date = split_cfg["split_date"]
    n_cv_splits = split_cfg.get("cv_n_splits", cfg["split"].get("cv_n_splits", 3))
    rolling_window = cfg["split"]["rolling_window"]
    selection_metric = cfg["modeling"]["selection_metric"] if target == "recession" else cfg["modeling"].get(
        "market_stress_selection_metric", cfg["modeling"]["selection_metric"])
    random_state = cfg["modeling"]["random_state"]

    X_pre = X[X.index < split_date]
    y_pre = y[y.index < split_date]
    logger.info("Pre-OOS (CV) period: %d rows, %s to %s, positive rate=%.3f",
                len(X_pre), X_pre.index.min().date(), X_pre.index.max().date(), y_pre.mean())

    if save_outputs:
        # Cleaned + transformed macro panel (pre-feature-engineering, pre-label,
        # pre-horizon-alignment) is target-independent, so it is written once
        # to data/processed/ as the intermediate artifact between raw pulls
        # and the final per-target modeling datasets in data/datasets/.
        processed_dir = ROOT / cfg["data"]["processed_dir"]
        processed_dir.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(processed_dir / "fredmd_cleaned_transformed.csv")
        with open(processed_dir / "cleaning_log.txt", "w") as f:
            f.write(clean_log)

        datasets_dir = ROOT / cfg["data"]["datasets_dir"]
        datasets_dir.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(datasets_dir / f"{target}_dataset.csv")
        # Target-prefixed: recession and market_stress have different
        # feature sets (VIX columns dropped for recession; each has its own
        # autoregressive column), so a single shared filename would let
        # whichever target ran second silently overwrite the other's metadata.
        metadata_for_target.to_csv(datasets_dir / f"{target}_feature_metadata.csv")

    best_params = {}
    best_thresholds = {}
    cv_results = {}
    for model_name in ["DT", "RF", "XGB"]:
        params, cv_df = time_series_cv_search(
            X_pre, y_pre, model_name=model_name, n_splits=n_cv_splits,
            selection_metric=selection_metric, random_state=random_state,
        )
        best_params[model_name] = params
        cv_results[model_name] = cv_df
        if tune_decision_threshold:
            # EXPERIMENTAL, off by default. See README "What I tried that
            # didn't work". Empirically made every metric worse on the real
            # OOS backtest for both targets, despite looking better on the
            # pre-OOS folds themselves: it overfits the threshold to a very
            # small pre-OOS sample (153-168 rows; 2-3 TimeSeriesSplit folds
            # is not enough data to reliably pick 1-of-19 threshold
            # candidates). Kept available via this flag for transparency and
            # reproducibility of that experiment, not because it helped.
            best_thresholds[model_name] = select_decision_threshold(
                X_pre, y_pre, model_name=model_name, params=params, n_splits=n_cv_splits, random_state=random_state,
            )
        else:
            # sklearn default; already imbalance-corrected via
            # class_weight="balanced" / scale_pos_weight, not via threshold.
            best_thresholds[model_name] = 0.5

    preds_by_model = {}
    fi_df = None
    for model_name in ["DT", "RF", "XGB"]:
        preds, fi = rolling_backtest(
            X, y, model_name=model_name, params=best_params[model_name], rolling_window=rolling_window,
            split_date=split_date, random_state=random_state, collect_feature_importance=(model_name == "XGB"),
            threshold=best_thresholds[model_name],
        )
        preds_by_model[model_name] = preds
        if model_name == "XGB":
            fi_df = fi

    metrics_df = metrics_table_with_baseline(preds_by_model)
    logger.info("Rolling-OOS metrics for target=%s:\n%s", target, metrics_df.round(4).to_string())

    if save_outputs:
        results_dir = ROOT / "results"
        predictions_dir = ROOT / cfg["data"]["predictions_dir"]
        predictions_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "confusion_matrices").mkdir(parents=True, exist_ok=True)
        (results_dir / "figures").mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(results_dir / f"metrics_{target}.csv")
        for model_name, params in best_params.items():
            cv_results[model_name].to_csv(predictions_dir / f"{target}_{model_name}_cv_search.csv", index=False)
        with open(predictions_dir / f"{target}_best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)
        with open(predictions_dir / f"{target}_best_thresholds.json", "w") as f:
            json.dump(best_thresholds, f, indent=2)

        for model_name, preds in preds_by_model.items():
            preds.to_csv(predictions_dir / f"{target}_{model_name}_oos_predictions.csv")

        class_names = ["Normal", "Recession"] if target == "recession" else ["Normal", "Stress"]
        for model_name, preds in preds_by_model.items():
            plot_confusion_matrix(
                preds["y_true"], preds["y_pred"], class_names=class_names,
                title=f"{model_name} - {target} - Confusion Matrix",
                out_path=str(results_dir / "confusion_matrices" / f"{target}_{model_name}_cm.png"),
            )
        plot_metrics_bar(metrics_df, title=f"Rolling-OOS Evaluation Metrics: {target}",
                          out_path=str(results_dir / "figures" / f"{target}_metrics_bar.png"))

        any_model = next(iter(preds_by_model.values()))
        plot_regime_probabilities(
            dates=any_model.index, y_true=any_model["y_true"],
            y_prob_by_model={m: p["y_prob"] for m, p in preds_by_model.items()},
            title=f"Predicted probability vs actual regime: {target}",
            out_path=str(results_dir / "figures" / f"{target}_regime_probabilities.png"),
        )

        if fi_df is not None:
            top_feat, group_rollup, type_rollup = aggregate_feature_importance(fi_df, metadata_for_target)
            top_feat.to_csv(results_dir / f"feature_importance_{target}.csv")
            group_rollup.to_csv(predictions_dir / f"{target}_group_importance.csv")
            type_rollup.to_csv(predictions_dir / f"{target}_type_importance.csv")
            plot_importance_rollup(group_rollup, title=f"XGBoost importance by macro group: {target}",
                                    out_path=str(results_dir / "figures" / f"{target}_importance_by_group.png"))
            plot_importance_rollup(type_rollup, title=f"XGBoost importance by engineered-feature type: {target}",
                                    out_path=str(results_dir / "figures" / f"{target}_importance_by_type.png"))

    return {
        "dataset": dataset, "best_params": best_params, "best_thresholds": best_thresholds,
        "preds_by_model": preds_by_model, "metrics_df": metrics_df, "fi_df": fi_df,
        "metadata": metadata_for_target, "clean_log": clean_log,
    }


if __name__ == "__main__":
    cfg = load_config()
    run_target("recession", cfg)
    run_target("market_stress", cfg)
