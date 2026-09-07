"""
evaluation.py
=============
Metrics, confusion matrices, and XGBoost feature-attribution rollups
(by macro group AND by engineered-feature type) on rolling-OOS predictions.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

logger = logging.getLogger(__name__)

METRIC_ORDER = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1", "MCC", "ROC-AUC"]


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    try:
        roc_auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    except ValueError:
        roc_auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0,
        "ROC-AUC": roc_auc,
    }


def metrics_table(preds_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cols = {}
    for model_name, preds in preds_by_model.items():
        m = compute_metrics(preds["y_true"], preds["y_pred"], preds["y_prob"])
        cols[model_name] = [m[k] for k in METRIC_ORDER]
    return pd.DataFrame(cols, index=METRIC_ORDER)


def compute_persistence_baseline(y_true: pd.Series) -> dict:
    """The single most important baseline for a 1-month-ahead regime
    classifier: predict that this month's regime is whatever it actually
    was LAST month, using no macro data and no model, just the label's own
    autocorrelation. Recessions and market-stress periods are multi-month
    episodes, so this "nothing changed" rule is often hard to beat at a
    1-month horizon. A model that fails to beat it means regimes persist
    on their own; it doesn't necessarily mean the engineered features add
    real forecasting value. Always report this alongside the trained
    models, since a strong metric in isolation (high accuracy or F1, say)
    can look impressive while still underperforming this trivial rule,
    which is exactly what happens for the recession target here (see the
    README's "How impressive are these numbers" discussion)."""
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = y_true.shift(1)
    mask = y_pred.notna()
    yt = y_true[mask].astype(int).to_numpy()
    yp = y_pred[mask].astype(int).to_numpy()
    return compute_metrics(yt, yp, yp.astype(float))


def metrics_table_with_baseline(preds_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Same as metrics_table, plus a 'Persistence' column computed from
    whichever model's y_true series (they all share the same OOS index)."""
    table = metrics_table(preds_by_model)
    any_preds = next(iter(preds_by_model.values()))
    baseline = compute_persistence_baseline(any_preds["y_true"])
    table["Persistence"] = [baseline[k] for k in METRIC_ORDER]
    return table


def plot_confusion_matrix(y_true, y_pred, class_names: list[str], title: str, out_path: str):
    cm = confusion_matrix(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int))
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names,
                cbar=False, ax=ax, annot_kws={"size": 13})
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_metrics_bar(metrics_df: pd.DataFrame, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    metrics_df.plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(title="Model")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_regime_probabilities(dates, y_true, y_prob_by_model: dict[str, pd.Series], title: str, out_path: str, ylabel: str = "Predicted probability"):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    # shade actual regime periods
    y_true = pd.Series(y_true, index=dates).astype(float)
    in_regime = False
    start = None
    for d, v in y_true.items():
        if v == 1 and not in_regime:
            in_regime, start = True, d
        elif v == 0 and in_regime:
            ax.axvspan(start, d, color="grey", alpha=0.25)
            in_regime = False
    if in_regime:
        ax.axvspan(start, y_true.index[-1], color="grey", alpha=0.25)
    for model_name, probs in y_prob_by_model.items():
        ax.plot(dates, probs, label=model_name, linewidth=1.2)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Feature attribution
# ---------------------------------------------------------------------------
def aggregate_feature_importance(fi_df: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """fi_df: rows=OOS dates, cols=feature importances from each rolling-window
    XGBoost fit. metadata: feature -> (source_series, group, feat_type).

    Returns (top_features_df, group_rollup_df, type_rollup_df).
    """
    avg_importance = fi_df.mean(axis=0).rename("Importance").to_frame()
    avg_importance.index.name = "Feature"
    avg_importance = avg_importance.join(metadata, how="left")
    avg_importance = avg_importance.sort_values("Importance", ascending=False)

    group_rollup = (
        avg_importance.groupby("group")["Importance"].sum().sort_values(ascending=False).rename("Importance").to_frame()
    )
    type_rollup = (
        avg_importance.groupby("feat_type")["Importance"].sum().sort_values(ascending=False).rename("Importance").to_frame()
    )
    return avg_importance, group_rollup, type_rollup


def plot_importance_rollup(rollup_df: pd.DataFrame, title: str, out_path: str, xlabel: str = "Importance (share of total)"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    (rollup_df["Importance"] / rollup_df["Importance"].sum()).sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
