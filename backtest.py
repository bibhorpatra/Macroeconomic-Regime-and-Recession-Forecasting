"""
backtest.py
===========
Rolling-window walk-forward out-of-sample backtest: at each OOS month,
retrain on the trailing `rolling_window` months and predict the single next
(unseen) month, then roll forward one month and repeat. This is the same
methodology as the old project (kept because it is the right way to
simulate real deployment), applied here to the enriched feature set and
imbalance-aware models.

Leakage note: at step i, the model is fit on rows [i, i+rolling_window) and
predicts row i+rolling_window only. The training window never includes the
row being predicted, and X has already been shifted forward by the
forecast horizon in feature_engineering.align_features_to_horizon, so the
prediction for month t only ever used information available at t-horizon.
The `threshold` argument (probability -> hard class cutoff) is likewise
tuned only on the pre-OOS period (modeling.select_decision_threshold) and
held fixed for the entire OOS backtest; it is never re-selected using OOS
data.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from modeling import build_model

logger = logging.getLogger(__name__)


def rolling_backtest(
    X: pd.DataFrame, y: pd.Series, model_name: str, params: dict, rolling_window: int,
    split_date: str, random_state: int = 42, collect_feature_importance: bool = False,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """X, y must already be aligned (same index, sorted by date, no NaNs).
    Returns (predictions_df[Date, y_true, y_prob, y_pred], feature_importance_df or None).
    """
    dates = X.index
    split_ts = pd.Timestamp(split_date)
    oos_candidates = dates[dates >= split_ts]
    if len(oos_candidates) == 0:
        raise ValueError(f"No rows on/after split_date={split_date}")
    oos_start_pos = dates.get_loc(oos_candidates[0])
    train_start_pos = oos_start_pos - rolling_window
    if train_start_pos < 0:
        raise ValueError(
            f"Not enough pre-OOS history for rolling_window={rolling_window}: "
            f"OOS starts at position {oos_start_pos}, need {rolling_window} rows before it."
        )

    n_steps = len(dates) - oos_start_pos
    logger.info("Rolling backtest: model=%s, %d OOS steps, rolling_window=%d, OOS starts %s",
                model_name, n_steps, rolling_window, oos_candidates[0].date())

    rows = []
    fi_rows = []
    for i in range(train_start_pos, len(dates) - rolling_window):
        train_slice = slice(i, i + rolling_window)
        predict_pos = i + rolling_window
        X_train, y_train = X.iloc[train_slice], y.iloc[train_slice]
        X_pred = X.iloc[predict_pos:predict_pos + 1]
        y_true = y.iloc[predict_pos]
        date = dates[predict_pos]

        if y_train.nunique() < 2:
            # degenerate all-one-class training window: predict the
            # majority class with prob 0/1 rather than fitting a model that
            # sklearn would refuse / silently misbehave on.
            majority = float(y_train.iloc[0])
            rows.append({"Date": date, "y_true": y_true, "y_prob": majority, "y_pred": majority})
            continue

        model = build_model(model_name, params, y_train, random_state=random_state)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_pred)[:, 1][0]
        # `threshold` is tuned once on the pre-OOS period (see
        # modeling.select_decision_threshold) rather than left at the
        # sklearn default of 0.5, and it is never re-tuned using OOS data.
        y_pred = int(y_prob > threshold)
        rows.append({"Date": date, "y_true": y_true, "y_prob": y_prob, "y_pred": y_pred})

        if collect_feature_importance and model_name == "XGB":
            importances = model.feature_importances_
            fi_row = {"Date": date}
            fi_row.update(dict(zip(X.columns, importances)))
            fi_rows.append(fi_row)

    preds = pd.DataFrame(rows).set_index("Date")
    fi_df = pd.DataFrame(fi_rows).set_index("Date") if fi_rows else None
    return preds, fi_df
