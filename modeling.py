"""
modeling.py
===========
Hyperparameter search (TimeSeriesSplit CV, selecting on an imbalance-aware
metric) and imbalance-aware model construction for Decision Tree, Random
Forest, and XGBoost.

What changed vs. the old project
---------------------------------
* The old project selected hyperparameters by raw accuracy in CV. Under
  ~10-15% positive-class prevalence, a model that always predicts the
  majority class scores ~85-90% "accuracy" while being useless. This
  version selects on F1 / balanced accuracy / MCC (configurable via
  config.yaml `modeling.selection_metric`, default F1), computed on the
  held-out fold of each TimeSeriesSplit split.
* Every model is imbalance-aware: `class_weight="balanced"` for the tree
  models (DecisionTree, RandomForest), and `scale_pos_weight` (the
  negative:positive class ratio of the CURRENT training window, recomputed
  every fit rather than fixed once) for XGBoost. This is applied both
  during hyperparameter search and later during the rolling OOS backtest.
"""
from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

logger = logging.getLogger(__name__)

def _safe_ranking_metric(fn):
    def wrapped(y_true, y_pred, y_prob):
        if len(set(y_true)) < 2:
            return 0.0
        try:
            return fn(y_true, y_prob)
        except ValueError:
            return 0.0
    return wrapped


SELECTION_METRICS = {
    "f1": lambda y_true, y_pred, y_prob: f1_score(y_true, y_pred, zero_division=0),
    "balanced_accuracy": lambda y_true, y_pred, y_prob: balanced_accuracy_score(y_true, y_pred),
    "mcc": lambda y_true, y_pred, y_prob: matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0,
    "accuracy": lambda y_true, y_pred, y_prob: accuracy_score(y_true, y_pred),  # kept only for comparison/ablation
    # Ranking-based (continuous-probability) metrics. These stay informative
    # even in folds so sparse that every hard 0.5-threshold prediction comes
    # out all-negative (F1/accuracy/MCC then tie at a constant value for
    # EVERY hyperparameter candidate, making selection arbitrary). See
    # config.yaml's note on `market_stress.cv_n_splits` / `modeling` for
    # where this was needed in practice.
    "pr_auc": _safe_ranking_metric(average_precision_score),
    "roc_auc": _safe_ranking_metric(roc_auc_score),
}


def compute_scale_pos_weight(y: np.ndarray | pd.Series) -> float:
    y = np.asarray(y)
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    return n_neg / n_pos


PARAM_GRIDS = {
    "DT": {
        "max_depth": [3, 5, 8, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 5, 10],
    },
    "RF": {
        "n_estimators": [100, 200, 400],
        "max_depth": [3, 5, 8, 10],
        "min_samples_leaf": [1, 5, 10],
    },
    "XGB": {
        "max_depth": [3, 5, 8],
        "n_estimators": [100, 200, 400],
        "learning_rate": [0.05, 0.1],
    },
}


def build_model(model_name: str, params: dict, y_train_for_weight: np.ndarray | pd.Series, random_state: int = 42):
    """Constructs a fresh, imbalance-aware model instance. `y_train_for_weight`
    is used ONLY to compute XGBoost's scale_pos_weight for this specific
    training set; it is not a hyperparameter, it is recomputed every fit."""
    if model_name == "DT":
        return DecisionTreeClassifier(random_state=random_state, class_weight="balanced", **params)
    elif model_name == "RF":
        return RandomForestClassifier(random_state=random_state, class_weight="balanced", n_jobs=-1, **params)
    elif model_name == "XGB":
        spw = compute_scale_pos_weight(y_train_for_weight)
        return xgb.XGBClassifier(
            random_state=random_state, booster="gbtree", objective="binary:logistic",
            eval_metric="logloss", scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, **params,
        )
    else:
        raise ValueError(model_name)


def _param_grid_to_list(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]


def time_series_cv_search(
    X: pd.DataFrame, y: pd.Series, model_name: str, n_splits: int, selection_metric: str,
    random_state: int = 42, param_grid: dict | None = None,
) -> tuple[dict, pd.DataFrame]:
    """TimeSeriesSplit hyperparameter search. Returns (best_params, results_df)
    where results_df has one row per candidate with its mean CV score."""
    grid = param_grid or PARAM_GRIDS[model_name]
    candidates = _param_grid_to_list(grid)
    metric_fn = SELECTION_METRICS[selection_metric]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    rows = []
    for cand in candidates:
        fold_scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            if y_train.nunique() < 2:
                continue  # degenerate fold, skip
            model = build_model(model_name, cand, y_train, random_state=random_state)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            fold_scores.append(metric_fn(y_test.values, y_pred, y_prob))
        mean_score = float(np.mean(fold_scores)) if fold_scores else float("-inf")
        rows.append({**cand, "mean_cv_score": mean_score})

    results_df = pd.DataFrame(rows).sort_values("mean_cv_score", ascending=False).reset_index(drop=True)
    best_row = results_df.iloc[0]
    best_params = {k: best_row[k] for k in grid.keys()}
    # cast numeric params back to int where the grid was integer-valued
    for k, v in best_params.items():
        if isinstance(grid[k][0], int):
            best_params[k] = int(v)
    logger.info("%s best params (selection_metric=%s, score=%.4f): %s", model_name, selection_metric, best_row["mean_cv_score"], best_params)
    return best_params, results_df


def select_decision_threshold(
    X: pd.DataFrame, y: pd.Series, model_name: str, params: dict, n_splits: int, random_state: int = 42,
    thresholds: np.ndarray | None = None,
) -> float:
    """Tunes the probability->class decision threshold on the pre-OOS period
    ONLY (never on the OOS backtest itself), instead of leaving every model
    at the sklearn default of 0.5.

    Why this matters: ROC-AUC / PR-AUC score a model's probability RANKING
    across every possible threshold, but F1/MCC/accuracy are computed at
    whatever threshold turns probabilities into hard 0/1 predictions. A
    model can rank risk well (high ROC-AUC) while still doing worse than a
    trivial rule on hard-decision metrics simply because 0.5 is not where
    that model's probabilities happen to separate the classes best under
    heavy imbalance. This was observed directly on the recession target: a
    fixed persistence baseline is impossible to beat on F1/MCC (it only
    ever outputs a hard 0 or 1), yet XGBoost's ROC-AUC beats it, meaning its
    probabilities do carry usable extra information that a blanket 0.5
    threshold was not exploiting.

    Method: re-run the SAME TimeSeriesSplit folds used by
    time_series_cv_search, but for the one already-chosen `params`, pool
    every fold's held-out (y_true, y_prob) pairs, then pick whichever
    candidate threshold in `thresholds` maximizes F1 over the pooled
    out-of-fold predictions. This is still entirely pre-OOS: nothing from
    the walk-forward backtest period is used to pick the threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    pooled_true, pooled_prob = [], []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if y_train.nunique() < 2:
            continue
        model = build_model(model_name, params, y_train, random_state=random_state)
        model.fit(X_train, y_train)
        pooled_prob.append(model.predict_proba(X_test)[:, 1])
        pooled_true.append(y_test.values)
    if not pooled_true:
        return 0.5
    yt = np.concatenate(pooled_true)
    yp = np.concatenate(pooled_prob)
    if len(set(yt)) < 2:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(yt, (yp > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    logger.info("%s decision threshold selected on pre-OOS pooled folds: %.2f (F1=%.4f at that threshold)", model_name, best_t, best_f1)
    return best_t
