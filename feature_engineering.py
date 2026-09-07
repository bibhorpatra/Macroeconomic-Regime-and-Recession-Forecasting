"""
feature_engineering.py
=======================
The core "fix" of this rebuild. The old project applied FRED-MD's
stationarity transforms and then only added raw multi-horizon lags: no
spreads, momentum, rolling statistics, drawdowns, or volatility. This module
adds all of them, on top of correctly-implemented FRED-MD transform codes,
and returns a feature -> (source series, macro group, engineered-feature
type) mapping table alongside every feature, for later attribution.

Leakage discipline
-------------------
Every transform in this module (`.diff()`, `.pct_change()`, `.rolling()`,
`.shift()`) is backward-looking by construction: the value written at row t
uses only data at or before t. Aligning features to the t+h label is the one
place leakage could creep in, and it is deliberately not done here; it is
done once, explicitly, in `align_features_to_horizon()` at the bottom of
this file, by shifting FEATURES forward rather than shifting the label
backward, per the project brief.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RATE_LIKE_TCODES = {2}  # series already expressed as a rate/spread/level-diff; can be negative/near-zero


# ---------------------------------------------------------------------------
# 1. FRED-MD stationarity transforms (corrected)
# ---------------------------------------------------------------------------
def apply_transform_codes(levels: pd.DataFrame, transform_codes: dict[str, float]) -> pd.DataFrame:
    """Apply the FRED-MD transform code (1-7) to each column of `levels`.

    NOTE on a bug fixed vs. the old project's implementation: the old code
    used `.diff(periods=2)` for codes 3, 5 AND 6, which is wrong for codes 5
    and 6 (a first-difference-of-log should be a *single*-period diff of
    log(x), and a second-difference should be diff(diff(x)), not x(t)-x(t-2)
    which skips the intermediate period entirely). This implementation uses
    the textbook (McCracken & Ng, 2016) definitions.
    """
    out = {}
    for col in levels.columns:
        code = int(transform_codes.get(col, 1))
        x = levels[col]
        if code == 1:
            out[col] = x
        elif code == 2:
            out[col] = x.diff()
        elif code == 3:
            out[col] = x.diff().diff()
        elif code == 4:
            out[col] = np.log(x.where(x > 0))
        elif code == 5:
            out[col] = np.log(x.where(x > 0)).diff()
        elif code == 6:
            out[col] = np.log(x.where(x > 0)).diff().diff()
        elif code == 7:
            out[col] = x.pct_change().diff()
        else:
            logger.warning("Unknown transform code %s for %s -- using level", code, col)
            out[col] = x
    return pd.DataFrame(out, index=levels.index)


# ---------------------------------------------------------------------------
# 2. Multi-horizon lags (kept from the old version)
# ---------------------------------------------------------------------------
def add_lags(transformed: pd.DataFrame, lag_windows: list[int]) -> pd.DataFrame:
    lagged = {}
    for col in transformed.columns:
        for n in lag_windows:
            lagged[f"{col}__lag{n}"] = transformed[col].shift(n)
    return pd.DataFrame(lagged, index=transformed.index)


# ---------------------------------------------------------------------------
# 3. Spreads (new)
# ---------------------------------------------------------------------------
def add_spreads(levels: pd.DataFrame, spread_specs: list[dict]) -> pd.DataFrame:
    out = {}
    for spec in spread_specs:
        name, a, b = spec["name"], spec["minuend"], spec["subtrahend"]
        if a not in levels.columns or b not in levels.columns:
            logger.warning("Spread %s skipped: %s or %s not present in cleaned panel", name, a, b)
            continue
        out[name] = levels[a] - levels[b]
    return pd.DataFrame(out, index=levels.index)


# ---------------------------------------------------------------------------
# 4. Momentum / rate-of-change over multiple windows (new)
# ---------------------------------------------------------------------------
def add_momentum(levels: pd.DataFrame, transform_codes: dict[str, float], momentum_windows: list[int]) -> pd.DataFrame:
    """Rate-of-change on top of the RAW level, distinct from the single-
    period FRED-MD transform. Percent change is only meaningful for series
    that are strictly positive throughout their history (indices, price
    levels, counts); for anything that can be zero or negative, such as interest
    rate SPREADS in particular (e.g. the reconstructed *FFM series, which
    can cross zero), where a pct_change would blow up or flip sign
    nonsensically near zero, we use a simple point change over the window
    (x(t) - x(t-w)) instead. This is decided empirically per-series (any
    non-positive historical value routes to the point-change branch) rather
    than off the FRED-MD transform code alone, since a handful of tcode=1
    ("no transform") series are themselves already rate spreads."""
    out = {}
    for col in levels.columns:
        x = levels[col]
        use_diff = bool((x.dropna() <= 0).any())
        for w in momentum_windows:
            if use_diff:
                out[f"{col}__mom{w}"] = x.diff(w)
            else:
                out[f"{col}__mom{w}"] = x.pct_change(w).replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(out, index=levels.index)


# ---------------------------------------------------------------------------
# 5. Rolling statistics: mean, std, z-score (new)
# ---------------------------------------------------------------------------
def add_rolling_stats(transformed: pd.DataFrame, rolling_windows: list[int]) -> pd.DataFrame:
    """Computed on the already-stationary (transformed) series. Rolling
    mean/std of a nonstationary raw level would just track the trend."""
    out = {}
    for col in transformed.columns:
        x = transformed[col]
        for w in rolling_windows:
            roll_mean = x.rolling(w, min_periods=max(3, w // 2)).mean()
            roll_std = x.rolling(w, min_periods=max(3, w // 2)).std()
            out[f"{col}__rollmean{w}"] = roll_mean
            out[f"{col}__rollstd{w}"] = roll_std
            # A zero-std window means the series was exactly flat over the
            # whole window (e.g. WTI oil price under pre-1973 regulation), so
            # x_t then also equals that constant mean, so the deviation is
            # genuinely zero; fill 0/0 as 0 rather than leaving NaN (which
            # would otherwise silently drop an entire row later on).
            safe_std = roll_std.replace(0, np.nan)
            zscore = (x - roll_mean) / safe_std
            zscore = zscore.where(roll_std != 0, 0.0)
            out[f"{col}__rollz{w}"] = zscore
    return pd.DataFrame(out, index=transformed.index)


# ---------------------------------------------------------------------------
# 6. Drawdowns: % off trailing rolling max (new)
# ---------------------------------------------------------------------------
def add_drawdowns(levels: pd.DataFrame, source_series: list[str], drawdown_windows: list[int]) -> pd.DataFrame:
    out = {}
    for col in source_series:
        if col not in levels.columns:
            logger.warning("Drawdown source %s not present in panel -- skipped", col)
            continue
        x = levels[col]
        for w in drawdown_windows:
            roll_max = x.rolling(w, min_periods=max(3, w // 3)).max()
            out[f"{col}__drawdown{w}"] = x / roll_max - 1.0
    return pd.DataFrame(out, index=levels.index)


# ---------------------------------------------------------------------------
# 7. Volatility: realized vol of returns (new)
# ---------------------------------------------------------------------------
def add_volatility(sp500_close: pd.Series, vix_monthly: pd.DataFrame, volatility_windows: list[int]) -> pd.DataFrame:
    out = {}
    sp_ret = sp500_close.pct_change()
    out["SP500_RETURN"] = sp_ret
    for w in volatility_windows:
        out[f"SP500__realizedvol{w}"] = sp_ret.rolling(w, min_periods=max(2, w // 2)).std()

    if vix_monthly is not None and "VIX_MEAN" in vix_monthly.columns:
        vix = vix_monthly["VIX_MEAN"]
        out["VIX_LEVEL"] = vix
        vix_chg = vix.diff()
        out["VIX_CHANGE"] = vix_chg
        for w in volatility_windows:
            out[f"VIX__vol{w}"] = vix_chg.rolling(w, min_periods=max(2, w // 2)).std()
    return pd.DataFrame(out, index=sp500_close.index)


# ---------------------------------------------------------------------------
# Feature metadata (for attribution) + orchestration
# ---------------------------------------------------------------------------
@dataclass
class FeatureBuildResult:
    features: pd.DataFrame
    metadata: pd.DataFrame  # columns: feature, source_series, group, feat_type


def _series_group_lookup(series_spec: pd.DataFrame) -> dict[str, str]:
    return dict(zip(series_spec["ID"], series_spec["Group"]))


def build_feature_panel(
    cleaned_levels: pd.DataFrame,
    transform_codes: dict[str, float],
    series_spec: pd.DataFrame,
    sp500_close: pd.Series,
    vix_monthly: pd.DataFrame,
    lag_windows: list[int],
    momentum_windows: list[int],
    rolling_windows: list[int],
    drawdown_windows: list[int],
    volatility_windows: list[int],
    spread_specs: list[dict],
    drawdown_source_series: list[str],
) -> FeatureBuildResult:
    """Builds the full engineered feature panel + a feature->(source,group,type)
    metadata table, from the cleaned raw-level macro panel plus SP500/VIX."""
    group_lookup = _series_group_lookup(series_spec)

    transformed = apply_transform_codes(cleaned_levels, transform_codes)

    base = transformed.copy()
    base.columns = [f"{c}__base" for c in base.columns]

    lags = add_lags(transformed, lag_windows)
    spreads = add_spreads(cleaned_levels, spread_specs)
    momentum = add_momentum(cleaned_levels, transform_codes, momentum_windows)
    rolling = add_rolling_stats(transformed, rolling_windows)
    drawdowns = add_drawdowns(cleaned_levels, drawdown_source_series, drawdown_windows)
    volatility = add_volatility(sp500_close, vix_monthly, volatility_windows)

    all_blocks = [base, lags, spreads, momentum, rolling, drawdowns, volatility]
    features = pd.concat(all_blocks, axis=1)
    features = features.sort_index()

    # --- build metadata: feature -> (source_series, group, feat_type) ---
    meta_rows = []

    def _source_series_of(feat_name: str) -> str:
        for tag in ("__base", "__lag", "__mom", "__rollmean", "__rollstd", "__rollz", "__drawdown"):
            if tag in feat_name:
                return feat_name.split(tag)[0]
        return feat_name

    def _group_of(source: str) -> str:
        if source in group_lookup:
            return group_lookup[source]
        if source in ("SP500", "SP500_CLOSE", "SP500_RETURN"):
            return "Stock Market"
        if source.startswith("VIX"):
            return "Stock Market"
        return "Engineered / Cross-series"

    for col in base.columns:
        src = _source_series_of(col)
        meta_rows.append({"feature": col, "source_series": src, "group": _group_of(src), "feat_type": "base"})
    for col in lags.columns:
        src = _source_series_of(col)
        meta_rows.append({"feature": col, "source_series": src, "group": _group_of(src), "feat_type": "lag"})
    for col in spreads.columns:
        meta_rows.append({"feature": col, "source_series": col, "group": "Interest and Exchange Rates", "feat_type": "spread"})
    for col in momentum.columns:
        src = _source_series_of(col)
        meta_rows.append({"feature": col, "source_series": src, "group": _group_of(src), "feat_type": "momentum"})
    for col in rolling.columns:
        src = _source_series_of(col)
        meta_rows.append({"feature": col, "source_series": src, "group": _group_of(src), "feat_type": "rolling_stat"})
    for col in drawdowns.columns:
        src = _source_series_of(col)
        meta_rows.append({"feature": col, "source_series": src, "group": _group_of(src), "feat_type": "drawdown"})
    for col in volatility.columns:
        meta_rows.append({"feature": col, "source_series": col, "group": "Stock Market", "feat_type": "volatility"})

    metadata = pd.DataFrame(meta_rows).drop_duplicates(subset="feature").set_index("feature")
    metadata = metadata.loc[[c for c in features.columns if c in metadata.index]]

    logger.info(
        "Built %d engineered features (base=%d, lag=%d, spread=%d, momentum=%d, rolling_stat=%d, drawdown=%d, volatility=%d)",
        features.shape[1], base.shape[1], lags.shape[1], spreads.shape[1], momentum.shape[1], rolling.shape[1], drawdowns.shape[1], volatility.shape[1],
    )
    return FeatureBuildResult(features=features, metadata=metadata)


# ---------------------------------------------------------------------------
# Horizon alignment (leakage-critical): shift FEATURES forward, not the label
# ---------------------------------------------------------------------------
def align_features_to_horizon(features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Shifts every feature column forward by `horizon` rows so that the
    value stored at row t is the value that was actually observed at
    t - horizon. When later joined against a label that is NOT shifted, row
    t then pairs "information available at t-horizon" with "the outcome at
    t", i.e. exactly a horizon-step-ahead forecast, and it is built by
    moving the features (as the project brief requires), not by shifting the
    label backward, which is easy to get backwards accidentally."""
    return features.shift(horizon)
