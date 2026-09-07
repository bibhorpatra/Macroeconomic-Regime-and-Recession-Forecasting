"""
labeling.py
===========
Builds the two forecasting targets.

Recession label
---------------
The NBER US recession indicator (USREC), pulled directly from FRED,
reindexed onto the same monthly index as the feature panel. This is
unchanged from the old project (it was correct).

Market-stress label (rebuilt)
------------------------------
The old project derived "crash" periods purely from S&P 500 price via
L1 trend filtering (a cvxpy convex optimization that detects trend
break-points). That's a purely price-based, somewhat black-box signal with
no reference to actual market-stress/volatility indicators, and the
project brief calls this out as a weakness.

This version instead builds an explicit, interpretable RULE that requires
BOTH legs to be true in the same month:
  1. A drawdown leg: S&P 500 is down at least `drawdown_threshold` (e.g.
     -15%) from its trailing `drawdown_window`-month rolling high.
  2. A stress leg: VIX is elevated, strictly above its own trailing
     `vix_threshold_percentile` (e.g. 85th) percentile, computed via an
     EXPANDING (not fixed-window) quantile so the "elevated" bar is set
     relative to all VIX history available up to that month (avoids
     sacrificing an arbitrary multi-year warm-up window purely to a fixed
     lookback, while still being fully point-in-time / leakage-safe: the
     quantile at row t uses only VIX observations at or before t).

Requiring both legs (rather than either alone) is deliberately more
conservative than pure price-based trend filtering: a large price pullback
with LOW VIX (e.g. a slow grind lower) or a VIX spike with no material
drawdown (e.g. a one-day volatility scare) do not by themselves get labeled
"market stress"; both the price action and the market's own fear gauge
have to agree. This ties the label directly to a well-understood, standard
stress proxy (VIX) instead of relying solely on a convex-optimization
artifact of the price series.

Caveat: VIX only exists on FRED from 1990-01, so the market-stress label
(and every model trained on it) is only defined from 1990-01 onward, much
shorter than the recession label's history back to the 1960s. See
README for how this shapes the market-stress train/OOS split.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_recession_label(usrec: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    s = usrec["USREC"].copy()
    s.index = s.index.to_period("M").to_timestamp()
    s = s[~s.index.duplicated(keep="last")]
    aligned = s.reindex(index)
    aligned.name = "EconRegime"
    return aligned


def build_market_stress_label(
    sp500_close: pd.Series,
    vix_monthly: pd.DataFrame,
    drawdown_threshold: float,
    drawdown_window: int,
    vix_threshold_percentile: float,
    vix_min_periods: int,
    index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Returns a DataFrame with the label plus its two component legs (for
    transparency / plotting), indexed like `sp500_close`."""
    sp500_close = sp500_close.copy()
    roll_max = sp500_close.rolling(drawdown_window, min_periods=max(3, drawdown_window // 3)).max()
    drawdown = sp500_close / roll_max - 1.0
    drawdown_flag = (drawdown <= drawdown_threshold).astype(float)

    vix = vix_monthly["VIX_MEAN"].copy()
    vix = vix.reindex(sp500_close.index)
    # Expanding (point-in-time) quantile: at row t, uses only VIX observations
    # up to and including t, so there is no lookahead.
    vix_expanding_q = vix.expanding(min_periods=vix_min_periods).quantile(vix_threshold_percentile)
    # Strict ">" (not ">="): "elevated" should mean genuinely above the
    # historical percentile, not merely at-or-above it. With ">=", a VIX
    # series that happened to be perfectly flat over its whole history would
    # have quantile(q) == every observation, so every month would trivially
    # satisfy "at or above the 85th percentile", a degenerate tie that
    # never actually occurs in the real VIX series (verified empirically:
    # zero rows differ between ">" and ">=" on the real data) but is still
    # the more correct definition and is what the unit tests assume.
    vix_flag = (vix > vix_expanding_q).astype(float)

    label = drawdown_flag * vix_flag
    # Undefined (NaN) wherever either leg is undefined (VIX history too
    # short, or drawdown window not yet full), left explicit rather than silently 0.
    undefined = drawdown.isna() | vix.isna() | vix_expanding_q.isna()
    label = label.where(~undefined, other=pd.NA)

    out = pd.DataFrame({
        "SP500_Drawdown": drawdown,
        "DrawdownFlag": drawdown_flag.where(~drawdown.isna()),
        "VIX_Level": vix,
        "VIX_ExpandingThreshold": vix_expanding_q,
        "VIXFlag": vix_flag.where(~vix.isna()),
        "MktRegime": label,
    })
    if index is not None:
        out = out.reindex(index)
    logger.info(
        "Market-stress label: %d/%d months labeled (rest undefined pre-warm-up), %.1f%% positive (stress) among labeled",
        out["MktRegime"].notna().sum(), len(out),
        100.0 * out["MktRegime"].dropna().mean() if out["MktRegime"].notna().any() else float("nan"),
    )
    return out
