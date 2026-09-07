# Macro-Regime Forecasting

A macro-financial ML pipeline for one-month-ahead recession and market-regime
forecasting using FRED-MD, S&P 500, and VIX data. Two binary targets are
forecast one month ahead:

- **Recession** (`EconRegime`): NBER US recession indicator (`USREC`).
- **Market stress** (`MktRegime`): a rule combining an S&P 500 drawdown with
  an elevated VIX, defined below. It's not based on price alone.

For each target, Decision Tree, Random Forest, and XGBoost are benchmarked
with imbalance-aware hyperparameter search (`TimeSeriesSplit` CV, selecting
on F1 / PR-AUC rather than raw accuracy), then evaluated on a rolling-window
walk-forward out-of-sample backtest. XGBoost's feature importances are
aggregated across every rolling refit and rolled up by macro group and by
engineered-feature type.

This is a from-scratch rebuild of an earlier (Nov-Dec 2024) version of this
project. See "What changed vs. the earlier version" below for what was
fixed and why.

## Results (rolling-window walk-forward OOS backtest)

### Recession (778 months, 1961-09 to 2026-07; 150-month rolling training window; OOS from 1974-06)

| Metric | Decision Tree | Random Forest | XGBoost | Persistence baseline |
|---|---|---|---|---|
| Accuracy | 0.9424 | 0.9552 | 0.9680 | **0.9792** |
| Balanced Accuracy | 0.8257 | 0.9297 | 0.9175 | **0.9489** |
| Precision | 0.7667 | 0.7439 | 0.8529 | **0.8971** |
| Recall | 0.6765 | 0.8971 | 0.8529 | **0.9104** |
| F1 | 0.7188 | 0.8133 | 0.8529 | **0.9037** |
| MCC | 0.6885 | 0.7926 | 0.8350 | **0.8921** |
| ROC-AUC | 0.8256 | 0.9699 | 0.9772 | 0.9489 |

### Market stress (403 months, 1992-12 to 2026-07; 150-month rolling training window; OOS from 2007-01)

| Metric | Decision Tree | Random Forest | XGBoost | Persistence baseline |
|---|---|---|---|---|
| Accuracy | 0.9487 | 0.9573 | **0.9701** | 0.9657 |
| Balanced Accuracy | 0.8484 | 0.7598 | **0.8598** | 0.8575 |
| Precision | 0.5789 | 0.7273 | **0.7857** | 0.7333 |
| Recall | 0.7333 | 0.5333 | 0.7333 | 0.7333 |
| F1 | 0.6471 | 0.6154 | **0.7586** | 0.7333 |
| MCC | 0.6249 | 0.6013 | **0.7432** | 0.7150 |
| ROC-AUC | 0.8484 | 0.8945 | **0.8916** | 0.8575 |

Read the persistence baseline column first. It predicts that this month's
regime is whatever it actually was last month: no macro data, no model,
just the label's own autocorrelation (`evaluation.compute_persistence_baseline`).
Recessions and stress periods are multi-month episodes, so "nothing changed
since last month" is a strong rule at a 1-month horizon, and any serious
evaluation of a short-horizon regime classifier should report it alongside
the model's own metrics rather than showing those metrics in isolation.

For recession, the persistence baseline beats every model on every metric,
including XGBoost (F1 0.904 vs 0.853, MCC 0.892 vs 0.835). That's a real,
negative finding for the models: at this horizon, with this feature set,
none of the three add measurable value over simply assuming the recession
indicator doesn't flip this month. For market stress, XGBoost does beat the
baseline, though only modestly (F1 0.759 vs 0.733, MCC 0.743 vs 0.715) -
a genuine improvement, but smaller than the model-only table would suggest
on its own.

XGBoost is still the best of the three trained models on both targets, on
every imbalance-aware metric, which is why it's the model used for the
feature-attribution analysis below. But "best model" and "beats the
trivial baseline" are different claims, and only market stress supports
the second one here. Full metrics: `results/metrics_recession.csv` and
`results/metrics_market_stress.csv`. Confusion matrices:
`results/confusion_matrices/`. Figures (metric bars, predicted-probability
timelines, importance rollups): `results/figures/`.

### Top predictors, market-stress model (XGBoost, importance aggregated across every rolling refit)

| Feature | Source series | Macro group | Engineered type | Importance |
|---|---|---|---|---|
| `SP500__drawdown12` | SP500 | Stock Market | drawdown | 0.0532 |
| `SP500__mom12` | SP500 | Stock Market | momentum | 0.0445 |
| `DMANEMP__base` | DMANEMP (manuf. employment) | Labor Market | base | 0.0342 |
| `IPMANSICS__mom3` | IPMANSICS (manuf. production) | Output and Income | momentum | 0.0300 |
| `T1YFFM__rollstd24` | T1YFFM (1Y Treasury - Fed Funds spread) | Interest and Exchange Rates | rolling_stat | 0.0271 |
| `TB3SMFFM__lag12` | TB3SMFFM (3M T-bill - Fed Funds spread) | Interest and Exchange Rates | lag | 0.0260 |
| `IPFPNSS__mom3` | IPFPNSS (industrial production, final products) | Output and Income | momentum | 0.0227 |
| `PAYEMS__base` | PAYEMS (nonfarm payrolls) | Labor Market | base | 0.0227 |

By macro group: Labor Market (0.291), Interest and Exchange Rates (0.227),
Stock Market (0.148), Output and Income (0.145), Prices (0.108), Housing
(0.052), Money and Credit (0.028), Consumption/Orders/Inventories (0.001).
By engineered-feature type: momentum (0.298), rolling_stat (0.259),
base/contemporaneous (0.213), lag (0.154), drawdown (0.054), volatility
(0.011), spread (0.010). Full tables: `results/feature_importance_market_stress.csv`,
`data/predictions/market_stress_group_importance.csv`,
`data/predictions/market_stress_type_importance.csv`.

## What changed vs. the earlier version

The earlier (Nov-Dec 2024) version of this project had five specific
weaknesses. Each is addressed below.

1. **Naive feature engineering, replaced with a richer, typed feature set.**
   The old pipeline had only FRED-MD's own stationarity transforms plus raw
   lags (1/3/6/9/12mo): about 118 series times 6 lags, not the "~700+
   features" it was described as. This version adds five new feature
   families on top of the base transform and lags: **spreads** (term
   spreads, credit spreads, a policy-vs-market short-rate spread, 5 total),
   **momentum** (3/6/12mo rate-of-change, computed as `diff()` for any
   series that can be zero or negative and `pct_change()` otherwise; see
   `feature_engineering.add_momentum`), **rolling statistics** (6/12/24mo
   rolling mean, std, and z-score), **drawdowns** (percent off a trailing
   rolling max, at 12/36/60mo windows, for SP500/INDPRO/PAYEMS/HOUST), and
   **volatility** (realized S&P 500 volatility and VIX-based
   volatility/level/change features). Total: 1,880 engineered features
   (103 base, 515 lag, 5 spread, 309 momentum, 927 rolling_stat, 12
   drawdown, 9 volatility), with a full feature-to-(source series, FRED-MD
   macro group, engineered type) mapping persisted at
   `data/datasets/{target}_feature_metadata.csv`.
2. **No real imbalance handling, now imbalance-aware throughout.** The old
   project selected hyperparameters by raw CV accuracy, which under ~7-11%
   positive-class prevalence rewards a model that just predicts the
   majority class. This version selects hyperparameters on F1 (recession)
   or PR-AUC (market stress; see the `config.yaml` comment on why F1/MCC
   tie at an identical, uninformative score for every candidate on this
   target's sparse pre-OOS period). Every model is imbalance-aware:
   `class_weight="balanced"` for Decision Tree and Random Forest, and
   `scale_pos_weight` for XGBoost, recomputed from the current training
   window's class ratio rather than fixed once on the full sample.
3. **Price-only market label, now incorporating VIX as both a label input
   and a feature source.** The old "crash" label was pure L1 trend-filtering
   on SP500 price alone. This version's market-stress label requires both
   an S&P 500 drawdown of 15% or more from its trailing 12-month high and
   VIX strictly above its own trailing, expanding, point-in-time 85th
   percentile (see "Labels" below for the exact rule). VIX is also a
   first-class feature source: level, month-over-month change, and
   3/6/12-month realized volatility of VIX itself.
4. **Feature importance previously reported only by raw macro group, now
   also by engineered type.** The old project stopped at FRED-MD
   group-level importance. `evaluation.aggregate_feature_importance` here
   rolls XGBoost's importances, aggregated across every one of the ~600+
   rolling-window refits, up two ways: by macro group and by engineered
   type (lag, spread, momentum, rolling_stat, drawdown, volatility, base).
   That's how you can see, for example, that momentum and rolling-stat
   features drive the market-stress model more than raw levels or lags do
   (see the table above).
5. **Loose notebooks and no packaging, replaced by this repo.** The old
   project was two `.py` files and a few standalone notebooks, with no
   `requirements.txt`, no README, no saved artifacts, and no tests. This
   version is a proper package: `src/` modules, `config.yaml` for every
   tunable, `requirements.txt` pinned to the environment it was actually
   run in, a `tests/` suite with 13 passing unit tests targeting the
   highest-stakes correctness properties, and every intermediate and final
   artifact persisted under `data/` and `results/`.

What was already correct in the old version, and so was preserved: the
FRED-MD monthly dataset and its per-series stationarity transform codes;
the NBER recession indicator; S&P 500 monthly price; the one-month-ahead
horizon, implemented by shifting features forward
(`feature_engineering.align_features_to_horizon`) rather than shifting the
label backward. Shifting the label backward would be a leakage bug (it
would let the model see the label's own future value indirectly through
unshifted contemporaneous features), so that was deliberately avoided.
Also preserved: a pre-OOS `TimeSeriesSplit` CV period for hyperparameter
selection, with no shuffling (shuffled CV would leak future folds into
past training), and the rolling-window walk-forward OOS backtest
methodology - a 150-month trailing window, retrain every step, predict
only the single next unseen month, roll forward.

Two bugs turned up in the FRED-MD transform-code math itself while
rebuilding this (see `src/feature_engineering.py::apply_transform_codes`
and the regression tests in `tests/test_leakage_and_alignment.py`): code 3
(second difference) has to be `diff().diff()`, not `diff(periods=2)`,
which skips the intermediate period and produces a materially different,
wrong quantity; and code 5 (single-period log-difference) has to be
`log(x).diff()`, not `log(x).diff(periods=2)`.

## Data sources, date ranges, and vintage

Pulled **2026-09-06** (see `config.yaml: data.vintage_pull_date`). All raw
pulls are persisted, untouched, under `data/raw/`. See "Honest caveats"
below for exactly how each was obtained in this environment.

| Source | Series | Range in this pull | Where |
|---|---|---|---|
| FRED-MD macro panel | ~110 of the standard ~128 FRED-MD series (see below) | 1871-01 to 2026-09 (raw); 1960-01 to 2026-07 after cleaning | `data/raw/fredmd_current.csv` |
| NBER recession indicator | `USREC` | full FRED history | `data/raw/nber_usrec.csv` |
| VIX | `VIXCLS` | 1990-01-02 daily, resampled to monthly | `data/raw/vix_daily.csv`, `data/raw/vix_monthly.csv` |
| S&P 500 | monthly close (see caveat below) | 1871-01 to 2026-08 | `data/raw/sp500_monthly.csv` |

FRED-MD series coverage in this pull (`data/raw/fredmd_ingestion_log.csv`):
101 series pulled directly by their standard FRED ticker, 8 reconstructed
from raw FRED levels (term and credit spreads that FRED-MD itself derives
rather than sourcing directly), 1 pulled via a documented substitute
ticker, 11 dropped for insufficient history in their only free substitute
(listed in `src/fredmd_spec.py::DROPPED_SHORT_HISTORY` with the reason for
each), and 5 dropped for having no free source at all
(`DROPPED_NO_FREE_SOURCE`: `HWI`, `HWIURATIO`, `SP_INDUST`, `SP_DIV_YIELD`,
`SP_PE_RATIO`, all proprietary Conference Board / S&P index-constituent
series).

## Cleaning

`src/data_cleaning.py::DataCleaning` drops any row (month) with more than
`cleaning.max_null_rows` (6) null series, drops any column (series) with
more than `cleaning.max_null_cols` (10) null months, then forward-fills
remaining gaps. Both thresholds and the resulting drop log are in
`config.yaml` and `data/processed/cleaning_log.txt`. Result: 798 rows by
103 columns, 1960-01 to 2026-07, zero remaining nulls after forward-fill.

One residual gap is real, not a bug: 2025-10 is missing from the final
modeling datasets because 21 labor-market/CPI series were null that month,
since the 2025 US government shutdown delayed that month's BLS data
releases. The cleaning threshold correctly excludes that row rather than
forward-filling over a genuine reporting gap.

## Feature engineering (exact definitions)

All features are computed from the FRED-MD transformed (stationary)
series unless noted; "levels" below means the pre-transform raw series.

- **Base**: the series' own FRED-MD stationarity transform (code 1-7; see
  `TRANSFORM_CODE_MEANING` in `src/fredmd_spec.py`), untouched.
- **Lags**: `x.shift(n)` for n in {1, 3, 6, 9, 12} months, on the
  transformed series.
- **Spreads** (on raw levels): `TERM_SPREAD_10Y3M` = GS10 - TB3MS,
  `TERM_SPREAD_10YFF` = GS10 - FEDFUNDS, `CREDIT_SPREAD_BAA_AAA` = BAA - AAA,
  `CREDIT_SPREAD_BAA_10Y` = BAA - GS10, `FF_TBILL_SPREAD` = FEDFUNDS - TB3MS.
- **Momentum**: `x.diff(w)` for w in {3, 6, 12} months if the series ever
  takes a non-positive value historically (spreads, rates that can hit
  zero), otherwise `x.pct_change(w)`. This is decided empirically per
  series rather than by transform code, specifically to avoid
  divide-by-near-zero blowups in spread-like series (see the regression
  tests in `tests/test_leakage_and_alignment.py`).
- **Rolling stats**: rolling mean, std, and z-score
  (`(x - rollmean) / rollstd`, set to 0 rather than NaN/inf when
  `rollstd == 0`, which happens for example during WTI spot price's
  regulated, genuinely flat stretches in the early 1970s), over windows
  {6, 12, 24} months.
- **Drawdowns**: `x / x.rolling(w).max() - 1` for SP500, INDPRO, PAYEMS,
  HOUST (levels), at windows {12, 36, 60} months.
- **Volatility**: S&P 500 realized volatility (rolling std of monthly
  returns) at windows {3, 6, 12}; VIX level, month-over-month change, and
  its own rolling std at the same windows.

**Horizon alignment (leakage guard):** every feature is shifted forward by
the 1-month forecast horizon (`align_features_to_horizon`), so the row
labeled for month *t* contains only information available at month *t-1*.
The label itself is never shifted; shifting it backward instead would be a
leakage bug, and there's a dedicated regression test for that in
`tests/test_leakage_and_alignment.py`.

**Recession-target caveat:** VIX only exists from 1990-01 onward. Including
VIX-derived features in the recession model would truncate its ~65-year
usable history down to the VIX era, discarding five recessions (1960,
1969-70, 1973-75, 1980, 1981-82) from both training and OOS evaluation. So
the recession model excludes all VIX-derived columns, while the
market-stress model, whose own label already depends on VIX and whose
usable sample already starts in 1992, keeps them.

## Labels

- **Recession**: `USREC`, reindexed onto the modeling calendar, unchanged
  from the earlier version since it was already correct.
- **Market stress**: `MktRegime = DrawdownFlag AND VIXFlag`, both legs
  required in the same month:
  - `DrawdownFlag`: S&P 500 close is down 15% or more from its trailing
    12-month rolling high.
  - `VIXFlag`: VIX's monthly mean is strictly above its own trailing,
    expanding (not fixed-window), point-in-time 85th percentile, computed
    only from VIX observations at or before that month so the "elevated"
    bar can never see the future, and requiring at least 36 months of VIX
    history before the flag is defined.

  Requiring both legs, rather than either alone or price alone as in the
  earlier version, is deliberately conservative: a slow drift lower with a
  calm VIX, or a one-day VIX scare with no material drawdown, don't get
  labeled "stress." Both the price action and the market's own fear gauge
  have to agree. The label is undefined (NaN, not silently 0) wherever
  either leg lacks sufficient history. Result: 403 labeled months (1992-12
  to 2026-07), 29 positive (7.2%).

## Modeling

For each target and each of Decision Tree, Random Forest, and XGBoost:

1. **Hyperparameter search** on the pre-OOS period only (`TimeSeriesSplit`,
   no shuffling), selecting on F1 (recession) or PR-AUC (market stress;
   see `config.yaml`'s comment on why F1/accuracy/MCC tie at an identical,
   uninformative score for every candidate on this target's very sparse,
   14-positive-month, pre-OOS period). Grids: `src/modeling.py::PARAM_GRIDS`.
2. **Imbalance handling**: `class_weight="balanced"` for DT/RF;
   `scale_pos_weight` for XGBoost, equal to the negative-to-positive ratio
   of the current training window, recomputed at every single fit, both
   during CV search and during the rolling backtest below, never a fixed
   constant computed once on the full sample.
3. **Rolling-window walk-forward OOS backtest** (`src/backtest.py`): a
   150-month trailing window trains a fresh model, predicts the single
   next, strictly out-of-sample month, then rolls forward by one month and
   repeats for every remaining month in the sample. The model never sees
   the row it's predicting, or any row after it.

Recession pre-OOS/CV split: before 1974-06 (3 `TimeSeriesSplit` folds).
Market-stress pre-OOS/CV split: before 2007-01 (2 folds; 3 folds would put
the period's only pre-split stress episode, the 2001-03 dot-com bust,
entirely inside a single test fold every time, making every candidate
score 0 on every fold). Both choices, and the diagnostic that found the
market-stress issue, are documented in `config.yaml`.

## What I tried that didn't work

After finding that a naive persistence baseline (predict this month = last
month, no macro data at all) beats every model on the recession target and
edges out even XGBoost on some metrics for market stress (see the Results
table above), I tried two natural fixes. Both are implemented and
available (`run_target(..., add_autoregressive_feature=True,
tune_decision_threshold=True)`), but both are off by default, because both
empirically made the rolling-OOS metrics worse rather than better. This
section keeps that on record rather than deleting the code, since a tested
negative result is worth more than a positive one that was never checked.

**1. Autoregressive feature.** Added the target's own actual prior-month
regime as one more input feature (`src/pipeline.py`, gated by
`add_autoregressive_feature`), a standard ARX-style lagged-dependent-
variable term, leakage-safe by construction since it goes through the same
forward-shift as every other feature. The hope was that the model could
combine this persistence signal with the macro features and do at least as
well as pure persistence, plus do better at turning points. Instead:
recession XGBoost F1 dropped from 0.853 to 0.827 (MCC from 0.835 to
0.807); market-stress XGBoost F1 dropped from 0.759 to 0.690 (MCC from
0.743 to 0.670). Both got worse. Working theory: with only ~150-170
pre-OOS rows to select hyperparameters on, the grid search is already
picking among near-tied, noisy candidates (the recession target's own best
CV F1 scores during search were 0.0-0.27 out of 1.0, barely informative).
Adding one more feature shifts which specific hyperparameter combination
wins that noisy race, and a shallow tree given a near-perfect
single-feature shortcut may lean on it early in each tree at the expense
of the depth budget it would otherwise spend combining the genuinely
informative macro features that, apparently, work better on their own for
this dataset and model family.

**2. Decision-threshold tuning.** ROC-AUC, a threshold-free ranking
metric, shows the recession XGBoost model's probabilities do carry more
information than persistence (0.977 vs. 0.949), even though its default
0.5-cutoff hard predictions score worse on F1/MCC. That gap suggested the
fix might just be picking a better cutoff. I tuned it
(`src/modeling.py::select_decision_threshold`, gated by
`tune_decision_threshold`) by pooling out-of-fold probabilities across the
same pre-OOS `TimeSeriesSplit` folds used for hyperparameter search and
picking whichever of 19 threshold candidates maximized F1 there, still
entirely pre-OOS, no leakage. Result: worse across the board for both
targets (recession XGBoost F1 dropped to 0.805, market-stress XGBoost F1
dropped to 0.470 when combined with the AR feature above). The threshold
that looked best on ~150-170 pre-OOS rows (2-3 folds) didn't generalize to
the 234-625-row OOS backtest: a textbook case of overfitting a
hyperparameter, here the threshold, to too small a validation set.

Neither attempted fix reliably closes the gap with the persistence
baseline given this project's sample sizes and search procedure. That's a
legitimate finding on its own: it suggests short-horizon regime
persistence is a genuinely hard bar to clear here, rather than a sign of
an unoptimized pipeline. A more promising direction not yet attempted:
nested or expanding-window threshold selection, re-tuning per
rolling-backtest step using only data available at that step, rather than
once on the whole pre-OOS period, or using a longer pre-OOS period
specifically for threshold selection.

## Feature ablation

The per-window importance breakdown (see "Top predictors" above and
`results/feature_importance_{target}.csv`) showed some lag windows
carrying much less weight than others. For recession, the 6-month lag
alone accounts for 58% of everything the "lag" feature family
contributes, while windows 3 and 12 contribute only 8.1% and 7.3%
respectively. That raised an obvious question: would dropping the weakest
windows lose nothing and simplify the model? Tested directly, via a config
override rather than a change to the shipped `config.yaml`, `results/`,
or `data/predictions/` (this was a side-by-side comparison run, not a
replacement of the shipped numbers):

| | Recession XGBoost | Market-stress XGBoost |
|---|---|---|
| Full config (`lag_windows: [1,3,6,9,12]`, 1,880 features) | F1=0.853, MCC=0.835 | F1=0.759, MCC=0.743 |
| Reduced config (weakest 2 windows dropped per target, 1,674 features) | F1=0.818, MCC=0.797 | F1=0.690, MCC=0.670 |

Recession dropped windows 3 and 12 (`lag_windows: [1,6,9]`); market-stress
dropped windows 3 and 9 (`lag_windows: [1,6,12]`), its own weakest per the
same per-window breakdown. Both got worse, on both targets, across nearly
every metric, not just XGBoost: DT and RF also declined on most metrics in
both runs. It wasn't a close call in either direction.

Not adopted. `config.yaml` is unchanged; all five lag windows stay.

Why, despite the low importance-share diagnostic that motivated the test:
a feature's importance share mostly measures redundancy with its
correlated siblings, not whether it's worthless. A lag window can still
carry small, genuinely non-redundant information at specific decision
boundaries that the tree ensemble uses in combination with its correlated
neighbors, so removing it doesn't cleanly hand its value to the remaining
windows; it can just delete information outright. There's also a
compounding effect already documented above in "What I tried that didn't
work": with only 153-169 pre-OOS rows to select hyperparameters on,
changing the feature set shifts which specific hyperparameter combination
wins that already-noisy search, which can move the final OOS numbers for
reasons unrelated to whether the dropped features were valuable. Either
way, the tested result is unambiguous: pruning these windows hurt, so they
stay. That makes three consistent data points now, after the autoregressive
feature and the decision-threshold tuning, that this pipeline's
feature/hyperparameter surface is more sensitive to small changes than the
individual-feature importance numbers alone would suggest.

## Evaluation & feature attribution

All 7 metrics (accuracy, balanced accuracy, precision, recall, F1, MCC,
ROC-AUC) plus a confusion matrix are computed on the full rolling-OOS
backtest for every model and target combination (`results/metrics_*.csv`,
`results/confusion_matrices/`). For XGBoost, `feature_importances_` is
collected at every one of the rolling refits and averaged
(`evaluation.aggregate_feature_importance`), then rolled up three ways:
top individual features, by FRED-MD macro group, and by engineered-feature
type (`results/feature_importance_*.csv`,
`data/predictions/*_group_importance.csv`, `*_type_importance.csv`,
`results/figures/*_importance_by_group.png`, `*_importance_by_type.png`).

## Honest caveats

**Statistical significance and sample size.** The recession OOS backtest
covers only 7 distinct recession episodes over roughly 52 years; the
market-stress OOS backtest covers only 4 distinct stress episodes over
roughly 19 years (the count of positive months is not the same as the
count of independent events here; see the notebook for the episode-count
computation). With that few independent events, a single differently-timed
or differently-shaped episode could move MCC or F1 by a meaningful amount.
Treat the point estimates above as indicative rather than precise. The
more important caveat is still the persistence baseline discussion above:
for recession specifically, none of the three trained models beat a rule
that uses no macro data at all.

This project was also built in a sandboxed environment with restricted
network egress, which shaped how a few data sources were obtained:

- **FRED-MD's own bundled `current.csv`**, the canonical source, returns
  HTTP 403 from this environment (S3/CloudFront bot protection), and the
  Wayback Machine mirror is blocked by the sandbox's egress policy.
  Instead, the panel is reconstructed live, series by series, from FRED's
  own public `fredgraph.csv` endpoint, the same underlying data FRED-MD
  itself is built from. See `src/fredmd_spec.py` for the complete,
  explicit per-series mapping, and the "FRED-MD series coverage" table
  above for what was dropped and why. This is a faithful reconstruction,
  not a fallback with faked numbers, but it isn't byte-identical to the
  official FRED-MD release, and a handful of series (see
  `DROPPED_SHORT_HISTORY` / `DROPPED_NO_FREE_SOURCE`) are absent that the
  official file would include.
- **S&P 500**: FRED's own `SP500` series is license-restricted to the
  trailing ~10 years. Standard historical vendors (Stooq, Yahoo Finance,
  macrotrends) were blocked or JS-gated in this environment, so the full
  1871-present history was instead pulled from multpl.com's monthly-close
  table (server-rendered HTML, no JS challenge encountered). That gives
  monthly close only, not true OHLC, so every place that would otherwise
  use intra-month high/low, such as a max-drawdown-within-month feature,
  uses close-to-close returns instead.
- **Network flakiness inside long-lived Python processes**: this
  sandbox's outbound requests intermittently timed out when made
  repeatedly from a single Python process, for both `requests` and
  `subprocess`-invoked `curl`, while a fresh `curl` invocation from the
  shell was completely reliable. The practical workaround was pre-fetching
  every needed series with `fetch_raw_data.sh`, caching it under
  `data/raw/fred_components/`, and having `data_ingestion.py` check that
  cache before attempting any network call. That's documented in
  `src/data_ingestion.py`'s module docstring, and it's a reasonable
  reproducibility practice on its own merits, separate from the workaround
  it originated from.
- **A genuinely negative-valued FRED-MD series**: `NONBORRES` (nonborrowed
  reserves) went sharply negative during the Fed's 2008-09 emergency
  lending, which breaks FRED-MD's published log-based transform for that
  series (code 6), since the log of a negative number is undefined. That
  propagated as NaN through every downstream lag and rolling feature,
  which in turn silently dropped the entire 2008-09 window from both
  datasets via the NaN-drop step. An explicit index-gap check caught this,
  and the fix was overriding `NONBORRES`'s transform to a first difference
  (code 2, well-defined for negative values) instead of the published code
  6; see the comment in `src/fredmd_spec.py`. Every other series was
  checked for the same failure mode, and none had it.
- **All numbers in this README and in `results/` are real, computed
  outputs of an actual, executed run of this pipeline** in this sandbox on
  the vintage date above, not hypothetical or hand-estimated. A re-run with
  unrestricted network access (via `fetch_raw_data.sh --force` or
  `src/data_ingestion.py` directly) would likely recover the small number
  of dropped series and could shift the numbers slightly, but shouldn't
  change their order of magnitude or the qualitative conclusions above.

## Project layout

```
macro-regime-forecasting/
├── README.md
├── requirements.txt
├── config.yaml                  # every tunable: windows, thresholds, splits, metrics
├── fetch_raw_data.sh             # one-time raw data pull (curl-based, see caveats above)
├── data/
│   ├── raw/                      # untouched raw pulls + ingestion log
│   ├── processed/                # cleaned + transformed macro panel
│   ├── datasets/                 # final per-target modeling datasets + feature metadata
│   └── predictions/               # CV search results, best params, OOS predictions, importance rollups
├── src/
│   ├── fredmd_spec.py            # FRED-MD series spec, substitutions, drops
│   ├── data_ingestion.py         # raw data pulls
│   ├── data_cleaning.py          # DataCleaning class
│   ├── feature_engineering.py    # lags/spreads/momentum/rolling stats/drawdowns/volatility + horizon alignment
│   ├── labeling.py                # recession + market-stress label construction
│   ├── modeling.py                # CV search, imbalance-aware model construction
│   ├── backtest.py                # rolling-window walk-forward OOS backtest
│   ├── evaluation.py              # metrics, confusion matrices, feature-importance rollups
│   └── pipeline.py                # end-to-end orchestration (single source of truth)
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_recession_model.ipynb
│   └── 03_market_stress_model.ipynb
├── results/
│   ├── metrics_recession.csv, metrics_market_stress.csv
│   ├── feature_importance_recession.csv, feature_importance_market_stress.csv
│   ├── confusion_matrices/
│   └── figures/
└── tests/
    └── test_leakage_and_alignment.py
```

## How to run

```bash
pip install -r requirements.txt --break-system-packages   # or: pip install -r requirements.txt

# 1. Pull raw data (only needed once; reuses data/raw/fred_components/ cache if present)
python3 src/data_ingestion.py
# (or, if the cache is empty and requests-based fetch is unreliable in your environment:)
./fetch_raw_data.sh && python3 src/data_ingestion.py

# 2. Run the full pipeline for both targets
cd src && python3 pipeline.py

# 3. Or run the notebooks directly (equivalent, with inline plots/tables)
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

# 4. Run the test suite
pytest tests/ -v
```

All outputs (`data/processed/`, `data/datasets/`, `data/predictions/`,
`results/`) are regenerated deterministically (`modeling.random_state: 42`
in `config.yaml`) by step 2 or 3 above.
