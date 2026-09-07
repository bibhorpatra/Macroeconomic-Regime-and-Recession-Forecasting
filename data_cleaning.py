"""
data_cleaning.py
=================
Reimplements (and fixes/configures) the old project's `DataCleaning` class.

What changed vs. the old version
---------------------------------
* Thresholds are configurable (via config.yaml) instead of hardcoded at the
  call site, and every drop is logged (row/column counts, which columns).
* The FRED-MD "Transform:" code row is handled explicitly and kept separate
  from the data rows throughout, instead of being sliced off ad hoc in a
  notebook cell.
* Forward-fill is applied only *within* the surviving date range (no
  fabricated backward-fill of leading NaNs), and remaining leading NaNs are
  left as NaN for the caller to handle explicitly (avoids silently seeding
  a series' history with a forward-filled value that never actually
  occurred).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


def load_fredmd_raw(path: str) -> tuple[pd.DataFrame, dict[str, float]]:
    """Load a FRED-MD-style raw CSV (first data row = transform codes) and
    split it into (data_df indexed by Date, transform_codes dict)."""
    raw = pd.read_csv(path)
    tcode_row = raw.iloc[0]
    transform_codes = {col: float(tcode_row[col]) for col in raw.columns if col != "sasdate"}
    data = raw.iloc[1:].copy()
    data["sasdate"] = pd.to_datetime(data["sasdate"])
    data = data.rename(columns={"sasdate": "Date"}).set_index("Date")
    data = data.apply(pd.to_numeric, errors="coerce")
    return data, transform_codes


@dataclass
class DataCleaning:
    """Configurable null-thresholding + forward-fill cleaner.

    Attributes:
        data: the DataFrame being cleaned (mutated in place by each step
            when inplace=True, matching the old API).
        log_: list of human-readable strings describing every drop made,
            for inclusion in the README / run logs.
    """

    data: pd.DataFrame
    log_: list[str] = field(default_factory=list)

    def remove_null_rows(self, max_null: int, inplace: bool = True) -> pd.DataFrame:
        null_counts = self.data.isnull().sum(axis=1)
        keep_mask = null_counts <= max_null
        n_dropped = (~keep_mask).sum()
        cleaned = self.data[keep_mask]
        msg = (f"remove_null_rows(max_null={max_null}): dropped {n_dropped}/{len(self.data)} rows "
               f"(date range kept: {cleaned.index.min().date()} to {cleaned.index.max().date()})")
        logger.info(msg)
        self.log_.append(msg)
        if inplace:
            self.data = cleaned
        return cleaned

    def remove_null_features(self, max_null: int, inplace: bool = True) -> pd.DataFrame:
        null_counts = self.data.isnull().sum(axis=0)
        keep_cols = null_counts[null_counts <= max_null].index.tolist()
        dropped_cols = [c for c in self.data.columns if c not in keep_cols]
        cleaned = self.data[keep_cols]
        msg = (f"remove_null_features(max_null={max_null}): dropped {len(dropped_cols)}/{self.data.shape[1]} "
               f"columns: {dropped_cols}")
        logger.info(msg)
        self.log_.append(msg)
        if inplace:
            self.data = cleaned
        return cleaned

    def fill_null_obs(self, inplace: bool = True) -> pd.DataFrame:
        n_null_before = int(self.data.isnull().sum().sum())
        filled = self.data.ffill()
        n_null_after = int(filled.isnull().sum().sum())
        msg = f"fill_null_obs: forward-filled {n_null_before - n_null_after} values ({n_null_after} remain null, typically leading-edge NaNs before a series' inception)"
        logger.info(msg)
        self.log_.append(msg)
        if inplace:
            self.data = filled
        return filled

    def summary(self) -> str:
        return "\n".join(self.log_)
