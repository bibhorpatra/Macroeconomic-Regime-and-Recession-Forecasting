"""
data_ingestion.py
==================
Pulls every raw data source this project needs and writes them, untouched,
under `data/raw/`.

Sources & honesty notes
------------------------
1. FRED-MD macro panel.
   The official bundled file
   (https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv)
   returns HTTP 403 from this sandboxed environment (CloudFront/S3 bot
   protection) and the Wayback Machine is blocked by this sandbox's egress
   policy, so neither is reachable here. Instead we reconstruct the panel
   LIVE, series-by-series, from FRED's public, keyless `fredgraph.csv`
   endpoint, the same underlying data FRED-MD itself is built from. See
   `src/fredmd_spec.py` for the full per-series mapping, substitutions, and
   the handful of series dropped for lack of a free source.

2. NBER recession indicator (USREC).
   Pulled directly from FRED (`fredgraph.csv?id=USREC`), full history.

3. VIX.
   Pulled directly from FRED (`fredgraph.csv?id=VIXCLS`), daily since
   1990-01-02, resampled to monthly.

4. S&P 500.
   FRED's own "SP500" series is license-restricted to the trailing ~10
   years, too short for this project. Standard historical vendors
   (stooq.com, Yahoo Finance, macrotrends.net) are blocked or JS-gated from
   this sandbox. We use multpl.com's "S&P 500 Historical Prices By Month"
   page, which serves the complete 1871-present monthly close series as
   plain server-rendered HTML (no API key, no JS challenge encountered).
   NOTE: this gives monthly CLOSE only. True daily-derived OHLC is not
   freely available without a paid vendor, so "S&P 500 monthly OHLC" in the
   original project brief is satisfied here as "S&P 500 monthly close"
   only; every place that would have used Open/High/Low instead derives
   drawdown/volatility features from Close-to-Close returns.

Every network call is wrapped so a failure is logged and skipped rather
than silently faking data. Run this file directly to (re)pull everything.

Caching (`data/raw/fred_components/<TICKER>.csv`, `data/raw/sp500_multpl_raw.html`):
Every raw FRED series pull and the multpl.com page are cached to disk on
first successful pull. Re-running this script reuses those cached raw files
instead of re-hitting the network (they ARE the "raw pull", persisted
exactly per the project brief) unless `force_refresh=True` is passed. This
is also a practical necessity in this specific sandboxed environment, where
this host's network proxy has shown intermittent read-timeouts under rapid
sequential requests from a long-lived Python process; a plain `curl` from a
fresh shell has been completely reliable, so the one-time pull in this repo
was performed with short-lived `curl` calls (see `fetch_raw_data.sh`) and
cached here, and `fetch_fred_series` below transparently reuses that cache.
A fresh environment with normal network access can simply delete
`data/raw/fred_components/` and re-run this script to pull everything live
via `requests`.
"""
from __future__ import annotations

import io
import logging
import re
import subprocess
import time
from pathlib import Path

import pandas as pd
import requests

from fredmd_spec import FRED_MD_SERIES, RECONSTRUCTED, DROPPED_NO_FREE_SOURCE, DROPPED_SHORT_HISTORY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
COMPONENTS_DIR = RAW_DIR / "fred_components"
RAW_DIR.mkdir(parents=True, exist_ok=True)
COMPONENTS_DIR.mkdir(parents=True, exist_ok=True)

FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
MULTPL_URL = "https://www.multpl.com/s-p-500-historical-prices/table/by-month"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) research-data-pull/1.0"}
TIMEOUT = 25


def _curl_get(url: str, timeout: int = TIMEOUT) -> tuple[int, bytes]:
    """Shell out to curl as a fallback HTTP client (see module docstring)."""
    proc = subprocess.run(
        ["curl", "-s", "--http1.1", "-A", HEADERS["User-Agent"], "--max-time", str(timeout),
         "--retry", "2", "--retry-delay", "1",
         "-w", "\n%{http_code}", url],
        capture_output=True, timeout=timeout + 15,
    )
    out = proc.stdout
    if b"\n" not in out:
        return 0, b""
    body, _, code = out.rpartition(b"\n")
    try:
        status = int(code.decode().strip())
    except ValueError:
        status = 0
    return status, body


def fetch_fred_series(ticker: str, retries: int = 2, force_refresh: bool = False) -> pd.Series | None:
    """Fetch one FRED series as a Date-indexed pd.Series via the public,
    keyless fredgraph.csv endpoint, using the on-disk cache described in the
    module docstring. Returns None on any failure (404, empty, network
    error) rather than raising, so the caller can log & skip."""
    cache_path = COMPONENTS_DIR / f"{ticker}.csv"
    raw_text = None
    if cache_path.exists() and not force_refresh:
        raw_text = cache_path.read_text()
    else:
        url = f"{FRED_GRAPH_URL}?id={ticker}"
        for attempt in range(retries + 1):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
                if resp.status_code == 200 and resp.content:
                    raw_text = resp.text
                    break
            except Exception as exc:  # noqa: BLE001
                logger.warning("FRED ticker %s (requests) attempt %d failed: %s", ticker, attempt, exc)
            time.sleep(1.5 * (attempt + 1))
        if raw_text is None:
            # fall back to curl (see module docstring)
            status, body = _curl_get(f"{FRED_GRAPH_URL}?id={ticker}")
            if status == 200 and body:
                raw_text = body.decode()
        if raw_text is not None:
            cache_path.write_text(raw_text)

    if raw_text is None:
        logger.warning("FRED ticker %s -> could not fetch (no cache, network failed)", ticker)
        return None
    try:
        df = pd.read_csv(io.StringIO(raw_text))
        if df.shape[1] < 2:
            return None
        df.columns = ["Date", ticker]
        df["Date"] = pd.to_datetime(df["Date"])
        df[ticker] = pd.to_numeric(df[ticker], errors="coerce")
        return df.set_index("Date")[ticker]
    except Exception as exc:  # noqa: BLE001
        logger.warning("FRED ticker %s -> parse error: %s", ticker, exc)
        return None


def pull_nber_recession() -> pd.DataFrame:
    logger.info("Pulling NBER recession indicator (USREC) from FRED ...")
    s = fetch_fred_series("USREC")
    if s is None:
        raise RuntimeError("Could not fetch USREC from FRED; no fallback available for the recession label.")
    df = s.to_frame("USREC")
    df.index.name = "Date"
    out_path = RAW_DIR / "nber_usrec.csv"
    df.to_csv(out_path)
    logger.info("Saved %s (%d rows, %s to %s)", out_path, len(df), df.index.min().date(), df.index.max().date())
    return df


def pull_vix() -> pd.DataFrame:
    logger.info("Pulling VIX (VIXCLS, daily) from FRED ...")
    s = fetch_fred_series("VIXCLS")
    if s is None:
        raise RuntimeError("Could not fetch VIXCLS from FRED; no fallback available for VIX.")
    daily_path = RAW_DIR / "vix_daily.csv"
    s.to_frame("VIXCLS").to_csv(daily_path)

    monthly = s.resample("MS").agg(["mean", "max", "last"])
    monthly.columns = ["VIX_MEAN", "VIX_MAX", "VIX_CLOSE"]
    monthly.index.name = "Date"
    monthly_path = RAW_DIR / "vix_monthly.csv"
    monthly.to_csv(monthly_path)
    logger.info(
        "Saved %s (daily, %d rows) and %s (monthly, %d rows), %s to %s",
        daily_path, len(s), monthly_path, len(monthly), monthly.index.min().date(), monthly.index.max().date(),
    )
    return monthly


def pull_sp500_multpl() -> pd.DataFrame:
    logger.info("Pulling S&P 500 monthly close history from multpl.com ...")
    cache_path = RAW_DIR / "sp500_multpl_raw.html"
    html = None
    if cache_path.exists():
        html = cache_path.read_text(encoding="utf-8", errors="ignore")
    else:
        try:
            resp = requests.get(MULTPL_URL, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                html = resp.text
        except Exception as exc:  # noqa: BLE001
            logger.warning("multpl.com (requests) failed: %s", exc)
        if html is None:
            status, body = _curl_get(MULTPL_URL, timeout=30)
            if status == 200:
                html = body.decode("utf-8", errors="ignore")
        if html is not None:
            cache_path.write_text(html, encoding="utf-8")
    if html is None:
        raise RuntimeError("multpl.com unreachable and no cached copy found; no fallback available for S&P 500 history.")
    rows = re.findall(r"<td>([A-Za-z]{3} \d{1,2}, \d{4})</td>\s*<td>\s*[^<]*?([\d,]+\.\d+)\s*</td>", html, re.S)
    if len(rows) < 100:
        raise RuntimeError(f"multpl.com page parsed to only {len(rows)} rows; page layout may have changed.")
    dates = pd.to_datetime([d for d, _ in rows], format="%b %d, %Y")
    values = [float(v.replace(",", "")) for _, v in rows]
    df = pd.DataFrame({"Date": dates, "SP500_CLOSE": values}).sort_values("Date")
    # Collapse to one observation per month (the table is daily-as-of for the
    # current/incomplete month and month-start historically); keep the last
    # available observation in each month as that month's "close".
    df["month"] = df["Date"].dt.to_period("M")
    monthly = df.sort_values("Date").groupby("month")["SP500_CLOSE"].last().to_frame()
    monthly.index = monthly.index.to_timestamp()  # month-start timestamp
    monthly.index.name = "Date"
    out_path = RAW_DIR / "sp500_monthly.csv"
    monthly.to_csv(out_path)
    logger.info("Saved %s (%d rows, %s to %s)", out_path, len(monthly), monthly.index.min().date(), monthly.index.max().date())
    return monthly


def pull_fredmd_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstructs the FRED-MD panel series-by-series. Returns
    (raw_levels_df, coverage_log_df)."""
    logger.info("Reconstructing FRED-MD macro panel from %d specified series ...", len(FRED_MD_SERIES))
    raw = {}
    coverage_rows = []

    # Pass 1: fetch every series that has a direct FRED ticker.
    for fmd_id, spec in FRED_MD_SERIES.items():
        ticker = spec["fred_ticker"]
        if ticker is None:
            continue  # reconstructed in pass 2
        s = fetch_fred_series(ticker)
        status = "ok" if s is not None else "FAILED"
        if s is not None:
            raw[fmd_id] = s
        coverage_rows.append({
            "fredmd_id": fmd_id, "fred_ticker": ticker, "group": spec["group"],
            "tcode": spec["tcode"], "description": spec["description"], "status": status,
            "n_obs": len(s) if s is not None else 0,
            "start": s.index.min().date().isoformat() if s is not None and len(s) else None,
            "end": s.index.max().date().isoformat() if s is not None and len(s) else None,
        })

    # Special-case CLAIMSx: weekly ICSA -> monthly average
    if "CLAIMSx" in raw:
        raw["CLAIMSx"] = raw["CLAIMSx"].resample("MS").mean()
    # Special-case VXOCLSx: daily -> monthly average
    if "VXOCLSx" in raw:
        raw["VXOCLSx"] = raw["VXOCLSx"].resample("MS").mean()

    # Pass 2: reconstructed series (ratios / spreads of already-fetched raw levels)
    for fmd_id, (kind, a_key, b_key) in RECONSTRUCTED.items():
        spec = FRED_MD_SERIES[fmd_id]
        if a_key not in raw or b_key not in raw:
            coverage_rows.append({
                "fredmd_id": fmd_id, "fred_ticker": f"reconstructed({a_key},{b_key})", "group": spec["group"],
                "tcode": spec["tcode"], "description": spec["description"], "status": "FAILED (missing inputs)",
                "n_obs": 0, "start": None, "end": None,
            })
            continue
        a, b = raw[a_key].align(raw[b_key], join="inner")
        if kind == "spread":
            s = a - b
        elif kind == "ratio":
            s = a / b
        else:
            raise ValueError(kind)
        raw[fmd_id] = s
        coverage_rows.append({
            "fredmd_id": fmd_id, "fred_ticker": f"reconstructed({a_key}-{b_key})" if kind == "spread" else f"reconstructed({a_key}/{b_key})",
            "group": spec["group"], "tcode": spec["tcode"], "description": spec["description"], "status": "ok (reconstructed)",
            "n_obs": len(s), "start": s.index.min().date().isoformat(), "end": s.index.max().date().isoformat(),
        })

    # Pass 3: SP500 comes from multpl.com, already pulled; inject if available
    sp500_path = RAW_DIR / "sp500_monthly.csv"
    if sp500_path.exists():
        sp = pd.read_csv(sp500_path, index_col="Date", parse_dates=True)["SP500_CLOSE"]
        sp.index = sp.index.to_period("M").to_timestamp()  # normalize to month-start
        raw["SP500"] = sp
        coverage_rows.append({
            "fredmd_id": "SP500", "fred_ticker": "multpl.com (S&P 500 monthly close)", "group": "Stock Market",
            "tcode": 5, "description": FRED_MD_SERIES["SP500"]["description"], "status": "ok (substitute source)",
            "n_obs": len(sp), "start": sp.index.min().date().isoformat(), "end": sp.index.max().date().isoformat(),
        })

    for fmd_id, reason in DROPPED_NO_FREE_SOURCE.items():
        coverage_rows.append({
            "fredmd_id": fmd_id, "fred_ticker": None, "group": None, "tcode": None,
            "description": reason, "status": "DROPPED (no free source)", "n_obs": 0, "start": None, "end": None,
        })

    for fmd_id, reason in DROPPED_SHORT_HISTORY.items():
        spec = FRED_MD_SERIES.get(fmd_id, {})
        s_dropped = raw.pop(fmd_id, None)
        coverage_rows = [r for r in coverage_rows if r["fredmd_id"] != fmd_id]
        coverage_rows.append({
            "fredmd_id": fmd_id, "fred_ticker": spec.get("fred_ticker"), "group": spec.get("group"), "tcode": spec.get("tcode"),
            "description": reason, "status": "DROPPED (short history in free substitute)",
            "n_obs": len(s_dropped) if s_dropped is not None else 0,
            "start": s_dropped.index.min().date().isoformat() if s_dropped is not None and len(s_dropped) else None,
            "end": s_dropped.index.max().date().isoformat() if s_dropped is not None and len(s_dropped) else None,
        })

    # Normalize all series to month-start timestamps and assemble wide panel
    normalized = {}
    for k, s in raw.items():
        s = s.copy()
        s.index = s.index.to_period("M").to_timestamp()
        s = s[~s.index.duplicated(keep="last")]
        normalized[k] = s
    panel = pd.DataFrame(normalized).sort_index()
    panel.index.name = "sasdate"

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_path = RAW_DIR / "fredmd_ingestion_log.csv"
    coverage_df.to_csv(coverage_path, index=False)

    n_ok = (coverage_df["status"].str.startswith("ok")).sum()
    n_fail = coverage_df["status"].str.startswith("FAILED").sum()
    n_drop = coverage_df["status"].str.startswith("DROPPED").sum()
    logger.info("FRED-MD panel reconstruction: %d ok, %d failed, %d dropped (see %s)", n_ok, n_fail, n_drop, coverage_path)
    logger.info("Panel shape: %s, date range %s to %s", panel.shape, panel.index.min().date(), panel.index.max().date())

    # Write FRED-MD-style raw file: first data row holds the transform codes,
    # exactly mirroring the format of the original bundled file so that
    # data_cleaning.py / feature_engineering.py can consume it unchanged.
    tcode_row = {fmd_id: FRED_MD_SERIES[fmd_id]["tcode"] for fmd_id in panel.columns}
    out_df = panel.copy()
    out_df = out_df.reset_index()
    out_df["sasdate"] = out_df["sasdate"].dt.strftime("%m/%d/%Y")
    tcode_record = {"sasdate": "Transform:", **tcode_row}
    out_df = pd.concat([pd.DataFrame([tcode_record]), out_df], ignore_index=True)
    fredmd_path = RAW_DIR / "fredmd_current.csv"
    out_df.to_csv(fredmd_path, index=False)
    logger.info("Saved reconstructed FRED-MD panel to %s", fredmd_path)

    # Also persist a clean series-spec table for downstream feature attribution
    spec_rows = []
    for fmd_id in panel.columns:
        spec = FRED_MD_SERIES[fmd_id]
        spec_rows.append({"ID": fmd_id, "Feature": fmd_id, "Description": spec["description"], "Group": spec["group"], "Transform": spec["tcode"]})
    spec_df = pd.DataFrame(spec_rows)
    spec_path = RAW_DIR / "fredmd_series_spec.csv"
    spec_df.to_csv(spec_path, index=False)

    return out_df, coverage_df


def main():
    pull_nber_recession()
    pull_vix()
    pull_sp500_multpl()
    pull_fredmd_panel()
    logger.info("Data ingestion complete. Raw files written to %s", RAW_DIR)


if __name__ == "__main__":
    main()
