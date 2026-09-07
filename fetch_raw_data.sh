#!/usr/bin/env bash
# fetch_raw_data.sh
# ==================
# Pulls every raw FRED ticker this project needs, plus the S&P 500 monthly
# history page, via plain `curl` from a fresh shell. That was the one
# network path that proved fully reliable in the sandbox this project was
# originally built in (see the "Caching" note in src/data_ingestion.py's docstring for
# why: the same requests made from inside a long-lived Python process saw
# intermittent read-timeouts through this environment's egress proxy, while
# a `curl` invoked directly from the shell did not).
#
# Output: data/raw/fred_components/<TICKER>.csv (one file per FRED series
# below, in FRED's own fredgraph.csv format) and
# data/raw/sp500_multpl_raw.html (multpl.com's monthly S&P 500 close page).
#
# Once these files exist, `python3 src/data_ingestion.py` (or any pipeline
# run) reuses them as a cache and does not re-hit the network at all. This
# script is what to re-run (or adapt) to refresh the raw pull with a new
# vintage: delete the target file(s) first, or pass --force to overwrite
# everything.
#
# In an environment with normal, reliable network access, this script is
# not required: src/data_ingestion.py's `requests`-based fetch works
# directly and will populate the same cache on first run.
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="data/raw/fred_components"
mkdir -p "$OUT_DIR"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

# Every FRED ticker actually used by src/fredmd_spec.py (direct series plus
# the raw inputs to every RECONSTRUCTED series), plus USREC (NBER recession
# indicator) and VIXCLS (VIX).
TICKERS=(
  AAA ACOGNO AMBSL AMDMUO ANDENO AWHMAN AWOTMAN BAA BUSINV BUSLOANS CE16OV
  CES0600000007 CES0600000008 CES1021000001 CES2000000008 CES3000000008
  CLF16OV CMRMTSPL CPF3M CPIAPPSL CPIAUCSL CPIMEDSL CPITRNSL CPIULFSL CUMFNS
  CUSR0000SA0L5 CUSR0000SAC CUSR0000SAS CUUR0000SA0L2 CUUR0000SAD
  DDURRG3M086SBEA DGORDER DMANEMP DNDGRG3M086SBEA DPCERA3M086SBEA
  DSERRG3M086SBEA DTCOLNVHFNM DTCTHFNM EXCAUS EXJPUS EXSZUS EXUSUK FEDFUNDS
  GS1 GS10 GS5 HOUST HOUSTMW HOUSTNE HOUSTS HOUSTW ICSA INDPRO INVEST
  IPB51222S IPBUSEQ IPCONGD IPDCONGD IPDMAT IPFINAL IPFPNSS IPFUELS
  IPMANSICS IPMAT IPNCONGD IPNMAT ISRATIO M1SL M2REAL M2SL MANEMP NDMANEMP
  NONBORRES NONREVSL PAYEMS PCEPI PERMIT PERMITMW PERMITNE PERMITS PERMITW
  PPICMM REALLN RPI RSAFS SRVPRD TB3MS TB6MS TOTRESNS TWEXAFEGSMTH UEMP15OV
  UEMP15T26 UEMP27OV UEMP5TO14 UEMPLT5 UEMPMEAN UNRATE USCONS USFIRE USGOOD
  USGOVT USREC USTPU USTRADE USWTRADE VIXCLS VXOCLS W875RX1 WPSFD49207
  WPSFD49502 WPSID61 WPSID62 WTISPLC
)

echo "Fetching ${#TICKERS[@]} FRED series into $OUT_DIR ..."
n_ok=0; n_skip=0; n_fail=0
for t in "${TICKERS[@]}"; do
  dest="$OUT_DIR/$t.csv"
  if [[ -f "$dest" && "$FORCE" -eq 0 ]]; then
    n_skip=$((n_skip+1)); continue
  fi
  url="https://fred.stlouisfed.org/graph/fredgraph.csv?id=${t}"
  if curl -sf --max-time 20 "$url" -o "$dest"; then
    n_ok=$((n_ok+1))
  else
    echo "  FAILED: $t"
    rm -f "$dest"
    n_fail=$((n_fail+1))
  fi
done
echo "FRED pulls: $n_ok fetched, $n_skip already cached (skipped), $n_fail failed"

sp500_dest="data/raw/sp500_multpl_raw.html"
if [[ -f "$sp500_dest" && "$FORCE" -eq 0 ]]; then
  echo "S&P 500 page already cached at $sp500_dest (skipped)"
else
  echo "Fetching S&P 500 monthly history from multpl.com ..."
  curl -sf --max-time 20 "https://www.multpl.com/s-p-500-historical-prices/table/by-month" -o "$sp500_dest" \
    && echo "  saved to $sp500_dest" \
    || echo "  FAILED to fetch S&P 500 page"
fi

echo "Done. Now run: python3 src/data_ingestion.py  (assembles data/raw/fredmd_current.csv from this cache)"
