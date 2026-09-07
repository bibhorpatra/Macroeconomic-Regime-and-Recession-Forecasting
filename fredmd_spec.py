"""
FRED-MD series specification table.

This module hardcodes the McCracken & Ng (2016) "FRED-MD: A Monthly Database
for Macroeconomic Research" appendix: each series' FRED-MD mnemonic, its
economic group (one of the 8 standard FRED-MD groups), and its standard
stationarity transformation code (1-7, see `TRANSFORM_CODE_MEANING` below).

IMPORTANT / HONESTY NOTE (see README "Data availability caveats"):
This project runs in a sandboxed environment where the official bundled
FRED-MD file (files.stlouisfed.org/.../fred-md/monthly/current.csv) is
blocked by the host's bot-protection (HTTP 403), and the Wayback Machine is
blocked by this sandbox's egress policy. Instead, we reconstruct the FRED-MD
panel LIVE, series-by-series, by pulling each component series directly from
FRED's public `fredgraph.csv` endpoint (the same data FRED-MD itself is built
from), which IS reachable. A handful of FRED-MD's own mnemonics carry a
trailing "x" that denotes McCracken & Ng's own internal splice of a
discontinued series with its successor (e.g. CLAIMSx, RETAILx); FRED does not
publish these spliced series under an "x" ticker, so we substitute the
closest current, freely-available FRED series (documented per-row in
`FRED_TICKER_OVERRIDE` / `notes` below). A few series have no free public
substitute at all (HWI, HWIURATIO, S&P: indust, S&P div yield, S&P PE ratio);
those are dropped entirely rather than having values fabricated for them. A handful of official
FRED-MD "*FFM" interest-rate-spread series (TB3SMFFM, T10YFFM, AAAFFM, ...)
are, by McCracken & Ng's own published construction, simply another rate
minus Fed Funds; we reconstruct those directly from their two components
(both of which ARE plain FRED series), which is not an approximation but the
literal definition.

Transform codes (applied in feature_engineering.py):
    1 = no transformation                          x(t)
    2 = first difference                            x(t) - x(t-1)
    3 = second difference                            (x(t)-x(t-1)) - (x(t-1)-x(t-2))
    4 = natural log                                  log(x(t))
    5 = first difference of log                      log(x(t)) - log(x(t-1))
    6 = second difference of log                     [log(x(t))-log(x(t-1))] - [log(x(t-1))-log(x(t-2))]
    7 = first difference of percent change            (x(t)/x(t-1) - 1) - (x(t-1)/x(t-2) - 1)
"""

TRANSFORM_CODE_MEANING = {
    1: "level (no transform)",
    2: "first difference",
    3: "second difference",
    4: "log level",
    5: "first difference of log (~ growth rate)",
    6: "second difference of log (~ change in growth rate)",
    7: "first difference of percent change",
}

GROUPS = [
    "Output and Income",
    "Labor Market",
    "Housing",
    "Consumption, Orders and Inventories",
    "Money and Credit",
    "Interest and Exchange Rates",
    "Prices",
    "Stock Market",
]

# Each entry: fredmd_id -> dict(
#   fred_ticker: the ticker to actually request from FRED's fredgraph.csv
#                (None => reconstructed from other series, handled specially
#                 in data_ingestion.py; see `RECONSTRUCTED`)
#   description, group, tcode, notes
# )
FRED_MD_SERIES = {
    # ---------------------------- Output and Income ----------------------------
    "RPI":              dict(fred_ticker="RPI",              tcode=5, group="Output and Income", description="Real Personal Income"),
    "W875RX1":          dict(fred_ticker="W875RX1",          tcode=5, group="Output and Income", description="Real personal income ex transfer receipts"),
    "INDPRO":           dict(fred_ticker="INDPRO",           tcode=5, group="Output and Income", description="IP Index"),
    "IPFPNSS":          dict(fred_ticker="IPFPNSS",          tcode=5, group="Output and Income", description="IP: Final Products and Nonindustrial Supplies"),
    "IPFINAL":          dict(fred_ticker="IPFINAL",          tcode=5, group="Output and Income", description="IP: Final Products"),
    "IPCONGD":          dict(fred_ticker="IPCONGD",          tcode=5, group="Output and Income", description="IP: Consumer Goods"),
    "IPDCONGD":         dict(fred_ticker="IPDCONGD",         tcode=5, group="Output and Income", description="IP: Durable Consumer Goods"),
    "IPNCONGD":         dict(fred_ticker="IPNCONGD",         tcode=5, group="Output and Income", description="IP: Nondurable Consumer Goods"),
    "IPBUSEQ":          dict(fred_ticker="IPBUSEQ",          tcode=5, group="Output and Income", description="IP: Business Equipment"),
    "IPMAT":            dict(fred_ticker="IPMAT",            tcode=5, group="Output and Income", description="IP: Materials"),
    "IPDMAT":           dict(fred_ticker="IPDMAT",           tcode=5, group="Output and Income", description="IP: Durable Materials"),
    "IPNMAT":           dict(fred_ticker="IPNMAT",           tcode=5, group="Output and Income", description="IP: Nondurable Materials"),
    "IPMANSICS":        dict(fred_ticker="IPMANSICS",        tcode=5, group="Output and Income", description="IP: Manufacturing (SIC)"),
    "IPB51222S":        dict(fred_ticker="IPB51222S",        tcode=5, group="Output and Income", description="IP: Residential Utilities"),
    "IPFUELS":          dict(fred_ticker="IPFUELS",          tcode=5, group="Output and Income", description="IP: Fuels"),
    "CUMFNS":           dict(fred_ticker="CUMFNS",           tcode=2, group="Output and Income", description="Capacity Utilization: Manufacturing"),

    # ------------------------------- Labor Market -------------------------------
    "CLF16OV":          dict(fred_ticker="CLF16OV",          tcode=5, group="Labor Market", description="Civilian Labor Force"),
    "CE16OV":           dict(fred_ticker="CE16OV",           tcode=5, group="Labor Market", description="Civilian Employment"),
    "UNRATE":           dict(fred_ticker="UNRATE",           tcode=2, group="Labor Market", description="Civilian Unemployment Rate"),
    "UEMPMEAN":         dict(fred_ticker="UEMPMEAN",         tcode=2, group="Labor Market", description="Average Duration of Unemployment (Weeks)"),
    "UEMPLT5":          dict(fred_ticker="UEMPLT5",          tcode=5, group="Labor Market", description="Civilians Unemployed - Less Than 5 Weeks"),
    "UEMP5TO14":        dict(fred_ticker="UEMP5TO14",        tcode=5, group="Labor Market", description="Civilians Unemployed for 5-14 Weeks"),
    "UEMP15OV":         dict(fred_ticker="UEMP15OV",         tcode=5, group="Labor Market", description="Civilians Unemployed - 15 Weeks & Over"),
    "UEMP15T26":        dict(fred_ticker="UEMP15T26",        tcode=5, group="Labor Market", description="Civilians Unemployed for 15-26 Weeks"),
    "UEMP27OV":         dict(fred_ticker="UEMP27OV",         tcode=5, group="Labor Market", description="Civilians Unemployed for 27 Weeks and Over"),
    "CLAIMSx":          dict(fred_ticker="ICSA",             tcode=5, group="Labor Market", description="Initial Claims (weekly ICSA resampled to monthly avg; FRED-MD's own CLAIMSx splice is not separately published)"),
    "PAYEMS":           dict(fred_ticker="PAYEMS",           tcode=5, group="Labor Market", description="All Employees: Total nonfarm"),
    "USGOOD":           dict(fred_ticker="USGOOD",           tcode=5, group="Labor Market", description="All Employees: Goods-Producing Industries"),
    "CES1021000001":    dict(fred_ticker="CES1021000001",    tcode=5, group="Labor Market", description="All Employees: Mining and Logging"),
    "USCONS":           dict(fred_ticker="USCONS",           tcode=5, group="Labor Market", description="All Employees: Construction"),
    "MANEMP":           dict(fred_ticker="MANEMP",           tcode=5, group="Labor Market", description="All Employees: Manufacturing"),
    "DMANEMP":          dict(fred_ticker="DMANEMP",          tcode=5, group="Labor Market", description="All Employees: Durable goods"),
    "NDMANEMP":         dict(fred_ticker="NDMANEMP",         tcode=5, group="Labor Market", description="All Employees: Nondurable goods"),
    "SRVPRD":           dict(fred_ticker="SRVPRD",           tcode=5, group="Labor Market", description="All Employees: Service-Providing Industries"),
    "USTPU":            dict(fred_ticker="USTPU",            tcode=5, group="Labor Market", description="All Employees: Trade, Transportation & Utilities"),
    "USWTRADE":         dict(fred_ticker="USWTRADE",         tcode=5, group="Labor Market", description="All Employees: Wholesale Trade"),
    "USTRADE":          dict(fred_ticker="USTRADE",          tcode=5, group="Labor Market", description="All Employees: Retail Trade"),
    "USFIRE":           dict(fred_ticker="USFIRE",           tcode=5, group="Labor Market", description="All Employees: Financial Activities"),
    "USGOVT":           dict(fred_ticker="USGOVT",           tcode=5, group="Labor Market", description="All Employees: Government"),
    "CES0600000007":    dict(fred_ticker="CES0600000007",    tcode=1, group="Labor Market", description="Avg Weekly Hours: Goods-Producing"),
    "AWOTMAN":          dict(fred_ticker="AWOTMAN",          tcode=2, group="Labor Market", description="Avg Weekly Overtime Hours: Manufacturing"),
    "AWHMAN":           dict(fred_ticker="AWHMAN",           tcode=1, group="Labor Market", description="Avg Weekly Hours: Manufacturing"),

    # --------------------------------- Housing ----------------------------------
    "HOUST":            dict(fred_ticker="HOUST",            tcode=4, group="Housing", description="Housing Starts: Total"),
    "HOUSTNE":          dict(fred_ticker="HOUSTNE",          tcode=4, group="Housing", description="Housing Starts: Northeast"),
    "HOUSTMW":          dict(fred_ticker="HOUSTMW",          tcode=4, group="Housing", description="Housing Starts: Midwest"),
    "HOUSTS":           dict(fred_ticker="HOUSTS",           tcode=4, group="Housing", description="Housing Starts: South"),
    "HOUSTW":           dict(fred_ticker="HOUSTW",           tcode=4, group="Housing", description="Housing Starts: West"),
    "PERMIT":           dict(fred_ticker="PERMIT",           tcode=4, group="Housing", description="New Private Housing Permits: Total"),
    "PERMITNE":         dict(fred_ticker="PERMITNE",         tcode=4, group="Housing", description="New Private Housing Permits: Northeast"),
    "PERMITMW":         dict(fred_ticker="PERMITMW",         tcode=4, group="Housing", description="New Private Housing Permits: Midwest"),
    "PERMITS":          dict(fred_ticker="PERMITS",          tcode=4, group="Housing", description="New Private Housing Permits: South"),
    "PERMITW":          dict(fred_ticker="PERMITW",          tcode=4, group="Housing", description="New Private Housing Permits: West"),

    # --------------------- Consumption, Orders and Inventories -------------------
    "DPCERA3M086SBEA":  dict(fred_ticker="DPCERA3M086SBEA",  tcode=5, group="Consumption, Orders and Inventories", description="Real personal consumption expenditures"),
    "CMRMTSPLx":        dict(fred_ticker="CMRMTSPL",         tcode=5, group="Consumption, Orders and Inventories", description="Real Manufacturing and Trade Industries Sales"),
    "RETAILx":          dict(fred_ticker="RSAFS",            tcode=5, group="Consumption, Orders and Inventories", description="Retail and Food Services Sales (FRED-MD's RETAILx splice not separately published; RSAFS used as substitute)"),
    "ACOGNO":           dict(fred_ticker="ACOGNO",           tcode=5, group="Consumption, Orders and Inventories", description="New Orders for Consumer Goods"),
    "AMDMNOx":          dict(fred_ticker="DGORDER",          tcode=5, group="Consumption, Orders and Inventories", description="New Orders for Durable Goods (DGORDER substituted for FRED-MD's AMDMNOx splice)"),
    "ANDENOx":          dict(fred_ticker="ANDENO",           tcode=5, group="Consumption, Orders and Inventories", description="New Orders for Nondefense Capital Goods"),
    "AMDMUOx":          dict(fred_ticker="AMDMUO",           tcode=5, group="Consumption, Orders and Inventories", description="Unfilled Orders for Durable Goods"),
    "BUSINVx":          dict(fred_ticker="BUSINV",           tcode=5, group="Consumption, Orders and Inventories", description="Total Business Inventories"),
    "ISRATIOx":         dict(fred_ticker="ISRATIO",          tcode=2, group="Consumption, Orders and Inventories", description="Total Business: Inventories to Sales Ratio"),

    # ------------------------------ Money and Credit ------------------------------
    "M1SL":             dict(fred_ticker="M1SL",             tcode=6, group="Money and Credit", description="M1 Money Stock"),
    "M2SL":             dict(fred_ticker="M2SL",             tcode=6, group="Money and Credit", description="M2 Money Stock"),
    "M2REAL":           dict(fred_ticker="M2REAL",           tcode=5, group="Money and Credit", description="Real M2 Money Stock"),
    "AMBSL":            dict(fred_ticker="AMBSL",            tcode=6, group="Money and Credit", description="St. Louis Adjusted Monetary Base"),
    "TOTRESNS":         dict(fred_ticker="TOTRESNS",         tcode=6, group="Money and Credit", description="Total Reserves of Depository Institutions"),
    # tcode overridden from FRED-MD's published 6 (second-diff-of-log) to 2
    # (first difference): during the 2008-09 financial crisis, banks
    # borrowed so heavily from the Fed's emergency facilities that
    # NONBORROWED reserves went sharply NEGATIVE (as low as -$333B in
    # 2008-10). log(negative) is undefined, so a log-based transform
    # produces a multi-month NaN gap over precisely the 2008-09 crisis,
    # which is exactly the period this project most needs. A first
    # difference handles negative levels correctly and is well-defined
    # throughout.
    "NONBORRES":        dict(fred_ticker="NONBORRES",        tcode=2, group="Money and Credit", description="Reserves Of Depository Institutions, Nonborrowed"),
    "BUSLOANS":         dict(fred_ticker="BUSLOANS",         tcode=6, group="Money and Credit", description="Commercial and Industrial Loans"),
    "REALLN":           dict(fred_ticker="REALLN",           tcode=6, group="Money and Credit", description="Real Estate Loans at All Commercial Banks"),
    "NONREVSL":         dict(fred_ticker="NONREVSL",         tcode=6, group="Money and Credit", description="Total Nonrevolving Credit"),
    "CONSPI":           dict(fred_ticker=None,                tcode=2, group="Money and Credit", description="Nonrevolving consumer credit to Personal Income (reconstructed = NONREVSL / RPI)"),
    "DTCOLNVHFNM":      dict(fred_ticker="DTCOLNVHFNM",      tcode=6, group="Money and Credit", description="Consumer Motor Vehicle Loans Outstanding"),
    "DTCTHFNM":         dict(fred_ticker="DTCTHFNM",         tcode=6, group="Money and Credit", description="Total Consumer Loans and Leases at Finance Companies"),
    "INVEST":           dict(fred_ticker="INVEST",           tcode=6, group="Money and Credit", description="Securitized Consumer Credit"),

    # ------------------------- Interest and Exchange Rates -------------------------
    "FEDFUNDS":         dict(fred_ticker="FEDFUNDS",         tcode=2, group="Interest and Exchange Rates", description="Effective Federal Funds Rate"),
    "CP3Mx":            dict(fred_ticker="CPF3M",            tcode=2, group="Interest and Exchange Rates", description="3-Month AA Financial Commercial Paper Rate"),
    "TB3MS":            dict(fred_ticker="TB3MS",            tcode=2, group="Interest and Exchange Rates", description="3-Month Treasury Bill Rate"),
    "TB6MS":            dict(fred_ticker="TB6MS",            tcode=2, group="Interest and Exchange Rates", description="6-Month Treasury Bill Rate"),
    "GS1":              dict(fred_ticker="GS1",              tcode=2, group="Interest and Exchange Rates", description="1-Year Treasury Rate"),
    "GS5":              dict(fred_ticker="GS5",              tcode=2, group="Interest and Exchange Rates", description="5-Year Treasury Rate"),
    "GS10":             dict(fred_ticker="GS10",             tcode=2, group="Interest and Exchange Rates", description="10-Year Treasury Rate"),
    "AAA":              dict(fred_ticker="AAA",              tcode=2, group="Interest and Exchange Rates", description="Moody's Seasoned Aaa Corporate Bond Yield"),
    "BAA":              dict(fred_ticker="BAA",              tcode=2, group="Interest and Exchange Rates", description="Moody's Seasoned Baa Corporate Bond Yield"),
    "COMPAPFFx":        dict(fred_ticker=None,                tcode=1, group="Interest and Exchange Rates", description="3M Commercial Paper minus Fed Funds (reconstructed = CPF3M - FEDFUNDS)"),
    "TB3SMFFM":         dict(fred_ticker=None,                tcode=1, group="Interest and Exchange Rates", description="3M T-bill minus Fed Funds (reconstructed = TB3MS - FEDFUNDS)"),
    "TB6SMFFM":         dict(fred_ticker=None,                tcode=1, group="Interest and Exchange Rates", description="6M T-bill minus Fed Funds (reconstructed = TB6MS - FEDFUNDS)"),
    "T1YFFM":           dict(fred_ticker=None,                tcode=1, group="Interest and Exchange Rates", description="1Y Treasury minus Fed Funds (reconstructed = GS1 - FEDFUNDS)"),
    "T5YFFM":           dict(fred_ticker=None,                tcode=1, group="Interest and Exchange Rates", description="5Y Treasury minus Fed Funds (reconstructed = GS5 - FEDFUNDS)"),
    "T10YFFM":          dict(fred_ticker=None,                tcode=1, group="Interest and Exchange Rates", description="10Y Treasury minus Fed Funds (reconstructed = GS10 - FEDFUNDS)"),
    "AAAFFM":           dict(fred_ticker=None,                tcode=1, group="Interest and Exchange Rates", description="Aaa yield minus Fed Funds (reconstructed = AAA - FEDFUNDS)"),
    "BAAFFM":           dict(fred_ticker=None,                tcode=1, group="Interest and Exchange Rates", description="Baa yield minus Fed Funds (reconstructed = BAA - FEDFUNDS)"),
    "TWEXAFEGSMTHx":    dict(fred_ticker="TWEXAFEGSMTH",     tcode=5, group="Interest and Exchange Rates", description="Trade Weighted U.S. Dollar Index: Advanced Foreign Economies"),
    "EXSZUSx":          dict(fred_ticker="EXSZUS",           tcode=5, group="Interest and Exchange Rates", description="Switzerland / U.S. Foreign Exchange Rate"),
    "EXJPUSx":          dict(fred_ticker="EXJPUS",           tcode=5, group="Interest and Exchange Rates", description="Japan / U.S. Foreign Exchange Rate"),
    "EXUSUKx":          dict(fred_ticker="EXUSUK",           tcode=5, group="Interest and Exchange Rates", description="U.S. / U.K. Foreign Exchange Rate"),
    "EXCAUSx":          dict(fred_ticker="EXCAUS",           tcode=5, group="Interest and Exchange Rates", description="Canada / U.S. Foreign Exchange Rate"),

    # ---------------------------------- Prices ------------------------------------
    "WPSFD49207":       dict(fred_ticker="WPSFD49207",       tcode=6, group="Prices", description="PPI: Final Demand: Finished Goods"),
    "WPSFD49502":       dict(fred_ticker="WPSFD49502",       tcode=6, group="Prices", description="PPI: Final Demand: Finished Consumer Goods"),
    "WPSID61":          dict(fred_ticker="WPSID61",          tcode=6, group="Prices", description="PPI: Intermediate Demand by Commodity: Materials/Supplies"),
    "WPSID62":          dict(fred_ticker="WPSID62",          tcode=6, group="Prices", description="PPI: Intermediate Demand by Commodity: Materials/Components"),
    "OILPRICEx":        dict(fred_ticker="WTISPLC",          tcode=6, group="Prices", description="WTI Crude Oil Spot Price (substituted for FRED-MD's OILPRICEx)"),
    "PPICMM":           dict(fred_ticker="PPICMM",           tcode=6, group="Prices", description="PPI: Metals and Metal Products"),
    "CPIAUCSL":         dict(fred_ticker="CPIAUCSL",         tcode=6, group="Prices", description="CPI: All Items"),
    "CPIAPPSL":         dict(fred_ticker="CPIAPPSL",         tcode=6, group="Prices", description="CPI: Apparel"),
    "CPITRNSL":         dict(fred_ticker="CPITRNSL",         tcode=6, group="Prices", description="CPI: Transportation"),
    "CPIMEDSL":         dict(fred_ticker="CPIMEDSL",         tcode=6, group="Prices", description="CPI: Medical Care"),
    "CUSR0000SAC":      dict(fred_ticker="CUSR0000SAC",      tcode=6, group="Prices", description="CPI: Commodities"),
    "CUUR0000SAD":      dict(fred_ticker="CUUR0000SAD",      tcode=6, group="Prices", description="CPI: Durables"),
    "CUSR0000SAS":      dict(fred_ticker="CUSR0000SAS",      tcode=6, group="Prices", description="CPI: Services"),
    "CPIULFSL":         dict(fred_ticker="CPIULFSL",         tcode=6, group="Prices", description="CPI: All Items Less Food"),
    "CUUR0000SA0L2":    dict(fred_ticker="CUUR0000SA0L2",    tcode=6, group="Prices", description="CPI: All items less shelter"),
    "CUSR0000SA0L5":    dict(fred_ticker="CUSR0000SA0L5",    tcode=6, group="Prices", description="CPI: All items less medical care"),
    "PCEPI":            dict(fred_ticker="PCEPI",            tcode=6, group="Prices", description="Personal Cons. Expenditures: Chain Price Index"),
    "DDURRG3M086SBEA":  dict(fred_ticker="DDURRG3M086SBEA",  tcode=6, group="Prices", description="PCE: Durable goods (chain price index)"),
    "DNDGRG3M086SBEA":  dict(fred_ticker="DNDGRG3M086SBEA",  tcode=6, group="Prices", description="PCE: Nondurable goods (chain price index)"),
    "DSERRG3M086SBEA":  dict(fred_ticker="DSERRG3M086SBEA",  tcode=6, group="Prices", description="PCE: Services (chain price index)"),
    "CES0600000008":    dict(fred_ticker="CES0600000008",    tcode=6, group="Prices", description="Avg Hourly Earnings: Goods-Producing"),
    "CES2000000008":    dict(fred_ticker="CES2000000008",    tcode=6, group="Prices", description="Avg Hourly Earnings: Construction"),
    "CES3000000008":    dict(fred_ticker="CES3000000008",    tcode=6, group="Prices", description="Avg Hourly Earnings: Manufacturing"),

    # ------------------------------- Stock Market ---------------------------------
    "SP500":            dict(fred_ticker=None,                tcode=5, group="Stock Market", description="S&P 500 monthly closing price (sourced from multpl.com, see data_ingestion.py)"),
    "VXOCLSx":          dict(fred_ticker="VXOCLS",            tcode=1, group="Stock Market", description="CBOE S&P 100 Volatility Index (VXO), monthly avg of daily"),
}

# Series where a free substitute EXISTS (see FRED_MD_SERIES above) but its
# history is dramatically shorter than FRED-MD's own proprietary splice,
# to the point that including it would truncate the ENTIRE panel's usable
# sample (the row-level null-threshold in data_cleaning.py requires nearly
# all columns populated). FRED-MD's own construction splices these onto a
# discontinued predecessor series (not separately available for free) to
# reach back to 1959-67; the modern successor alone only starts in the
# 1990s/2000s. Excluded from the assembled panel so the other ~110 series'
# much longer history is not sacrificed for a handful of columns.
DROPPED_SHORT_HISTORY = {
    "RETAILx":       "Substitute RSAFS only starts 1992-01 (FRED-MD's own splice reaches back to 1947 via a discontinued predecessor).",
    "ACOGNO":        "FRED series ACOGNO itself only starts 1992-02.",
    "AMDMNOx":       "Substitute DGORDER only starts 1992-02.",
    "ANDENOx":       "Substitute ANDENO only starts 1992-02.",
    "AMDMUOx":       "Substitute AMDMUO only starts 1992-01.",
    "BUSINVx":       "Substitute BUSINV only starts 1992-01.",
    "ISRATIOx":      "Substitute ISRATIO only starts 1992-01.",
    "CP3Mx":         "Substitute CPF3M only starts 1997-01.",
    "COMPAPFFx":     "Depends on CP3Mx (see above), so inherits the 1997-01 start.",
    "TWEXAFEGSMTHx": "This FRED broad-dollar-index vintage only starts 2006-01 (methodology redesign).",
    "VXOCLSx":       "CBOE discontinued the VXO index in Sept 2021 (superseded by VIX, which this project already sources directly) and it only starts 1986-01, so it would both truncate history AND go null for the last ~5 years.",
}

# Series that are documented in the official FRED-MD appendix but for which
# no freely-available, non-proprietary current substitute could be found in
# this environment. Dropped explicitly, and logged, rather than silently; see README.
DROPPED_NO_FREE_SOURCE = {
    "HWI": "Help-Wanted Index discontinued by Conference Board in 2006; Barnichon's reconstruction is not freely redistributable.",
    "HWIURATIO": "Depends on HWI (see above).",
    "SP_INDUST": "S&P's Common Stock Price Index: Industrials. Proprietary S&P data, no free substitute found.",
    "SP_DIV_YIELD": "S&P Composite dividend yield. Proprietary S&P data; Shiller's public series is a reasonable proxy but is a stale, manually-updated file (last refreshed 2023-09 as pulled in this environment), so it was excluded rather than silently mixing vintages.",
    "SP_PE_RATIO": "S&P Composite P/E ratio. Same proprietary/staleness issue as dividend yield above.",
}

# Reconstructed series: computed directly from the RAW (pre-transform) levels
# of two other FRED-MD series already fetched in the same pull, rather than
# pulled from a single FRED ticker of their own. `fred_ticker=None` above
# marks these; this dict names the two FRED_MD_SERIES keys (raw levels) used:
# ("ratio"|"spread", numerator_or_minuend_key, denominator_or_subtrahend_key).
RECONSTRUCTED = {
    "CONSPI":    ("ratio", "NONREVSL", "RPI"),
    "COMPAPFFx": ("spread", "CP3Mx", "FEDFUNDS"),
    "TB3SMFFM":  ("spread", "TB3MS", "FEDFUNDS"),
    "TB6SMFFM":  ("spread", "TB6MS", "FEDFUNDS"),
    "T1YFFM":    ("spread", "GS1", "FEDFUNDS"),
    "T5YFFM":    ("spread", "GS5", "FEDFUNDS"),
    "T10YFFM":   ("spread", "GS10", "FEDFUNDS"),
    "AAAFFM":    ("spread", "AAA", "FEDFUNDS"),
    "BAAFFM":    ("spread", "BAA", "FEDFUNDS"),
}
