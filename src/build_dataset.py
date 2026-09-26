"""
build_dataset.py - Two-pass IV surface data pipeline

See PROJECT_PLAN.md for full documentation of every design decision.

Pass 1 (extract): streams through raw zip files without fully decompressing
                  to disk, keeps only rows for chosen tickers, writes an
                  intermediate raw Parquet file (~manageable size).

Pass 2 (featurize): loads the intermediate file, applies quality filters,
                    computes all derived features, assigns split labels,
                    writes the final dataset.

The two-pass design means you only need to run the slow extraction once.
After that, re-running featurize is fast.

Usage:
    # Full pipeline (both passes)
    python src/build_dataset.py

    # Skip extraction if intermediate file already exists
    python src/build_dataset.py --skip_extract

    # Custom paths
    python src/build_dataset.py --raw_dir data/raw --out data/processed/iv_surface.parquet
"""

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ticker universe - must match PROJECT_PLAN.md Section 3
# ---------------------------------------------------------------------------

TRAIN_TICKERS   = ["SPY", "NVDA", "AAPL", "LLY", "GS", "COST", "BA", "EOG"]
HOLDOUT_TICKERS = ["MSFT", "JPM", "TSLA", "XOM"]
ALL_TICKERS     = TRAIN_TICKERS + HOLDOUT_TICKERS

# ---------------------------------------------------------------------------
# Chronological split boundaries - PROJECT_PLAN.md Section 7
# Jan–Aug  → train
# Sep–Oct  → val
# Nov–Dec  → test (temporal for training tickers, ticker for held-out)
# ---------------------------------------------------------------------------

VAL_START  = pd.Timestamp("2024-09-01")
TEST_START = pd.Timestamp("2024-11-01")

# ---------------------------------------------------------------------------
# Raw columns to keep during extraction - PROJECT_PLAN.md Section 2
# Grab generously now; going back to 150GB zips is expensive.
# ---------------------------------------------------------------------------

RAW_COLS = [
    "undSecKey_tk",                       # underlying ticker
    "tradingDate",
    "okey_cp",                            # Call / Put
    "okey_xx",                            # strike price
    "okey_yr", "okey_mn", "okey_dy",     # expiry date components
    "years",                              # time to expiry in years
    "uClose",                             # underlying close price
    "bidIV", "askIV",                     # bid and ask implied volatility
    "rate",                               # risk-free rate
    "error",                              # data quality flag (0 = clean)
    "openInterest",                       # for liquidity filter
    "ve",                                 # vega - for vega-weighted error metrics
]

# ---------------------------------------------------------------------------
# Quality filter thresholds - PROJECT_PLAN.md Section 6
# ---------------------------------------------------------------------------

MIN_DAYS_TO_EXPIRY  = 7          # drop options expiring in less than 7 days
MAX_IV_SPREAD_RATIO = 0.50       # drop if (askIV - bidIV) / midIV > 50%

# Provider sentinel bounds. Implied volatility is not observed - it is solved for
# by inverting Black-Scholes numerically, and that root-finder returns its own
# bound rather than a null when the solve fails to converge. 10.0 (1000%
# annualised) is the ceiling and near-zero is the floor, so these values are
# error codes sitting in a price field, not quotes.
#
# Evidence in this dataset: ZERO rows with midIV in [5, 9.99) against 11 at
# >= 9.99. A continuous distribution has no such gap below a round number. At the
# floor, ~370 rows pile onto 0.0001 and 0.0002 while any genuine IV value appears
# at most ~30 times. The failures are all deep-OTM contracts with negligible vega
# (worst case: an XOM call struck at 223% of spot, vega 0.08) where the option
# price is almost unresponsive to volatility and the inversion is degenerate.
#
# Checked on bidIV and askIV rather than midIV alone: if one leg is capped and
# the other is not, their average can fall below the threshold and slip through
# while still being contaminated.
IV_SENTINEL_HIGH = 9.99
IV_SENTINEL_LOW  = 0.001

# ---------------------------------------------------------------------------
# ATM IV reference maturity window (days) - PROJECT_PLAN.md Section 5
# We look for options in the 20-40 day window to define the ATM IV level.
# ---------------------------------------------------------------------------

ATM_WINDOW_LO = 20 / 365
ATM_WINDOW_HI = 40 / 365
ATM_FALLBACK  = 30 / 365   # target maturity when window is empty

# ---------------------------------------------------------------------------
# Maturity buckets for SVI - PROJECT_PLAN.md Section 5
# SVI fits one curve per maturity slice; we discretise into standard tenors.
# ---------------------------------------------------------------------------

MATURITY_BINS   = [0, 14, 21, 45, 75, 135, 225, np.inf]  # days
MATURITY_LABELS = ["7d", "14d", "30d", "60d", "90d", "180d", "1Y"]


# ===========================================================================
# PASS 1 - EXTRACTION
# ===========================================================================

def extract(raw_dir: Path, intermediate: Path) -> None:
    """
    Stream through all outer/inner zips.  Keep only rows for ALL_TICKERS.
    Write a raw intermediate Parquet - no filtering or feature computation yet.
    """
    outer_zips = sorted(raw_dir.glob("*.zip"))
    if not outer_zips:
        raise FileNotFoundError(f"No zip files in {raw_dir}")

    frames = []
    for outer_path in outer_zips:
        print(f"[extract] {outer_path.name}")
        with zipfile.ZipFile(outer_path) as outer:
            for inner_name in outer.namelist():
                if not inner_name.endswith(".zip"):
                    continue
                inner_bytes = outer.read(inner_name)
                with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                    for txt_name in inner.namelist():
                        if not txt_name.endswith(".txt"):
                            continue
                        df = pd.read_csv(
                            io.BytesIO(inner.read(txt_name)),
                            sep="\t",
                            usecols=RAW_COLS,
                            low_memory=False,
                        )
                        df = df[df["undSecKey_tk"].isin(ALL_TICKERS)]
                        if not df.empty:
                            frames.append(df)
                            print(f"  {txt_name}: {len(df):,} rows")

    out = pd.concat(frames, ignore_index=True)
    intermediate.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(intermediate, index=False)
    print(f"[extract] done - {len(out):,} rows saved to {intermediate}\n")


# ===========================================================================
# PASS 2 - FEATURE ENGINEERING
# ===========================================================================

def _assign_maturity_bucket(years: pd.Series) -> pd.Series:
    days = years * 365
    return pd.cut(days, bins=MATURITY_BINS, labels=MATURITY_LABELS).astype(str)


def _compute_atm_iv(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (ticker, tradingDate): find the option nearest to at-the-money
    within the 20-40 day maturity window.  Falls back to the option whose
    maturity is closest to 30 days when the window contains no contracts.

    Result merged back as column 'atmIV'.

    Motivation: atmIV anchors the overall volatility level for that stock on
    that day.  normalizedIV = midIV / atmIV then isolates surface shape.
    """
    key = ["ticker", "tradingDate"]

    # --- Primary: 20-40 day window ---
    ref = df[(df["years"] >= ATM_WINDOW_LO) & (df["years"] < ATM_WINDOW_HI)].copy()
    ref["_absM"] = ref["logMoneyness"].abs()
    atm_primary = (
        ref.sort_values("_absM")
        .groupby(key, as_index=False)
        .first()[key + ["midIV"]]
        .rename(columns={"midIV": "atmIV"})
    )

    # --- Fallback: ticker-dates not covered by primary window ---
    all_keys    = df[key].drop_duplicates()
    missing     = all_keys.merge(atm_primary[key], on=key, how="left", indicator=True)
    missing     = missing[missing["_merge"] == "left_only"][key]

    if not missing.empty:
        fb = df.merge(missing, on=key)
        fb = fb.copy()
        fb["_mat_dist"] = (fb["years"] - ATM_FALLBACK).abs()
        fb["_absM"]     = fb["logMoneyness"].abs()
        atm_fallback = (
            fb.sort_values(["_mat_dist", "_absM"])
            .groupby(key, as_index=False)
            .first()[key + ["midIV"]]
            .rename(columns={"midIV": "atmIV"})
        )
        atm = pd.concat([atm_primary, atm_fallback], ignore_index=True)
    else:
        atm = atm_primary

    return df.merge(atm, on=key, how="left")


SPLIT_THRESHOLD = 0.40  # flag any single-day price drop > 40% as a stock split


def _mask_split_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Zero out log returns on days where the raw price ratio implies a stock
    split or reverse split.  A genuine >40% single-day move is essentially
    impossible for a liquid large-cap; the only real cause is an unadjusted
    price series jumping by the split factor.

    Zeroing (not dropping) preserves the continuity of the rolling window so
    the 30-day std is not thrown off by a gap in the series.
    """
    raw_return = prices.groupby("ticker")["uClose"].transform(
        lambda s: np.log(s / s.shift(1))
    )
    is_split = raw_return.abs() > np.log(1 / (1 - SPLIT_THRESHOLD))
    n = is_split.sum()
    if n:
        flagged = prices.loc[is_split, ["ticker", "tradingDate"]].values.tolist()
        for ticker, date in flagged:
            print(f"[rolling] split detected - {ticker} on {date}, return zeroed")
    prices = prices.copy()
    prices["_raw_logReturn"] = raw_return
    prices["logReturn"] = raw_return.where(~is_split, 0.0)
    return prices


def _compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-ticker daily rolling features from the underlying close price.

    Both features derive from the same split-masked log return series produced by
    _mask_split_returns, so neither is contaminated by unadjusted price jumps.

    realizedVol:  30-day rolling annualised volatility of masked log returns.
    recentReturn: 20-day log return, as the rolling sum of masked log returns.

    Both computed from the uClose embedded in the options data (same value for all
    options on a given ticker-date), so no external data needed.
    """
    prices = (
        df[["ticker", "tradingDate", "uClose"]]
        .drop_duplicates(subset=["ticker", "tradingDate"])
        .sort_values(["ticker", "tradingDate"])
        .copy()
    )

    prices = _mask_split_returns(prices)

    prices["realizedVol"] = prices.groupby("ticker")["logReturn"].transform(
        lambda s: s.rolling(30, min_periods=15).std() * np.sqrt(252)
    )
    prices["recentReturn"] = prices.groupby("ticker")["logReturn"].transform(
        lambda s: s.rolling(20, min_periods=20).sum()
    )

    # Guardrail: a 20-day equity log return outside +/-100% is not a price move - it is
    # an unhandled split leaking through the mask. Fail loudly rather than silently
    # poison a model input.
    extreme = prices["recentReturn"].abs() > 1.0
    if extreme.any():
        raise ValueError(
            f"recentReturn implausible for {int(extreme.sum())} ticker-days - check split masking:\n"
            f"{prices.loc[extreme, ['ticker', 'tradingDate', 'recentReturn']]}"
        )

    return df.merge(
        prices[["ticker", "tradingDate", "realizedVol", "recentReturn"]],
        on=["ticker", "tradingDate"],
        how="left",
    )


def _assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each row with its data split:
      train         - Jan–Aug,  training tickers
      val           - Sep–Oct,  training tickers
      test_temporal - Nov–Dec,  training tickers  (temporal generalisation test)
      test_ticker   - Nov–Dec,  held-out tickers  (cross-sectional generalisation test)

    Held-out ticker rows outside Nov–Dec are dropped - the model must never
    see them during training or hyperparameter tuning.
    """
    d = df["tradingDate"]
    t = df["ticker"]

    is_train   = t.isin(TRAIN_TICKERS)
    is_holdout = t.isin(HOLDOUT_TICKERS)

    keep = (
        (is_train   & (d < VAL_START))                          |  # train
        (is_train   & (d >= VAL_START)  & (d < TEST_START))    |  # val
        (is_train   & (d >= TEST_START))                        |  # test_temporal
        (is_holdout & (d >= TEST_START))                           # test_ticker
    )
    df = df[keep].copy()

    df["split"] = np.select(
        [
            (df["tradingDate"] < VAL_START),
            (df["tradingDate"] >= VAL_START) & (df["tradingDate"] < TEST_START),
            (df["tradingDate"] >= TEST_START) & df["ticker"].isin(TRAIN_TICKERS),
            (df["tradingDate"] >= TEST_START) & df["ticker"].isin(HOLDOUT_TICKERS),
        ],
        ["train", "val", "test_temporal", "test_ticker"],
        default="",
    )

    return df


def featurize(intermediate: Path, final_out: Path) -> None:
    print(f"[featurize] loading {intermediate}")
    df = pd.read_parquet(intermediate)

    # --- Rename and types ---
    df = df.rename(columns={"undSecKey_tk": "ticker"})
    df["tradingDate"] = pd.to_datetime(df["tradingDate"])
    df["okey_cp"]     = df["okey_cp"].str.strip()

    # --- Expiry date from components ---
    df["expiryDate"] = pd.to_datetime(
        df[["okey_yr", "okey_mn", "okey_dy"]].rename(
            columns={"okey_yr": "year", "okey_mn": "month", "okey_dy": "day"}
        )
    )
    df = df.drop(columns=["okey_yr", "okey_mn", "okey_dy"])

    # --- Per-contract features ---
    df["midIV"]        = (df["bidIV"] + df["askIV"]) / 2
    df["logMoneyness"] = np.log(df["okey_xx"] / df["uClose"])
    df["callPut"]      = (df["okey_cp"] == "Call").astype(int)  # 1=Call, 0=Put

    # --- Quality filters (PROJECT_PLAN.md Section 6) ---
    n0 = len(df)
    min_years = MIN_DAYS_TO_EXPIRY / 365
    spread_ratio = (df["askIV"] - df["bidIV"]) / df["midIV"]
    n_sentinel = int((
        (df["bidIV"] >= IV_SENTINEL_HIGH) |
        (df["askIV"] >= IV_SENTINEL_HIGH) |
        (df["midIV"] <= IV_SENTINEL_LOW)
    ).sum())
    df = df[
        (df["error"]        == 0)                &
        (df["bidIV"]        >  0)                &
        (df["askIV"]        >  0)                &
        (df["bidIV"]        <  IV_SENTINEL_HIGH) &
        (df["askIV"]        <  IV_SENTINEL_HIGH) &
        (df["midIV"]        >  IV_SENTINEL_LOW)  &
        (df["years"]        >= min_years)        &
        (df["uClose"]       >  0)                &
        (df["openInterest"] >  0)                &
        (spread_ratio       <= MAX_IV_SPREAD_RATIO)
    ]
    print(f"[featurize] quality filters: {n0:,} → {len(df):,} ({n0 - len(df):,} dropped)")
    print(f"[featurize]   of which IV solver sentinels: {n_sentinel:,}")

    # --- SVI inputs ---
    df["totalVariance"]  = df["midIV"] ** 2 * df["years"]
    df["maturityBucket"] = _assign_maturity_bucket(df["years"])

    # --- Rolling cross-day features (uses full 2024 price history) ---
    # Computed before split assignment so held-out tickers have full history
    # for their Nov–Dec test window.
    df = _compute_rolling_features(df)

    # --- ATM IV per (ticker, tradingDate) ---
    df = _compute_atm_iv(df)

    # --- Normalised target: isolates surface shape from vol level ---
    df["normalizedIV"] = df["midIV"] / df["atmIV"]

    # --- Drop rows where rolling warmup leaves NaNs (first ~30 days per ticker) ---
    n1 = len(df)
    df = df.dropna(subset=["realizedVol", "recentReturn", "atmIV", "normalizedIV"])
    print(f"[featurize] NaN drop (rolling warmup): {n1:,} → {len(df):,} ({n1 - len(df):,} dropped)")

    # --- Split labels (also drops held-out rows outside test window) ---
    df = _assign_splits(df)

    # --- Final column order ---
    df = df[[
        # Identifiers
        "tradingDate", "ticker", "split",
        # Inputs shared by all models
        "logMoneyness", "years", "maturityBucket",
        # Inputs for NN only (PROJECT_PLAN.md Section 5)
        "callPut", "atmIV", "realizedVol", "recentReturn",
        # Targets
        "normalizedIV",   # NN and polynomial regression
        "totalVariance",  # SVI
        # Metadata - kept for evaluation, not used as model inputs
        "midIV", "expiryDate", "okey_xx", "uClose", "rate", "ve",
    ]].sort_values(["tradingDate", "ticker"]).reset_index(drop=True)

    final_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(final_out, index=False)

    print(f"\n[featurize] final dataset: {len(df):,} rows → {final_out}")
    print("\nRows per split × ticker:")
    print(df.groupby(["split", "ticker"]).size().unstack(fill_value=0).to_string())


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build IV surface dataset from raw instructor zips.")
    parser.add_argument("--raw_dir",      type=Path, default=Path("data/raw"),
                        help="Directory containing the outer .zip files")
    parser.add_argument("--out",          type=Path, default=Path("data/processed/iv_surface.parquet"),
                        help="Path for the final processed Parquet file")
    parser.add_argument("--skip_extract", action="store_true",
                        help="Skip pass 1 if the intermediate file already exists")
    args = parser.parse_args()

    intermediate = args.out.parent / "iv_surface_raw.parquet"

    if args.skip_extract:
        if not intermediate.exists():
            raise FileNotFoundError(f"--skip_extract set but {intermediate} not found. Run without the flag first.")
        print(f"[extract] skipped - using {intermediate}")
    else:
        extract(args.raw_dir, intermediate)

    featurize(intermediate, args.out)
