"""
aggregate_results.py - Stitches per-model result CSVs into one comparison table.

Reads (must exist - run the notebooks first):
    poly_test_results.csv        - polynomial, 2 features (logMoneyness, years)
    poly_test_results_5feat.csv  - polynomial, 5 features (+ atmIV, realizedVol, recentReturn)
    svi_test_results.csv         - SVI, in-sample calibration floor (all four splits)
    nn_test_results.csv          - NN rungs 2/4 (nn_raw / nn_full)

Writes:
    model_comparison.csv - one row per (model, split), common column set:
        model, split, r2, rmse_volpts, vw_rmse_volpts, bf_violation_pct, cal_violation_pct

Notes:
- vw_rmse_volpts (vega-weighted RMSE) is only computed in the NN notebook - poly and
  SVI rows carry NaN there rather than a fabricated number.
- SVI has no R2 - it is a per-slice fitting residual, not a variance-explained metric
  (see svi.ipynb "Evaluation" section). Its r2 column is NaN by construction.
- Split labels in the poly/SVI CSVs are long descriptive strings (e.g.
  "test_temporal  (trained tickers, Nov-Dec)"); these are normalised to the plain
  split names (train/val/test_temporal/test_ticker) used by the NN CSV.
"""

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).parent

SPLIT_NAMES = ["train", "val", "test_temporal", "test_ticker"]


def _normalise_split(raw: str) -> str:
    """Map a messy split label (possibly with trailing description) to a canonical name."""
    raw = str(raw).strip()
    for name in SPLIT_NAMES:
        if raw.startswith(name):
            return name
    raise ValueError(f"Unrecognised split label: {raw!r}")


def _load_poly(path: Path, model_label: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df = df.rename(columns={
        "r2": "r2",
        "rmse_volpts": "rmse_volpts",
        "bf_violation_pct": "bf_violation_pct",
        "cal_violation_pct": "cal_violation_pct",
    })
    df["model"] = model_label
    df["split"] = [_normalise_split(s) for s in df.index]
    df["vw_rmse_volpts"] = float("nan")
    return df[["model", "split", "r2", "rmse_volpts", "vw_rmse_volpts",
               "bf_violation_pct", "cal_violation_pct"]]


def _load_svi(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df["model"] = "svi"
    df["split"] = [_normalise_split(s) for s in df.index]
    df["r2"] = float("nan")
    df["vw_rmse_volpts"] = float("nan")
    return df[["model", "split", "r2", "rmse_volpts", "vw_rmse_volpts",
               "bf_violation_pct", "cal_violation_pct"]]


def _load_nn(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "R2": "r2",
        "butterfly_%": "bf_violation_pct",
        "calendar_%": "cal_violation_pct",
    })
    return df[["model", "split", "r2", "rmse_volpts", "vw_rmse_volpts",
               "bf_violation_pct", "cal_violation_pct"]]


def main() -> None:
    poly_2feat = _load_poly(RESULTS_DIR / "poly_test_results.csv", "poly_2feat")
    poly_5feat = _load_poly(RESULTS_DIR / "poly_test_results_5feat.csv", "poly_5feat")
    svi        = _load_svi(RESULTS_DIR / "svi_test_results.csv")
    nn         = _load_nn(RESULTS_DIR / "nn_test_results.csv")

    combined = pd.concat([poly_2feat, poly_5feat, svi, nn], ignore_index=True)

    model_order = ["poly_2feat", "nn_raw", "poly_5feat", "nn_full", "svi"]
    combined["model"] = pd.Categorical(combined["model"], categories=model_order, ordered=True)
    combined["split"] = pd.Categorical(combined["split"], categories=SPLIT_NAMES, ordered=True)
    combined = combined.sort_values(["split", "model"]).reset_index(drop=True)

    out_path = RESULTS_DIR / "model_comparison.csv"
    combined.to_csv(out_path, index=False)
    print(f"Wrote {out_path}\n")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
