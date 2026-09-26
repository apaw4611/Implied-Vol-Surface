"""
make_summary_table.py - Renders a single report-ready table image summarizing
every test metric (statistical + economic) for every model in the ladder,
across both test splits.

Reads results/model_comparison.csv (+ FLAT_volpts from nn_test_results.csv for
the constant baseline, which isn't itself a row in model_comparison.csv).

Columns: R2, RMSE (vol pts), vega-weighted RMSE (vol pts), butterfly %, calendar %
Rows: constant, poly (2-feat), NN (2-feat), poly (5-feat), NN (5-feat, final), SVI (floor)
One table per test split (test_temporal, test_ticker), stacked in one figure.

Run from the project root:
    python results/make_summary_table.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent
FIG_DIR = RESULTS_DIR / "figures" / "results_summary"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["test_temporal", "test_ticker"]
SPLIT_TITLES = {
    "test_temporal": "test_temporal  (trained tickers, Nov–Dec)",
    "test_ticker": "test_ticker  (held-out tickers, Nov–Dec)",
}

ROWS = [
    ("constant",  "0  Constant (train mean)"),
    ("poly_2feat", "1  Polynomial (2-feat)"),
    ("nn_raw",    "2  NN (2-feat)"),
    ("poly_5feat", "3  Polynomial (5-feat)"),
    ("nn_full",   "4  NN (5-feat, final)"),
    ("svi",       "5  SVI (in-sample floor)"),
]
COLUMNS = ["R²", "RMSE (vol pts)", "Vega-wtd RMSE", "Butterfly %", "Calendar %"]


def _fmt(v, pct=False, dash_if_nan=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–" if dash_if_nan else "0.00"
    return f"{v:.2f}%" if pct else f"{v:.3f}"


def load_data():
    comparison = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    nn = pd.read_csv(RESULTS_DIR / "nn_test_results.csv")

    flat = (
        nn[["split", "FLAT_volpts"]]
        .drop_duplicates()
        .rename(columns={"FLAT_volpts": "rmse_volpts"})
    )
    flat["model"] = "constant"
    flat["r2"] = 0.0
    flat["vw_rmse_volpts"] = np.nan
    flat["bf_violation_pct"] = np.nan
    flat["cal_violation_pct"] = np.nan

    return pd.concat([comparison, flat], ignore_index=True)


def build_table_data(df: pd.DataFrame, split: str):
    rows = []
    for model_key, row_label in ROWS:
        r = df[(df.model == model_key) & (df.split == split)]
        if len(r) == 0:
            rows.append([row_label, "–", "–", "–", "–", "–"])
            continue
        r = r.iloc[0]
        rows.append([
            row_label,
            _fmt(r.r2),
            _fmt(r.rmse_volpts),
            _fmt(r.vw_rmse_volpts),
            _fmt(r.bf_violation_pct, pct=True),
            _fmt(r.cal_violation_pct, pct=True),
        ])
    return rows


MODEL_COL_WIDTH = 0.30
OTHER_COL_WIDTH = (1.0 - MODEL_COL_WIDTH) / len(COLUMNS)


def draw_table(ax, split: str, df: pd.DataFrame):
    data = build_table_data(df, split)
    col_labels = ["Model"] + COLUMNS
    col_widths = [MODEL_COL_WIDTH] + [OTHER_COL_WIDTH] * len(COLUMNS)

    tbl = ax.table(
        cellText=data,
        colLabels=col_labels,
        colWidths=col_widths,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", weight="bold")
        else:
            model_key = ROWS[row - 1][0]
            if model_key == "nn_full":
                cell.set_facecolor("#eaf2fb")
                cell.set_text_props(weight="bold")
            elif model_key == "svi":
                cell.set_facecolor("#fdf2e9")
            elif row % 2 == 0:
                cell.set_facecolor("#f7f7f7")
        if col == 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.02

    ax.axis("off")
    ax.set_title(SPLIT_TITLES[split], fontsize=12, fontweight="bold", pad=14)


def main():
    df = load_data()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5))
    for ax, split in zip(axes, SPLITS):
        draw_table(ax, split, df)

    fig.suptitle("Test Set Metrics — Statistical and Economic Summary",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = FIG_DIR / "results_summary_table.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
