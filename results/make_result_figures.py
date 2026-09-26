"""
make_result_figures.py - Builds the two headline results figures for the
presentation/report from results/model_comparison.csv and results/nn_test_results.csv.

Figure 1 (statistical) - results_ladder_rmse.png
    Grouped bar chart of vol-pt RMSE across the ablation ladder
    (constant -> poly 2-feat -> NN 2-feat -> poly 5-feat -> NN 5-feat (final) ->
    SVI in-sample floor), split by test_temporal / test_ticker. Matches the
    six-rung ladder in neural_network.ipynb section 5 (see nn_ladder.csv).

Figure 2 (economic) - results_economic.png
    Two panels:
    (a) butterfly + calendar arbitrage violation rate by rung
    (b) unweighted vs. vega-weighted RMSE for the final model (nn_full, 5-feat) -
        the error that actually costs money vs. the raw number

Run from the notebooks/results working tree:
    python results/make_result_figures.py
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
SPLIT_LABELS = {"test_temporal": "test_temporal\n(trained tickers)",
                "test_ticker": "test_ticker\n(held-out tickers)"}

# The six-rung ladder as validated in neural_network.ipynb section 5 / nn_ladder.csv.
RUNGS = [
    ("constant",  "0  constant"),
    ("poly_2feat", "1  poly (2-feat)"),
    ("nn_raw",    "2  NN (2-feat)"),
    ("poly_5feat", "3  poly (5-feat)"),
    ("nn_full",   "4  NN (5-feat, final)"),
    ("svi",       "5  SVI (floor)"),
]


def load_data():
    comparison = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    nn = pd.read_csv(RESULTS_DIR / "nn_test_results.csv")

    # "constant" rows aren't in model_comparison.csv - pull FLAT_volpts from the
    # NN results (identical across nn_raw/nn_full for a given split).
    flat = (
        nn[["split", "FLAT_volpts"]]
        .drop_duplicates()
        .rename(columns={"FLAT_volpts": "rmse_volpts"})
    )
    flat["model"] = "constant"
    flat["bf_violation_pct"] = 0.0
    flat["cal_violation_pct"] = 0.0
    flat["vw_rmse_volpts"] = np.nan

    combined = pd.concat([comparison, flat], ignore_index=True)
    return combined


def figure1_ladder(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))

    n_rungs = len(RUNGS)
    x = np.arange(n_rungs)
    width = 0.35
    colors = {"test_temporal": "steelblue", "test_ticker": "tomato"}

    for i, split in enumerate(SPLITS):
        vals = []
        for model_key, _ in RUNGS:
            row = df[(df.model == model_key) & (df.split == split)]
            vals.append(row.rmse_volpts.iloc[0] if len(row) else np.nan)
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=SPLIT_LABELS[split],
                      color=colors[split], alpha=0.85)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.15,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in RUNGS], fontsize=9)
    ax.set_ylabel("RMSE (vol points)")
    ax.set_title("Ablation ladder: prediction error by rung")
    ax.axvline(4.5, color="gray", linestyle=":", alpha=0.6)
    ax.legend(fontsize=9, loc="upper right", title="  ↑ rung 5 is in-sample",
              title_fontsize=7.5, alignment="left")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "results_ladder_rmse.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def figure2_economic(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Panel (a): arbitrage violations by rung, averaged across test splits ---
    ax = axes[0]
    rung_keys = [k for k, _ in RUNGS if k != "constant"]
    rung_labels = [lbl for k, lbl in RUNGS if k != "constant"]
    bf_vals, cal_vals = [], []
    for model_key in rung_keys:
        sub = df[(df.model == model_key) & (df.split.isin(SPLITS))]
        bf_vals.append(sub.bf_violation_pct.mean())
        cal_vals.append(sub.cal_violation_pct.mean())

    x = np.arange(len(rung_keys))
    width = 0.35
    ax.bar(x - width / 2, bf_vals, width, label="Butterfly", color="darkorange", alpha=0.85)
    ax.bar(x + width / 2, cal_vals, width, label="Calendar", color="seagreen", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(rung_labels, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Violation rate (%, avg of both test sets)")
    ax.set_title("(a) Arbitrage violations by rung")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Panel (b): unweighted vs vega-weighted RMSE, final model only ---
    ax = axes[1]
    final = df[df.model == "nn_full"].set_index("split")
    x = np.arange(len(SPLITS))
    width = 0.35
    unweighted = [final.loc[s, "rmse_volpts"] for s in SPLITS]
    weighted = [final.loc[s, "vw_rmse_volpts"] for s in SPLITS]
    bars1 = ax.bar(x - width / 2, unweighted, width, label="Unweighted", color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + width / 2, weighted, width, label="Vega-weighted", color="mediumpurple", alpha=0.85)
    for bars in (bars1, bars2):
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLITS], fontsize=9)
    ax.set_ylabel("RMSE (vol points)")
    ax.set_title("(b) Raw error vs. the error that costs money")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Economic context: arbitrage cost and vega-weighted error", fontsize=11, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "results_economic.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    df = load_data()
    figure1_ladder(df)
    figure2_economic(df)


if __name__ == "__main__":
    main()
