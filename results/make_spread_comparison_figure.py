"""
make_spread_comparison_figure.py - Builds the Robustness & Discussion exhibit
comparing typical quoted market bid-ask spreads against the final model's
economic (vega-weighted) error, both in volatility points.

Spreads are computed directly from data/processed/iv_surface_raw.parquet
(median askIV - bidIV for near-ATM, ~30-day contracts) rather than asserted,
per the source cited in the report (Amaya, 2026 - instructor-provided options
data). Model error is read from results/nn_test_results.csv (nn_full row) so
the figure always reflects the currently trained model rather than a
hardcoded number.

Run from the project root:
    python results/make_spread_comparison_figure.py
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

NEAR_ATM_LOGMONEYNESS = 0.03
MATURITY_WINDOW_DAYS = (25, 35)


def median_spread_volpts(df: pd.DataFrame, ticker: str) -> float:
    d = df[df["undSecKey_tk"] == ticker].copy()
    d["logMoneyness"] = np.log(d["okey_xx"] / d["uClose"])
    lo, hi = (n / 365 for n in MATURITY_WINDOW_DAYS)
    mask = (
        (d["logMoneyness"].abs() < NEAR_ATM_LOGMONEYNESS)
        & d["years"].between(lo, hi)
        & (d["error"] == 0)
        & (d["bidIV"] > 0)
        & (d["askIV"] > 0)
    )
    spread = (d.loc[mask, "askIV"] - d.loc[mask, "bidIV"]) * 100
    return float(spread.median())


def main():
    raw = pd.read_parquet(RESULTS_DIR.parent / "data" / "processed" / "iv_surface_raw.parquet")
    spy_spread = median_spread_volpts(raw, "SPY")
    nvda_spread = median_spread_volpts(raw, "NVDA")

    nn = pd.read_csv(RESULTS_DIR / "nn_test_results.csv")
    final = nn[nn.model == "nn_full"].set_index("split")
    model_temporal = float(final.loc["test_temporal", "vw_rmse_volpts"])
    model_ticker = float(final.loc["test_ticker", "vw_rmse_volpts"])

    labels = [
        "SPY spread\n(median, near-ATM 30d)",
        "NVDA spread\n(median, near-ATM 30d)",
        "Model error\n(temporal, vega-wtd)",
        "Model error\n(ticker, vega-wtd)",
    ]
    values = [spy_spread, nvda_spread, model_temporal, model_ticker]
    colors = ["silver", "darkgray", "steelblue", "tomato"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, values, color=colors, alpha=0.9, width=0.55)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Volatility points")
    ax.set_title("Typical Quoted Spread vs. Model Error (Rung 4, final NN)")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(fontsize=9)

    plt.tight_layout()
    out = FIG_DIR / "spread_vs_error.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"SPY median spread (vol pts):  {spy_spread:.4f}")
    print(f"NVDA median spread (vol pts): {nvda_spread:.4f}")
    print(f"Model vega-wtd RMSE (temporal): {model_temporal:.4f}")
    print(f"Model vega-wtd RMSE (ticker):   {model_ticker:.4f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
