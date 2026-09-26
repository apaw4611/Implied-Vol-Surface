"""
make_poly_fit_figure.py - Builds a fitted-smile figure for the 2-feature
polynomial baseline, in the same style as results/figures/svi/svi_example_fit.png,
for direct visual comparison against the SVI curve on the same slice.

Refits the polynomial exactly as in polynomial.ipynb (degree 5, alpha=2.8943e5,
log-target, standardized features) since the fitted model itself isn't
persisted anywhere. Then, for one (ticker, date, maturityBucket) slice, plots
the actual market quotes against the polynomial's predicted curve across
logMoneyness, holding years fixed at that slice's average maturity - the same
NVDA / 30d-bucket slice SVI's example figure uses.

Run from the project root:
    python results/make_poly_fit_figure.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

RESULTS_DIR = Path(__file__).parent
FIG_DIR = RESULTS_DIR / "figures" / "polynomial"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEGREE = 5
ALPHA = 2.8943e5
EX_TICKER, EX_BUCKET = "NVDA", "30d"


def fit_polynomial(train_df: pd.DataFrame):
    X = train_df[["logMoneyness", "years"]]
    y = np.log(train_df[["normalizedIV"]])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    poly = PolynomialFeatures(degree=DEGREE, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    poly_scaler = StandardScaler()
    X_poly = poly_scaler.fit_transform(X_poly)

    model = Ridge(alpha=ALPHA)
    model.fit(X_poly, y)
    return model, scaler, poly, poly_scaler


def predict(model, scaler, poly, poly_scaler, log_moneyness, years):
    X = np.column_stack([log_moneyness, years])
    X_scaled = scaler.transform(X)
    X_poly = poly.transform(X_scaled)
    X_poly = poly_scaler.transform(X_poly)
    return np.exp(model.predict(X_poly)).ravel()


def main():
    df = pd.read_parquet(RESULTS_DIR.parent / "data" / "processed" / "iv_surface.parquet")
    train_df = df[df.split == "train"]
    test_temp = df[df.split == "test_temporal"]

    print("Fitting 2-feature polynomial (degree 5, alpha=2.8943e5) on train...")
    model, scaler, poly, poly_scaler = fit_polynomial(train_df)

    grp = test_temp[(test_temp.ticker == EX_TICKER) & (test_temp.maturityBucket == EX_BUCKET)]
    best_date = grp.groupby("tradingDate").size().idxmax()
    slice_df = grp[grp.tradingDate == best_date].sort_values("logMoneyness")
    years_avg = slice_df["years"].mean()
    atm_iv = slice_df["atmIV"].iloc[0]

    k_data = slice_df["logMoneyness"].values
    pad = (k_data.max() - k_data.min()) * 0.08
    k_grid = np.linspace(k_data.min() - pad, k_data.max() + pad, 300)
    years_grid = np.full_like(k_grid, years_avg)

    pred_norm_iv = predict(model, scaler, poly, poly_scaler, k_grid, years_grid)
    pred_iv = pred_norm_iv * atm_iv * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(slice_df["logMoneyness"], slice_df["midIV"] * 100,
               s=25, color="steelblue", zorder=5, label="Market quotes", alpha=0.85)
    ax.plot(k_grid, pred_iv, color="tomato", linewidth=2, label="Polynomial fit (deg 5)")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.4)
    ax.set_xlabel("Log-moneyness  k = log(K/S)")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title(f"Polynomial baseline fit — {EX_TICKER}  |  "
                 f"{pd.Timestamp(best_date).strftime('%Y-%m-%d')}  |  {EX_BUCKET} bucket")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = FIG_DIR / "poly_example_fit.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}  ({len(slice_df)} contracts)")


if __name__ == "__main__":
    main()
