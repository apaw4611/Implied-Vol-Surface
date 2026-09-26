"""
nn_utils.py - shared machinery for the feedforward IV surface network.

Kept out of the notebook so the ablation ladder reads as a sequence of
experiments rather than a wall of boilerplate. Arbitrage checks are lifted
verbatim from notebooks/polynomial.ipynb so the two models are scored by
identical code.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

SEED = 42


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_std_moneyness(df: pd.DataFrame) -> pd.DataFrame:
    """
    stdMoneyness = logMoneyness / (atmIV * sqrt(years))

    This is -d2 from Black-Scholes up to the drift term: the strike's distance
    from the money measured in STANDARD DEVIATIONS of the log price over the
    option's life, rather than in raw log-units.

    Motivation: a smile's width scales with atmIV*sqrt(T). At logMoneyness
    = -0.20, a 10-day SPY put (atmIV 0.124) sits 5.6 sigma out while a 1-year
    NVDA call (atmIV 0.471) sits 0.4 sigma out. Any model whose only inputs are
    (logMoneyness, years) is required to return the same value for both.
    """
    df = df.copy()
    df["stdMoneyness"] = df["logMoneyness"] / (df["atmIV"] * np.sqrt(df["years"]))
    return df


def support_bounds(train_df: pd.DataFrame, col: str = "stdMoneyness",
                   q: tuple[float, float] = (0.001, 0.999)) -> tuple[float, float]:
    lo, hi = train_df[col].quantile(list(q))
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SurfaceNet(nn.Module):
    """
    Feedforward net, 3 hidden layers. Course material (classes 8-9) covers
    exactly this architecture; project brief C3 asks for two to three hidden
    layers.

    Predicts log(normalizedIV). Exponentiating the output guarantees a strictly
    positive IV without a constrained output layer, and matches the polynomial
    baseline's target transform so the two are directly comparable.
    """

    def __init__(self, n_features: int, hidden=(64, 32, 16), dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    hidden=(64, 32, 16),
    dropout: float = 0.0,
    epochs: int = 40,
    batch_size: int = 8192,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 5,
    clip_col: str | None = None,
    clip_bounds: tuple[float, float] | None = None,
    sample_frac: float | None = None,
    verbose: bool = True,
):
    """
    Fit on train, early-stop on val. Returns (model, scaler, history).

    Early stopping uses validation MSE in log space - the training objective -
    not vol-point RMSE. Selecting on a different metric than you optimise is a
    common way to pick a model that is worse at the thing you actually trained.

    clip_col/clip_bounds: clip a feature to the training support. Mandatory for
    stdMoneyness. For a 7-day contract on a low-vol name atmIV*sqrt(T) is about
    0.026, so logMoneyness -0.5 maps to stdMoneyness -19. A network extrapolating
    that far outside its training hull produces unbounded output.
    """
    set_seed()
    tr = train_df.sample(frac=sample_frac, random_state=SEED) if sample_frac else train_df

    def prep(d):
        X = d[features].copy()
        if clip_col and clip_col in features and clip_bounds:
            X[clip_col] = X[clip_col].clip(*clip_bounds)
        return X.to_numpy(dtype=np.float32)

    Xtr_raw, Xva_raw = prep(tr), prep(val_df)
    scaler = StandardScaler().fit(Xtr_raw)
    Xtr = torch.from_numpy(scaler.transform(Xtr_raw).astype(np.float32))
    Xva = torch.from_numpy(scaler.transform(Xva_raw).astype(np.float32))
    ytr = torch.from_numpy(np.log(tr["normalizedIV"].to_numpy(dtype=np.float32)))
    yva = torch.from_numpy(np.log(val_df["normalizedIV"].to_numpy(dtype=np.float32)))

    model = SurfaceNet(len(features), hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.MSELoss()

    n = len(Xtr)
    best_val, best_state, bad, history = float("inf"), None, 0, []
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        running = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        train_loss = running / n

        model.eval()
        with torch.no_grad():
            val_loss = float(lossf(model(Xva), yva))
        history.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss})

        if verbose:
            print(f"  epoch {ep:3d}  train {train_loss:.5f}  val {val_loss:.5f}"
                  f"{'  *' if val_loss < best_val else ''}")

        if val_loss < best_val - 1e-6:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep} (best val {best_val:.5f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    if verbose:
        print(f"  trained in {time.time()-t0:.1f}s on {n:,} rows")
    return model, scaler, pd.DataFrame(history)


def predict(model, scaler, df, features, clip_col=None, clip_bounds=None):
    """Return predicted normalizedIV. Output clipped to [0.1, 20] - values
    outside that range are not a price under any surface."""
    X = df[features].copy()
    if clip_col and clip_col in features and clip_bounds:
        X[clip_col] = X[clip_col].clip(*clip_bounds)
    Xs = torch.from_numpy(scaler.transform(X.to_numpy(dtype=np.float32)).astype(np.float32))
    with torch.no_grad():
        out = model(Xs).numpy()
    return np.exp(np.clip(out, np.log(0.1), np.log(20)))


# ---------------------------------------------------------------------------
# Arbitrage checks - identical to notebooks/polynomial.ipynb
# ---------------------------------------------------------------------------

def _bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _bs_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def compute_pred_prices(df, pred_iv):
    S, K, T, r = (df["uClose"].values, df["okey_xx"].values,
                  df["years"].values, df["rate"].values)
    is_call = df["callPut"].values == 1
    return np.where(is_call, _bs_call(S, K, T, r, pred_iv), _bs_put(S, K, T, r, pred_iv))


def butterfly_violations(df, tol=1e-8):
    """Second derivative of price w.r.t. strike must be >= 0. This is a
    no-arbitrage condition, not a modelling convention: a negative second
    derivative means a butterfly spread has negative cost and positive payoff."""
    grp = ["ticker", "tradingDate", "expiryDate", "callPut"]
    d = df.sort_values(grp + ["okey_xx"])
    p_prev = d.groupby(grp)["pred_price"].shift(1)
    p_next = d.groupby(grp)["pred_price"].shift(-1)
    k_prev = d.groupby(grp)["okey_xx"].shift(1)
    k_next = d.groupby(grp)["okey_xx"].shift(-1)
    h1, h2 = d["okey_xx"] - k_prev, k_next - d["okey_xx"]
    valid = (h1 > 0) & (h2 > 0)
    sd = (2.0 / (h1 + h2)) * ((p_next - d["pred_price"]) / h2
                              - (d["pred_price"] - p_prev) / h1)
    checks = int(valid.sum())
    viol = int((valid & (sd < -tol)).sum())
    return (viol / checks if checks else 0.0), checks


def calendar_violations(df, tol=1e-8):
    """Total implied variance must be non-decreasing in maturity at a fixed
    strike, else a calendar spread is riskless profit."""
    grp = ["ticker", "tradingDate", "okey_xx", "callPut"]
    d = df.sort_values(grp + ["years"])
    nxt = d.groupby(grp)["pred_total_var"].shift(-1)
    valid = nxt.notna()
    checks = int(valid.sum())
    viol = int((valid & (nxt < d["pred_total_var"] - tol)).sum())
    return (viol / checks if checks else 0.0), checks


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(df, pred_norm, label, train_mean_normIV, check_arbitrage=True):
    """
    Full scorecard for one split.

    rmse_volpts is the headline unit but is dominated by deep-OTM contracts with
    negligible vega, so vega-weighted RMSE is reported alongside it - that is the
    error that has economic consequence.
    """
    act = df["normalizedIV"].to_numpy()
    atm = df["atmIV"].to_numpy()
    err_vol = (act - pred_norm) * atm
    mse = mean_squared_error(act, pred_norm)

    row = {
        "split": label,
        "R2": 1 - mse / np.var(act),
        "rmse_normIV": np.sqrt(mse),
        "rmse_volpts": 100 * np.sqrt(mean_squared_error(act * atm, pred_norm * atm)),
        "FLAT_volpts": 100 * np.sqrt(mean_squared_error(
            act * atm, np.full(len(df), train_mean_normIV) * atm)),
    }

    if "ve" in df.columns:
        w = df["ve"].abs().to_numpy()
        if w.sum() > 0:
            row["vw_rmse_volpts"] = 100 * np.sqrt(np.average(err_vol ** 2, weights=w))

    if check_arbitrage:
        d = df.copy()
        pred_iv = pred_norm * atm
        d["pred_price"] = compute_pred_prices(df, pred_iv)
        d["pred_total_var"] = pred_iv ** 2 * df["years"].to_numpy()
        bf, _ = butterfly_violations(d)
        cal, _ = calendar_violations(d)
        row["butterfly_%"] = 100 * bf
        row["calendar_%"] = 100 * cal

    return row


def per_ticker(df, pred_norm):
    """Pooled R2 can hide negative within-ticker R2. On the polynomial baseline
    SPY, BA and LLY are all worse than predicting each ticker's own mean."""
    d = df.copy()
    d["_pred"] = pred_norm
    return d.groupby("ticker").apply(lambda g: pd.Series({
        "n": len(g),
        "share_%": 100 * len(g) / len(d),
        "atmIV": g["atmIV"].mean(),
        "R2": 1 - mean_squared_error(g["normalizedIV"], g["_pred"]) / np.var(g["normalizedIV"]),
        "rmse_normIV": np.sqrt(mean_squared_error(g["normalizedIV"], g["_pred"])),
        "rmse_volpts": 100 * np.sqrt(mean_squared_error(
            g["normalizedIV"] * g["atmIV"], g["_pred"] * g["atmIV"])),
    }), include_groups=False)


def bucket_errors(df, pred_norm):
    """RMSE by maturity and moneyness bucket - the error analysis the milestone
    listed as remaining work."""
    d = df.copy()
    d["_err_vol"] = (d["normalizedIV"] - pred_norm) * d["atmIV"]
    d["mBucket"] = pd.cut(d["logMoneyness"], [-np.inf, -0.3, -0.1, 0.1, 0.3, np.inf],
                          labels=["deep OTM put", "OTM put", "near ATM",
                                  "OTM call", "deep OTM call"])
    d["tBucket"] = pd.cut(d["years"], [0, 0.08, 0.25, 0.5, np.inf],
                          labels=["<1m", "1-3m", "3-6m", ">6m"])
    return (d.groupby(["tBucket", "mBucket"], observed=True)["_err_vol"]
            .agg(n="size", rmse_volpts=lambda s: 100 * np.sqrt((s ** 2).mean()))
            .reset_index()
            .pivot(index="tBucket", columns="mBucket", values="rmse_volpts"))
