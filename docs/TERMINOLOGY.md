# Project Terminology

Definitions for terms used in this project's code, notebooks, and documentation. Standard options concepts that any finance practitioner would know (delta, vega, strike, expiry) are not included.

---

## Target Variable

### `midIV`
The midpoint of the bid-ask spread of implied volatility for a specific option contract:
```
midIV = (bidIV + askIV) / 2
```
Implied volatility (IV) is the volatility parameter that makes the Black-Scholes formula reproduce the option's market price. Every contract has its own IV. `midIV` is the best single-number estimate of the market's consensus IV for that contract.

### `atmIV`
The implied volatility of the nearest at-the-money option in the **20–40 day maturity window** for a given ticker on a given trading day. It anchors the overall level of volatility for that stock-day.

Implementation: for each (ticker, tradingDate) pair, find all contracts with `years` between 20/365 and 40/365, pick the one with the smallest `|logMoneyness|` (closest to ATM), and take its `midIV`. If no contract falls in the 20–40 day window, fall back to the contract whose maturity is closest to 30 days.

### `normalizedIV`
The surface shape target - the primary variable predicted by both the polynomial and neural network models:
```
normalizedIV = midIV / atmIV
```
Dividing by `atmIV` removes the overall vol level and leaves only the shape of the surface relative to ATM. A value of 1.3 means that contract trades at 30% above the ATM vol level for that stock on that day, regardless of whether the market is calm (atmIV=15%) or stressed (atmIV=50%). This allows the model to train jointly across tickers with very different vol levels.

---

## SVI-Specific

### `totalVariance`
Total implied variance - the target variable used by the SVI model instead of IV directly:
```
totalVariance = midIV² × years
```
SVI fits total variance rather than IV because working in variance space has desirable mathematical properties for enforcing no-arbitrage constraints. Standard in the SVI literature (Gatheral & Jacquier, 2014).

### `maturityBucket`
A discretised maturity label used to group contracts into standard tenor slices for SVI fitting. SVI must be fit independently per maturity slice; `maturityBucket` defines those slices:

| Bucket | Days to expiry |
|---|---|
| 7d | 0–14 |
| 14d | 14–21 |
| 30d | 21–45 |
| 60d | 45–75 |
| 90d | 75–135 |
| 180d | 135–225 |
| 1Y | 225+ |

---

## Features

### `logMoneyness`
The log ratio of strike to spot price:
```
logMoneyness = log(K / S)
```
Negative values = out-of-the-money puts (strike below spot), zero = at-the-money, positive = out-of-the-money calls. Standard in the options literature; preferred over raw strike because it is scale-invariant across stocks.

### `recentReturn`
The 20-trading-day log return of the underlying:
```
recentReturn = log(S_t / S_{t-20})
```
Captures momentum and the leverage effect: sharp falls steepen put skew; strong rallies can build call skew (as seen in NVDA in 2024). Named `recentReturn` in the dataset rather than something more generic because its role is specifically to capture recent directional momentum.

### `realizedVol`
30-trading-day rolling annualised volatility of log returns:
```
realizedVol = std(log(S_t / S_{t-1}), trailing 30 days) × sqrt(252)
```
Computed from the `uClose` column (unadjusted close price). The gap between `realizedVol` and `atmIV` is the variance risk premium - one of the strongest known drivers of skew steepness. **Note:** a stock split detection heuristic is applied before computing log returns to prevent split-day price jumps from contaminating the rolling window (see `build_dataset.py`).

---

## Data Splits

### `train`
January–August 2024, training tickers only. Used to fit all models. Contains the August 5 VIX spike (largest single-day VIX move in history), which makes this a challenging and diverse training set.

### `val`
September–October 2024, training tickers only. Used exclusively for hyperparameter selection (degree, alpha, NN architecture). **Not used for final reported performance.** Any number reported from val is a tuning metric, not an out-of-sample result.

### `test_temporal`
November–December 2024, training tickers (SPY, NVDA, AAPL, LLY, GS, COST, BA, EOG). Measures **temporal generalisation**: does the model hold up on future market conditions it has never seen? The post-election vol collapse in Nov–Dec makes this a genuinely different regime from training.

### `test_ticker`
November–December 2024, held-out tickers (MSFT, JPM, TSLA, XOM). Measures **cross-sectional generalisation**: does the model transfer to stocks it was never trained on? These tickers are never seen during training or validation.

Both test sets cover the same calendar period intentionally - any performance difference between `test_temporal` and `test_ticker` is attributable to ticker familiarity, not market conditions.

---

## Arbitrage Conditions

### Butterfly arbitrage (across strikes)
Within a fixed maturity, option prices must be convex in strike - the second derivative of price with respect to strike must be non-negative. A violation means a butterfly spread (long low-strike, short two mid-strike, long high-strike) can be bought for a negative price: a riskless profit. Checked separately for calls and puts.

### Calendar spread arbitrage (across maturities)
For a fixed strike, total implied variance must be non-decreasing in time to expiry. A violation means a calendar spread (long long-dated, short short-dated at the same strike) can be bought for a negative price: a riskless profit. Checked using `pred_total_var = pred_midIV² × years`.

SVI is butterfly-free by construction (when parameter constraints are satisfied) but calendar arbitrage across maturity slices must be checked separately. The polynomial and NN have no structural guarantee on either condition.
