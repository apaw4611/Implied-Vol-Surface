# BU493 Project Plan - Neural Network Pricing of the Implied Volatility Surface

## 1. The Financial Question

The implied volatility (IV) surface maps every option contract on a given stock - across all strikes and maturities - to a single number: the volatility implied by its market price via the Black-Scholes formula. The surface is not flat. It has shape: a downward slope across strikes (the skew) and a term structure across maturities. Parametric models like SVI impose a fixed functional form on this shape. The question we are asking is whether a feedforward neural network, which imposes no such constraints, can learn the shape of the IV surface more accurately than these parametric benchmarks - and where and when it fails.

---

## 2. The Data

### Source
The instructor provided 12 zip files (~23GB compressed) containing daily close mark history for US equity options across all of 2024. Each outer zip contains ~21 inner daily zips, one per trading day, giving **252 files** covering the full calendar year. Each daily file is a tab-delimited text file with ~1.5 million rows and 59 columns.

**Important:** The original proposal described this as "5-minute intraday quotes for one month." The actual data is end-of-day close marks for a full year. EOD data is cleaner (consistent timestamps, less microstructure noise) and is the standard in academic IV surface literature. The proposal description should be updated accordingly.

### Key Columns
| Column | Description |
|---|---|
| `undSecKey_tk` | Underlying ticker |
| `okey_xx` | Strike price |
| `okey_yr/mn/dy` | Expiry date |
| `okey_cp` | Call or Put |
| `years` | Time to expiry in years |
| `uClose` | Underlying close price |
| `bidIV`, `askIV` | Bid and ask implied volatility |
| `rate` | Risk-free rate |
| `error` | Data quality flag |
| `openInterest` | Open interest |
| `ve` | Vega |

### Scale
Full extraction of all tickers is ~150GB uncompressed. The pipeline extracts only chosen tickers to keep the working dataset manageable.

### Data Use Policy
The instructor-provided data cannot be redistributed. Raw zip files are gitignored and must never be committed to the repository. Submit only code, not data.

---

## 3. Ticker Selection

### Why We Can't Use Everything
With 1.5 million rows per day across thousands of tickers, we must scope to a manageable subset. The choice of tickers drives the financial story, the model inputs, and the interpretation of results.

### Training Tickers
We selected a basket designed to give the model diversity across three dimensions: volatility level, sector, and what drives the underlying stock. A model trained only on high-vol names learns that all surfaces are spiky. A model trained only on SPY learns nothing cross-sectional.

| Ticker | Sector | Vol Profile | Key Driver |
|---|---|---|---|
| **SPY** | S&P 500 Index | Low-medium | Broad macro - the anchor |
| **NVDA** | Semiconductors | Very high | AI capex cycle, earnings-driven spikes, strong call skew |
| **AAPL** | Consumer tech | Low | Consumer spending, product cycles - calm surface |
| **LLY** | Pharma | Medium-high | GLP-1 narrative, drug approval risk |
| **GS** | Financials | Medium | Interest rates, deal flow |
| **COST** | Retail staples | Low | Consumer staples, membership model - very flat surface |
| **BA** | Industrials | High | Supply chain crises, regulatory, geopolitical - high vol for non-tech reasons |
| **EOG** | Energy | Medium | Oil prices, macro cycle - commodity-driven |

### Held-Out Ticker Test Set
Four additional tickers are held entirely out of training. They are used only in the November–December test window to evaluate cross-sectional generalisation - whether the model transfers what it learned to stocks it has never seen.

| Ticker | Sector | What It Tests |
|---|---|---|
| **MSFT** | Tech | Within-sector transfer from AAPL |
| **JPM** | Financials | Within-sector transfer from GS |
| **TSLA** | Consumer/EV | Out-of-distribution - nothing in training is like TSLA |
| **XOM** | Energy | Within-sector transfer from EOG |

**Why test only in Nov–Dec:** The macro regime is held constant between the temporal test (trained tickers, Nov–Dec) and the ticker test (held-out tickers, Nov–Dec). Any performance difference is then cleanly attributable to whether the model has seen the ticker before, not to differences in market conditions.

---

## 4. Financial Background

### Options Basics

An **option contract** gives the buyer the right - but not the obligation - to buy (call) or sell (put) an underlying asset at a fixed price (the strike, K) on or before an expiry date (T). The buyer pays a premium upfront for this right.

Key contract identifiers:
- **S** - current price of the underlying stock
- **K** - strike price
- **T** - time to expiry in years
- **r** - risk-free interest rate
- **σ** - volatility of the underlying (the key unknown)

### The Black-Scholes Formula

Black-Scholes (1973) gives a closed-form price for a European option assuming the underlying follows geometric Brownian motion with constant volatility σ:

**Call price:**
```
C(S, K, T, r, σ) = S·N(d₁) - K·e^{-rT}·N(d₂)
```

**Put price:**
```
P(S, K, T, r, σ) = K·e^{-rT}·N(-d₂) - S·N(-d₁)
```

where:
```
d₁ = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
d₂ = d₁ - σ·√T
```

and N(·) is the cumulative standard normal distribution function.

The formula has one unobservable input: σ. Everything else - S, K, T, r - is directly observable.

### What Implied Volatility Is

IV is a single number attached to one specific option contract. Rather than using an assumed σ to compute a price, you observe the market price and solve backwards for the σ that makes Black-Scholes reproduce it:

```
C_market = C_BS(S, K, T, r, σ_implied)  →  solve for σ_implied
```

There is no closed-form solution. IV is found numerically - typically via Newton-Raphson iteration - by repeatedly guessing σ until the formula price matches the market price.

IV is a property of a specific contract, not the stock. Every combination of strike and expiry has its own IV. If Black-Scholes were a perfect model, every contract on the same stock would imply the same σ - because the model assumes volatility is constant. In practice they don't, and the pattern of disagreement is what makes IV surfaces interesting.

### Put-Call Parity

In a frictionless market, call and put prices on the same strike and expiry are linked by:

```
C - P = S - K·e^{-rT}
```

This means the IV of a call and its corresponding put at the same strike must theoretically be equal. In practice small violations occur due to liquidity differences and borrowing costs - which is why we include the call/put indicator as a model input.

### The Greeks

The Greeks are partial derivatives of the option price with respect to its inputs. They measure sensitivity:

| Greek | Symbol | Definition | Meaning |
|---|---|---|---|
| Delta | Δ | ∂C/∂S = N(d₁) | How much the option price moves per $1 move in the stock |
| Gamma | Γ | ∂²C/∂S² | Rate of change of delta - curvature of the price-spot relationship |
| Vega | ν | ∂C/∂σ = S·√T·n(d₁) | How much the option price moves per 1% move in volatility |
| Theta | Θ | ∂C/∂t | Time decay - how much value the option loses per day |
| Rho | ρ | ∂C/∂r | Sensitivity to interest rates |

Vega is particularly important here: it tells you how sensitive an option price is to changes in IV. High-vega contracts (long-dated, near-ATM) are the most sensitive to the shape of the IV surface - which is where model accuracy matters most in practice.

### The Volatility Smile and Skew

If Black-Scholes were correct, a plot of IV against strike for a fixed expiry would be a flat horizontal line - all contracts would imply the same σ. In reality the curve is not flat:

**Equity options (indices like SPY):** IV is highest for low-strike puts (OTM puts, crash protection) and lowest near-ATM, then rises slightly for high-strike calls. The curve slopes downward to the right - called the **volatility smirk** or **skew**. The economic reason: investors pay a premium to protect against market crashes, inflating the price - and therefore the IV - of OTM puts.

**Single-name stocks:** The smile is less regular. High-momentum stocks like NVDA can develop elevated call-side IV when investors speculate on upside. Stocks approaching earnings show a symmetric spike in IV across all strikes near the earnings date.

### The Term Structure

IV also varies across maturities. For a fixed strike, a plot of IV against time-to-expiry is the **volatility term structure**:

- **Normal (contango):** Long-dated IV > short-dated IV. Markets expect vol to be higher further out than it is today.
- **Inverted (backwardation):** Short-dated IV > long-dated IV. Happens during stress - investors are panicking about the next few weeks, not next year.

August 5, 2024 (the yen carry trade unwind) is a clear example of term structure inversion: near-term SPY options spiked dramatically while long-dated options moved less.

### The IV Surface

Put the smile and term structure together. The **IV surface** is a 2D landscape:
- X-axis: log-moneyness (where the strike is relative to spot)
- Y-axis: time to expiry
- Z-axis: implied volatility

Each row in our dataset is one point on this surface, observed at market close on one trading day. The surface shifts shape every day as the market reprices risk. Our models are learning the mapping from (logMoneyness, years, stock characteristics) to the height of the surface at that point.

### The Variance Risk Premium

A key economic concept motivating several of our features:

```
VRP ≈ atmIV - realizedVol
```

The variance risk premium is the difference between what the market implies future volatility will be (atmIV) and what volatility has actually been (realizedVol). It is almost always positive for equity indices - investors systematically overpay for protection relative to realised outcomes. This premium:

- Is the primary reason put skew exists (crash protection is expensive relative to its realised payoff)
- Varies over time and across stocks
- Is a strong predictor of the steepness of the volatility smile

Including both atmIV and realizedVol as features gives the model direct access to the VRP, which is one of the strongest known drivers of surface shape.

---

## 5. What Implied Volatility Is (Summary)

IV is a single number attached to one specific option contract. You take the market price of the option, plug it into Black-Scholes, and solve backwards for the volatility input that reproduces that price. It is a property of the contract, not the stock.

If Black-Scholes were perfect, every contract on the same stock would imply the same volatility. In practice they don't:

- **Across strikes (the smile/skew):** Deep OTM puts have higher IV than ATM options. Investors pay a crash protection premium. For equity indices this creates a pronounced downward slope - the volatility smirk.
- **Across maturities (the term structure):** Near-dated and long-dated options on the same stock imply different volatilities. In calm markets long-dated IV is typically higher. In stressed markets near-dated IV spikes.

The **IV surface** is the 2D landscape of IV across all strikes and maturities for a given stock on a given day. Each row in the dataset is one point on that surface.

---

## 5. Feature Engineering

### Why We Need More Than Strike and Maturity

When training on a single underlying (SPY only), log-moneyness and time-to-expiry fully describe where a contract sits on the surface - the only variation across rows is strike and expiry. When training on multiple underlyings, two contracts with identical log-moneyness and time-to-expiry can have wildly different IVs (e.g. SPY ATM 30-day at 15% vs NVDA ATM 30-day at 45%). The model needs additional inputs to understand why.

### The Target Variable

Rather than predicting raw mid-IV directly, we predict **normalizedIV**:

```
midIV       = (bidIV + askIV) / 2
normalizedIV = midIV / atmIV
```

This separates the model's job into two clean pieces: `atmIV` captures the overall level of volatility for that stock-day, and `normalizedIV` captures the shape of the surface around that level. A value of 1.2 means the contract trades 20% above ATM vol - that is the skew premium the model is learning to predict.

### The Six Inputs

| Feature | Formula | Source | Why It's Needed |
|---|---|---|---|
| `logMoneyness` | `log(K / S)` | Raw data | Where on the strike axis the contract sits. The primary driver of smile shape. Negative = OTM put, zero = ATM, positive = OTM call |
| `years` | - | Raw data | Which maturity slice the contract is on. Drives term structure |
| `callPut` | P=0, C=1 | Raw data | Puts and calls can diverge slightly in practice due to liquidity differences |
| `atmIV` | midIV of nearest-to-ATM contract for this ticker-date | Computed | The level anchor. Even after normalising the target, vol level affects smile shape - high vol regimes tend to have steeper skews |
| `realizedVol` | `std(log(S_t / S_{t-1}), trailing 30 days) × sqrt(252)` | Computed from `uClose` | What volatility has actually been. The gap between realizedVol and atmIV is the variance risk premium - one of the strongest drivers of skew steepness |
| `recentReturn` | `log(S_t / S_{t-20})` | Computed from `uClose` | Momentum and the leverage effect. Sharp falls steepen put skew; strong rallies can create call skew (as seen in NVDA in 2024) |

### Additional Columns for Parametric Models

| Column | Formula | Used By |
|---|---|---|
| `totalVariance` | `midIV² × years` | SVI - which fits variance, not vol |
| `maturityBucket` | Round `years` to nearest standard expiry (7d, 14d, 30d, 60d, 90d, 180d, 1y) | SVI - which fits one curve per maturity slice |

### Final Dataset Schema

| Column | Type | Purpose |
|---|---|---|
| `tradingDate` | Date | Identification |
| `ticker` | String | Identification |
| `logMoneyness` | Float | Input - NN, Polynomial, SVI |
| `years` | Float | Input - NN, Polynomial, SVI |
| `callPut` | Binary | Input - NN |
| `atmIV` | Float | Input - NN |
| `realizedVol` | Float | Input - NN |
| `recentReturn` | Float | Input - NN |
| `normalizedIV` | Float | Target - NN, Polynomial |
| `totalVariance` | Float | Target - SVI |
| `maturityBucket` | String | Grouping - SVI |

---

## 6. Data Quality Filters

Applied after extraction, before feature computation. Drop any row where:

- `error != 0` - provider flagged the quote as unreliable
- `bidIV <= 0` or `askIV <= 0` - no valid quote on one side
- `years <= 0` - expired or same-day options
- Bid-ask spread on IV is excessively wide - stale or illiquid quotes make the mid meaningless
- `openInterest == 0` - contracts nobody is actively trading

---

## 7. Train / Validation / Test Split

All splits are **chronological**. A random split would allow the model to train on December and test on January - that is look-ahead bias.

| Split | Period | Purpose |
|---|---|---|
| **Train** | January – August 2024 | Model learns surface structure |
| **Validation** | September – October 2024 | Hyperparameter tuning, early stopping |
| **Test - temporal** | November – December 2024 | Final evaluation on trained tickers |
| **Test - ticker** | November – December 2024 | Final evaluation on held-out tickers |

### Why These Boundaries

2024 contains distinct market regimes that make the split financially meaningful:

- **Jan–July:** Calm bull market. Smooth, well-behaved surfaces.
- **August 5:** Yen carry trade unwind. VIX hits ~65 intraday - the largest single-day VIX spike in history. The surface changes shape dramatically overnight.
- **September–October:** Election uncertainty premium builds into near-dated options.
- **November–December:** Post-election vol collapse. Surfaces rapidly reprice.

The validation set immediately stress-tests the model on conditions it has never seen. The test set includes a second distinct regime shift. Results are not just numbers - they are conditioned on specific market environments, and the analysis must reflect this.

### Note on Contracts Spanning the Boundary

Options traded in July with October expiries appear in training data but reference future periods. We originally planned to drop these from training to eliminate any implicit forward-looking information (a conservative but clean choice). **This was reversed:** dropping boundary-spanning contracts disproportionately removes long-maturity contracts from training (any contract with more than a few weeks left to expiry near the Aug 31 cutoff spans the boundary), which left the training set skewed toward short-maturity contracts and produced a model that generalised poorly across the maturity axis. Boundary-spanning contracts are therefore kept in training. The minor look-ahead risk (the model sees that a contract priced in July still has time value in October) is judged less harmful than training a maturity-biased model - see [DECISIONS.md](DECISIONS.md) for the full rationale.

---

## 8. The Three Models

### Polynomial Regression
Fits normalizedIV as a polynomial function of logMoneyness and years. The simplest possible nonlinear baseline. Sets the floor - if the NN can't beat this, something is wrong.

### SVI (Stochastic Volatility Inspired)
A parametric model widely used in industry (Gatheral & Jacquier, 2014). Rather than fitting IV directly, SVI fits **total implied variance** - defined as:

```
w(k, T) = σ_BS(k, T)² × T
```

where σ_BS is the Black-Scholes IV and T is time to expiry. Working in variance space rather than vol space has desirable mathematical properties for enforcing no-arbitrage.

SVI fits the following curve to total variance for each maturity slice separately:

```
w(k) = a + b[ρ(k − m) + √((k − m)² + σ²)]
```

**Parameters and their financial interpretation:**

| Parameter | Role | Interpretation |
|---|---|---|
| `a` | Level | Overall variance level - shifts the whole curve up or down |
| `b` | Slope/wings | Controls how steeply the tails rise - larger b means more pronounced wings |
| `ρ` | Skew asymmetry | Correlation parameter (-1 < ρ < 1). Negative ρ tilts the curve left, producing put skew |
| `m` | ATM shift | Horizontal shift of the minimum - where the smile bottoms out |
| `σ` | Curvature | Controls the smoothness of the transition between put and call wings |

**No-arbitrage constraints on parameters:**
```
b ≥ 0
|ρ| < 1
σ > 0
a + b·σ·√(1 - ρ²) ≥ 0
```

When these constraints are satisfied, the fitted slice is guaranteed to be free of butterfly arbitrage within that maturity. Calendar spread arbitrage across maturities must be checked separately.

**Key limitation:** SVI must be re-fit independently for every ticker, every maturity slice, and every trading day. A full-year dataset with 8 tickers, 7 maturity buckets, and 252 days requires approximately 14,000 separate SVI fits. The NN replaces all of these with a single trained model.

### Feedforward Neural Network
Two to three hidden layers. Takes all six inputs and predicts normalizedIV. Trained once across all training tickers and dates. No guarantee of arbitrage-freedom - this is a key limitation to address.

---

## 9. Evaluation

### Source of Truth
Observed market prices in the test set - mid-IV of options the model never saw during training.

### Statistical Metrics
- **RMSE** - primary metric. Average prediction error in normalizedIV units. Also reported in vol points: rather than a flat rescale by the average atmIV, RMSE is recomputed after multiplying each row's prediction and actual by *that row's* atmIV. This weights each row's contribution to the error by that row's vol level (effectively by atmIV²), so the vol-point RMSE is dominated by whichever days/tickers have the most extreme vol - an economically weighted number, not just a unit conversion of the normalizedIV RMSE.
- **R²** - proportion of cross-sectional variation in normalizedIV explained by the model.
- **Error map** - RMSE broken down by logMoneyness bucket and maturity bucket. Shows *where* on the surface each model wins and loses. The most informative output of the project.

### Economic Metrics

**Butterfly spread violations (static arbitrage across strikes)**

A butterfly spread is constructed by buying one low-strike option, selling two mid-strike options, and buying one high-strike option - all at the same maturity. This spread must always have non-negative value (you cannot lose money with certainty). In terms of total implied variance w(k), this requires:

```
For K₁ < K₂ < K₃ at the same maturity T:
w(k₂) ≤ [w(k₁)·(k₃ - k₂) + w(k₃)·(k₂ - k₁)] / (k₃ - k₁)
```

In plain terms: the total variance at the middle strike must be below the weighted average of the two outer strikes. If this is violated, a trader can buy the butterfly spread for a negative price - a riskless profit.

**Calendar spread violations (static arbitrage across maturities)**

A calendar spread buys a long-dated option and sells a short-dated option at the same strike. Its value must be non-negative. In total variance terms:

```
For T₁ < T₂ at the same strike K:
w(k, T₁) ≤ w(k, T₂)
```

Total implied variance must be non-decreasing in time to expiry. If it decreases, you can construct a riskless profit by trading calendar spreads.

**How we measure violations**

For each model, on each test-set day, for each ticker:
1. Group contracts into maturity slices
2. Check butterfly condition across all strike triplets within each slice
3. Check calendar condition across all consecutive maturity pairs at each strike
4. Report the fraction of checks that fail as the violation rate

SVI is butterfly-free by construction within each slice (when parameter constraints are imposed). The NN and polynomial regression have no such guarantee. Higher violation rates in the NN are an honest limitation - and a real tradeoff against the flexibility that produces lower RMSE.

### The Two Test Dimensions

| Test | Tickers | Period | Question |
|---|---|---|---|
| Temporal | SPY, NVDA, AAPL, LLY, GS, COST, BA, EOG | Nov–Dec | Does the model generalise to future market conditions? |
| Cross-sectional | MSFT, JPM, TSLA, XOM | Nov–Dec | Does the model generalise to stocks it has never seen? |

Holding the time period constant between these two tests means any performance difference is cleanly attributable to ticker familiarity, not market regime.

---

## 10. The Financial Story

The central tension is **flexibility vs. structural constraints**:

- SVI imposes a parametric shape that is arbitrage-free by construction but may miss nonlinear surface features
- The NN imposes nothing and may fit better - but risks overfitting and arbitrage violations

2024 is an ideal year to study this because it contains multiple distinct regimes. The analysis should report not just aggregate RMSE but performance broken down by:

- **Market regime** (calm vs. stressed vs. post-event)
- **Ticker type** (index vs. single-name, low-vol vs. high-vol)
- **Surface region** (deep OTM puts, ATM, OTM calls, short vs. long maturity)

Either direction of results is interesting. If the NN wins, it is capturing nonlinear surface structure that parametric models miss - most likely in the wings and around event-driven regimes. If SVI wins or matches, structural constraints are a feature, not a limitation - providing robustness that flexible models lose when tested outside the training distribution.

---

## 11. Robustness Checks

- **Model depth:** Does a one-layer NN perform similarly to a three-layer NN? If yes, additional complexity adds nothing.
- **Regime subsample:** Report RMSE separately for calm days (VIX < 20), elevated (VIX 20–30), and stressed (VIX > 30). Does NN performance degrade faster than SVI as vol rises?
- **Normalisation choice:** Is normalizedIV actually the right target? Level-dependent smile shape means normalising may discard information. Worth testing raw midIV as an alternative target.

---

## 12. Repository Structure

```
bu493_project/
├── data/
│   ├── raw/          # Raw zip files (gitignored - do not commit)
│   └── processed/    # Extracted Parquet datasets (gitignored)
├── notebooks/        # EDA and modelling notebooks
├── src/
│   └── build_dataset.py   # Extraction and feature engineering pipeline
├── results/
│   └── figures/      # Saved plots and output figures
├── PROJECT_PLAN.md   # This document
└── README.md
```

---

## 13. Key Deadlines

| Deliverable | Due | Requirements |
|---|---|---|
| Milestone report | July 7 | Working baseline, descriptive stats, code archive |
| Final presentation | July 21, 9:00 AM | 5-minute slide deck |
| Final written report | July 21, 11:59 PM | Max 10-page white paper + code archive |

The milestone requires at least one fully working model (polynomial regression suffices) with out-of-sample results. Build the data pipeline first, then get the baseline running end-to-end before adding the NN.

---

## 14. AI Use Disclosure

Per course policy, the final report must include a paragraph describing how AI tools were used. It is not permitted to submit code or text that group members cannot explain under questioning. Every member should be able to defend every part of the project independently.

---

## References

- Cao, J., Chen, J., & Hull, J. (2019). A neural network approach to understanding implied volatility movements. University of Toronto.
- Gatheral, J., & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance*, 14(1), 59–71.
- Horvath, B., Muguruza, A., & Tomas, M. (2021). Deep learning volatility. *Quantitative Finance*, 21(1), 11–27.
