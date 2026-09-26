# Design Decisions and Considerations

This document records every significant design choice made in the project, the alternatives that were considered, and the reasoning behind each decision. It is intended for use in the final write-up, Q&A preparation, and as a reference when revisiting earlier choices.

---

## Data Pipeline

### Two-pass extraction architecture

**Decision:** Separate extraction (pass 1) from feature engineering (pass 2), with an intermediate raw Parquet file saved between them.

**Why:** Feature engineering requires cross-day data (rolling volatility, recent return) that cannot be computed while streaming one day at a time. A single-pass design would require holding the entire dataset in memory. The two-pass approach keeps extraction fast and streaming, while pass 2 operates on the already-filtered intermediate file which fits comfortably in memory.

**Consequence:** Running `--skip_extract` reruns only pass 2, making feature engineering iteration fast without repeating the ~1-2 hour extraction.

---

### Quality filters

Each filter removes a specific class of bad data that would corrupt the target variable (mid-IV):

| Filter | Threshold | Reason |
|---|---|---|
| `error != 0` | - | Provider flagged the quote as unreliable |
| `bidIV <= 0` | - | No valid market bid; mid-IV would be meaningless |
| `askIV <= 0` | - | No valid market ask |
| `years < 7/365` | 7 days | Near-expiry options behave erratically; gamma risk dominates and surfaces are unreliable |
| `uClose <= 0` | - | No valid underlying price; logMoneyness cannot be computed |
| `openInterest == 0` | - | Nobody is actively trading this contract; quote is likely stale |
| Bid-ask spread ratio | > 0.5 | `(askIV - askIV) / midIV > 50%` indicates an illiquid or stale quote where the mid is not a reliable price |

**Observed impact:** 10.28M raw rows → 5.92M after filtering (42% dropped). This is expected - the raw data includes the entire US options market across thousands of tickers, many with no valid quotes. The 5.92M retained rows represent clean, liquid, actively-traded contracts for our 12 chosen tickers.

---

### Stock-split masking in rolling features

**Decision:** Before computing `realizedVol` and `recentReturn`, zero out (not drop) any single-day log return where the raw price ratio implies a stock split or reverse split - flagged when the implied move exceeds 40% in a single day (`SPLIT_THRESHOLD = 0.40`).

**Why:** `uClose` is the raw, unadjusted close price embedded in the options data. A genuine >40% single-day move is essentially impossible for the liquid large-caps in this dataset; the only realistic cause is an unadjusted price series jumping by the split factor. Without masking, a split would produce one enormous fake return that corrupts the 30-day rolling volatility and the 20-day return for a month afterward.

**Why zero instead of drop:** Dropping the row would leave a gap in the rolling window, throwing off the 30-day standard deviation calculation for reasons unrelated to the split itself. Zeroing preserves window continuity - a return of 0 is also the economically correct interpretation of a split day (the stock didn't actually move, only its quoted price did).

**Implementation:** `_mask_split_returns()` in `build_dataset.py`, called from `_compute_rolling_features()`.

---

### Rolling warmup drop

**Decision:** Drop rows where `realizedVol` or `recentReturn` are NaN due to insufficient price history.

**Why:** The 30-day rolling volatility requires 15+ observations (min_periods=15) and the 20-day return requires 20 prior days. The first ~30 trading days of each ticker in the dataset therefore have no valid rolling features.

**Observed impact:** 5.92M → 5.52M (398k dropped, ~7%). Small and unavoidable.

---

## Ticker Selection

### Training tickers

**Decision:** SPY, NVDA, AAPL, LLY, GS, COST, BA, EOG.

**Why:** Diversity across three dimensions - volatility level, sector, and what drives the underlying. A basket skewed toward high-vol names teaches the model that all surfaces are spiky. A basket limited to one sector cannot learn cross-sectional patterns.

| Dimension | Low | Medium | High |
|---|---|---|---|
| Vol level | AAPL, COST | GS, EOG, LLY | NVDA, BA |
| - | SPY (index anchor) | - | - |

Sectors covered: tech/semiconductors, consumer staples, pharma, financials, industrials/aerospace, energy, broad market index.

**EOG sparsity note:** EOG has significantly fewer option rows than other tickers (~90k training rows vs ~790k for SPY). This is expected - energy E&P names have fewer listed contracts than mega-cap tech or index ETFs. It is not a data quality issue. Worth mentioning in the write-up when describing the dataset.

---

### Held-out ticker test set

**Decision:** MSFT, JPM, TSLA, XOM - held entirely out of training and validation, evaluated only in November–December.

**Why:** Tests cross-sectional generalisation separately from temporal generalisation. If the model only performs well on tickers it trained on, it has memorised per-ticker surface patterns rather than learning general surface structure.

| Ticker | Tests |
|---|---|
| MSFT | Within-sector transfer from AAPL (large-cap stable tech) |
| JPM | Within-sector transfer from GS (financials, different business model) |
| TSLA | Out-of-distribution - nothing in training resembles TSLA's surface |
| XOM | Within-sector transfer from EOG (energy, integrated major vs. E&P) |

**Why only November–December for held-out tickers:** The macro regime is held constant between the temporal test (trained tickers, Nov–Dec) and the ticker test (held-out tickers, Nov–Dec). Any performance difference between the two test sets is then cleanly attributable to ticker familiarity, not differences in market conditions. Testing held-out tickers across the full year would confound ticker generalisation with regime generalisation.

---

## Feature Engineering

### Target variable: normalizedIV vs raw midIV

**Decision:** Predict `normalizedIV = midIV / atmIV` rather than raw `midIV`.

**Why:** When training across multiple underlyings, the model sees SPY with ATM IV of ~15% and NVDA with ATM IV of ~45%. Two contracts with identical logMoneyness and years will have very different raw IVs purely because of their vol level, not because of their surface shape. Using raw midIV forces the model to learn both the level and the shape simultaneously, conflating two distinct things.

Normalising by atmIV separates the problem: `atmIV` captures the level, `normalizedIV` captures the shape. A value of 1.2 means the contract trades 20% above ATM vol - that is the skew premium the model is learning to predict.

**Known limitation:** Surface shape is not entirely independent of vol level - high-vol regimes tend to have steeper skews. By normalising, we may discard some level-dependent shape information. This is worth testing as a robustness check: rerun with raw midIV as target and compare results.

---

### ATM IV definition

**Decision:** For each (ticker, tradingDate), find the option with minimum |logMoneyness| among contracts with 20–40 days to expiry. Fall back to the option whose maturity is nearest to 30 days when no contracts exist in that window.

**Why:** ATM IV is the market's best estimate of near-term volatility for that stock. A 30-day reference is conventional in academic literature and practitioner usage (the VIX is constructed on 30-day options). Using the option nearest-to-money within this window gives a clean, consistent ATM anchor.

**Alternative considered:** Using the 30-day ATM IV via interpolation across strikes and maturities. Rejected as overly complex for marginal gain.

---

### realizedVol window: 30 days

**Decision:** 30-day rolling standard deviation of log returns, annualised by ×√252.

**Why:** 30 days matches the ATM IV reference maturity, making the variance risk premium (`atmIV - realizedVol`) a meaningful comparison between what the market expects and what has recently occurred over the same horizon.

`min_periods=15` allows the rolling calculation to begin after 15 days rather than requiring a full 30, reducing data loss at the start of the sample.

---

### recentReturn window: 20 days

**Decision:** 20-day log return of the underlying: `log(S_t / S_{t-20})`.

**Why:** Captures the leverage effect and momentum over roughly one calendar month. Shorter windows (5–10 days) are too noisy; longer windows (60+ days) are too slow to capture regime changes. 20 days is a standard lookback in options literature for momentum-related skew effects.

---

### Bid-ask spread ratio threshold: 50%

**Decision:** Drop contracts where `(askIV - bidIV) / midIV > 0.50`.

**Why:** A spread wider than 50% of mid means the ask is at least 1.5× the bid. The mid-point of such a quote is not a reliable estimate of fair value - it could easily be off by 20–25 vol points. Training on these targets would add significant noise.

**Alternative considered:** An absolute spread threshold (e.g. drop if askIV - bidIV > 0.10). Rejected because the same absolute spread is inconsequential for a high-IV contract (e.g. NVDA at 60% IV) but severe for a low-IV contract (e.g. COST at 15% IV). The ratio is scale-invariant across tickers.

---

### Minimum days to expiry: 7 days

**Decision:** Drop options with fewer than 7 calendar days to expiry.

**Why:** Near-expiry options are dominated by gamma risk and binary event outcomes (e.g. earnings). Their IV surface is extremely irregular, driven by short-term mechanics rather than the structural surface shape we are modelling. Including them would add noise without adding signal.

---

### Maturity buckets for SVI

**Decision:** 7d, 14d, 30d, 60d, 90d, 180d, 1Y - based on standard market conventions for option expiry cycles.

**Why:** SVI is fit per maturity slice. The buckets must be coarse enough that each slice contains enough contracts for a reliable fit, but fine enough to capture term structure variation. These tenors correspond to weekly, bi-weekly, monthly, quarterly, and longer cycles - the natural grid of listed option expiries.

---

## Data Split Design

### Chronological split - not random

**Decision:** All splits assigned by `tradingDate`, never randomly.

**Why:** Option surfaces on adjacent days are highly correlated - today's surface looks almost identical to yesterday's. A random split would allow the model to train on data from December and test on January, creating the illusion of generalisation when it is actually interpolating between near-identical observations. This is the most common methodological error in time-series ML projects.

---

### Split boundaries: September 1 and November 1

**Decision:** Train ends August 31, validation September 1 – October 31, test starts November 1.

**Why - financial calendar:** 2024 contains four distinct market regimes:
- **Jan–July:** Calm bull market, smooth surfaces
- **August 5:** Yen carry trade unwind - largest single-day VIX spike in history (~65 intraday). Surface changes shape overnight.
- **September–October:** Election uncertainty premium builds into near-dated options
- **November–December:** Post-election vol collapse, surfaces rapidly reprice

Placing the train/val boundary at September 1 means the model is validated immediately on the first major stress event it has never seen (August spike + election buildup). The test set (November–December) adds a second distinct regime. Results are therefore conditioned on specific, identifiable market environments - not arbitrary date ranges.

---

### Same test window (Nov–Dec) for both test sets

**Decision:** Both temporal test (trained tickers) and ticker test (held-out tickers) use only November–December data.

**Why:** Holding the time period constant between the two test sets isolates the variable of interest. Any performance difference between trained and held-out tickers in the same period is attributable purely to ticker familiarity - not to differences in market conditions. Using the full year for held-out tickers would conflate ticker generalisation with regime generalisation, making it impossible to cleanly interpret the results.

---

### Boundary-spanning contracts: kept in training (reversed from original plan)

**Decision:** Contracts traded before the train/val boundary (Aug 31) with expiries after it are **kept** in the training set. The originally planned filter (PROJECT_PLAN.md §7) that dropped these contracts has been removed from `build_dataset.py`.

**Why:** Any contract with meaningful time-to-expiry near the Aug 31 cutoff spans the boundary by construction, so dropping them removed long-maturity contracts from training far more than short-maturity ones. The training set became skewed toward short-maturity contracts and the model generalised poorly across the maturity axis as a result. Keeping these contracts introduces a minor look-ahead risk (the model implicitly sees that a contract priced in July still carries time value into October), but this is judged less harmful than training a maturity-biased model - full maturity coverage in training is required for the model to learn the term structure at all.

**Consequence:** Training set size increased from ~945k rows (with the filter) to ~2.65M rows (without it) - see updated dataset characteristics below. Val, test_temporal, and test_ticker splits are unaffected since the filter only ever applied to training rows.

---

## Evaluation Metrics

### Vol-point RMSE: per-row atmIV weighting, not a flat rescale

**Decision:** RMSE in vol points is computed by multiplying each row's prediction and actual normalizedIV by *that row's* atmIV, then taking RMSE on the resulting series - not by taking the normalizedIV RMSE and multiplying by a single average atmIV.

**Why:** A flat rescale (RMSE_normIV × mean(atmIV)) would just be a constant unit conversion and would always agree in direction with the normalizedIV RMSE. The per-row approach instead weights each row's contribution to the squared-error sum by that row's atmIV² - so a miss on a high-vol day (e.g. NVDA, or the Aug 5 VIX spike) counts far more than an equally-sized relative miss on a calm, low-vol day. This is the economically meaningful version of the metric: it reflects where the model's errors actually matter in vega-dollar terms, rather than treating a 20% relative miss on COST the same as a 20% relative miss on NVDA.

**Consequence:** Because of this weighting, `rmse_normIV` and `rmse_volpts` can disagree on which split (train vs. val) performed worse - the split with the heavier tail of high-atmIV days dominates the vol-point number regardless of the average relative fit. Both metrics are reported since they answer different questions: `rmse_normIV` is the fair, scale-invariant comparison of shape-fitting quality; `rmse_volpts` is the economically weighted number that matters for practical use.

---

## Observed Dataset Characteristics

| Split | Rows | Notes |
|---|---|---|
| train | ~2.65M | 8 tickers, Jan–Aug, includes boundary-spanning contracts (see decision above) |
| val | ~799k | 8 tickers, Sep–Oct |
| test_temporal | ~718k | 8 training tickers, Nov–Dec |
| test_ticker | ~277k | 4 held-out tickers, Nov–Dec |
| **Total** | **~4.45M** | |

- SPY and NVDA dominate by row count due to their extremely deep option chains
- EOG is the sparsest training ticker (~90k train rows) due to fewer listed contracts
- TSLA is the largest held-out ticker (~131k test rows)
- The quality filter drop (~42%) is large but expected given the breadth of the raw data
- Train is much larger than val/test_temporal/test_ticker in row-count terms because it spans 8 months vs 2 months each, and (since the boundary filter reversal above) is no longer pruned of long-maturity contracts
