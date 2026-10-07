# fe5209
# Deep Momentum Networks on ETFs — replication, cost analysis and live pipeline

This repository replicates the core of **"Enhancing Time Series Momentum Strategies Using Deep Neural Networks"** (Lim, Zohren & Roberts, 2019) on a universe of ~51 ETFs, and extends it with transaction-cost analysis, position smoothing, diagnostics, and a daily-update pipeline for a trading competition.

* **Models:** an MLP and an LSTM that output a position in [-1, 1] for each asset directly, trained by maximising the Sharpe ratio (no separate trend estimator or position-sizing rule).
* **Framework:** volatility scaling as in time-series momentum — each asset's position is scaled by `sigma_tgt / sigma_t`, and the portfolio is equal-weighted across assets.
* **Benchmarks:** Long-Only (volatility scaled), `Sgn(Returns)` (Moskowitz et al. 2012) and MACD (Baz et al. 2015).
* **Evaluation:** walk-forward, 5-year expanding windows; all results are out-of-sample.

> All numbers in this project are historical backtests. They are not forecasts of future performance.

---

## 1. Repository layout

```
.
├── README.md
├── ohlcv_data_by_date_adjusted.xlsx        # input: adjusted close prices (sheet "Close")
├── trading_cost_50_etfs_1.xlsx             # input: per-asset one-way costs (bp)
├── processed_data/
│   ├── full_feature_panel.csv              # all features (incl. rows without a forward return)
│   └── model_ready_dataset.csv             # rows used for training / backtesting
│
├── pipeline_data_processing.py / .ipynb    # 1. feature engineering
│
├── neural_config.py                        # 2. shared settings
├── neural_data.py                          #    data loading, splits, datasets
├── neural_models.py                        #    MLP / LSTM definitions
├── neural_loss.py                          #    Sharpe loss
├── train_utils.py                          #    training loop with early stopping
├── neural_search.py                        #    random search + top-K + multi-seed ensemble
├── neural_backtest.py                      #    prediction, portfolio construction, metrics
├── train_mlp.ipynb / train_lstm.ipynb      # 3. walk-forward training scripts
├── benchmark_baz.ipynb                     # 4. benchmarks
├── model_compare.ipynb                     # 5. summary table and equity curves
│
├── cost_eval.ipynb                         # 6. analysis: uniform-bp cost sweep
├── cost_eval_assets.ipynb                  #    analysis: per-asset costs, two universes
├── baz_strategy_transactioncost.ipynb      #    analysis: stand-alone MACD strategy with a fee schedule
├── smooth_eval.ipynb                       #    analysis: EWM smoothing + no-trade band
├── class_eval.ipynb                        #    analysis: results by asset-class subset
├── sig_test.py                             #    analysis: yearly Sharpe and bootstrap tests
├── universe_check.py                       #    analysis: correlation / effective number of assets
├── group_share.py                          #    analysis: contribution of a group of assets
├── risk_budget_table.py                    #    analysis: return vs risk at different vol targets
├── competition_metrics.py                  #    analysis: the course's performance metrics
│
├── live_config.py                          # 7. live / competition pipeline
├── live_engine.py
├── final_fit.py
├── daily_update.py
│
├── models/                                 # created by final_fit.py
└── state/, output/                         # created by daily_update.py
```

**Python modules vs notebooks.** Anything that is `import`ed by other files must be a `.py` file (`neural_*.py`, `train_utils.py`, `pipeline_data_processing.py`, `live_*.py`). Notebooks are runnable analyses that import those modules; keep everything in **one folder**. If you edit a module, restart the notebook kernel (or use `%load_ext autoreload` / `%autoreload 2`).

---

## 2. Requirements

* Python 3.10+ (developed with 3.11)
* `numpy`, `pandas`, `torch`, `matplotlib`, `openpyxl`
* Jupyter or VS Code for the notebooks

```
pip install numpy pandas torch matplotlib openpyxl
```

---

## 3. Quick start (research workflow)

| Step | Run | Produces |
|---|---|---|
| 1 | `pipeline_data_processing` | `processed_data/model_ready_dataset.csv` |
| 2 | Set `QUICK_TEST = True` in `neural_config.py`, run `train_mlp` and `train_lstm` | smoke test only — checks nothing crashes |
| 3 | Set `QUICK_TEST = False`, restart the kernel, run `train_mlp`, then `train_lstm` | OOS predictions and returns for each model |
| 4 | `benchmark_baz` | benchmark returns and positions (run after step 3 so it uses the same sample as the LSTM) |
| 5 | `model_compare` | summary table and equity curves |
| 6 | Any analysis script (Section 5) | cost, smoothing and diagnostic tables |

`benchmark_baz` also runs before training: if `lstm_oos_predictions.csv` is missing it falls back to a date cut-off. Re-run it after training to align the samples exactly.

---

## 4. Configuration

All research settings live in `neural_config.py`.

| Setting | Meaning |
|---|---|
| `DATA_PATH` | path to `model_ready_dataset.csv` |
| `FEATURES` | the 8 input features (see Section 6) |
| `TARGET_ANNUAL_VOL`, `TRADING_DAYS` | 15% volatility target, 252 days |
| `MLP_LOOKBACK`, `LSTM_LOOKBACK` | 5 and 63 days of history per sample |
| `QUICK_TEST` | `True` = tiny smoke-test settings; `False` = full training |
| `VAL_FRACTION` | last 10% of each training block's dates is the validation set |
| `SEARCH_SPACE` | random-search grid: hidden size, dropout, batch size, learning rate, gradient-clip norm, weight decay |
| `SEARCH_ITER`, `SEARCH_ITER_LSTM` | random-search trials per fold (20 / 10 in full mode) |
| `TOP_K`, `N_SEEDS` | best configurations kept (3) × random seeds each (3) |
| `MAX_EPOCHS`, `PATIENCE` | 100 epochs, early stopping after 25 epochs without improvement |

---

## 5. File reference

### 5.1 Data and features

| File | What it contains |
|---|---|
| `pipeline_data_processing.py` / `.ipynb` | Feature engineering. Reads the `Close` sheet of `ohlcv_data_by_date_adjusted.xlsx`; computes daily returns, 60-day EWM volatility, volatility-normalised returns over 1d/1m/3m/6m/1y, MACD signals, winsorisation, 1-day forward returns, and quality checks. Writes `full_feature_panel.csv` and `model_ready_dataset.csv`. The `.py` version is required by the live pipeline. |
| `trading_cost_50_etfs_1.xlsx` | Input, one-way transaction cost per asset in basis points. Sheet `Cost(50支)`: all assets with a cost; `Cost(高流动性20支)`: the high-liquidity list (21 tickers); `Costs`: liquidity details and tier; plus `Tiers` and `ReadMe`. `FXS` has no cost and is excluded from the cost evaluation and the live pipeline. |

### 5.2 Model code (imported by the training scripts)

| File | What it contains |
|---|---|
| `neural_config.py` | All shared settings (Section 4). |
| `neural_data.py` | `load_model_data` (filters rows), `make_expanding_splits` (5-year expanding-window folds), `SequenceDataset` (lazy sliding windows per ticker, no look-ahead), `make_fold_datasets` (train / validation / test for one fold; windows use history from before the split so no test samples are lost). |
| `neural_models.py` | `MLPDirect` (single tanh hidden layer, tanh output, dropout on input and hidden layer), `LSTMDirect` (one LSTM layer, tanh output; input dropout with a mask shared across time, plus output dropout), `LockedDropout`. Recurrent-state dropout from the paper is not implemented. |
| `neural_loss.py` | `strategy_returns`, `sharpe_loss` (negative annualised Sharpe over a batch), `mse_return_loss` (optional baseline). There is **no** turnover or cost term in the loss. |
| `train_utils.py` | `train_model`: AdamW (equals Adam when weight decay is 0), gradient clipping, validation Sharpe computed on the whole validation set, early stopping, best-epoch weights restored. Returns `(model, info)`. |
| `neural_search.py` | `fit_ensemble`: random search over `SEARCH_SPACE` (shortened training), keep the top-K configurations, retrain each with several seeds, average positions across all models. Uses only the validation set; the test set is never touched. |
| `neural_backtest.py` | `predict_arrays` / `predict_dataset` (single model or ensemble average), `build_daily_portfolio` (equal-weight daily return), `rescale_to_target_vol` (scale to 15% volatility using the portfolio's own lagged 60-day EWM volatility), `summary_table` (return, volatility, downside deviation, max drawdown, Sharpe, Sortino, Calmar, % positive returns, average profit / average loss). |

### 5.3 Training, benchmarks, comparison

| File | Inputs | Outputs |
|---|---|---|
| `train_mlp` / `train_lstm` | `model_ready_dataset.csv` | `*_oos_predictions.csv` (Date, Ticker, signal, vol_daily, fwd_ret_1d, weight, contrib, fold); `*_oos_daily_returns.csv` (port_ret, raw_return, rescaled_return); `*_search_log.csv` (every search trial and final model: validation Sharpe, training Sharpe, best epoch, hyper-parameters) |
| `benchmark_baz` | `model_ready_dataset.csv`, `lstm_oos_predictions.csv` | for each of `longonly`, `sgn`, `macd` (MACD, signals **averaged** across the three time-scales), `macdsum` (signals summed, as the paper's equation (8) is literally written): `*_oos_daily_returns.csv` and `*_oos_predictions.csv` |
| `model_compare` | the `*_oos_daily_returns.csv` files | `neural_model_comparison.csv` and `neural_equity_curves.png` (rescaled returns, log scale) |

### 5.4 Analysis scripts

| File | What it answers | Main outputs |
|---|---|---|
| `cost_eval` | How does Sharpe change under a **uniform** one-way cost of 0–10 bp? Includes a "static position" diagnostic (each ETF held at its own average position). | `cost_sweep_sharpe.csv` |
| `cost_eval_assets` | Same, but with **per-asset** costs from the workbook, on two universes (all 50 assets, and the high-liquidity list); cost multipliers 0×, 0.5×, 1×, 2×; raw and smoothed MLP/LSTM positions; full sample and hold-out folds 3–4. | `cost_assets_<universe>_<period>.csv` |
| `baz_strategy_transactioncost` | Stand-alone MACD strategy with a market-specific fee schedule (rate, minimum fee) and a rebalance threshold. Not connected to the neural models. | its own console report |
| `smooth_eval` | Post-hoc position smoothing (EWM half-life and no-trade band). Parameters are chosen on folds 1–2 at an assumed 2 bp cost and reported only on folds 3–4. | `*_smoothing_grid.csv` |
| `class_eval` | Re-aggregates the existing positions by asset-class subset (all, equities, non-equities, without leveraged bonds, commodities + FX, leveraged bonds + UDN) to see where the value comes from. No retraining. | console tables |
| `sig_test.py` | Yearly Sharpe, paired block-bootstrap confidence intervals for Sharpe differences, correlation between model return series. | console tables |
| `universe_check.py` | Average pairwise correlation, share of the first principal component, effective number of independent assets, most correlated pairs, shortest histories. | console tables |
| `group_share.py` | Share of return, variance and turnover contributed by a group of tickers (default: leveraged/inverse bond ETFs + `UDN`). | console table |
| `risk_budget_table.py` | The same return series scaled to 15/25/40/60% volatility: return, drawdown, worst day, worst 50-day window, how often a 50-day window looks like Sharpe ≥ 2. | console table |
| `competition_metrics.py` | The course's performance table: annual return, Sharpe, max drawdown, win rate, annual volatility, drawdown episodes, number of trades, average holding period, profit per trade. A **trade** is defined as an uninterrupted period during which an asset's position keeps the same sign. | console table / `performance_report()` |

### 5.5 Live / competition pipeline

| File | What it contains |
|---|---|
| `live_config.py` | Paths, `LIVE_MODEL`, `EXCLUDE_TICKERS`, smoothing parameters (`SMOOTH_HALFLIFE`, `SMOOTH_BAND`), `LIVE_TARGET_VOL`, `MAX_ASSET_FRACTION`, `MAX_GROSS_LEVERAGE`, drawdown rules `DD_RULES`, covariance settings, `CAPITAL`. |
| `live_engine.py` | Torch-free logic: feature panel from prices, per-ticker feature windows, incremental smoothing, EWM covariance, target position sizing with caps, drawdown multiplier. |
| `final_fit.py` | Trains the final ensemble on **all** history and saves `models/<MODEL>_ensemble.pt`. Run once before going live, with `QUICK_TEST = False`. |
| `daily_update.py` | Run after each close: load prices → features → model signals → smoothing → target weights → risk controls. Updates `state/state.json` and `state/nav_log.csv`, and writes `output/target_weights_YYYYMMDD.csv` plus `output/latest_target_weights.csv`. |

Commands:

```
python final_fit.py                    # once, after refreshing the data
python daily_update.py --selfcheck     # checks live windows match the training windows
python daily_update.py --init          # first run: warm up the smoothing state
python daily_update.py                 # every trading day (missed days are caught up automatically)
```

Position sizing in `daily_update.py`:

1. Model output (ensemble average) → smoothed with EWM and a no-trade band.
2. `weight_i = position_i × sigma_slice / sigma_i`, each asset getting a 1/N slice.
3. The whole portfolio is scaled to `LIVE_TARGET_VOL` using an EWM covariance estimate.
4. Caps: `MAX_ASSET_FRACTION` per asset and `MAX_GROSS_LEVERAGE` in total; if the caps bind, realised volatility ends up below the target.
5. Drawdown rule: exposure is multiplied by a factor once drawdown from the peak exceeds the thresholds in `DD_RULES`.

---

## 6. Conventions and definitions

* **Features (8):** volatility-normalised returns over 1d, 1m, 3m, 6m, 1y; MACD signals with (short, long) time-scales (8, 24), (16, 48), (32, 96), normalised as in Baz et al. (2015).
* **Position and weight:** the model outputs `X` in [-1, 1]; the leveraged weight is `w = X × sigma_tgt / sigma_t`. Shorting and leverage are allowed.
* **Portfolio return:** equal-weight average over the assets available on each day; assets enter when they have enough history.
* **"Rescaled":** the portfolio return multiplied by `15% / (60-day EWM volatility, lagged one day)`.
* **Sharpe ratio:** `mean / std × sqrt(252)`, **without** subtracting a risk-free rate (as in the paper).
* **Downside deviation:** standard deviation of the negative daily returns (differs slightly from the usual definition).
* **Transaction cost:** `cost_t = sum_i c_i × |w_i,t − w_i,t−1| / N_t`, with `c_i` the one-way cost in decimal form (the paper's equation 35 with an asset-specific `c`). Each asset's first day is a free entry. Trades are assumed to execute at the close.
* **Walk-forward:** each fold trains on all earlier data (last 10% of dates as validation) and tests on the next 5 years. With data from 2002 this gives four folds: 2007–11, 2012–16, 2017–21, 2022–26.

---

## 7. Results snapshot

Reported from the runs of this project; re-run to refresh. Sharpe ratios are rescaled to 15% volatility. With roughly 19 years of out-of-sample data the standard error of a Sharpe ratio is about 0.25, so differences of 0.1 or less are not distinguishable from noise.

* **No costs, all ETFs:** LSTM 0.64, MLP 0.63, Long-Only 0.49, MACD 0.38.
* **Where the value came from:** by asset class, the models added value almost only on the leveraged/inverse bond ETFs plus `UDN`. On equities, commodities and FX they were close to a static or Long-Only position. Yearly breakdown of that group shows the gain concentrated in about five years (2019, 2020, 2022, 2023, 2026).
* **Costs (per-asset costs, 1× multiplier, raw positions):** on all 50 assets the neural networks lose money (LSTM −0.97, MLP −1.42) because costs exceed gross profits; Long-Only is best (0.46). On the 21 high-liquidity assets the picture is much milder (Long-Only 0.81, LSTM 0.55, MLP 0.19), but Long-Only still ranks first. Part of that improvement is the asset mix (mostly US equities in a bull market), not skill.
* **Smoothing:** on hold-out folds 3–4 (uniform costs) the smoothed LSTM cut turnover from about 54 to 1.65 per year while the no-cost Sharpe moved from 0.71 to 0.69; the MLP did not improve. Smoothing under per-asset costs is produced by `cost_eval_assets`.
* **Turnover regularisation (paper, section VI-A) is not implemented.** Training maximises the plain Sharpe ratio; post-hoc smoothing is used instead.

---

## 8. Limitations

* The universe is dominated by US equities (about 31 of 51 assets); the effective number of independent assets is small, much lower than in the paper's 88-futures universe.
* ETF returns are used instead of ratio-adjusted continuous futures.
* The cost workbook contains half the bid-ask spread only — no commissions, borrowing fees or market impact — and applies today's spreads to the whole sample, which understates early costs. Several 15 bp entries are tier assumptions, not observed values.
* Smoothing parameters were selected on folds 1–2 with a uniform 2 bp cost, not with per-asset costs.
* The model sees no ticker identity, so mirror products (bull/bear pairs) are treated as independent assets.
* Live pipeline status: the data, smoothing, position-sizing and bookkeeping logic were unit-tested on synthetic data. Training and loading of the neural networks on real data must be validated by running `final_fit.py` and `daily_update.py --selfcheck`. The risk thresholds in `live_config.py` are judgement calls, not validated on history.
* Choosing the universe, parameters or strategy by looking at test-period results invalidates the out-of-sample evaluation; settings should be fixed before the test period starts.

---
