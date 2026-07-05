# Portfolio Optimization — GMF Investments

Time series forecasting and portfolio optimization pipeline built for GMF Investments'
Financial Analyst challenge. Covers data extraction/cleaning/EDA, classical + deep learning forecasting models, future forecasting, Modern Portfolio Theory optimization, and strategy backtesting.

## Project Structure

````
portfolio-optimization/
├── .github/workflows/unittests.yml
├── .vscode/settings.json
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── scripts/
├── src/
├── README.md
└── tests/

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

````

## Running the Notebooks

```bash
jupyter notebook notebooks/preprocess_and_eda.ipynb
```

## Key Findings: Task 1 (Preprocessing & EDA)

**Data quality:** TSLA, BND, and SPY daily OHLCV data (2015-01-01 → 2026-06-30) was extracted, reindexed
onto the full business-day calendar, and cleaned via forward-fill + linear interpolation (never
backward-fill, to avoid look-ahead bias). No irrecoverable data-quality issues were found beyond routine
calendar gaps.

**Stationarity (Augmented Dickey-Fuller test):**

| Ticker | Adj Close                | Daily Return         |
| ------ | ------------------------ | -------------------- |
| TSLA   | Non-stationary (p=0.833) | Stationary (p<0.001) |
| BND    | Non-stationary (p=0.841) | Stationary (p<0.001) |
| SPY    | Non-stationary (p=0.994) | Stationary (p<0.001) |

**Risk metrics (95% VaR, annualized return/volatility, Sharpe Ratio @ 2% risk-free rate):**

| Ticker | Historical VaR | Ann. Return | Ann. Volatility | Sharpe Ratio |
| ------ | -------------- | ----------- | --------------- | ------------ |
| TSLA   | -6.01%         | 35.67%      | 60.31%          | 0.56         |
| BND    | -0.63%         | 0.79%       | 6.03%           | -0.20        |
| SPY    | -1.61%         | 16.61%      | 16.28%          | 0.90         |

## Key Findings: Task 2 (Forecasting Models)

**Setup:** chronological train/test split at `2025-01-01` (no shuffling); ARIMA/SARIMA order selected via
`auto_arima` (AIC-minimizing stepwise search); LSTM uses a 60-day lookback window with a scaler fit on
training data only.

**Model comparison (live TSLA data):**

| Metric   | ARIMA/SARIMA | Baseline LSTM | Optimized LSTM |
| -------- | ------------ | ------------- | -------------- |
| MAE ($)  | 55.48        | **13.03**     | 14.58          |
| RMSE ($) | 72.62        | **16.80**     | 18.80          |
| MAPE (%) | 17.69%       | **3.69%**     | 4.12%          |

## Next Steps (Tasks 3–5, not yet implemented here)

- **Task 3:** iteratively extend the best Task 2 model's forecast 6–12 months out
  (`LSTMForecaster.forecast_iteratively` and `ARIMAForecaster.predict(n_periods=...)`
  are already built to support this).
- **Task 4:** feed the Task 3 TSLA return forecast plus BND/SPY historical returns
  into `PyPortfolioOpt` for Efficient Frontier optimization.
- **Task 5:** backtest the Task 4 optimal weights against a 60/40 SPY/BND benchmark
  over the most recent year of data.
