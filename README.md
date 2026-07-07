# Portfolio Optimization & Time Series Forecasting

An end-to-end quantitative investment pipeline that combines time series forecasting and Modern Portfolio Theory (MPT) to forecast Tesla (TSLA) price dynamics, optimize asset allocation across TSLA, BND, and SPY, and evaluate portfolio performance through historical backtesting.

## Project Structure

```
portfolio-optimization/
├── .venv/
├── .vscode/
├── data/                      # Raw and processed market data
├── models/                    # Serialized/trained forecasting models
├── notebook/
│   ├── preprocess_and_eda.ipynb
│   ├── time_series_forecasting.ipynb
│   ├── forecasting.ipynb
│   ├── portfolio.ipynb
│   └── backtesting.ipynb
├── reports/                   # Generated reports, plots, and summaries
├── scripts/                   # Standalone/CLI-runnable scripts
├── src/
│   ├── forecaster.py          # ARIMA/SARIMA & LSTM model logic
│   ├── portfolio_optimizer.py # MPT / Efficient Frontier logic
│   └── backtester.py          # Strategy backtesting logic
├── tests/                     # Unit tests
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

## Workflow

### 1. Data Preprocessing & Exploratory Analysis

- Fetch historical OHLCV data for TSLA, BND, and SPY via `yfinance`
- Clean and validate data types, handle missing values
- Visualize closing prices, daily returns, and rolling volatility
- Detect outliers and analyze days with unusual returns
- Test stationarity (Augmented Dickey-Fuller test) on prices and returns
- Compute foundational risk metrics: Value at Risk (VaR) and Sharpe Ratio

### 2. Time Series Forecasting

- Chronological train/test split (no shuffling) to preserve temporal structure
- **ARIMA/SARIMA**: parameter selection via ACF/PACF and `auto_arima` (pmdarima)
- **LSTM**: sequence-based deep learning model (e.g., 60-day lookback window)
- Model evaluation via MAE, RMSE, and MAPE
- Comparative discussion of statistical vs. deep learning performance

### 3. Future Trend Forecasting

- Generate 6–12 month forward forecasts using the best-performing model
- Visualize forecasts with confidence intervals alongside historical data
- Analyze how forecast uncertainty widens over the horizon
- Summarize market opportunities and risks implied by the forecast

### 4. Portfolio Optimization (Modern Portfolio Theory)

- Use TSLA's forecasted return as its expected return; historical annualized returns for BND and SPY
- Compute the covariance matrix across all three assets
- Generate the Efficient Frontier using `PyPortfolioOpt` / `scipy.optimize`
- Identify the Maximum Sharpe Ratio (Tangency) Portfolio and Minimum Volatility Portfolio
- Recommend a final portfolio allocation with expected return, volatility, and Sharpe Ratio

### 5. Strategy Backtesting

- Backtest the optimized portfolio over a held-out period (e.g., the final year of data)
- Benchmark against a static 60% SPY / 40% BND portfolio
- Compare cumulative returns, total/annualized return, Sharpe Ratio, and maximum drawdown
- Reflect on strategy viability and backtest limitations

## Tech Stack

- **Data & Wrangling**: `pandas`, `numpy`, `yfinance`
- **Statistical Modeling**: `statsmodels`, `pmdarima`
- **Deep Learning**: `tensorflow` / `keras` (LSTM)
- **Portfolio Optimization**: `PyPortfolioOpt`, `scipy.optimize`
- **Visualization**: `matplotlib`, `seaborn`
- **Testing**: `pytest`

## Getting Started

```bash
# Clone the repository
git clone <repo-url>
cd portfolio-optimization

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Run the notebooks in `notebook/` sequentially (`preprocess_and_eda` → `time_series_forecasting` / `forecasting` → `portfolio` → `backtesting`), or use the modular pipeline components in `src/`.

## Testing

```bash
pytest
```
