import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def run_strategy_backtest(
    strategy_weights,
    benchmark_weights={"SPY": 0.60, "BND": 0.40},
    start_date="2025-01-01",
    end_date="2026-01-01",
    risk_free_rate=0.04,
):

    print("Initializing Strategy Backtesting Engine...")
    tickers = list(set(list(strategy_weights.keys()) + list(benchmark_weights.keys())))

    # 1. Ingest Historical Backtest Windows
    try:
        print(
            f"Downloading out-of-sample data ({start_date} to {end_date}) for {tickers}..."
        )
        raw_data = yf.download(
            tickers, start=start_date, end=end_date, group_by="ticker"
        )

        price_dict = {}
        for ticker in tickers:
            if ticker in raw_data:
                asset_df = raw_data[ticker]
                close_col = "Adj Close" if "Adj Close" in asset_df.columns else "Close"
                price_dict[ticker] = asset_df[close_col]

        prices = pd.DataFrame(price_dict).dropna()
        daily_returns = prices.pct_change().dropna()
        print(f"Extracted {len(prices)} trading days for backtest simulation.")
    except Exception as e:
        print(f"Data ingestion failed: {str(e)}")
        return None

    # 2. Simulation Loop (Monthly Rebalancing Execution Framework)
    try:
        # Rebalancing points occur at the start of every calendar month
        rebalance_dates = daily_returns.index[
            daily_returns.index.to_series().dt.to_period("M").diff() != 0
        ]

        def simulate_portfolio(weights_dict):
            # Convert weights dict to an ordered array matching daily_returns columns
            assets = daily_returns.columns
            w = np.array([weights_dict.get(asset, 0.0) for asset in assets])

            portfolio_returns = []
            current_weights = w.copy()

            for date, daily_ret in daily_returns.iterrows():
                # Check if today is a scheduled rebalancing day
                if date in rebalance_dates:
                    current_weights = w.copy()

                # Calculate daily return based on current asset drift weights
                day_return = np.dot(daily_ret, current_weights)
                portfolio_returns.append(day_return)

                # Update asset value weights based on market drift
                current_weights = current_weights * (1 + daily_ret.values)
                weight_sum = np.sum(current_weights)
                if weight_sum > 0:
                    current_weights /= weight_sum  # Re-normalize

            return pd.Series(portfolio_returns, index=daily_returns.index)

        # Execute simulations
        strat_daily_ret = simulate_portfolio(strategy_weights)
        bench_daily_ret = simulate_portfolio(benchmark_weights)

        # Calculate Compounded Cumulative Growth Curves
        strat_cum_ret = (1 + strat_daily_ret).cumprod() - 1
        bench_cum_ret = (1 + bench_daily_ret).cumprod() - 1
    except Exception as e:
        print(f"Portfolio simulation loop failed: {str(e)}")
        return None

    # 3. Risk and Performance Metric Sub-Engine
    def compute_performance_metrics(daily_ret, cum_ret):
        total_return = cum_ret.iloc[-1]

        # Annualized Return based on 252 business trading days
        days_count = len(daily_ret)
        annualized_return = (1 + total_return) ** (252 / days_count) - 1
        annualized_vol = daily_ret.std() * np.sqrt(252)

        # Sharpe Ratio calculation
        sharpe = (
            (annualized_return - risk_free_rate) / annualized_vol
            if annualized_vol > 0
            else 0
        )

        # Maximum Drawdown calculation
        equity_curve = 1 + cum_ret
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            "Total Return": total_return,
            "Annualized Return": annualized_return,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_drawdown,
        }

    metrics_strat = compute_performance_metrics(strat_daily_ret, strat_cum_ret)
    metrics_bench = compute_performance_metrics(bench_daily_ret, bench_cum_ret)

    metrics_df = pd.DataFrame(
        {
            "Model Strategy Portfolio": metrics_strat,
            "60/40 Benchmark Portfolio": metrics_bench,
        }
    ).T

    # 4. Data Visualization Engine
    try:
        os.makedirs("../reports/figures", exist_ok=True)
        plt.figure(figsize=(11, 6))
        plt.plot(
            strat_cum_ret.index,
            strat_cum_ret * 100,
            color="#2980b9",
            linewidth=2.5,
            label="Optimized Strategy Portfolio",
        )
        plt.plot(
            bench_cum_ret.index,
            bench_cum_ret * 100,
            color="#7f8c8d",
            linewidth=2.0,
            linestyle="--",
            label="Passive 60/40 Benchmark",
        )

        plt.title(
            "Out-of-Sample Performance Backtest Comparison (2025 - 2026)",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        plt.xlabel("Backtest Timeline Horizon", fontsize=10)
        plt.ylabel("Cumulative Growth Return (%)", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper left", fontsize=10)
        plt.tight_layout()
        plt.savefig("../reports/figures/portfolio_backtest_comparison.png", dpi=300)
        plt.show()
        print("Performance visualization charts saved to reports/figures/.")
    except Exception as e:
        print(f"Chart generation failed: {str(e)}")

    return metrics_df
