import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize


def run_portfolio_optimization(
    tsla_forecasted_return,
    tickers=["SPY", "BND"],
    start_date="2020-01-01",
    end_date="2026-01-01",
    risk_free_rate=0.04,
):
    """
    Production-grade module for Task 4: Portfolio Optimization using Modern Portfolio Theory (MPT).
    Handles yfinance multi-index structural transformations cleanly.
    """
    print("Initializing Production Portfolio Optimization Engine...")

    # 1. Download Historical Data for Benchmark Indices
    try:
        print(f"Fetching historical data for benchmark assets: {tickers}...")
        # Force grouping by ticker to make column parsing consistent
        historical_data = yf.download(
            tickers, start=start_date, end=end_date, group_by="ticker"
        )

        # Parse out the Close/Adj Close price for each asset dynamically
        bench_returns = {}
        for ticker in tickers:
            if ticker in historical_data:
                asset_df = historical_data[ticker]
                # Fallback from Adj Close to Close if needed
                close_col = "Adj Close" if "Adj Close" in asset_df.columns else "Close"
                bench_returns[ticker] = asset_df[close_col].pct_change()

        historical_returns = pd.DataFrame(bench_returns).dropna()
    except Exception as e:
        print(f"Data download failed: {str(e)}")
        return None

    # 2. Compute Annualized Historical Returns & Asset Merging
    try:
        print("Fetching historical data for TSLA...")
        tsla_df = yf.download("TSLA", start=start_date, end=end_date, group_by="ticker")

        # Handle single ticker structure vs multi-ticker structure if group_by is active
        if "TSLA" in tsla_df:
            tsla_data = tsla_df["TSLA"]
        else:
            tsla_data = tsla_df

        tsla_close_col = "Adj Close" if "Adj Close" in tsla_data.columns else "Close"
        tsla_hist_ret = tsla_data[tsla_close_col].pct_change()

        # Merge all three assets cleanly matching on date indexes
        all_returns = pd.DataFrame(
            {
                "TSLA": tsla_hist_ret,
                "SPY": historical_returns["SPY"],
                "BND": historical_returns["BND"],
            }
        ).dropna()

        # Calculate expected annualized returns vector
        spy_expected = all_returns["SPY"].mean() * 252
        bnd_expected = all_returns["BND"].mean() * 252

        expected_returns = np.array(
            [tsla_forecasted_return, spy_expected, bnd_expected]
        )
        asset_names = ["TSLA", "SPY", "BND"]
        print(
            f"Expected Returns Vector: TSLA={expected_returns[0]:.2%}, SPY={expected_returns[1]:.2%}, BND={expected_returns[2]:.2%}"
        )
    except Exception as e:
        print(f"Returns setup failed: {str(e)}")
        return None

    # 3. Compute Covariance Matrix
    try:
        cov_matrix_daily = all_returns.cov()
        cov_matrix_annualized = cov_matrix_daily * 252
        print("Annualized Covariance Matrix compiled successfully.")
    except Exception as e:
        print(f"Covariance matrix calculation failed: {str(e)}")
        return None

    # 4. Define MPT Optimization Objective Sub-Functions
    def portfolio_performance(weights, returns, cov_mat):
        p_ret = np.sum(returns * weights)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
        p_sharpe = (p_ret - risk_free_rate) / p_vol
        return p_ret, p_vol, p_sharpe

    def minimize_sharpe(weights):
        return -portfolio_performance(weights, expected_returns, cov_matrix_annualized)[
            2
        ]

    def minimize_volatility(weights):
        return portfolio_performance(weights, expected_returns, cov_matrix_annualized)[
            1
        ]

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(3))
    init_guess = [1 / 3, 1 / 3, 1 / 3]

    # 5. Locate Core Tactical Portfolios
    try:
        opt_sharpe = minimize(
            minimize_sharpe,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        w_max_sharpe = opt_sharpe.x
        ret_max_sharpe, vol_max_sharpe, sharpe_max_sharpe = portfolio_performance(
            w_max_sharpe, expected_returns, cov_matrix_annualized
        )

        opt_vol = minimize(
            minimize_volatility,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        w_min_vol = opt_vol.x
        ret_min_vol, vol_min_vol, sharpe_min_vol = portfolio_performance(
            w_min_vol, expected_returns, cov_matrix_annualized
        )
    except Exception as e:
        print(f"Matrix optimization routines failed: {str(e)}")
        return None

    # 6. Map Efficient Frontier Curve
    try:
        target_returns = np.linspace(
            expected_returns.min(), expected_returns.max(), 100
        )
        frontier_vols = []

        for target in target_returns:
            cons = (
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w: np.sum(expected_returns * w) - target},
            )
            res = minimize(
                minimize_volatility,
                init_guess,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
            )
            if res.success:
                frontier_vols.append(res.fun)
            else:
                frontier_vols.append(np.nan)

        valid_idx = ~np.isnan(frontier_vols)
        target_returns = target_returns[valid_idx]
        frontier_vols = np.array(frontier_vols)[valid_idx]
    except Exception as e:
        print(f"Frontier simulation failure: {str(e)}")
        return None

    # 7. Visualization Render Engines
    try:
        os.makedirs("../reports/figures", exist_ok=True)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cov_matrix_annualized,
            annot=True,
            fmt=".4f",
            cmap="coolwarm",
            cbar=True,
            square=True,
        )
        plt.title(
            "Asset Covariance Heatmap (Annualized Basis)",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )
        plt.tight_layout()
        plt.savefig("../reports/figures/portfolio_covariance_heatmap.png", dpi=300)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(
            frontier_vols,
            target_returns,
            color="#2c3e50",
            linestyle="-",
            linewidth=2.5,
            label="Efficient Frontier",
        )

        plt.scatter(
            vol_max_sharpe,
            ret_max_sharpe,
            color="#e74c3c",
            marker="*",
            s=200,
            zorder=5,
            label=f"Max Sharpe Ratio Portfolio\nTSLA: {w_max_sharpe[0]:.1%}, SPY: {w_max_sharpe[1]:.1%}, BND: {w_max_sharpe[2]:.1%}",
        )

        plt.scatter(
            vol_min_vol,
            ret_min_vol,
            color="#2ecc71",
            marker="o",
            s=120,
            zorder=5,
            label=f"Minimum Volatility Portfolio\nTSLA: {w_min_vol[0]:.1%}, SPY: {w_min_vol[1]:.1%}, BND: {w_min_vol[2]:.1%}",
        )

        plt.title(
            "Modern Portfolio Theory (MPT) Efficient Frontier Analysis",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        plt.xlabel("Portfolio Risk / Volatility (Standard Deviation)", fontsize=10)
        plt.ylabel("Portfolio Expected Annualized Return", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper left", fontsize=9)
        plt.tight_layout()
        plt.savefig("../reports/figures/portfolio_efficient_frontier.png", dpi=300)
        plt.show()
    except Exception as e:
        print(f"Visual generation routines failed: {str(e)}")

    return {
        "covariance_matrix": cov_matrix_annualized,
        "max_sharpe": {
            "weights": dict(zip(asset_names, w_max_sharpe)),
            "return": ret_max_sharpe,
            "volatility": vol_max_sharpe,
            "sharpe_ratio": sharpe_max_sharpe,
        },
        "min_vol": {
            "weights": dict(zip(asset_names, w_min_vol)),
            "return": ret_min_vol,
            "volatility": vol_min_vol,
            "sharpe_ratio": sharpe_min_vol,
        },
    }
