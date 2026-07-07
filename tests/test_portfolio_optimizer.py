import os
import sys
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Ensure runtime path can resolve the source code folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Import the targeted script function
from portfolio_optimizer import run_portfolio_optimization


@pytest.fixture
def mock_yfinance_multi_ticker_data():

    dates = pd.date_range(start="2025-01-01", periods=20, freq="B")

    # Simulating a sequential drift to establish a standard return history
    prices_spy = np.linspace(400, 420, 20)
    prices_bnd = np.linspace(70, 72, 20)

    # Constructing a hierarchical multi-index header structure
    columns = pd.MultiIndex.from_tuples(
        [
            ("SPY", "Adj Close"),
            ("SPY", "Volume"),
            ("BND", "Adj Close"),
            ("BND", "Volume"),
        ]
    )

    data = pd.DataFrame(index=dates, columns=columns)
    data[("SPY", "Adj Close")] = prices_spy
    data[("SPY", "Volume")] = 100000
    data[("BND", "Adj Close")] = prices_bnd
    data[("BND", "Volume")] = 50000
    return data


@pytest.fixture
def mock_yfinance_single_ticker_data():
    dates = pd.date_range(start="2025-01-01", periods=20, freq="B")
    prices_tsla = np.linspace(200, 220, 20)

    columns = pd.MultiIndex.from_tuples([("TSLA", "Adj Close"), ("TSLA", "Volume")])

    data = pd.DataFrame(index=dates, columns=columns)
    data[("TSLA", "Adj Close")] = prices_tsla
    data[("TSLA", "Volume")] = 300000
    return data


# --- TEST 1: ROBUST ERROR HANDLING FOR NETWORK OUTAGES ---
@patch("portfolio_optimizer.yf.download")
def test_optimization_handles_download_failures_gracefully(mock_download):
    mock_download.side_effect = Exception("Network connection timeout")

    result = run_portfolio_optimization(tsla_forecasted_return=0.15)
    assert result is None


# --- TEST 2: SUCCESSFUL EXECUTION, MATRIX OPTIMIZATION & DICTIONARY SCHEMAS ---
@patch("portfolio_optimizer.yf.download")
@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.savefig")
def test_optimization_successful_execution(
    mock_savefig,
    mock_show,
    mock_download,
    mock_yfinance_multi_ticker_data,
    mock_yfinance_single_ticker_data,
):

    # Instruct mock yfinance to deliver benchmark data first, then single asset data
    mock_download.side_effect = [
        mock_yfinance_multi_ticker_data,
        mock_yfinance_single_ticker_data,
    ]

    # Execute portfolio optimization engine
    result = run_portfolio_optimization(
        tsla_forecasted_return=0.18,
        tickers=["SPY", "BND"],
        start_date="2025-01-01",
        end_date="2025-02-01",
    )

    # 1. Broad output shape validations
    assert result is not None, "Optimization pipeline returned None unexpectedly."
    assert isinstance(
        result, dict
    ), "The optimization module must pass back a structured dictionary."

    # 2. Check dictionary key layout rules
    assert "covariance_matrix" in result
    assert "max_sharpe" in result
    assert "min_vol" in result

    # 3. Check mathematical matrix realities
    cov_matrix = result["covariance_matrix"]
    assert isinstance(cov_matrix, pd.DataFrame)
    assert cov_matrix.shape == (
        3,
        3,
    ), "Covariance metrics must map a clear 3x3 layout (TSLA, SPY, BND)."

    # 4. Confirm optimization logic targets sum-to-1 constraints
    for allocation_profile in ["max_sharpe", "min_vol"]:
        profile = result[allocation_profile]

        # Verify nested keys are mapped out cleanly
        assert "weights" in profile
        assert "return" in profile
        assert "volatility" in profile
        assert "sharpe_ratio" in profile

        # Pull allocation array weights and verify sum constraint total matches 100%
        weights = list(profile["weights"].values())
        assert len(weights) == 3, "Weights must map exactly to 3 assets."
        assert (
            pytest.approx(sum(weights), abs=1e-5) == 1.0
        ), f"MPT allocations for {allocation_profile} must equal 100%."

        # Confirm bounds compliance (No shorting permitted by setup rules)
        for w in weights:
            assert (
                0.0 <= w <= 1.0
            ), f"Weight {w} in {allocation_profile} violates long-only constraints [0, 1]."
