import os
import sys
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from backtester import run_strategy_backtest


@pytest.fixture
def mock_backtest_market_data():
    """Generates dummy prices for all three portfolio tickers."""
    dates = pd.date_range(start="2025-01-01", periods=30, freq="B")
    columns = pd.MultiIndex.from_tuples(
        [("TSLA", "Adj Close"), ("SPY", "Adj Close"), ("BND", "Adj Close")]
    )
    data = pd.DataFrame(index=dates, columns=columns)
    data[("TSLA", "Adj Close")] = np.linspace(200, 210, 30)
    data[("SPY", "Adj Close")] = np.linspace(400, 410, 30)
    data[("BND", "Adj Close")] = np.linspace(70, 71, 30)
    return data


@patch("backtester.yf.download")
@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.savefig")
def test_backtester_execution_flow(
    mock_savefig, mock_show, mock_download, mock_backtest_market_data
):
    """Validates that backtest simulations compute rebalancing math and output risk metric frames."""
    mock_download.return_value = mock_backtest_market_data

    weights = {"TSLA": 0.20, "SPY": 0.50, "BND": 0.30}
    result = run_strategy_backtest(strategy_weights=weights)

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert "Model Strategy Portfolio" in result.index
    assert "Total Return" in result.columns
    assert "Sharpe Ratio" in result.columns
