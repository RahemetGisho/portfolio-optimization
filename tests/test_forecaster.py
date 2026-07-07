import os
import sys
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Ensure the runtime can discover the source code module folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from forecaster import generate_future_pipeline_forecast


@pytest.fixture
def mock_historical_dataframe():
    """Generates a dummy DataFrame simulating the cleaned historical asset csv structure."""
    dates = pd.date_range(start="2025-01-01", periods=100, freq="B")
    data = pd.DataFrame({"Adj Close": np.linspace(150, 250, 100)}, index=dates)
    return data


# --- TEST 1: ROBUST ERROR HANDLING (MISSING INGESTION TARGETS) ---
@patch("os.path.exists")
def test_pipeline_handles_missing_file_gracefully(mock_exists):
    """Verifies that the function gracefully handles missing data instead of crashing."""
    mock_exists.return_value = False
    result = generate_future_pipeline_forecast(processed_data_path="invalid_path.csv")
    assert result is None


# --- TEST 2: SUCCESSFUL COMPONENT INTEGRATION AND PIPELINE ROUTING ---
@patch("os.path.exists")
@patch("pandas.read_csv")
@patch("joblib.load")
@patch("forecaster.load_model")
@patch("matplotlib.pyplot.show")  # Intercept plotting windows during tests
@patch("matplotlib.pyplot.savefig")  # Intercept saving plots during tests
def test_pipeline_successful_execution(
    mock_plt_savefig,
    mock_plt_show,
    mock_load_model,
    mock_joblib_load,
    mock_read_csv,
    mock_exists,
    mock_historical_dataframe,
):
    """
    Validates that the forecasting engine parses inputs correctly, computes
    the multi-step shapes, and formats the output DataFrame matching parameters.
    """
    horizon = 10
    lookback = 60

    # 1. Setup mock states for the underlying dependencies
    mock_exists.return_value = True
    mock_read_csv.return_value = mock_historical_dataframe

    # Mock Scaler behaviors
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.linspace(-1, 1, 100).reshape(-1, 1)
    mock_scaler.inverse_transform.return_value = np.linspace(250, 300, horizon).reshape(
        -1, 1
    )

    # Mock ARIMA model behavior to return an array matching our forecast horizon length
    mock_arima = MagicMock()
    mock_arima.forecast.return_value = np.linspace(240, 260, horizon)
    mock_arima.predict.return_value = np.linspace(240, 260, horizon)

    # joblib.load is called twice: first for the scaler, second for the arima model
    mock_joblib_load.side_effect = [mock_scaler, mock_arima]

    # Mock LSTM neural network predictions matrix response
    mock_lstm = MagicMock()
    mock_lstm.predict.return_value = np.array([[0.5]])
    mock_load_model.return_value = mock_lstm

    # 2. Execute target function
    result = generate_future_pipeline_forecast(
        processed_data_path="../data/processed/TSLA_clean.csv",
        forecast_horizon=horizon,
        lookback_window=lookback,
    )

    # 3. Assert structural compliance of the resulting analytical data matrix
    assert result is not None, "Pipeline execution returned None unexpectedly."
    assert isinstance(
        result, pd.DataFrame
    ), "Returned entity must be a Pandas DataFrame."
    assert (
        len(result) == horizon
    ), f"Expected data rows to equal horizon length ({horizon})."

    # Ensure expected data properties are completely mapped
    expected_columns = [
        "LSTM_Future_Forecast",
        "ARIMA_Future_Forecast",
        "Lower_Bound",
        "Upper_Bound",
    ]
    for column in expected_columns:
        assert (
            column in result.columns
        ), f"Missing required metric target header: {column}"

    assert (
        result["Lower_Bound"] >= 0
    ).all(), "Lower bounding limit dropped below corporate liquidation floor ($0)."
