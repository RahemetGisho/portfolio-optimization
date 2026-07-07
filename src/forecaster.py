import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def generate_future_pipeline_forecast(
    processed_data_path="../data/processed/TSLA_clean.csv",
    scaler_path="../models/tsla_scaler.pkl",
    lstm_path="../models/lstm_tsla_model.h5",
    arima_path="../models/arima_tsla_model.pkl",
    forecast_horizon=252,
    lookback_window=60,
):

    print("Starting Modular Production Forecasting Pipeline...")

    # 1. Ingestion & Robust Error Handling
    try:
        if not os.path.exists(processed_data_path):
            raise FileNotFoundError(
                f"Processed historical data missing at {processed_data_path}"
            )

        raw_data = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)
        data_baseline = raw_data[["Adj Close"]].copy()

        scaler = joblib.load(scaler_path)
        lstm_model = load_model(lstm_path)
        arima_model = joblib.load(arima_path)
        print("Data structures and pre-trained models safely restored.")
    except Exception as e:
        print(f"Error during component ingestion: {str(e)}")
        return None

    # 2. LSTM Multi-Step Recursive Engine
    try:
        scaled_historical_data = scaler.transform(data_baseline)
        last_historical_date = data_baseline.index[-1]

        # Build forward business timeline index
        future_date_range = pd.date_range(
            start=last_historical_date + pd.Timedelta(days=1),
            periods=forecast_horizon,
            freq="B",
        )

        # Track sliding lookback sequence tensor
        current_window = scaled_historical_data[-lookback_window:].reshape(
            1, lookback_window, 1
        )
        lstm_predictions_scaled = []

        for step in range(forecast_horizon):
            next_pred_scaled = lstm_model.predict(current_window, verbose=0)
            lstm_predictions_scaled.append(next_pred_scaled[0, 0])

            # Slide window matrix (drop t=0, append predicted point)
            next_pred_reshaped = next_pred_scaled.reshape(1, 1, 1)
            current_window = np.append(
                current_window[:, 1:, :], next_pred_reshaped, axis=1
            )

        lstm_predictions_scaled = np.array(lstm_predictions_scaled).reshape(-1, 1)
        lstm_future_prices = scaler.inverse_transform(lstm_predictions_scaled).flatten()
        print("✅ Recursive multi-step deep learning paths calculated.")
    except Exception as e:
        print(f"Failure during deep learning simulation: {str(e)}")
        return None

    # 3. Statistical Comparison and Volatility Boundaries
    try:
        try:
            arima_future_prices = arima_model.forecast(steps=forecast_horizon)
        except AttributeError:
            arima_future_prices = arima_model.predict(n_periods=forecast_horizon)

        # Calculate trailing empirical volatility from residuals
        historical_residuals = data_baseline["Adj Close"].pct_change().dropna()
        daily_volatility = historical_residuals.std()

        # Compound error over forward horizon using square-root-of-time rule
        compounding_error = daily_volatility * np.sqrt(
            np.arange(1, forecast_horizon + 1)
        )
        current_price_baseline = data_baseline["Adj Close"].iloc[-1]

        # Derive 95% confidence intervals (Z=1.96)
        lower_bound = lstm_future_prices - (
            1.96 * compounding_error * current_price_baseline
        )
        upper_bound = lstm_future_prices + (
            1.96 * compounding_error * current_price_baseline
        )
        lower_bound = np.clip(lower_bound, 0, None)  # Guard asset boundary

        forecast_df = pd.DataFrame(
            {
                "LSTM_Future_Forecast": lstm_future_prices,
                "ARIMA_Future_Forecast": np.array(arima_future_prices).flatten(),
                "Lower_Bound": lower_bound,
                "Upper_Bound": upper_bound,
            },
            index=future_date_range,
        )
        print("Statistical uncertainty risk boundaries computed.")
    except Exception as e:
        print(f"Failure during confidence calculation: {str(e)}")
        return None

    # 4. Professional Visualization Suite
    try:
        plt.figure(figsize=(14, 7))

        # Render past historical curve (truncated to 18 months for crisp plotting scale)
        plt.plot(
            data_baseline.index[-378:],
            data_baseline["Adj Close"].tail(378),
            label="Historical Observed Prices",
            color="#2c3e50",
            linewidth=2,
        )

        # Render forecast targets
        plt.plot(
            forecast_df.index,
            forecast_df["LSTM_Future_Forecast"],
            label="LSTM Future Projection Baseline (Primary Engine)",
            color="#2980b9",
            linewidth=2.5,
        )
        plt.plot(
            forecast_df.index,
            forecast_df["ARIMA_Future_Forecast"],
            label="ARIMA Comparative Structural Path",
            color="#7f8c8d",
            linestyle="--",
            alpha=0.7,
        )

        # Render uncertainty cloud
        plt.fill_between(
            forecast_df.index,
            forecast_df["Lower_Bound"],
            forecast_df["Upper_Bound"],
            color="#2980b9",
            alpha=0.12,
            label="95% Confidence Risk Horizon Boundary",
        )

        plt.axvline(
            x=last_historical_date,
            color="#e74c3c",
            linestyle=":",
            linewidth=2,
            label="Forecast Boundary Line",
        )
        plt.title(
            "Tesla (TSLA) Strategic Market Forecasting Profile — 12-Month Horizon",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        plt.xlabel("Timeline Target Date", fontsize=11)
        plt.ylabel("Asset Valuation Price in USD ($)", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.legend(loc="upper left", fontsize=10)
        plt.tight_layout()

        # Save visualization to file
        os.makedirs("../reports/figures", exist_ok=True)
        plt.savefig("../reports/figures/tsla_future_forecast.png", dpi=300)
        plt.show()
        print(
            "Success: Strategic visualization asset generated and saved to reports/figures/."
        )

        return forecast_df
    except Exception as e:
        print(f"Failure during visualization rendering: {str(e)}")
        return forecast_df


if __name__ == "__main__":
    # Self-test code logic
    generate_future_pipeline_forecast(
        processed_data_path="TSLA_clean.csv",  # fallback context
        scaler_path="tsla_scaler.pkl",
        lstm_path="lstm_tsla_model.h5",
        arima_path="arima_tsla_model.pkl",
    )
