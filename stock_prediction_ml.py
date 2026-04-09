import warnings
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import requests
import os
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "GMURP7PRCNKS4VRQ")
SYMBOL = "AAPL"

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
MODELS_DIR = PROJECT_DIR / "models"

try:
    from tensorflow.keras.layers import LSTM, Dense  # type: ignore[reportMissingImports]
    from tensorflow.keras.models import Sequential  # type: ignore[reportMissingImports]

    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


def fetch_stock_data(symbol: str = SYMBOL) -> pd.DataFrame:
    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}"
        f"&apikey={API_KEY}&outputsize=full"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "Time Series (Daily)" in data:
        df = pd.DataFrame(data["Time Series (Daily)"]).T.astype(float)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df.columns = ["Open", "High", "Low", "Close", "Adj_Close", "Volume", "Dividend", "Split"]
        return df[["Close", "Volume", "Adj_Close"]]

    # Alpha Vantage may throttle free keys; fallback keeps the pipeline runnable.
    try:
        import yfinance as yf

        yf_df = yf.download(symbol, period="max", interval="1d", auto_adjust=False, progress=False)
        if yf_df.empty:
            raise RuntimeError("yfinance returned no data.")
        yf_df = yf_df.rename(columns={"Adj Close": "Adj_Close"})
        yf_df = yf_df[["Close", "Volume", "Adj_Close"]].copy()
        yf_df.index = pd.to_datetime(yf_df.index)
        yf_df.sort_index(inplace=True)
        return yf_df
    except Exception as exc:
        raise RuntimeError(
            f"Alpha Vantage unavailable and fallback failed. Response keys: {list(data.keys())}; "
            f"fallback error: {exc}"
        ) from exc


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["Returns"] = df["Adj_Close"].pct_change()
    df["MA_7"] = df["Adj_Close"].rolling(7).mean()
    df["MA_30"] = df["Adj_Close"].rolling(30).mean()
    df["RSI"] = rsi(df["Adj_Close"])
    df["Volatility"] = df["Returns"].rolling(20).std()
    df.dropna(inplace=True)
    return df


def train_and_predict(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    features = ["Volume", "MA_7", "MA_30", "RSI", "Volatility"]
    X = df[features]
    y = df["Adj_Close"]

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # LSTM (or MLP fallback if TensorFlow is unavailable)
    model_lstm = None
    if TF_AVAILABLE:
        X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
        model_lstm = Sequential(
            [
                LSTM(50, input_shape=(1, X_train_scaled.shape[1])),
                Dense(25),
                Dense(1),
            ]
        )
        model_lstm.compile(optimizer="adam", loss="mse")
        model_lstm.fit(X_train_lstm, y_train, epochs=30, batch_size=32, verbose=0)
        lstm_pred = model_lstm.predict(
            X_test_scaled.reshape(-1, 1, X_test_scaled.shape[1]), verbose=0
        ).flatten()
    else:
        # Keeps output schema stable for Power BI when TensorFlow isn't supported.
        fallback_lstm = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            random_state=42,
            max_iter=500,
        )
        fallback_lstm.fit(X_train_scaled, y_train)
        lstm_pred = fallback_lstm.predict(X_test_scaled)

    # Random Forest
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    rf_pred = np.asarray(model_rf.predict(X_test)).reshape(-1)

    # XGBoost
    model_xgb = XGBRegressor(
        n_estimators=100,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1,
    )
    model_xgb.fit(X_train, y_train)
    xgb_pred = np.asarray(model_xgb.predict(X_test)).reshape(-1)

    y_test_values = np.asarray(y_test).reshape(-1)

    results = pd.DataFrame(
        {
            "Date": df.index[split:],
            "Actual": y_test_values,
            "LSTM_Pred": lstm_pred,
            "RF_Pred": rf_pred,
            "XGB_Pred": xgb_pred,
            "LSTM_Error": np.abs(lstm_pred - y_test_values),
            "RF_Error": np.abs(rf_pred - y_test_values),
            "XGB_Error": np.abs(xgb_pred - y_test_values),
        }
    )

    metrics = pd.DataFrame(
        {
            "Model": ["LSTM", "RF", "XGB"],
            "RMSE": [
                np.sqrt(mean_squared_error(y_test, lstm_pred)),
                np.sqrt(mean_squared_error(y_test, rf_pred)),
                np.sqrt(mean_squared_error(y_test, xgb_pred)),
            ],
            "MAE": [
                mean_absolute_error(y_test, lstm_pred),
                mean_absolute_error(y_test, rf_pred),
                mean_absolute_error(y_test, xgb_pred),
            ],
        }
    )

    feature_importance = (
        pd.DataFrame({"Feature": features, "Importance": model_rf.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    # Dashboard-friendly long-format table for model-wise visuals in Power BI.
    predictions_long = pd.concat(
        [
            results[["Date", "Actual", "LSTM_Pred", "LSTM_Error"]]
            .rename(columns={"LSTM_Pred": "Predicted", "LSTM_Error": "Abs_Error"})
            .assign(Model="LSTM"),
            results[["Date", "Actual", "RF_Pred", "RF_Error"]]
            .rename(columns={"RF_Pred": "Predicted", "RF_Error": "Abs_Error"})
            .assign(Model="RF"),
            results[["Date", "Actual", "XGB_Pred", "XGB_Error"]]
            .rename(columns={"XGB_Pred": "Predicted", "XGB_Error": "Abs_Error"})
            .assign(Model="XGB"),
        ],
        ignore_index=True,
    )[["Date", "Model", "Actual", "Predicted", "Abs_Error"]]

    # One-row-per-model summary for cards and matrix visuals.
    model_summary = metrics.copy()
    model_summary["Best_Model_By_RMSE"] = model_summary["RMSE"] == model_summary["RMSE"].min()
    model_summary["Best_Model_By_MAE"] = model_summary["MAE"] == model_summary["MAE"].min()

    # Persist models for reuse
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_rf, MODELS_DIR / "rf_model.joblib")
    joblib.dump(model_xgb, MODELS_DIR / "xgb_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "feature_scaler.joblib")
    if TF_AVAILABLE and model_lstm is not None:
        model_lstm.save(MODELS_DIR / "lstm_model.keras")

    return {
        "results": results,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "predictions_long": predictions_long,
        "model_summary": model_summary,
    }


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = fetch_stock_data(SYMBOL)
    df = build_features(raw_df)

    # This is the history table requested for Power BI
    df.to_csv(OUTPUTS_DIR / "historical_data.csv", index=True)

    outputs = train_and_predict(df)
    outputs["results"].to_csv(OUTPUTS_DIR / "predictions.csv", index=False)
    outputs["metrics"].to_csv(OUTPUTS_DIR / "model_metrics.csv", index=False)
    outputs["feature_importance"].to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)
    outputs["predictions_long"].to_csv(OUTPUTS_DIR / "predictions_long.csv", index=False)
    outputs["model_summary"].to_csv(OUTPUTS_DIR / "model_summary.csv", index=False)

    print("Generated files:")
    print(f"- {OUTPUTS_DIR / 'historical_data.csv'}")
    print(f"- {OUTPUTS_DIR / 'predictions.csv'}")
    print(f"- {OUTPUTS_DIR / 'model_metrics.csv'}")
    print(f"- {OUTPUTS_DIR / 'feature_importance.csv'}")
    print(f"- {OUTPUTS_DIR / 'predictions_long.csv'}")
    print(f"- {OUTPUTS_DIR / 'model_summary.csv'}")
    print("\nModel metrics:")
    print(outputs["metrics"])


if __name__ == "__main__":
    main()
