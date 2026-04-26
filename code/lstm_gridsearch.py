import os
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

# ── GPU CHECK ──────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {[gpu.name for gpu in gpus]}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected, running on CPU.")

# ── CONFIG ────────────────────────────────────────────────
INPUT_FILE = "cleaned_output.xlsx"
TARGET_COL = "log_out"
TRAIN_RATIO = 0.8
DROP_COLS = {"location", "timestamp", "tracked", "out", "state_x", "state_y"}
FORECAST_HORIZONS = [24, 48]

PARAM_GRID = {
    "look_back": [12, 24],
    "lstm_units": [32, 64],
    "dropout": [0.1, 0.2],
    "batch_size": [16, 32],
    "epochs": [30]
}

# ── OUTPUT FOLDERS ───────────────────────────────────────
LOSS_DIR = "loss"
PREDICT_DIR = "predict"
BEST_PARAM_DIR = "best_params"
FORECAST_DIR = "forecast"
FORECAST_VISUAL_DIR = "forecast_visual"   # ← NEW

for d in [LOSS_DIR, PREDICT_DIR, BEST_PARAM_DIR, FORECAST_DIR, FORECAST_VISUAL_DIR]:
    os.makedirs(d, exist_ok=True)

# ── DATA PREP ─────────────────────────────────────────────
def prepare_data(df, look_back):
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in DROP_COLS]
    data = df[feature_cols].copy().ffill().bfill().fillna(0)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(data)
    y_scaled = scaler_y.fit_transform(data[[TARGET_COL]])
    X, y = [], []
    for i in range(look_back, len(X_scaled)):
        X.append(X_scaled[i - look_back:i])
        y.append(y_scaled[i])
    X, y = np.array(X), np.array(y)
    split = int(len(X) * TRAIN_RATIO)
    return X[:split], y[:split], X[split:], y[split:], scaler_y, X_scaled

# ── MODEL ────────────────────────────────────────────────
def build_model(input_shape, units, dropout):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units // 2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# ── FORECAST ─────────────────────────────────────────────
def forecast(model, last_window, scaler_y, steps):
    window = last_window.copy()
    preds = []
    for _ in range(steps):
        pred = model.predict(window[np.newaxis, :, :], verbose=0)[0, 0]
        preds.append(pred)
        new_row = window[-1].copy()
        new_row[0] = pred
        window = np.vstack([window[1:], new_row])
    return scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

# ── PLOTS ───────────────────────────────────────────────
def plot_loss(history, name):
    plt.figure()
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.legend(["Train", "Val"])
    plt.savefig(os.path.join(LOSS_DIR, f"{name}.png"))
    plt.close()

def plot_pred(y_true, y_pred, name):
    plt.figure()
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.savefig(os.path.join(PREDICT_DIR, f"{name}.png"))
    plt.close()

# ── NEW: CONTINUOUS ACTUAL + FORECAST PLOT ──────────────
def plot_pred_with_forecast(y_true, y_pred, forecasts, name):
    """
    Plots:
      - Actual test values (blue)
      - Model fit on test set (orange, overlaid on actual)
      - 24h forecast continuing after actual (green)
      - 48h forecast continuing after actual (red)
    A vertical dashed line marks where forecasts begin.
    """
    n_actual = len(y_true)
    actual_idx = np.arange(n_actual)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Actual and fitted
    ax.plot(actual_idx, y_true, color="steelblue", linewidth=1.5, label="Actual")
    ax.plot(actual_idx, y_pred, color="darkorange", linewidth=1.2,
            linestyle="--", alpha=0.8, label="Predicted (test fit)")

    # Forecast lines — each starts right after the last actual point
    colors = {24: "seagreen", 48: "crimson"}
    for h, fc_vals in forecasts.items():
        fc_idx = np.arange(n_actual - 1, n_actual - 1 + len(fc_vals) + 1)
        # prepend last actual value so the line connects smoothly
        fc_line = np.concatenate([[y_true[-1]], fc_vals])
        ax.plot(fc_idx, fc_line, color=colors[h], linewidth=1.8,
                label=f"{h}h Forecast")

    # Vertical separator at forecast start
    ax.axvline(x=n_actual - 1, color="gray", linestyle=":", linewidth=1.2, label="Forecast start")

    ax.set_title(f"{name} — Actual vs Predicted vs Forecast")
    ax.set_xlabel("Time step")
    ax.set_ylabel(TARGET_COL)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FORECAST_VISUAL_DIR, f"{name}_forecast_plot.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved forecast plot → {FORECAST_VISUAL_DIR}/{name}_forecast_plot.png")

# ── GRID SEARCH ─────────────────────────────────────────
def run_grid_search(df):
    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    best_mse = float("inf")
    best_params = None
    for combo in combos:
        params = dict(zip(keys, combo))
        print(f"Testing params: {params}")
        try:
            X_tr, y_tr, X_val, y_val, scaler_y, _ = prepare_data(df, params["look_back"])
        except:
            continue
        tf.keras.backend.clear_session()
        model = build_model((params["look_back"], X_tr.shape[2]), params["lstm_units"], params["dropout"])
        es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=[es],
            verbose=0
        )
        pred = model.predict(X_val, verbose=0)
        y_true = scaler_y.inverse_transform(y_val)
        y_pred = scaler_y.inverse_transform(pred)
        mse = mean_squared_error(y_true, y_pred)
        if mse < best_mse:
            best_mse = mse
            best_params = params
            best_model = model
            best_history = history
            best_y_true = y_true
            best_y_pred = y_pred
            best_scaler = scaler_y
    return best_params, best_model, best_history, best_y_true, best_y_pred, best_scaler

# ── MAIN ────────────────────────────────────────────────
def main():
    sheets = pd.read_excel(INPUT_FILE, sheet_name=None)
    results = []
    all_forecasts = {}

    for name, df in sheets.items():
        print(f"\n=== {name} ===")
        best_params, model, history, y_true, y_pred, scaler_y = run_grid_search(df)
        print("Best params:", best_params)

        # standard plots
        plot_loss(history.history, name)
        plot_pred(y_true, y_pred, name)

        # metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        results.append({
            "Sheet": name,
            "MSE": round(mse, 4),
            "MAE": round(mae, 4),
            "R2": round(r2, 4),
            **best_params
        })

        # forecast
        _, _, _, _, _, X_scaled = prepare_data(df, best_params["look_back"])
        last_window = X_scaled[-best_params["look_back"]:]
        forecasts = {h: forecast(model, last_window, scaler_y, h) for h in FORECAST_HORIZONS}
        all_forecasts[name] = forecasts

        # ── NEW: combined actual + forecast plot ──────────
        plot_pred_with_forecast(
            y_true.flatten(),
            y_pred.flatten(),
            forecasts,
            name
        )

    # save metrics
    pd.DataFrame(results).to_excel("sheet_metrics.xlsx", index=False)

    # save forecasts separately
    for h in FORECAST_HORIZONS:
        rows = []
        for loc, fc in all_forecasts.items():
            for i, v in enumerate(fc[h], 1):
                rows.append({"Location": loc, "Hour": i, "Value": v})
        pd.DataFrame(rows).to_excel(f"forecast_{h}h.xlsx", index=False)

    print("Done")

if __name__ == "__main__":
    main()