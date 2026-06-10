"""
src/preprocess.py
─────────────────────────────────────────────────────────
Loads historical_90days.csv, cleans it, and creates
sliding window sequences for LSTM training.

Input  : data/raw/historical_90days.csv
Output : models/scaler.pkl  (saved MinMaxScaler)
         Returns X, y arrays ready for PyTorch

Run to verify: python src/preprocess.py
─────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler


# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "raw", "historical_90days.csv")
SCALER_PATH = os.path.join("models", "scaler.pkl")

# Features the LSTM will learn from
FEATURES    = ["pm25", "pm10", "no2"]

# Target column we are predicting
TARGET      = "pm25"

# How many past hours the LSTM looks at to make a prediction
INPUT_HOURS = 24

# How many future hours we predict
OUTPUT_HOURS = 24
# ──────────────────────────────────────────────────────────────────────────────


def load_data() -> pd.DataFrame:
    """
    Load the historical CSV and do basic cleaning.

    Steps:
    1. Parse timestamp column as datetime
    2. Sort by city + timestamp (chronological order is critical for time series)
    3. Forward-fill missing values within each city
    4. Drop any remaining NaN rows
    """
    print(f"Loading data from {DATA_PATH}...")

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    print(f"  Raw shape: {df.shape}")

    # Sort chronologically within each city
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    # Forward fill missing sensor readings within each city group
    # This handles gaps where a sensor temporarily went offline
    df[FEATURES] = df.groupby("city")[FEATURES].transform(
        lambda x: x.fillna(method="ffill")
    )

    # Drop rows that still have NaN (e.g. NaN at the very start with no prior value)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    print(f"  Cleaned shape: {df.shape}")
    print(f"  Cities: {df['city'].unique().tolist()}")
    print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    return df


def scale_data(df: pd.DataFrame) -> tuple:
    """
    Normalize feature values to [0, 1] range using MinMaxScaler.

    WHY NORMALIZE?
    LSTMs train much better on normalized data.
    PM2.5 can range from 5 to 500 — raw values cause unstable gradients.
    After scaling, all values are between 0 and 1.

    Saves the scaler to disk so we can inverse-transform predictions later
    (convert predictions back to real µg/m³ values).

    Returns:
    - df with scaled feature columns
    - fitted scaler object
    """
    scaler = MinMaxScaler()

    df = df.copy()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])

    # Save scaler — we need it later to convert predictions back to real units
    os.makedirs("models", exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n  Scaler saved to {SCALER_PATH}")

    return df, scaler


def create_sequences(city_df: pd.DataFrame) -> tuple:
    """
    Create sliding window sequences for one city.

    HOW THIS WORKS:
    Given hourly data: [h1, h2, h3, h4, ... h200]

    We create overlapping windows:
    X[0] = hours 1-24   → y[0] = hours 25-48  (PM2.5 only)
    X[1] = hours 2-25   → y[1] = hours 26-49
    X[2] = hours 3-26   → y[2] = hours 27-50
    ... and so on

    X shape: (num_samples, INPUT_HOURS, num_features)
    y shape: (num_samples, OUTPUT_HOURS)

    The LSTM sees 24 hours of [pm25, pm10, no2] and predicts
    the next 24 hours of pm25.
    """
    values = city_df[FEATURES].values   # shape: (timesteps, 3)
    target = city_df[TARGET].values     # shape: (timesteps,)

    X, y = [], []

    total_hours = INPUT_HOURS + OUTPUT_HOURS  # need 48 hours per sample

    for i in range(len(values) - total_hours + 1):
        # Input: 24 hours of all features
        X.append(values[i : i + INPUT_HOURS])
        # Output: next 24 hours of PM2.5 only
        y.append(target[i + INPUT_HOURS : i + INPUT_HOURS + OUTPUT_HOURS])

    return np.array(X), np.array(y)


def train_val_test_split(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Split into train / validation / test sets CHRONOLOGICALLY.

    WHY NOT RANDOM SPLIT?
    For time series, random split causes data leakage —
    the model would see future data during training.
    We always split by time: train on past, test on future.

    Split: 70% train | 15% validation | 15% test
    """
    n = len(X)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_train, y_train = X[:train_end],        y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:],          y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_all_cities(df: pd.DataFrame) -> tuple:
    """
    Run sequence creation for all cities and combine.
    Each city contributes its own set of training sequences.
    """
    all_X, all_y = [], []

    for city in df["city"].unique():
        city_df = df[df["city"] == city].reset_index(drop=True)
        X, y    = create_sequences(city_df)
        all_X.append(X)
        all_y.append(y)
        print(f"  {city}: {X.shape[0]} sequences created")

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    return X_all, y_all


def main():
    print("=" * 55)
    print(" AirCast — Preprocessing")
    print("=" * 55)

    # Step 1: Load and clean
    df = load_data()

    # Step 2: Normalize
    print("\nScaling features...")
    df_scaled, scaler = scale_data(df)

    # Step 3: Create sequences
    print("\nCreating LSTM sequences...")
    X, y = prepare_all_cities(df_scaled)
    print(f"\n  Total X shape: {X.shape}")
    print(f"  Total y shape: {y.shape}")

    # Step 4: Split
    print("\nSplitting into train / val / test...")
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

    print(f"  Train : X={X_train.shape}  y={y_train.shape}")
    print(f"  Val   : X={X_val.shape}    y={y_val.shape}")
    print(f"  Test  : X={X_test.shape}   y={y_test.shape}")

    print("\n✓ Preprocessing complete. Ready for model training.")

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


if __name__ == "__main__":
    main()