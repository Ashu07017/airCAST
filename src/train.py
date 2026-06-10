"""
src/train.py
─────────────────────────────────────────────────────────
Trains the AirCast LSTM model on historical pollution data.

Saves:
- models/aircast_lstm.pt     (trained model weights)
- results/training_loss.png  (train vs val loss plot)
- results/eval_metrics.txt   (MAE, RMSE on test set)

Run: python src/train.py
─────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from preprocess import main as load_preprocessed
from model import AirCastLSTM, count_parameters


# ── CONFIG ────────────────────────────────────────────────────────────────────
BATCH_SIZE   = 32
EPOCHS       = 50
LEARNING_RATE = 0.001
PATIENCE     = 7        # early stopping — stop if val loss doesn't improve

MODEL_PATH   = os.path.join("models", "aircast_lstm.pt")
SCALER_PATH  = os.path.join("models", "scaler.pkl")
LOSS_PLOT    = os.path.join("results", "training_loss.png")
METRICS_FILE = os.path.join("results", "eval_metrics.txt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────


def make_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test):
    """Convert numpy arrays to PyTorch tensors and wrap in DataLoaders."""

    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32)

    train_ds = TensorDataset(to_tensor(X_train), to_tensor(y_train))
    val_ds   = TensorDataset(to_tensor(X_val),   to_tensor(y_val))
    test_ds  = TensorDataset(to_tensor(X_test),  to_tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion):
    """Run one full pass over the training data."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()

        # Gradient clipping — prevents exploding gradients in LSTMs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion):
    """Run one full pass over validation or test data. No gradient updates."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()

    return total_loss / len(loader)


def compute_metrics(model, loader, scaler):
    """
    Compute MAE and RMSE on real (unscaled) PM2.5 values.

    We inverse-transform predictions back to µg/m³
    so the metrics are meaningful and reportable.
    """
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch.to(DEVICE)).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())

    preds   = np.concatenate(all_preds,   axis=0)   # (N, 24)
    targets = np.concatenate(all_targets, axis=0)   # (N, 24)

    # Inverse transform PM2.5 column only (index 0 in FEATURES)
    # We need to pad to match scaler's expected 3-column input
    def inverse_pm25(arr_2d):
        """arr_2d shape: (N, 24) — pad to (N*24, 3) for scaler"""
        flat = arr_2d.reshape(-1, 1)
        padded = np.zeros((flat.shape[0], 3))
        padded[:, 0] = flat[:, 0]   # pm25 is first column
        unscaled = scaler.inverse_transform(padded)
        return unscaled[:, 0].reshape(arr_2d.shape)

    preds_real   = inverse_pm25(preds)
    targets_real = inverse_pm25(targets)

    mae  = np.mean(np.abs(preds_real - targets_real))
    rmse = np.sqrt(np.mean((preds_real - targets_real) ** 2))

    return mae, rmse, preds_real, targets_real


def plot_loss(train_losses, val_losses):
    """Save training vs validation loss curve to results/."""
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", color="#2563EB")
    plt.plot(val_losses,   label="Val Loss",   color="#DC2626")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("AirCast LSTM — Training Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT, dpi=150)
    plt.close()
    print(f"  Loss plot saved → {LOSS_PLOT}")


def save_metrics(mae, rmse):
    """Save evaluation metrics to a text file for the README."""
    os.makedirs("results", exist_ok=True)
    with open(METRICS_FILE, "w") as f:
        f.write("AirCast LSTM — Test Set Evaluation\n")
        f.write("=" * 40 + "\n")
        f.write(f"MAE  (Mean Absolute Error) : {mae:.2f} µg/m³\n")
        f.write(f"RMSE (Root Mean Sq. Error) : {rmse:.2f} µg/m³\n")
        f.write("\nInterpretation:\n")
        f.write("MAE = average prediction error in µg/m³\n")
        f.write("Lower is better. WHO safe limit = 15 µg/m³ daily avg.\n")
    print(f"  Metrics saved → {METRICS_FILE}")


def main():
    print("=" * 55)
    print(f" AirCast — Model Training")
    print(f" Device: {DEVICE}")
    print("=" * 55)

    # ── Load preprocessed data ────────────────────────────────────────────────
    print("\nLoading preprocessed data...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_preprocessed()

    train_loader, val_loader, test_loader = make_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # ── Build model ───────────────────────────────────────────────────────────
    model = AirCastLSTM().to(DEVICE)
    print(f"\nModel parameters: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for up to {EPOCHS} epochs (early stopping patience={PATIENCE})...\n")

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss   = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"  Epoch {epoch:>3}/{EPOCHS}  |  "
              f"Train Loss: {train_loss:.5f}  |  "
              f"Val Loss: {val_loss:.5f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    # ── Evaluation ────────────────────────────────────────────────────────────
    print(f"\nLoading best model from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH))

    print("\nEvaluating on test set...")
    mae, rmse, _, _ = compute_metrics(model, test_loader, scaler)

    print(f"\n{'='*55}")
    print(f"  MAE  : {mae:.2f} µg/m³")
    print(f"  RMSE : {rmse:.2f} µg/m³")
    print(f"{'='*55}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\nSaving outputs...")
    plot_loss(train_losses, val_losses)
    save_metrics(mae, rmse)

    print("\n✓ Training complete.")
    print(f"  Model  → {MODEL_PATH}")
    print(f"  Plot   → {LOSS_PLOT}")
    print(f"  Metrics→ {METRICS_FILE}")


if __name__ == "__main__":
    main()