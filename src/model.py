"""
src/model.py
─────────────────────────────────────────────────────────
Defines the PyTorch LSTM model for AirCast.

Architecture:
- 2-layer LSTM with dropout
- Fully connected output layer
- Input:  (batch, 24 hours, 3 features)
- Output: (batch, 24 hours) — PM2.5 forecast

Run to verify: python src/model.py
─────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn


class AirCastLSTM(nn.Module):
    def __init__(
        self,
        input_size: int  = 3,    # pm25, pm10, no2
        hidden_size: int = 64,   # neurons per LSTM layer
        num_layers: int  = 2,    # stacked LSTM layers
        output_size: int = 24,   # hours to forecast
        dropout: float   = 0.2,  # dropout between LSTM layers
    ):
        super(AirCastLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # ── LSTM layers ───────────────────────────────────────────────────────
        # batch_first=True means input shape is (batch, timesteps, features)
        # dropout only applied between layers (not after last layer)
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        # ── Output layer ──────────────────────────────────────────────────────
        # Takes the final LSTM hidden state and maps to 24-hour forecast
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        x shape: (batch_size, 24, 3)
        
        Step 1: Pass through LSTM
          lstm_out shape: (batch_size, 24, hidden_size)
          We only care about the LAST timestep's output
          because it summarizes the entire 24-hour input sequence.

        Step 2: Pass last timestep through FC layer
          output shape: (batch_size, 24) — the 24-hour PM2.5 forecast
        """
        lstm_out, _ = self.lstm(x)

        # Take only the last timestep output
        last_hidden = lstm_out[:, -1, :]   # shape: (batch_size, hidden_size)

        # Map to forecast
        output = self.fc(last_hidden)      # shape: (batch_size, 24)

        return output


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 55)
    print(" AirCast — Model Architecture Check")
    print("=" * 55)

    # Build model
    model = AirCastLSTM(
        input_size  = 3,
        hidden_size = 64,
        num_layers  = 2,
        output_size = 24,
        dropout     = 0.2,
    )

    print(f"\nModel architecture:")
    print(model)

    # Test with a dummy batch
    batch_size = 16
    dummy_input = torch.randn(batch_size, 24, 3)
    dummy_output = model(dummy_input)

    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape}")
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")
    print("\n✓ Model architecture verified.")