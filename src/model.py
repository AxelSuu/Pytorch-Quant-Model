"""LSTM model for stock price prediction."""

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """Simple LSTM model for stock prediction."""
    
    def __init__(self, input_size: int = 5, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def load_model(checkpoint_path: str, device: torch.device) -> StockLSTM:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = StockLSTM(
        input_size=checkpoint.get("input_size", 5),
        hidden_size=checkpoint.get("hidden_size", 128),
        num_layers=checkpoint.get("num_layers", 2),
        dropout=checkpoint.get("dropout", 0.2)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_checkpoint(checkpoint_path: str, device: torch.device) -> tuple:
    """Load model and scaler data from checkpoint.
    
    Returns:
        tuple: (model, scaler_min, scaler_max, input_size, use_indicators)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    input_size = checkpoint.get("input_size", 5)
    model = StockLSTM(
        input_size=input_size,
        hidden_size=checkpoint.get("hidden_size", 128),
        num_layers=checkpoint.get("num_layers", 2),
        dropout=checkpoint.get("dropout", 0.2)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    scaler_min = checkpoint.get("scaler_min")
    scaler_max = checkpoint.get("scaler_max")
    use_indicators = checkpoint.get("use_indicators", False)
    
    return model, scaler_min, scaler_max, input_size, use_indicators


def reconstruct_scaler(scaler_min: list, scaler_max: list):
    """Reconstruct MinMaxScaler from saved min/max values."""
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    
    scaler = MinMaxScaler()
    scaler.data_min_ = np.array(scaler_min)
    scaler.data_max_ = np.array(scaler_max)
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    scaler.scale_ = 1.0 / scaler.data_range_
    scaler.min_ = -scaler.data_min_ * scaler.scale_
    scaler.n_features_in_ = len(scaler_min)
    scaler.feature_range = (0, 1)
    return scaler
