"""
Stock Price Prediction Model using PyTorch.

This module contains an enhanced LSTM neural network architecture for stock price forecasting
with attention mechanisms, residual connections, and uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class StockLSTM(nn.Module):
    """
    Enhanced LSTM-based model for stock price prediction with residual connections and attention.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units in LSTM layers
        num_layers: Number of LSTM layers
        output_size: Number of output predictions (default: 1)
        dropout: Dropout probability
        bidirectional: Whether to use bidirectional LSTM
    """
    
    def __init__(self, 
                 input_size: int = 15,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 bidirectional: bool = False):
        super(StockLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        # Input projection for residual connection
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Attention mechanism for temporal importance
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Fully connected layers with residual connections
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(lstm_output_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size // 2)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'fc' in name and 'weight' in name:
                nn.init.xavier_uniform_(param.data)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network with attention and residual connections.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # Initialize hidden state with Xavier initialization
        h0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            batch_size,
            self.hidden_size
        ).to(x.device)
        
        c0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            batch_size,
            self.hidden_size
        ).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Apply attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection
        lstm_out = self.layer_norm1(lstm_out + attn_out)
        
        # Use the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers with residual connections and normalization
        out = self.dropout(last_output)
        out = F.relu(self.layer_norm2(self.fc1(out)))
        out = self.dropout(out)
        out = F.relu(self.layer_norm3(self.fc2(out)))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out
    
    def predict_sequence(self, x: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Predict multiple steps into the future using autoregressive approach.
        
        Args:
            x: Input sequence of shape (batch_size, sequence_length, input_size)
            steps: Number of future steps to predict
            
        Returns:
            Predicted sequence of shape (batch_size, steps)
        """
        self.eval()
        predictions = []
        current_input = x.clone()
        
        with torch.no_grad():
            for step in range(steps):
                # Get prediction for current input
                pred = self.forward(current_input)
                predictions.append(pred)
                
                # Prepare next input by sliding window and appending prediction
                if step < steps - 1:  # Don't update for last iteration
                    # Create new timestep with prediction as the 'close' price
                    # and repeat other features from last timestep
                    last_features = current_input[:, -1:, :].clone()
                    
                    # Update the close price (assuming it's the first feature)
                    last_features[:, 0, 0] = pred.squeeze(-1)
                    
                    # For simplicity, keep other technical indicators the same
                    # In production, you'd recalculate technical indicators
                    
                    # Slide the window: remove first timestep, add new one
                    current_input = torch.cat([
                        current_input[:, 1:, :],
                        last_features
                    ], dim=1)
                
        return torch.cat(predictions, dim=1)
    
    def predict_next_day(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the next day's price.
        
        Args:
            x: Input sequence of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Next day price prediction of shape (batch_size, 1)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class ModelEnsemble(nn.Module):
    """
    Ensemble of multiple LSTM models for improved prediction accuracy.
    
    Args:
        models: List of LSTM models to ensemble
        weights: Optional weights for each model
    """
    
    def __init__(self, models: list, weights: Optional[list] = None):
        super(ModelEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = torch.ones(len(models)) / len(models)
        else:
            self.weights = torch.tensor(weights)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Weighted average of all model predictions
        """
        predictions = []
        
        for model in self.models:
            predictions.append(model(x))
            
        # Stack predictions and compute weighted average
        stacked_predictions = torch.stack(predictions, dim=0)
        weights = self.weights.to(x.device).view(-1, 1, 1)
        
        return torch.sum(stacked_predictions * weights, dim=0)


def create_model(model_type: str = "lstm", 
                input_size: int = 15,
                **kwargs) -> nn.Module:
    """
    Factory function to create LSTM models.
    
    Args:
        model_type: Type of model (only "lstm" is supported)
        input_size: Number of input features
        **kwargs: Additional arguments for LSTM model
        
    Returns:
        Initialized LSTM model
    """
    if model_type.lower() == "lstm":
        return StockLSTM(input_size=input_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Only 'lstm' is supported.")


def predict_with_uncertainty(model: nn.Module, 
                             x: torch.Tensor, 
                             n_samples: int = 30,
                             dropout_rate: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Monte Carlo Dropout for uncertainty estimation in predictions.
    
    This method enables dropout during inference and samples multiple predictions
    to estimate uncertainty, providing confidence intervals for predictions.
    
    Args:
        model: The trained model
        x: Input tensor of shape (batch_size, sequence_length, input_size)
        n_samples: Number of MC samples to generate
        dropout_rate: Dropout rate for MC sampling
        
    Returns:
        Tuple of (mean_prediction, std_prediction) representing mean and uncertainty
    """
    model.train()  # Enable dropout layers
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x)
            predictions.append(pred)
    
    predictions = torch.stack(predictions)
    mean_pred = predictions.mean(dim=0)
    std_pred = predictions.std(dim=0)
    
    model.eval()
    return mean_pred, std_pred


# Alias for backward compatibility
Model = StockLSTM
