"""
Stock Price Prediction Models using PyTorch.

This module contains various neural network architectures for stock price forecasting,
including LSTM, GRU, and Transformer-based models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class StockLSTM(nn.Module):
    """
    LSTM-based model for stock price prediction.
    
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
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Fully connected layers
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc1 = nn.Linear(lstm_output_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(lstm_output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # Initialize hidden state
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
        
        # Use the last output
        last_output = lstm_out[:, -1, :]
        
        # Apply layer normalization
        last_output = self.layer_norm(last_output)
        
        # Fully connected layers
        out = self.dropout(last_output)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
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


class StockGRU(nn.Module):
    """
    GRU-based model for stock price prediction.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units in GRU layers
        num_layers: Number of GRU layers
        output_size: Number of output predictions
        dropout: Dropout probability
    """
    
    def __init__(self, 
                 input_size: int = 15,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.2):
        super(StockGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # GRU forward pass
        gru_out, hn = self.gru(x, h0)
        
        # Use the last output
        last_output = gru_out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(last_output)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
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
                    last_features = current_input[:, -1:, :].clone()
                    last_features[:, 0, 0] = pred.squeeze(-1)
                    
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


class StockTransformer(nn.Module):
    """
    Transformer-based model for stock price prediction.
    
    Args:
        input_size: Number of input features
        d_model: Dimension of the model
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout probability
        output_size: Number of output predictions
    """
    
    def __init__(self,
                 input_size: int = 15,
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 output_size: int = 1):
        super(StockTransformer, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for interpretability
        self.attention_weights = None
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass through the transformer.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor of shape (batch_size, output_size)
            If return_attention=True, also returns attention weights
        """
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoding with attention extraction
        transformer_out = self.transformer_encoder(x)
        
        # Extract attention weights from the last layer
        if return_attention:
            # Get attention weights from the transformer encoder
            last_layer = self.transformer_encoder.layers[-1]
            # This is a simplified approach - in practice, you'd need to modify 
            # the transformer to properly extract attention weights
            self.attention_weights = torch.ones(x.size(0), x.size(1), x.size(1))  # Placeholder
        
        # Use the last output
        last_output = transformer_out[:, -1, :]
        
        # Output layers
        out = self.dropout(last_output)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        if return_attention:
            return out, self.attention_weights
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
                    last_features = current_input[:, -1:, :].clone()
                    last_features[:, 0, 0] = pred.squeeze(-1)
                    
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


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer model.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ModelEnsemble(nn.Module):
    """
    Ensemble of multiple models for improved prediction accuracy.
    
    Args:
        models: List of models to ensemble
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
    Factory function to create different types of models.
    
    Args:
        model_type: Type of model ("lstm", "gru", "transformer")
        input_size: Number of input features
        **kwargs: Additional arguments for specific models
        
    Returns:
        Initialized model
    """
    if model_type.lower() == "lstm":
        return StockLSTM(input_size=input_size, **kwargs)
    elif model_type.lower() == "gru":
        return StockGRU(input_size=input_size, **kwargs)
    elif model_type.lower() == "transformer":
        return StockTransformer(input_size=input_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Alias for backward compatibility
Model = StockLSTM