"""Prediction and evaluation module."""

import os
import glob
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.model import load_model
from src.data import prepare_prediction_data, fetch_stock_data, create_sequences


def find_checkpoint(symbol: str) -> str:
    """Find the best checkpoint for a symbol."""
    pattern = f"checkpoints/{symbol}_lstm_*_best.pth"
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No checkpoint found for {symbol}")
    return matches[0]


def evaluate(symbol: str, config: dict):
    """Evaluate model on recent data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_length = config["training"]["sequence_length"]
    
    # Load model
    checkpoint_path = find_checkpoint(symbol)
    model = load_model(checkpoint_path, device)
    print(f"Loaded model from {checkpoint_path}")
    
    # Get data
    data, last_date = fetch_stock_data(symbol)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    
    X, y_true = create_sequences(scaled, seq_length)
    X = torch.FloatTensor(X).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        y_pred = model(X).cpu().numpy().flatten()
    
    # Inverse transform for Close price (index 3)
    y_true_prices = inverse_transform_close(y_true, scaler)
    y_pred_prices = inverse_transform_close(y_pred, scaler)
    
    # Calculate metrics
    mse = np.mean((y_true_prices - y_pred_prices) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true_prices - y_pred_prices))
    mape = np.mean(np.abs((y_true_prices - y_pred_prices) / y_true_prices)) * 100
    
    print(f"\n=== Evaluation Results for {symbol} ===")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAE:  ${mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Data points: {len(y_true)}")


def predict(symbol: str, config: dict):
    """Predict future stock prices."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_length = config["training"]["sequence_length"]
    forecast_days = config["predict"]["forecast_days"]
    
    # Load model
    checkpoint_path = find_checkpoint(symbol)
    model = load_model(checkpoint_path, device)
    
    # Get data
    sequence, scaler, current_price, last_date = prepare_prediction_data(symbol, seq_length)
    sequence = sequence.to(device)
    
    print(f"\n=== {symbol} Price Prediction ===")
    print(f"Current price: ${current_price:.2f}")
    print(f"Last data: {last_date.strftime('%Y-%m-%d')}")
    print(f"\nForecast for next {forecast_days} days:")
    
    model.eval()
    predictions = []
    current_seq = sequence.clone()
    
    with torch.no_grad():
        for day in range(forecast_days):
            pred = model(current_seq)
            pred_value = pred.item()
            predictions.append(pred_value)
            
            # Update sequence with prediction (shift and add new)
            new_row = current_seq[0, -1, :].clone()
            new_row[3] = pred_value  # Update Close price
            current_seq = torch.cat([current_seq[:, 1:, :], new_row.unsqueeze(0).unsqueeze(0)], dim=1)
    
    # Inverse transform predictions
    pred_prices = inverse_transform_close(np.array(predictions), scaler)
    
    for i, price in enumerate(pred_prices, 1):
        change = ((price - current_price) / current_price) * 100
        arrow = "↑" if change > 0 else "↓"
        print(f"  Day {i}: ${price:.2f} ({arrow} {abs(change):.2f}%)")


def inverse_transform_close(scaled_values: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Inverse transform scaled Close prices."""
    # Close is at index 3
    dummy = np.zeros((len(scaled_values), 5))
    dummy[:, 3] = scaled_values
    return scaler.inverse_transform(dummy)[:, 3]
