"""Training module for stock prediction model."""

import os
import torch
import torch.nn as nn
from src.model import StockLSTM
from src.data import prepare_data


def train(symbol: str, config: dict, start: str = None, end: str = None):
    """Train the model on stock data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get config values
    seq_length = config["training"]["sequence_length"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    train_split = config["training"]["train_split"]
    hidden_size = config["model"]["hidden_size"]
    num_layers = config["model"]["num_layers"]
    dropout = config["model"]["dropout"]
    
    # Prepare data
    print(f"Fetching data for {symbol}...")
    train_loader, val_loader, scaler = prepare_data(
        symbol, seq_length, batch_size, train_split, start, end
    )
    
    # Create model
    model = StockLSTM(
        input_size=5,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float("inf")
    
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X).squeeze()
                val_loss += criterion(pred, y).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, symbol, hidden_size, num_layers, seq_length, scaler)
    
    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")


def save_checkpoint(model, symbol, hidden_size, num_layers, seq_length, scaler):
    """Save model checkpoint."""
    os.makedirs("checkpoints", exist_ok=True)
    filename = f"checkpoints/{symbol}_lstm_{hidden_size}h_{num_layers}l_{seq_length}s_best.pth"
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_size": 5,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": model.lstm.dropout,
        "scaler_min": scaler.data_min_.tolist(),
        "scaler_max": scaler.data_max_.tolist(),
    }, filename)
