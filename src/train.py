"""Training module for stock prediction model."""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from src.model import StockLSTM
from src.data import prepare_data, fetch_stock_data, create_sequences


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
    use_indicators = config.get("data", {}).get("use_indicators", False)
    
    # Prepare data
    print(f"Fetching data for {symbol}...")
    train_loader, val_loader, scaler, input_size = prepare_data(
        symbol, seq_length, batch_size, train_split, start, end, use_indicators
    )
    print(f"Using {input_size} features" + (" (with technical indicators)" if use_indicators else ""))
    
    # Create model
    model = StockLSTM(
        input_size=input_size,
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
            save_checkpoint(model, symbol, hidden_size, num_layers, seq_length, 
                          scaler, input_size, use_indicators)
    
    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")


def walk_forward_validation(symbol: str, config: dict, n_splits: int = 5,
                           start: str = None, end: str = None):
    """Perform walk-forward cross-validation.
    
    This trains on expanding windows and tests on subsequent periods,
    providing a more realistic evaluation for time series data.
    
    Args:
        symbol: Stock ticker symbol
        config: Configuration dictionary
        n_splits: Number of train/test splits
        start: Start date (optional)
        end: End date (optional)
    
    Returns:
        dict: Validation results with metrics per fold
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get config values
    seq_length = config["training"]["sequence_length"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    hidden_size = config["model"]["hidden_size"]
    num_layers = config["model"]["num_layers"]
    dropout = config["model"]["dropout"]
    use_indicators = config.get("data", {}).get("use_indicators", False)
    
    # Fetch all data
    print(f"Fetching data for {symbol}...")
    data, last_date = fetch_stock_data(symbol, start, end, use_indicators)
    input_size = data.shape[1]
    print(f"Total data points: {len(data)}, Features: {input_size}")
    
    # Calculate split sizes
    total_sequences = len(data) - seq_length
    min_train_size = int(total_sequences * 0.5)  # Minimum 50% for initial training
    test_size = (total_sequences - min_train_size) // n_splits
    
    if test_size < 10:
        raise ValueError(f"Not enough data for {n_splits} splits. Reduce n_splits or get more data.")
    
    results = {
        "folds": [],
        "avg_val_loss": 0,
        "avg_rmse": 0,
        "avg_mae": 0,
    }
    
    print(f"\n=== Walk-Forward Validation ({n_splits} splits) ===")
    
    for fold in range(n_splits):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        
        # Define train/test boundaries
        train_end = min_train_size + (fold * test_size)
        test_end = train_end + test_size
        
        # Scale data (fit only on training portion)
        scaler = MinMaxScaler()
        train_data = data[:train_end + seq_length]
        scaler.fit(train_data)
        scaled_data = scaler.transform(data)
        
        # Create sequences
        X, y = create_sequences(scaled_data, seq_length)
        
        # Split for this fold
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[train_end:test_end], y[train_end:test_end]
        
        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Create fresh model for each fold
        model = StockLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Train
        best_loss = float("inf")
        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                pred = model(X_batch).squeeze()
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
            
            # Track best
            model.eval()
            with torch.no_grad():
                val_loss = sum(
                    criterion(model(X.to(device)).squeeze(), y.to(device)).item()
                    for X, y in test_loader
                ) / len(test_loader)
            if val_loss < best_loss:
                best_loss = val_loss
        
        # Final evaluation on test set
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch).squeeze().cpu().numpy()
                all_preds.extend(preds.flatten())
                all_true.extend(y_batch.numpy().flatten())
        
        all_preds = np.array(all_preds)
        all_true = np.array(all_true)
        
        # Inverse transform for metrics
        def inverse_close(vals):
            dummy = np.zeros((len(vals), input_size))
            dummy[:, 3] = vals
            return scaler.inverse_transform(dummy)[:, 3]
        
        pred_prices = inverse_close(all_preds)
        true_prices = inverse_close(all_true)
        
        rmse = np.sqrt(np.mean((pred_prices - true_prices) ** 2))
        mae = np.mean(np.abs(pred_prices - true_prices))
        
        fold_result = {
            "fold": fold + 1,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "val_loss": best_loss,
            "rmse": rmse,
            "mae": mae,
        }
        results["folds"].append(fold_result)
        
        print(f"Fold {fold + 1} - Val Loss: {best_loss:.6f}, RMSE: ${rmse:.2f}, MAE: ${mae:.2f}")
    
    # Compute averages
    results["avg_val_loss"] = np.mean([f["val_loss"] for f in results["folds"]])
    results["avg_rmse"] = np.mean([f["rmse"] for f in results["folds"]])
    results["avg_mae"] = np.mean([f["mae"] for f in results["folds"]])
    
    print(f"\n=== Walk-Forward Summary ===")
    print(f"Average Val Loss: {results['avg_val_loss']:.6f}")
    print(f"Average RMSE: ${results['avg_rmse']:.2f}")
    print(f"Average MAE: ${results['avg_mae']:.2f}")
    
    return results


def save_checkpoint(model, symbol, hidden_size, num_layers, seq_length, scaler,
                   input_size=5, use_indicators=False):
    """Save model checkpoint."""
    os.makedirs("checkpoints", exist_ok=True)
    indicator_suffix = "_ind" if use_indicators else ""
    filename = f"checkpoints/{symbol}_lstm_{hidden_size}h_{num_layers}l_{seq_length}s{indicator_suffix}_best.pth"
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": model.lstm.dropout,
        "scaler_min": scaler.data_min_.tolist(),
        "scaler_max": scaler.data_max_.tolist(),
        "use_indicators": use_indicators,
    }, filename)
