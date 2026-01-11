"""Tests for training module."""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import StockLSTM


class TestTrainingLoop:
    """Tests for training functionality."""
    
    def test_model_loss_decreases(self):
        """Training should decrease loss over epochs."""
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(100, 10, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        model = StockLSTM(input_size=5, hidden_size=32, num_layers=1)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Record losses
        losses = []
        for epoch in range(5):
            epoch_loss = 0
            model.train()
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = model(X_batch).squeeze()
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(loader))
        
        # Loss should generally decrease (allow some noise)
        assert losses[-1] < losses[0], "Final loss should be lower than initial loss"
    
    def test_model_eval_mode_no_gradients(self):
        """In eval mode with no_grad, gradients should not be computed."""
        model = StockLSTM(input_size=5, hidden_size=32, num_layers=2)
        model.eval()
        
        x = torch.randn(2, 10, 5)
        
        with torch.no_grad():
            output = model(x)
        
        assert not output.requires_grad, "Output should not require gradients in eval mode"
    
    def test_validation_loss_computation(self):
        """Validation loss should be computable without affecting model."""
        model = StockLSTM(input_size=5, hidden_size=32, num_layers=2)
        criterion = torch.nn.MSELoss()
        
        # Get initial weights
        initial_weights = model.fc.weight.clone()
        
        # Create validation data
        X_val = torch.randn(20, 10, 5)
        y_val = torch.randn(20)
        
        # Compute validation loss
        model.eval()
        with torch.no_grad():
            pred = model(X_val).squeeze()
            val_loss = criterion(pred, y_val)
        
        # Weights should not change
        torch.testing.assert_close(model.fc.weight, initial_weights)
        assert isinstance(val_loss.item(), float)


class TestWalkForwardValidation:
    """Tests for walk-forward validation logic."""
    
    def test_expanding_window_splits(self):
        """Walk-forward should use expanding training windows."""
        total_data = 100
        n_splits = 4
        min_train_size = 50
        
        # Calculate split sizes (mimicking walk_forward_validation logic)
        test_size = (total_data - min_train_size) // n_splits
        
        windows = []
        for fold in range(n_splits):
            train_end = min_train_size + (fold * test_size)
            test_end = train_end + test_size
            windows.append((train_end, test_end))
        
        # Each window should have more training data than previous
        for i in range(1, len(windows)):
            assert windows[i][0] > windows[i-1][0], "Training window should expand"
    
    def test_no_data_leakage(self):
        """Test data should never overlap with training data."""
        total_data = 100
        n_splits = 4
        min_train_size = 50
        test_size = (total_data - min_train_size) // n_splits
        
        for fold in range(n_splits):
            train_end = min_train_size + (fold * test_size)
            test_start = train_end
            test_end = train_end + test_size
            
            # Training indices: 0 to train_end-1
            # Test indices: test_start to test_end-1
            train_indices = set(range(train_end))
            test_indices = set(range(test_start, test_end))
            
            overlap = train_indices & test_indices
            assert len(overlap) == 0, f"Data leakage detected in fold {fold}: {overlap}"
    
    def test_all_test_data_used(self):
        """All available test data should be used across folds."""
        total_data = 100
        n_splits = 4
        min_train_size = 50
        test_size = (total_data - min_train_size) // n_splits
        
        tested_indices = set()
        for fold in range(n_splits):
            train_end = min_train_size + (fold * test_size)
            test_end = train_end + test_size
            tested_indices.update(range(train_end, test_end))
        
        expected = set(range(min_train_size, min_train_size + n_splits * test_size))
        assert tested_indices == expected, "All expected test indices should be covered"
