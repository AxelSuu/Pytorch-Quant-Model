"""Tests for model module."""

import pytest
import torch
import numpy as np
import tempfile
import os

from src.model import StockLSTM, load_model, load_checkpoint, reconstruct_scaler


class TestStockLSTM:
    """Tests for StockLSTM model."""
    
    def test_model_forward_shape(self):
        """Model output should have correct shape."""
        model = StockLSTM(input_size=5, hidden_size=64, num_layers=2)
        
        # Batch of 4, sequence length 10, 5 features
        x = torch.randn(4, 10, 5)
        output = model(x)
        
        assert output.shape == (4, 1), f"Expected (4, 1), got {output.shape}"
    
    def test_model_forward_with_indicators(self):
        """Model should work with more input features (technical indicators)."""
        model = StockLSTM(input_size=18, hidden_size=64, num_layers=2)
        
        # 18 features (OHLCV + 13 technical indicators)
        x = torch.randn(2, 60, 18)
        output = model(x)
        
        assert output.shape == (2, 1)
    
    def test_model_single_layer_no_dropout(self):
        """Single layer model should not use dropout (PyTorch LSTM requirement)."""
        model = StockLSTM(input_size=5, hidden_size=32, num_layers=1, dropout=0.5)
        
        # Dropout should be 0 for single layer
        assert model.lstm.dropout == 0, "Single layer LSTM should have dropout=0"
    
    def test_model_multi_layer_with_dropout(self):
        """Multi-layer model should use dropout."""
        model = StockLSTM(input_size=5, hidden_size=32, num_layers=3, dropout=0.3)
        
        assert model.lstm.dropout == 0.3, "Multi-layer LSTM should preserve dropout"
    
    def test_model_deterministic_eval(self):
        """Model should give same output in eval mode with same input."""
        model = StockLSTM(input_size=5, hidden_size=32, num_layers=2, dropout=0.5)
        model.eval()
        
        x = torch.randn(1, 10, 5)
        
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        
        torch.testing.assert_close(out1, out2)
    
    def test_model_gradient_flow(self):
        """Gradients should flow through the model."""
        model = StockLSTM(input_size=5, hidden_size=32, num_layers=2)
        
        x = torch.randn(2, 10, 5, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None, "Gradients should flow to input"
        assert model.fc.weight.grad is not None, "Gradients should reach FC layer"


class TestCheckpointSaveLoad:
    """Tests for checkpoint saving and loading."""
    
    def test_save_and_load_model(self):
        """Model should maintain state after save/load cycle."""
        device = torch.device("cpu")
        
        # Create and train a simple model
        model = StockLSTM(input_size=5, hidden_size=32, num_layers=2, dropout=0.1)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "test_model.pth")
            
            # Save checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_size": 5,
                "hidden_size": 32,
                "num_layers": 2,
                "dropout": 0.1,
                "scaler_min": [0.0, 0.0, 0.0, 0.0, 0.0],
                "scaler_max": [100.0, 100.0, 100.0, 100.0, 1000000.0],
                "use_indicators": False,
            }, checkpoint_path)
            
            # Load checkpoint
            loaded_model = load_model(checkpoint_path, device)
            
            # Compare outputs
            x = torch.randn(1, 10, 5)
            model.eval()
            loaded_model.eval()
            
            with torch.no_grad():
                original_out = model(x)
                loaded_out = loaded_model(x)
            
            torch.testing.assert_close(original_out, loaded_out)
    
    def test_load_checkpoint_returns_all_data(self):
        """load_checkpoint should return model, scaler data, and indicators flag."""
        device = torch.device("cpu")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "test_model.pth")
            
            scaler_min = [0.0, 1.0, 2.0, 3.0, 4.0]
            scaler_max = [10.0, 11.0, 12.0, 13.0, 14.0]
            
            torch.save({
                "model_state_dict": StockLSTM(5, 32, 2).state_dict(),
                "input_size": 5,
                "hidden_size": 32,
                "num_layers": 2,
                "dropout": 0.2,
                "scaler_min": scaler_min,
                "scaler_max": scaler_max,
                "use_indicators": True,
            }, checkpoint_path)
            
            model, loaded_min, loaded_max, input_size, use_indicators = load_checkpoint(
                checkpoint_path, device
            )
            
            assert isinstance(model, StockLSTM)
            assert loaded_min == scaler_min
            assert loaded_max == scaler_max
            assert input_size == 5
            assert use_indicators is True


class TestScalerReconstruction:
    """Tests for scaler reconstruction from saved values."""
    
    def test_reconstruct_scaler_properties(self):
        """Reconstructed scaler should have correct properties."""
        scaler_min = [0.0, 10.0, 5.0, 50.0, 1000.0]
        scaler_max = [100.0, 110.0, 105.0, 150.0, 1000000.0]
        
        scaler = reconstruct_scaler(scaler_min, scaler_max)
        
        np.testing.assert_array_equal(scaler.data_min_, scaler_min)
        np.testing.assert_array_equal(scaler.data_max_, scaler_max)
        assert scaler.n_features_in_ == 5
        assert scaler.feature_range == (0, 1)
    
    def test_reconstruct_scaler_transform(self):
        """Reconstructed scaler should transform data correctly."""
        scaler_min = [0.0, 0.0, 0.0, 0.0, 0.0]
        scaler_max = [100.0, 100.0, 100.0, 100.0, 100.0]
        
        scaler = reconstruct_scaler(scaler_min, scaler_max)
        
        # Test data at min/max should transform to 0/1
        test_data = np.array([[0, 0, 0, 0, 0], [100, 100, 100, 100, 100]])
        transformed = scaler.transform(test_data)
        
        np.testing.assert_array_almost_equal(transformed[0], [0, 0, 0, 0, 0])
        np.testing.assert_array_almost_equal(transformed[1], [1, 1, 1, 1, 1])
    
    def test_reconstruct_scaler_inverse_transform(self):
        """Reconstructed scaler should inverse transform correctly."""
        scaler_min = [10.0, 20.0, 15.0, 50.0, 5000.0]
        scaler_max = [110.0, 120.0, 115.0, 150.0, 15000.0]
        
        scaler = reconstruct_scaler(scaler_min, scaler_max)
        
        # Transform and inverse should give back original
        original = np.array([[60.0, 70.0, 65.0, 100.0, 10000.0]])
        transformed = scaler.transform(original)
        recovered = scaler.inverse_transform(transformed)
        
        np.testing.assert_array_almost_equal(original, recovered)
    
    def test_reconstruct_scaler_matches_fitted_scaler(self):
        """Reconstructed scaler should match a freshly fitted scaler."""
        from sklearn.preprocessing import MinMaxScaler
        
        # Create and fit a real scaler
        data = np.array([
            [10, 20, 15, 50, 5000],
            [110, 120, 115, 150, 15000],
            [60, 70, 65, 100, 10000],
        ])
        
        original_scaler = MinMaxScaler()
        original_scaler.fit(data)
        
        # Reconstruct from saved values
        reconstructed = reconstruct_scaler(
            original_scaler.data_min_.tolist(),
            original_scaler.data_max_.tolist()
        )
        
        # Test on new data
        test_data = np.array([[80, 90, 85, 120, 12000]])
        
        original_result = original_scaler.transform(test_data)
        reconstructed_result = reconstructed.transform(test_data)
        
        np.testing.assert_array_almost_equal(original_result, reconstructed_result)
