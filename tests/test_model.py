"""
Test module for stock price prediction models.

This module contains unit tests for the stock price prediction models
and related functionality.
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import StockLSTM, StockGRU, StockTransformer, create_model
from train import StockTrainer
from utils import get_device, set_seed


class TestStockModels(unittest.TestCase):
    """Test cases for stock prediction models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')  # Use CPU for testing
        self.batch_size = 16
        self.sequence_length = 30
        self.input_size = 15
        self.hidden_size = 64
        self.num_layers = 2
        self.output_size = 1
        
        # Create dummy input
        self.dummy_input = torch.randn(
            self.batch_size, self.sequence_length, self.input_size
        )
        
        # Set seed for reproducibility
        set_seed(42)
    
    def test_stock_lstm_creation(self):
        """Test StockLSTM model creation."""
        model = StockLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        
        self.assertIsInstance(model, StockLSTM)
        self.assertEqual(model.input_size, self.input_size)
        self.assertEqual(model.hidden_size, self.hidden_size)
        self.assertEqual(model.num_layers, self.num_layers)
    
    def test_stock_lstm_forward(self):
        """Test StockLSTM forward pass."""
        model = StockLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        
        output = model(self.dummy_input)
        
        # Check output shape
        expected_shape = (self.batch_size, self.output_size)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output is finite
        self.assertTrue(torch.isfinite(output).all())
    
    def test_stock_lstm_bidirectional(self):
        """Test bidirectional StockLSTM."""
        model = StockLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True
        )
        
        output = model(self.dummy_input)
        
        # Check output shape
        expected_shape = (self.batch_size, self.output_size)
        self.assertEqual(output.shape, expected_shape)
    
    def test_stock_gru_creation(self):
        """Test StockGRU model creation."""
        model = StockGRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        
        self.assertIsInstance(model, StockGRU)
        self.assertEqual(model.input_size, self.input_size)
        self.assertEqual(model.hidden_size, self.hidden_size)
        self.assertEqual(model.num_layers, self.num_layers)
    
    def test_stock_gru_forward(self):
        """Test StockGRU forward pass."""
        model = StockGRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        
        output = model(self.dummy_input)
        
        # Check output shape
        expected_shape = (self.batch_size, self.output_size)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output is finite
        self.assertTrue(torch.isfinite(output).all())
    
    def test_stock_transformer_creation(self):
        """Test StockTransformer model creation."""
        model = StockTransformer(
            input_size=self.input_size,
            d_model=64,
            num_heads=4,
            num_layers=2
        )
        
        self.assertIsInstance(model, StockTransformer)
        self.assertEqual(model.input_size, self.input_size)
        self.assertEqual(model.d_model, 64)
    
    def test_stock_transformer_forward(self):
        """Test StockTransformer forward pass."""
        model = StockTransformer(
            input_size=self.input_size,
            d_model=64,
            num_heads=4,
            num_layers=2
        )
        
        output = model(self.dummy_input)
        
        # Check output shape
        expected_shape = (self.batch_size, self.output_size)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output is finite
        self.assertTrue(torch.isfinite(output).all())
    
    def test_create_model_function(self):
        """Test create_model factory function."""
        # Test LSTM creation
        lstm_model = create_model(
            "lstm",
            input_size=self.input_size,
            hidden_size=self.hidden_size
        )
        self.assertIsInstance(lstm_model, StockLSTM)
        
        # Test GRU creation
        gru_model = create_model(
            "gru",
            input_size=self.input_size,
            hidden_size=self.hidden_size
        )
        self.assertIsInstance(gru_model, StockGRU)
        
        # Test Transformer creation
        transformer_model = create_model(
            "transformer",
            input_size=self.input_size,
            d_model=64
        )
        self.assertIsInstance(transformer_model, StockTransformer)
        
        # Test invalid model type
        with self.assertRaises(ValueError):
            create_model("invalid_model")
    
    def test_model_parameters(self):
        """Test model parameter counting."""
        model = StockLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.assertGreater(total_params, 0)
        self.assertEqual(total_params, trainable_params)  # All parameters should be trainable
    
    def test_model_training_mode(self):
        """Test model training/evaluation mode switching."""
        model = StockLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        
        # Test training mode
        model.train()
        self.assertTrue(model.training)
        
        # Test evaluation mode
        model.eval()
        self.assertFalse(model.training)
    
    def test_model_device_placement(self):
        """Test model device placement."""
        model = StockLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        
        # Test CPU placement
        model.to(torch.device('cpu'))
        for param in model.parameters():
            self.assertEqual(param.device.type, 'cpu')
        
        # Test CUDA placement (if available)
        if torch.cuda.is_available():
            model.to(torch.device('cuda'))
            for param in model.parameters():
                self.assertEqual(param.device.type, 'cuda')
    
    def test_model_gradient_computation(self):
        """Test gradient computation."""
        model = StockLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        
        # Enable gradients
        model.train()
        
        # Forward pass
        output = model(self.dummy_input)
        loss = torch.mean(output)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_sequence_prediction(self):
        """Test sequence prediction functionality."""
        model = StockLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        
        # Test predict_sequence method
        single_input = self.dummy_input[:1]  # Single sample
        steps = 5
        
        predictions = model.predict_sequence(single_input, steps)
        
        # Check output shape
        expected_shape = (1, steps)
        self.assertEqual(predictions.shape, expected_shape)
        
        # Check predictions are finite
        self.assertTrue(torch.isfinite(predictions).all())


class TestStockTrainer(unittest.TestCase):
    """Test cases for StockTrainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.model = StockLSTM(input_size=15, hidden_size=32, num_layers=1)
        self.trainer = StockTrainer(
            model=self.model,
            device=self.device,
            model_name="test_model",
            save_dir="test_checkpoints"
        )
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        self.assertIsInstance(self.trainer, StockTrainer)
        self.assertEqual(self.trainer.model_name, "test_model")
        self.assertEqual(self.trainer.device, self.device)
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        # Create dummy predictions and targets
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.1, 2.1, 2.9, 3.8, 5.2])
        
        metrics = self.trainer.calculate_metrics(predictions, targets)
        
        # Check all metrics are present
        expected_metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'R2']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
            self.assertTrue(np.isfinite(metrics[metric]))
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        if os.path.exists("test_checkpoints"):
            shutil.rmtree("test_checkpoints")


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        self.assertIsInstance(device, torch.device)
        self.assertIn(device.type, ['cpu', 'cuda'])
    
    def test_set_seed(self):
        """Test seed setting for reproducibility."""
        set_seed(42)
        
        # Generate random numbers
        torch_rand1 = torch.randn(5)
        np_rand1 = np.random.randn(5)
        
        # Reset seed and generate again
        set_seed(42)
        torch_rand2 = torch.randn(5)
        np_rand2 = np.random.randn(5)
        
        # Check reproducibility
        self.assertTrue(torch.allclose(torch_rand1, torch_rand2))
        self.assertTrue(np.allclose(np_rand1, np_rand2))


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)