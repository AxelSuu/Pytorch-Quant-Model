"""
Evaluation script for stock price prediction models.

This script provides tools for evaluating trained models, making predictions,
and analyzing model performance.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os
import json
import argparse

from data.stock_data import create_dataloaders, StockDataLoader
from model import create_model
from train import StockTrainer
from utils import (
    plot_predictions, plot_stock_data, save_results,
    backtest_strategy, calculate_portfolio_metrics,
    get_device, set_seed
)


class StockPredictor:
    """
    Class for making predictions with trained stock models.
    
    Args:
        model_path: Path to the trained model checkpoint
        model_type: Type of model (lstm, gru, transformer)
        device: Device to run predictions on
    """
    
    def __init__(self, model_path: str, model_type: str = "lstm", device: torch.device = None):
        self.model_path = model_path
        self.model_type = model_type
        self.device = device or get_device()
        
        # Load model
        self.model = None
        self.data_loader = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model from checkpoint."""
        try:
            # Load checkpoint with weights_only=False for compatibility
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Extract model type from checkpoint if available, otherwise use provided model_type
            checkpoint_model_type = checkpoint.get('model_type', self.model_type)
            if checkpoint_model_type != self.model_type:
                print(f"🔄 Model type from checkpoint: {checkpoint_model_type}, overriding provided: {self.model_type}")
                self.model_type = checkpoint_model_type
            
            # Create model (assuming standard parameters for now)
            self.model = create_model(
                model_type=self.model_type,
                input_size=15,  # Standard number of features
                hidden_size=128,
                num_layers=2,
                dropout=0.2
            )
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded from {self.model_path}")
            print(f"   Validation loss: {checkpoint['val_loss']:.6f}")
            print(f"   Epoch: {checkpoint['epoch']}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def setup_data(self, symbol: str, period: str = "2y", sequence_length: int = 60):
        """Set up data loader for the specified symbol."""
        self.data_loader = StockDataLoader(symbol, period)
        self.data_loader.load_and_preprocess_data()
        self.sequence_length = sequence_length
        
    def predict_next_price(self, symbol: str, period: str = "2y", sequence_length: int = 60, use_realtime: bool = True) -> float:
        """
        Predict the next day's closing price.
        
        Args:
            symbol: Stock symbol
            period: Data period for training scaler (not used for real-time prediction)
            sequence_length: Length of input sequence
            use_realtime: Whether to fetch real-time data or use historical data
            
        Returns:
            Predicted price
        """
        # Setup data loader to fit the scaler (needed for real-time data processing)
        self.setup_data(symbol, period, sequence_length)
        
        # Get latest sequence (real-time or historical)
        if use_realtime:
            print("🔄 Using real-time data for prediction...")
            latest_sequence = self.data_loader.get_realtime_sequence(sequence_length)
        else:
            print("🔄 Using historical data for prediction...")
            latest_sequence = self.data_loader.get_latest_sequence(sequence_length)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(latest_sequence.to(self.device))
        
        # Convert to original scale
        prediction_original = self.data_loader.inverse_transform_predictions(
            prediction.cpu().numpy()
        )
        
        return prediction_original[0]
    
    def predict_sequence(self, symbol: str, steps: int = 5, 
                        period: str = "2y", sequence_length: int = 60, use_realtime: bool = True) -> List[float]:
        """
        Predict multiple future prices.
        
        Args:
            symbol: Stock symbol
            steps: Number of future steps to predict
            period: Data period for training scaler
            sequence_length: Length of input sequence
            use_realtime: Whether to fetch real-time data or use historical data
            
        Returns:
            List of predicted prices
        """
        # Setup data loader to fit the scaler
        self.setup_data(symbol, period, sequence_length)
        
        # Get latest sequence (real-time or historical)
        if use_realtime:
            print("🔄 Using real-time data for sequence prediction...")
            latest_sequence = self.data_loader.get_realtime_sequence(sequence_length)
        else:
            print("🔄 Using historical data for sequence prediction...")
            latest_sequence = self.data_loader.get_latest_sequence(sequence_length)
        
        # Make predictions
        if hasattr(self.model, 'predict_sequence'):
            # Use model's built-in sequence prediction
            with torch.no_grad():
                predictions = self.model.predict_sequence(
                    latest_sequence.to(self.device), steps
                )
        else:
            # Manual sequence prediction
            predictions = []
            current_sequence = latest_sequence.to(self.device)
            
            for _ in range(steps):
                with torch.no_grad():
                    pred = self.model(current_sequence)
                    predictions.append(pred)
                    
                    # Update sequence for next prediction
                    # Remove first timestep and add prediction
                    new_input = torch.cat([
                        current_sequence[:, 1:, :],
                        torch.cat([
                            current_sequence[:, -1:, :-1],
                            pred.unsqueeze(1)
                        ], dim=2)
                    ], dim=1)
                    current_sequence = new_input
            
            predictions = torch.cat(predictions, dim=1)
        
        # Convert to original scale
        predictions_original = self.data_loader.inverse_transform_predictions(
            predictions.cpu().numpy().flatten()
        )
        
        return predictions_original.tolist()
    
    def evaluate_on_test_data(self, symbol: str, period: str = "2y", 
                             sequence_length: int = 60, batch_size: int = 32) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            symbol: Stock symbol
            period: Data period
            sequence_length: Length of input sequence
            batch_size: Batch size
            
        Returns:
            Dictionary containing evaluation results
        """
        # Create data loaders
        train_loader, test_loader, data_loader = create_dataloaders(
            symbol=symbol,
            period=period,
            sequence_length=sequence_length,
            batch_size=batch_size,
            train_ratio=0.8
        )
        
        # Get predictions
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                predictions.extend(output.cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Convert to original scale
        predictions_original = data_loader.inverse_transform_predictions(predictions)
        targets_original = data_loader.inverse_transform_predictions(targets)
        
        # Calculate metrics
        from train import StockTrainer
        dummy_trainer = StockTrainer(self.model, self.device)
        metrics = dummy_trainer.calculate_metrics(predictions, targets)
        
        # Backtest strategy
        backtest_results = backtest_strategy(predictions_original, targets_original)
        
        return {
            'metrics': metrics,
            'predictions': predictions_original,
            'targets': targets_original,
            'backtest': backtest_results,
            'symbol': symbol
        }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Stock Price Prediction Model')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='lstm',
                       choices=['lstm', 'gru', 'transformer'],
                       help='Type of model')
    parser.add_argument('--symbol', type=str, default='AAPL',
                       help='Stock symbol to evaluate')
    parser.add_argument('--period', type=str, default='2y',
                       help='Data period')
    parser.add_argument('--sequence_length', type=int, default=60,
                       help='Sequence length')
    parser.add_argument('--predict_steps', type=int, default=5,
                       help='Number of future steps to predict')
    parser.add_argument('--save_results', action='store_true',
                       help='Save evaluation results')
    parser.add_argument('--use_realtime', action='store_true', default=True,
                       help='Use real-time data for predictions (default: True)')
    parser.add_argument('--historical_only', action='store_true',
                       help='Use only historical data (overrides --use_realtime)')
    
    args = parser.parse_args()
    
    # Determine whether to use real-time data
    use_realtime = args.use_realtime and not args.historical_only
    
    # Set up predictor
    set_seed(42)
    predictor = StockPredictor(args.model_path, args.model_type)
    
    print("🔍 STOCK PRICE PREDICTION EVALUATION")
    print("=" * 50)
    print(f"📊 Symbol: {args.symbol}")
    print(f"🤖 Model: {args.model_type.upper()}")
    print(f"📡 Data Mode: {'Real-time' if use_realtime else 'Historical'}")
    print("=" * 50)
    
    # Predict next price
    try:
        next_price = predictor.predict_next_price(
            args.symbol, args.period, args.sequence_length, use_realtime
        )
        print(f"\n📈 Next trading day prediction for {args.symbol}: ${next_price:.2f}")
    except Exception as e:
        print(f"❌ Error predicting next price: {e}")
    
    # Predict sequence
    try:
        future_prices = predictor.predict_sequence(
            args.symbol, args.predict_steps, args.period, args.sequence_length, use_realtime
        )
        print(f"\n📊 Future {args.predict_steps} trading days predictions:")
        for i, price in enumerate(future_prices, 1):
            print(f"   Day {i}: ${price:.2f}")
    except Exception as e:
        print(f"❌ Error predicting sequence: {e}")
    
    # Evaluate on test data
    try:
        print(f"\n🧪 Evaluating on test data for {args.symbol}...")
        results = predictor.evaluate_on_test_data(
            args.symbol, args.period, args.sequence_length
        )
        
        print("📊 Test Metrics:")
        for metric, value in results['metrics'].items():
            print(f"   {metric}: {value:.4f}")
        
        print(f"\n💰 Backtest Results:")
        backtest = results['backtest']
        print(f"   Total Return: {backtest['Total_Return']:.2%}")
        print(f"   Final Portfolio Value: ${backtest['Final_Value']:.2f}")
        print(f"   Number of Trades: {backtest['Number_of_Trades']}")
        
        # Plot results
        plot_predictions(
            actual=results['targets'],
            predicted=results['predictions'],
            title=f"{args.symbol} Model Evaluation - {args.model_type.upper()}",
            save_path=f"evaluation_{args.symbol}_{args.model_type}.png"
        )
        
        # Save results
        if args.save_results:
            save_results(results, f"evaluation_{args.symbol}_{args.model_type}")
            print(f"💾 Results saved to evaluation_{args.symbol}_{args.model_type}.json")
            
    except Exception as e:
        print(f"❌ Error evaluating model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
