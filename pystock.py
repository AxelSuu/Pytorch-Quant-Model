#!/usr/bin/env python3
"""
PyStock - Simple CLI for Stock Price Prediction

Usage:
    pystock train --symbol TICKER [--start DATE --end DATE]
    pystock evaluate --symbol TICKER
    pystock predict --symbol TICKER
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import torch

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.stock_data import create_dataloaders
from model import create_model
from train import StockTrainer
from evaluate import StockPredictor
from utils import get_device, set_seed


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def find_best_checkpoint(symbol: str, checkpoints_dir: str = "checkpoints") -> str:
    """Find the best checkpoint for a given symbol."""
    checkpoint_pattern = f"{symbol}_lstm_*_best.pth"
    checkpoints_path = Path(checkpoints_dir)
    
    best_checkpoints = list(checkpoints_path.glob(checkpoint_pattern))
    
    if not best_checkpoints:
        raise FileNotFoundError(
            f"No checkpoint found for {symbol}. "
            f"Please train a model first with: pystock train --symbol {symbol}"
        )
    
    # Return the most recent one if multiple exist
    return str(sorted(best_checkpoints)[-1])


def train_command(args):
    """Execute the train command."""
    config = load_config()
    set_seed(config['system']['seed'])
    device = get_device()
    
    print(f"Training model for {args.symbol}")
    print(f"Data period: {args.start or 'default'} to {args.end or 'default'}")
    print(f"Device: {device}")
    
    # Prepare data loading parameters
    period = None if args.start or args.end else config['data']['period']
    start_date = args.start
    end_date = args.end
    
    # Create dataloaders
    train_loader, val_loader, scaler = create_dataloaders(
        symbol=args.symbol,
        period=period,
        start=start_date,
        end=end_date,
        sequence_length=config['data']['sequence_length'],
        batch_size=config['data']['batch_size'],
        train_ratio=config['data']['train_ratio']
    )
    
    # Create model
    model = create_model(
        model_type='lstm',
        input_size=15,  # Number of features
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        bidirectional=config['model']['bidirectional']
    )
    
    # Create run name
    run_name = f"{args.symbol}_lstm_{config['model']['hidden_size']}h_{config['model']['num_layers']}l_{config['data']['sequence_length']}s"
    
    # Train model
    trainer = StockTrainer(
        model=model,
        device=device,
        model_name=run_name,
        save_dir=config['system']['save_dir']
    )
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        patience=config['training']['patience']
    )
    
    print(f"Training complete! Best model saved to checkpoints/{run_name}_best.pth")
    print(f"Best validation loss: {min(history['val_losses']):.6f}")


def evaluate_command(args):
    """Execute the evaluate command."""
    config = load_config()
    device = get_device()
    
    print(f"Evaluating model for {args.symbol}")
    
    # Find best checkpoint
    checkpoint_path = find_best_checkpoint(args.symbol, config['system']['save_dir'])
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Create predictor
    predictor = StockPredictor(checkpoint_path, device=device)
    
    # Evaluate
    results = predictor.evaluate_on_test_data(
        symbol=args.symbol,
        sequence_length=config['data']['sequence_length']
    )
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    # Get metrics from results
    metrics = results['metrics']
    print(f"RMSE: {metrics.get('rmse', metrics.get('RMSE', 'N/A')):.4f}")
    print(f"MAE:  {metrics.get('mae', metrics.get('MAE', 'N/A')):.4f}")
    print(f"R²:   {metrics.get('r2', metrics.get('R2', 'N/A')):.4f}")
    if 'direction_accuracy' in metrics or 'Directional_Accuracy' in metrics:
        dir_acc = metrics.get('direction_accuracy', metrics.get('Directional_Accuracy', 0))
        print(f"Direction Accuracy: {dir_acc:.2f}%")
    print("="*50)
    
    # Save results
    results_dir = Path(config['system']['results_dir'])
    results_dir.mkdir(exist_ok=True)
    
    import json
    import numpy as np
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    results_serializable = convert_to_json_serializable(results)
    
    results_file = results_dir / f"evaluation_{args.symbol}_lstm.json"
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to {results_file}")


def predict_command(args):
    """Execute the predict command."""
    config = load_config()
    device = get_device()
    
    print(f"Predicting next price for {args.symbol}")
    
    # Find best checkpoint
    checkpoint_path = find_best_checkpoint(args.symbol, config['system']['save_dir'])
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Create predictor
    predictor = StockPredictor(checkpoint_path, device=device)
    
    # Get prediction with uncertainty
    result = predictor.predict_next_price(
        symbol=args.symbol,
        sequence_length=config['data']['sequence_length'],
        return_confidence=True
    )
    
    # Unpack result (returns tuple when return_confidence=True)
    predicted_price, confidence_interval = result
    
    # Convert to Python floats
    predicted_price = float(predicted_price)
    confidence_interval = float(confidence_interval)
    
    # Get current price from data loader
    current_price = float(predictor.data_loader.data['Close'].iloc[-1])
    
    print("\n" + "="*50)
    print(f"PREDICTION FOR {args.symbol}")
    print("="*50)
    print(f"Current Price:    ${current_price:.2f}")
    print(f"Predicted Price:  ${predicted_price:.2f}")
    print(f"Change:           ${predicted_price - current_price:.2f} "
          f"({((predicted_price / current_price - 1) * 100):.2f}%)")
    
    print(f"\n95% Confidence Interval:")
    print(f"  Lower Bound: ${predicted_price - confidence_interval:.2f}")
    print(f"  Upper Bound: ${predicted_price + confidence_interval:.2f}")
    print(f"  Uncertainty: ±${confidence_interval:.2f}")
    
    print("="*50)
    
    from datetime import datetime
    print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='pystock',
        description='PyStock - Simple Stock Price Prediction CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pystock train --symbol AAPL
  pystock train --symbol TSLA --start 2020-01-01 --end 2023-12-31
  pystock evaluate --symbol AAPL
  pystock predict --symbol AAPL
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model for a stock symbol')
    train_parser.add_argument('--symbol', type=str, required=True,
                            help='Stock ticker symbol (e.g., AAPL, TSLA)')
    train_parser.add_argument('--start', type=str, default=None,
                            help='Start date (yyyy-mm-dd)')
    train_parser.add_argument('--end', type=str, default=None,
                            help='End date (yyyy-mm-dd)')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--symbol', type=str, required=True,
                           help='Stock ticker symbol (e.g., AAPL, TSLA)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict next price')
    predict_parser.add_argument('--symbol', type=str, required=True,
                              help='Stock ticker symbol (e.g., AAPL, TSLA)')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'train':
            train_command(args)
        elif args.command == 'evaluate':
            evaluate_command(args)
        elif args.command == 'predict':
            predict_command(args)
    except Exception as e:
        import traceback
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
