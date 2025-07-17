"""
Main entry point for the stock price prediction project.

This script demonstrates how to use the stock price prediction system
with different models and configurations.
"""

import torch
import numpy as np
import argparse
import os
from typing import Dict, Any

# Import project modules
from data.stock_data import create_dataloaders
from model import create_model, StockLSTM, StockGRU, StockTransformer
from train import StockTrainer
from utils import (
    get_device, set_seed, print_model_summary, 
    plot_predictions, save_results, plot_stock_data
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Stock Price Prediction with PyTorch')
    
    # Data parameters
    parser.add_argument('--symbol', type=str, default='AAPL',
                       help='Stock symbol (default: AAPL)')
    parser.add_argument('--period', type=str, default='2y',
                       help='Data period (default: 2y)')
    parser.add_argument('--sequence_length', type=int, default=60,
                       help='Sequence length for LSTM input (default: 60)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training data ratio (default: 0.8)')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='lstm',
                       choices=['lstm', 'gru', 'transformer'],
                       help='Model type (default: lstm)')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='Hidden size for LSTM/GRU (default: 128)')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of LSTM/GRU layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate (default: 0.2)')
    parser.add_argument('--bidirectional', action='store_true',
                       help='Use bidirectional LSTM')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay (default: 1e-5)')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience (default: 15)')
    
    # System parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (default: auto)')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model name for saving (default: auto-generated)')
    parser.add_argument('--no_plot', action='store_true',
                       help='Disable plotting')
    
    return parser.parse_args()


def setup_environment(args):
    """Set up the environment for training."""
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    return device


def create_model_name(args):
    """Create a model name based on arguments."""
    if args.model_name:
        return args.model_name
    
    return f"{args.symbol}_{args.model_type}_{args.hidden_size}h_{args.num_layers}l_{args.sequence_length}s"


def train_model(args, device):
    """Train the stock prediction model."""
    print("=" * 60)
    print("🚀 STOCK PRICE PREDICTION TRAINING")
    print("=" * 60)
    
    # Create data loaders
    print(f"📊 Loading data for {args.symbol} (period: {args.period})...")
    try:
        train_loader, val_loader, data_loader = create_dataloaders(
            symbol=args.symbol,
            period=args.period,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio
        )
        print(f"✅ Data loaded successfully!")
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Plot stock data if requested
    if not args.no_plot:
        try:
            plot_stock_data(data_loader.data, title=f"{args.symbol} Stock Data")
        except Exception as e:
            print(f"⚠️ Warning: Could not plot stock data: {e}")
    
    # Create model
    print(f"\n🧠 Creating {args.model_type.upper()} model...")
    input_size = 15  # Number of features from stock data
    
    model_kwargs = {
        'input_size': input_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout
    }
    
    if args.model_type == 'lstm':
        model_kwargs['bidirectional'] = args.bidirectional
    
    model = create_model(args.model_type, **model_kwargs)
    
    # Print model summary
    print_model_summary(model, (args.sequence_length, input_size))
    
    # Create trainer
    model_name = create_model_name(args)
    trainer = StockTrainer(
        model=model,
        device=device,
        model_name=model_name,
        save_dir=args.save_dir,
        model_type=args.model_type
    )
    
    # Train model
    print(f"\n🔥 Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience
    )
    
    # Plot training history
    if not args.no_plot:
        trainer.plot_training_history(
            save_path=f"results/{model_name}_training_history.png"
        )
    
    return trainer, history, data_loader


def evaluate_model(trainer, data_loader, args):
    """Evaluate the trained model."""
    print("\n📈 Evaluating model...")
    
    # Get predictions on test data
    trainer.model.eval()
    test_predictions = []
    test_targets = []
    
    # Create test loader
    _, test_loader, _ = create_dataloaders(
        symbol=args.symbol,
        period=args.period,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio
    )
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(trainer.device), target.to(trainer.device)
            output = trainer.model(data)
            test_predictions.extend(output.cpu().numpy())
            test_targets.extend(target.cpu().numpy())
    
    # Convert to numpy arrays
    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)
    
    # Inverse transform predictions to original scale
    test_predictions_original = data_loader.inverse_transform_predictions(test_predictions)
    test_targets_original = data_loader.inverse_transform_predictions(test_targets)
    
    # Calculate metrics
    metrics = trainer.calculate_metrics(test_predictions, test_targets)
    
    print("📊 Test Results:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot predictions
    if not args.no_plot:
        model_name = create_model_name(args)
        plot_predictions(
            actual=test_targets_original,
            predicted=test_predictions_original,
            title=f"{args.symbol} Price Predictions - {args.model_type.upper()}",
            save_path=f"results/{model_name}_predictions.png"
        )
    
    # Save results
    results = {
        'model_type': args.model_type,
        'symbol': args.symbol,
        'metrics': metrics,
        'predictions': test_predictions_original,
        'targets': test_targets_original,
        'args': vars(args)
    }
    
    save_results(results, create_model_name(args), "results")
    
    return results


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup environment
    device = setup_environment(args)
    
    try:
        # Train model
        trainer, history, data_loader = train_model(args, device)
        
        if trainer is None:
            print("❌ Training failed. Exiting.")
            return
        
        # Evaluate model
        results = evaluate_model(trainer, data_loader, args)
        
        print("\n🎉 Training and evaluation completed successfully!")
        print(f"📁 Results saved in: results/{create_model_name(args)}.json")
        print(f"💾 Model saved in: {args.save_dir}/{create_model_name(args)}_best.pth")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def demo_prediction():
    """Demonstrate prediction with a simple example."""
    print("🎯 DEMO: Quick Stock Prediction")
    print("=" * 40)
    
    # Set up for demo
    set_seed(42)
    device = get_device()
    
    # Create simple demo data
    try:
        train_loader, val_loader, data_loader = create_dataloaders(
            symbol="AAPL",
            period="1y",
            sequence_length=30,
            batch_size=16,
            train_ratio=0.8
        )
        
        # Create simple LSTM model
        model = StockLSTM(
            input_size=15,
            hidden_size=64,
            num_layers=1,
            dropout=0.1
        )
        
        # Quick training (just a few epochs for demo)
        trainer = StockTrainer(model, device, "demo_model", "demo_checkpoints")
        trainer.train(train_loader, val_loader, epochs=10, lr=0.001)
        
        # Make a prediction
        latest_sequence = data_loader.get_latest_sequence(30)
        with torch.no_grad():
            prediction = trainer.model(latest_sequence.to(device))
        
        # Convert to original scale
        prediction_original = data_loader.inverse_transform_predictions(
            prediction.cpu().numpy()
        )
        
        print(f"🔮 Predicted next day price: ${prediction_original[0]:.2f}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_prediction()
    else:
        main()