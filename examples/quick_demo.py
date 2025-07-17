"""
Quick Start Demo for Stock Price Prediction.

This script demonstrates how to quickly get started with stock price prediction
using the provided models and utilities.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.stock_data import create_dataloaders
from model import create_model
from train import StockTrainer
from utils import get_device, set_seed, plot_stock_data


def quick_demo():
    """Run a quick demo of stock price prediction."""
    
    print("🚀 Stock Price Prediction - Quick Demo")
    print("=" * 50)
    
    # Set up
    set_seed(42)
    device = get_device()
    
    # Parameters
    SYMBOL = "AAPL"
    PERIOD = "1y"
    SEQUENCE_LENGTH = 30
    BATCH_SIZE = 32
    EPOCHS = 10  # Quick demo with fewer epochs
    
    print(f"📊 Loading data for {SYMBOL} (last {PERIOD})...")
    
    try:
        # Create data loaders
        train_loader, val_loader, data_loader = create_dataloaders(
            symbol=SYMBOL,
            period=PERIOD,
            sequence_length=SEQUENCE_LENGTH,
            batch_size=BATCH_SIZE,
            train_ratio=0.8
        )
        
        print(f"✅ Data loaded successfully!")
        print(f"   Training samples: {len(train_loader.dataset)}")
        print(f"   Validation samples: {len(val_loader.dataset)}")
        
        # Plot stock data
        try:
            plot_stock_data(
                data_loader.data,
                title=f"{SYMBOL} Stock Data - Last {PERIOD}",
                save_path=f"demo_{SYMBOL}_data.png"
            )
            print("📈 Stock data plot saved as demo_AAPL_data.png")
        except Exception as e:
            print(f"⚠️ Could not plot data: {e}")
        
        # Create model
        print(f"\n🧠 Creating LSTM model...")
        model = create_model(
            "lstm",
            input_size=15,
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        )
        
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create trainer
        trainer = StockTrainer(
            model=model,
            device=device,
            model_name=f"demo_{SYMBOL}_lstm",
            save_dir="demo_checkpoints"
        )
        
        # Train model
        print(f"\n🔥 Training model for {EPOCHS} epochs...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=EPOCHS,
            lr=0.001,
            patience=5
        )
        
        print(f"✅ Training completed!")
        print(f"   Best validation loss: {trainer.best_val_loss:.6f}")
        
        # Make a prediction
        print(f"\n🔮 Making prediction for next day...")
        trainer.model.eval()
        
        # Get latest sequence
        latest_sequence = data_loader.get_latest_sequence(SEQUENCE_LENGTH)
        
        with torch.no_grad():
            prediction = trainer.model(latest_sequence.to(device))
        
        # Convert to original scale
        prediction_original = data_loader.inverse_transform_predictions(
            prediction.cpu().numpy()
        )
        
        # Get current price for comparison
        current_price = data_loader.data['Close'].iloc[-1]
        
        print(f"   Current price: ${current_price:.2f}")
        print(f"   Predicted next day price: ${prediction_original[0]:.2f}")
        print(f"   Predicted change: {((prediction_original[0] - current_price) / current_price * 100):+.2f}%")
        
        # Plot training history
        try:
            trainer.plot_training_history(
                save_path=f"demo_{SYMBOL}_training.png"
            )
            print(f"📊 Training history saved as demo_{SYMBOL}_training.png")
        except Exception as e:
            print(f"⚠️ Could not plot training history: {e}")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"📁 Model saved in: demo_checkpoints/demo_{SYMBOL}_lstm_best.pth")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def compare_models_demo():
    """Demo comparing different model types."""
    
    print("🔬 Model Comparison Demo")
    print("=" * 40)
    
    # Set up
    set_seed(42)
    device = get_device()
    
    # Parameters
    SYMBOL = "AAPL"
    PERIOD = "6mo"
    SEQUENCE_LENGTH = 30
    BATCH_SIZE = 32
    EPOCHS = 5  # Quick comparison
    
    model_types = ["lstm", "gru"]
    results = {}
    
    print(f"📊 Loading data for {SYMBOL}...")
    
    try:
        # Create data loaders
        train_loader, val_loader, data_loader = create_dataloaders(
            symbol=SYMBOL,
            period=PERIOD,
            sequence_length=SEQUENCE_LENGTH,
            batch_size=BATCH_SIZE,
            train_ratio=0.8
        )
        
        print(f"✅ Data loaded successfully!")
        
        # Compare models
        for model_type in model_types:
            print(f"\n🧠 Testing {model_type.upper()} model...")
            
            # Create model
            if model_type == "transformer":
                model = create_model(
                    model_type,
                    input_size=15,
                    d_model=64,
                    num_heads=4,
                    num_layers=2
                )
            else:
                model = create_model(
                    model_type,
                    input_size=15,
                    hidden_size=64,
                    num_layers=2,
                    dropout=0.2
                )
            
            # Create trainer
            trainer = StockTrainer(
                model=model,
                device=device,
                model_name=f"compare_{model_type}",
                save_dir="compare_checkpoints"
            )
            
            # Train model
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=EPOCHS,
                lr=0.001,
                patience=3
            )
            
            results[model_type] = {
                'best_val_loss': trainer.best_val_loss,
                'final_train_loss': history['train_losses'][-1],
                'final_val_loss': history['val_losses'][-1],
                'num_params': sum(p.numel() for p in model.parameters())
            }
            
            print(f"   Best validation loss: {trainer.best_val_loss:.6f}")
        
        # Compare results
        print(f"\n📊 Model Comparison Results:")
        print("-" * 50)
        print(f"{'Model':<12} {'Val Loss':<12} {'Parameters':<12}")
        print("-" * 50)
        
        for model_type, metrics in results.items():
            print(f"{model_type.upper():<12} {metrics['best_val_loss']:<12.6f} {metrics['num_params']:<12,}")
        
        # Find best model
        best_model = min(results.items(), key=lambda x: x[1]['best_val_loss'])
        print(f"\n🏆 Best model: {best_model[0].upper()} (Loss: {best_model[1]['best_val_loss']:.6f})")
        
        print(f"\n🎉 Model comparison completed!")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import torch
    
    # Check if running comparison demo
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_models_demo()
    else:
        quick_demo()
