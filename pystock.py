#!/usr/bin/env python3
"""
PyStock - Simple CLI for Stock Price Prediction

Usage:
    python pystock.py train --symbol TICKER [--start DATE --end DATE]
    python pystock.py evaluate --symbol TICKER
    python pystock.py predict --symbol TICKER
"""

import argparse
import yaml
from pathlib import Path


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="PyStock - Stock Price Prediction")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train model on stock data")
    train_parser.add_argument("--symbol", required=True, help="Stock ticker symbol")
    train_parser.add_argument("--start", help="Start date (yyyy-mm-dd)")
    train_parser.add_argument("--end", help="End date (yyyy-mm-dd)")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    eval_parser.add_argument("--symbol", required=True, help="Stock ticker symbol")
    
    # Predict command
    pred_parser = subparsers.add_parser("predict", help="Predict future prices")
    pred_parser.add_argument("--symbol", required=True, help="Stock ticker symbol")
    
    args = parser.parse_args()
    config = load_config()
    
    if args.command == "train":
        from src.train import train
        train(args.symbol, config, args.start, args.end)
    
    elif args.command == "evaluate":
        from src.predict import evaluate
        evaluate(args.symbol, config)
    
    elif args.command == "predict":
        from src.predict import predict
        predict(args.symbol, config)


if __name__ == "__main__":
    main()
