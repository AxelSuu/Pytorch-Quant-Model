"""
Industrial-Grade Model Evaluation and Analysis Script.

This script provides comprehensive evaluation including:
- Advanced metrics (regression, directional, financial)
- Sophisticated backtesting strategies
- Model interpretability analysis
- Cross-validation with time series data
- Model complexity analysis
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import argparse
import json
import os
from datetime import datetime

# Import project modules
from data.stock_data import create_dataloaders, StockDataLoader
from model import create_model
from train import StockTrainer
from advanced_metrics import (
    AdvancedMetrics, AdvancedBacktester, ModelComplexityAnalyzer,
    TimeSeriesCV, comprehensive_evaluation
)
from interpretability import ModelExplainer, analyze_feature_interactions
from utils import get_device, set_seed, save_results


class IndustrialEvaluator:
    """
    Comprehensive industrial-grade model evaluator.
    """
    
    def __init__(self, 
                 model_path: str,
                 model_type: str = "lstm",
                 device: torch.device = None):
        self.model_path = model_path
        self.model_type = model_type
        self.device = device or get_device()
        
        # Load model
        self.model = self._load_model()
        
        # Initialize analyzers
        self.explainer = ModelExplainer(self.model, self.device)
        self.complexity_analyzer = ModelComplexityAnalyzer()
        
    def _load_model(self):
        """Load the trained model from checkpoint."""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Extract model type from checkpoint if available
            checkpoint_model_type = checkpoint.get('model_type', self.model_type)
            
            # Create model
            model = create_model(
                model_type=checkpoint_model_type,
                input_size=15,
                hidden_size=128,
                num_layers=2,
                dropout=0.2
            )
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")
    
    def comprehensive_evaluation(self,
                               symbol: str = "AAPL",
                               period: str = "2y",
                               sequence_length: int = 60,
                               risk_free_rate: float = 0.02) -> Dict:
        """
        Perform comprehensive model evaluation.
        
        Args:
            symbol: Stock symbol to evaluate
            period: Data period
            sequence_length: Sequence length for predictions
            risk_free_rate: Risk-free rate for financial metrics
            
        Returns:
            Complete evaluation results
        """
        print(f"🔍 Starting comprehensive evaluation for {symbol}...")
        
        # Load data
        _, test_loader, data_loader = create_dataloaders(
            symbol=symbol,
            period=period,
            sequence_length=sequence_length,
            batch_size=32,
            train_ratio=0.8
        )
        
        # Get predictions and targets
        predictions, targets, prices = self._get_predictions(test_loader, data_loader)
        
        # Comprehensive evaluation using the new framework
        results = comprehensive_evaluation(
            model=self.model,
            predictions=predictions,
            targets=targets,
            prices=prices,
            risk_free_rate=risk_free_rate
        )
        
        # Add interpretability analysis
        if len(predictions) > 0:
            print("📊 Analyzing model interpretability...")
            results['Interpretability'] = self._analyze_interpretability(
                test_loader, data_loader
            )
        
        # Add cross-validation results
        print("🔄 Performing time series cross-validation...")
        results['Cross_Validation'] = self._time_series_cross_validation(
            symbol, period, sequence_length
        )
        
        return results
    
    def _get_predictions(self, test_loader, data_loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get model predictions and targets."""
        predictions = []
        targets = []
        
        self.model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                predictions.extend(output.cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        
        # Transform back to original scale for price-based calculations
        predictions_original = data_loader.inverse_transform_predictions(predictions)
        targets_original = data_loader.inverse_transform_predictions(targets)
        
        return predictions, targets, targets_original
    
    def _analyze_interpretability(self, test_loader, data_loader) -> Dict:
        """Analyze model interpretability."""
        # Get sample data for analysis
        sample_data, _ = next(iter(test_loader))
        sample_data = sample_data[:1].to(self.device)  # Single sample
        
        # Feature names (technical indicators)
        feature_names = [
            'Close', 'Volume', 'MA_5', 'MA_10', 'MA_20', 'EMA_12', 'EMA_26',
            'RSI', 'BB_upper', 'BB_middle', 'BB_lower', 'Volatility',
            'Price_Change', 'High_Low_Pct', 'Volume_MA'
        ]
        
        try:
            # Comprehensive explanation
            explanations = self.explainer.explain_prediction(
                sample_data, feature_names, sample_idx=0
            )
            
            # Feature interactions
            if len(feature_names) <= 15:  # Avoid computation explosion
                interactions = analyze_feature_interactions(
                    self.model, sample_data, feature_names, self.device
                )
                explanations['feature_interactions'] = interactions.to_dict()
            
            return explanations
            
        except Exception as e:
            print(f"⚠️  Interpretability analysis failed: {e}")
            return {"error": str(e)}
    
    def _time_series_cross_validation(self,
                                    symbol: str,
                                    period: str,
                                    sequence_length: int) -> Dict:
        """Perform time series cross-validation."""
        try:
            # Create full dataset
            train_loader, val_loader, data_loader = create_dataloaders(
                symbol=symbol,
                period=period,
                sequence_length=sequence_length,
                batch_size=32,
                train_ratio=1.0  # Use all data for CV
            )
            
            # Get all data
            all_data = []
            all_targets = []
            
            for data, target in train_loader:
                all_data.append(data)
                all_targets.append(target)
            
            if not all_data:
                return {"error": "No data available for cross-validation"}
            
            X = torch.cat(all_data, dim=0)
            y = torch.cat(all_targets, dim=0)
            
            # Time series cross-validation
            cv = TimeSeriesCV(n_splits=5, test_size=None, gap=5)
            cv_results = []
            
            for fold, (train_idx, test_idx) in enumerate(cv.split(X.numpy())):
                print(f"   Fold {fold + 1}/5...")
                
                # Split data
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Quick evaluation (without retraining for efficiency)
                self.model.eval()
                with torch.no_grad():
                    X_test = X_test.to(self.device)
                    y_test = y_test.to(self.device)
                    
                    predictions = self.model(X_test).cpu().numpy().flatten()
                    targets = y_test.cpu().numpy().flatten()
                
                # Calculate metrics
                metrics = AdvancedMetrics()
                fold_metrics = metrics.calculate_regression_metrics(predictions, targets)
                fold_metrics['fold'] = fold + 1
                
                cv_results.append(fold_metrics)
            
            # Calculate mean and std of CV metrics
            cv_summary = {}
            for metric in cv_results[0].keys():
                if metric != 'fold':
                    values = [result[metric] for result in cv_results]
                    cv_summary[f"{metric}_mean"] = np.mean(values)
                    cv_summary[f"{metric}_std"] = np.std(values)
            
            return {
                'cv_results': cv_results,
                'cv_summary': cv_summary
            }
            
        except Exception as e:
            print(f"⚠️  Cross-validation failed: {e}")
            return {"error": str(e)}
    
    def generate_report(self, results: Dict, save_path: str = None) -> str:
        """Generate a comprehensive evaluation report."""
        report = []
        report.append("=" * 80)
        report.append("INDUSTRIAL-GRADE MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Model: {self.model_type.upper()}")
        report.append(f"Model Path: {self.model_path}")
        report.append("")
        
        # Model Complexity
        if 'Model_Complexity' in results:
            report.append("📊 MODEL COMPLEXITY ANALYSIS")
            report.append("-" * 40)
            complexity = results['Model_Complexity']
            report.append(f"Total Parameters: {complexity['Total_Parameters']:,}")
            report.append(f"Trainable Parameters: {complexity['Trainable_Parameters']:,}")
            report.append(f"Parameter Memory: {complexity['Parameter_Memory_MB']:.2f} MB")
            report.append(f"Estimated FLOPs: {complexity['Estimated_FLOPs']:,}")
            report.append(f"Efficiency Score: {results.get('Efficiency_Score', 'N/A'):.4f}")
            report.append("")
        
        # Regression Metrics
        if 'Regression_Metrics' in results:
            report.append("📈 REGRESSION PERFORMANCE")
            report.append("-" * 40)
            reg_metrics = results['Regression_Metrics']
            report.append(f"MAE (Mean Absolute Error): {reg_metrics['MAE']:.4f}")
            report.append(f"RMSE (Root Mean Squared Error): {reg_metrics['RMSE']:.4f}")
            report.append(f"MAPE (Mean Absolute Percentage Error): {reg_metrics['MAPE']:.2f}%")
            report.append(f"R² (R-squared): {reg_metrics['R2']:.4f}")
            report.append(f"MASE (Mean Absolute Scaled Error): {reg_metrics.get('MASE', 'N/A')}")
            report.append(f"SMAPE (Symmetric MAPE): {reg_metrics.get('SMAPE', 'N/A'):.2f}%")
            report.append("")
        
        # Directional Metrics
        if 'Directional_Metrics' in results:
            report.append("🎯 DIRECTIONAL ACCURACY")
            report.append("-" * 40)
            dir_metrics = results['Directional_Metrics']
            report.append(f"Directional Accuracy: {dir_metrics['Directional_Accuracy']:.2%}")
            report.append(f"Classification Accuracy: {dir_metrics['Classification_Accuracy']:.2%}")
            report.append(f"Precision: {dir_metrics['Precision']:.4f}")
            report.append(f"Recall: {dir_metrics['Recall']:.4f}")
            report.append(f"F1-Score: {dir_metrics['F1_Score']:.4f}")
            report.append(f"Up Movement Accuracy: {dir_metrics['Up_Movement_Accuracy']:.2%}")
            report.append(f"Down Movement Accuracy: {dir_metrics['Down_Movement_Accuracy']:.2%}")
            report.append("")
        
        # Trading Strategy Performance
        if 'Momentum_Strategy' in results:
            report.append("💹 MOMENTUM STRATEGY PERFORMANCE")
            report.append("-" * 40)
            momentum = results['Momentum_Strategy']['Financial_Metrics']
            report.append(f"Total Return: {momentum['Total_Return']:.2%}")
            report.append(f"Annualized Return: {momentum['Annualized_Return']:.2%}")
            report.append(f"Sharpe Ratio: {momentum['Sharpe_Ratio']:.4f}")
            report.append(f"Maximum Drawdown: {momentum['Maximum_Drawdown']:.2%}")
            report.append(f"Calmar Ratio: {momentum['Calmar_Ratio']:.4f}")
            report.append(f"VaR (95%): {momentum['VaR_95']:.4f}")
            report.append(f"Win Rate: {momentum['Win_Rate']:.2%}")
            report.append("")
        
        if 'Mean_Reversion_Strategy' in results:
            report.append("🔄 MEAN REVERSION STRATEGY PERFORMANCE")
            report.append("-" * 40)
            mean_rev = results['Mean_Reversion_Strategy']['Financial_Metrics']
            report.append(f"Total Return: {mean_rev['Total_Return']:.2%}")
            report.append(f"Annualized Return: {mean_rev['Annualized_Return']:.2%}")
            report.append(f"Sharpe Ratio: {mean_rev['Sharpe_Ratio']:.4f}")
            report.append(f"Maximum Drawdown: {mean_rev['Maximum_Drawdown']:.2%}")
            report.append(f"Sortino Ratio: {mean_rev['Sortino_Ratio']:.4f}")
            report.append("")
        
        # Cross-validation Results
        if 'Cross_Validation' in results and 'cv_summary' in results['Cross_Validation']:
            report.append("🔄 CROSS-VALIDATION RESULTS")
            report.append("-" * 40)
            cv_summary = results['Cross_Validation']['cv_summary']
            report.append(f"CV MAE: {cv_summary.get('MAE_mean', 'N/A'):.4f} ± {cv_summary.get('MAE_std', 'N/A'):.4f}")
            report.append(f"CV RMSE: {cv_summary.get('RMSE_mean', 'N/A'):.4f} ± {cv_summary.get('RMSE_std', 'N/A'):.4f}")
            report.append(f"CV R²: {cv_summary.get('R2_mean', 'N/A'):.4f} ± {cv_summary.get('R2_std', 'N/A'):.4f}")
            report.append("")
        
        # Model Assessment
        report.append("🎯 MODEL ASSESSMENT")
        report.append("-" * 40)
        
        # Complexity assessment
        if 'Model_Complexity' in results:
            params = results['Model_Complexity']['Total_Parameters']
            if params < 50000:
                report.append("✅ Model Complexity: LOW - Efficient and fast")
            elif params < 200000:
                report.append("⚡ Model Complexity: MEDIUM - Good balance")
            else:
                report.append("🔥 Model Complexity: HIGH - May need optimization")
        
        # Performance assessment
        if 'Regression_Metrics' in results:
            r2 = results['Regression_Metrics']['R2']
            mape = results['Regression_Metrics']['MAPE']
            
            if r2 > 0.7 and mape < 5:
                report.append("🏆 Performance: EXCELLENT - Ready for production")
            elif r2 > 0.5 and mape < 10:
                report.append("✅ Performance: GOOD - Suitable for most applications")
            elif r2 > 0.3 and mape < 15:
                report.append("⚠️  Performance: MODERATE - Consider improvements")
            else:
                report.append("❌ Performance: POOR - Needs significant improvements")
        
        # Overfitting assessment
        if 'Cross_Validation' in results and 'cv_summary' in results['Cross_Validation']:
            cv_r2_std = results['Cross_Validation']['cv_summary'].get('R2_std', 0)
            if cv_r2_std < 0.1:
                report.append("✅ Overfitting: LOW - Model generalizes well")
            elif cv_r2_std < 0.2:
                report.append("⚠️  Overfitting: MODERATE - Monitor performance")
            else:
                report.append("❌ Overfitting: HIGH - Reduce model complexity")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"📄 Report saved to: {save_path}")
        
        return report_text


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Industrial-Grade Model Evaluation')
    
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
    parser.add_argument('--risk_free_rate', type=float, default=0.02,
                       help='Risk-free rate for financial metrics')
    parser.add_argument('--save_results', action='store_true',
                       help='Save evaluation results')
    parser.add_argument('--generate_report', action='store_true',
                       help='Generate comprehensive report')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(42)
    
    try:
        print("🚀 INDUSTRIAL-GRADE MODEL EVALUATION")
        print("=" * 60)
        print(f"📊 Model: {args.model_path}")
        print(f"📈 Symbol: {args.symbol}")
        print(f"📅 Period: {args.period}")
        print("=" * 60)
        
        # Initialize evaluator
        evaluator = IndustrialEvaluator(
            model_path=args.model_path,
            model_type=args.model_type
        )
        
        # Comprehensive evaluation
        results = evaluator.comprehensive_evaluation(
            symbol=args.symbol,
            period=args.period,
            sequence_length=args.sequence_length,
            risk_free_rate=args.risk_free_rate
        )
        
        # Generate and display report
        if args.generate_report:
            report_path = f"industrial_evaluation_{args.symbol}_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            report = evaluator.generate_report(results, report_path)
            print("\n" + report)
        
        # Save results
        if args.save_results:
            save_results(
                results,
                f"industrial_evaluation_{args.symbol}_{args.model_type}",
                "evaluation_results"
            )
        
        print("\n✅ Industrial-grade evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
