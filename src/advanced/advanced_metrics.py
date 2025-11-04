"""
Advanced Metrics and Evaluation Framework for Industrial-Grade Stock Prediction.

This module implements comprehensive evaluation metrics, backtesting strategies,
interpretability tools, and risk analysis for financial time series prediction.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


class AdvancedMetrics:
    """
    Comprehensive metrics suite for financial time series prediction.
    
    Implements industry-standard financial metrics including:
    - Regression metrics (MAE, RMSE, MAPE, R²)
    - Classification metrics (Accuracy, Precision, Recall, F1)
    - Financial metrics (Sharpe ratio, Maximum Drawdown, VaR)
    - Directional accuracy and trend prediction
    """
    
    @staticmethod
    def calculate_regression_metrics(predictions: np.ndarray, 
                                   targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            
        Returns:
            Dictionary of regression metrics
        """
        # Basic regression metrics
        mae = np.mean(np.abs(predictions - targets))
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((targets - predictions) / targets)) * 100
        
        # R-squared
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Mean Absolute Scaled Error (MASE)
        naive_forecast = targets[:-1]  # Lag-1 forecast
        mae_naive = np.mean(np.abs(targets[1:] - naive_forecast))
        mase = mae / mae_naive if mae_naive != 0 else float('inf')
        
        # Symmetric Mean Absolute Percentage Error
        smape = 100 * np.mean(2 * np.abs(predictions - targets) / 
                             (np.abs(predictions) + np.abs(targets)))
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'MASE': mase,
            'SMAPE': smape
        }
    
    @staticmethod
    def calculate_directional_metrics(predictions: np.ndarray,
                                    targets: np.ndarray,
                                    threshold: float = 0.0) -> Dict[str, float]:
        """
        Calculate directional accuracy and classification metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            threshold: Threshold for direction classification
            
        Returns:
            Dictionary of directional metrics
        """
        # Ensure predictions and targets are 1D arrays
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Remove any NaN or infinite values
        valid_mask = np.isfinite(predictions) & np.isfinite(targets)
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        # Need at least 2 points to calculate differences
        if len(predictions) < 2 or len(targets) < 2:
            return {
                'Directional_Accuracy': 0.0,
                'Classification_Accuracy': 0.0,
                'Precision': 0.0,
                'Recall': 0.0,
                'F1_Score': 0.0,
                'Up_Movement_Accuracy': 0.0,
                'Down_Movement_Accuracy': 0.0
            }
        
        # Calculate price changes
        pred_changes = np.diff(predictions)
        true_changes = np.diff(targets)
        
        # Convert to binary direction (up/down)
        pred_directions = (pred_changes > threshold).astype(int)
        true_directions = (true_changes > threshold).astype(int)
        
        # Classification metrics
        accuracy = accuracy_score(true_directions, pred_directions)
        precision = precision_score(true_directions, pred_directions, average='weighted', zero_division=0)
        recall = recall_score(true_directions, pred_directions, average='weighted', zero_division=0)
        f1 = f1_score(true_directions, pred_directions, average='weighted', zero_division=0)
        
        # Directional accuracy (percentage of correct direction predictions)
        directional_accuracy = np.mean(pred_directions == true_directions)
        
        # Up/Down accuracy separately
        up_mask = true_directions == 1
        down_mask = true_directions == 0
        
        up_accuracy = np.mean(pred_directions[up_mask] == true_directions[up_mask]) if np.any(up_mask) else 0.0
        down_accuracy = np.mean(pred_directions[down_mask] == true_directions[down_mask]) if np.any(down_mask) else 0.0
        
        return {
            'Directional_Accuracy': directional_accuracy,
            'Classification_Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'Up_Movement_Accuracy': up_accuracy,
            'Down_Movement_Accuracy': down_accuracy
        }
    
    @staticmethod
    def calculate_financial_metrics(returns: np.ndarray,
                                  risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate financial performance metrics.
        
        Args:
            returns: Portfolio returns
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            
        Returns:
            Dictionary of financial metrics
        """
        # Annualized return
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation != 0 else 0
        
        # Value at Risk (VaR) at 95% confidence
        var_95 = np.percentile(returns, 5)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else 0
        
        # Win rate
        win_rate = np.mean(returns > 0)
        
        return {
            'Total_Return': total_return,
            'Annualized_Return': annualized_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Maximum_Drawdown': max_drawdown,
            'Calmar_Ratio': calmar_ratio,
            'Sortino_Ratio': sortino_ratio,
            'VaR_95': var_95,
            'CVaR_95': cvar_95,
            'Win_Rate': win_rate
        }


class AdvancedBacktester:
    """
    Sophisticated backtesting framework with multiple trading strategies.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
    
    def simple_momentum_strategy(self,
                                predictions: np.ndarray,
                                actual_prices: np.ndarray,
                                lookback: int = 5) -> Dict[str, Union[float, List]]:
        """
        Simple momentum-based trading strategy.
        
        Args:
            predictions: Model predictions
            actual_prices: Actual price data
            lookback: Lookback period for momentum calculation
            
        Returns:
            Backtest results dictionary
        """
        portfolio_values = [self.initial_capital]
        positions = []  # Track positions: 0=cash, 1=long, -1=short
        trades = []
        cash = self.initial_capital
        shares = 0
        
        for i in range(lookback, len(predictions)):
            current_price = actual_prices[i]
            
            # Calculate momentum signals
            price_momentum = (actual_prices[i] - actual_prices[i-lookback]) / actual_prices[i-lookback]
            pred_momentum = (predictions[i] - actual_prices[i-1]) / actual_prices[i-1]
            
            # Generate trading signal
            if pred_momentum > 0.01 and price_momentum > 0:  # Strong buy signal
                if shares == 0:  # Buy
                    shares = cash / (current_price * (1 + self.transaction_cost + self.slippage))
                    cash = 0
                    trades.append(('BUY', current_price, shares, i))
                    positions.append(1)
                else:
                    positions.append(positions[-1])
            elif pred_momentum < -0.01 and price_momentum < 0:  # Strong sell signal
                if shares > 0:  # Sell
                    cash = shares * current_price * (1 - self.transaction_cost - self.slippage)
                    shares = 0
                    trades.append(('SELL', current_price, cash/current_price, i))
                    positions.append(0)
                else:
                    positions.append(positions[-1])
            else:
                positions.append(positions[-1] if positions else 0)
            
            # Calculate portfolio value
            portfolio_value = cash + shares * current_price
            portfolio_values.append(portfolio_value)
        
        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        return {
            'Portfolio_Values': portfolio_values,
            'Returns': returns,
            'Trades': trades,
            'Positions': positions,
            'Final_Value': portfolio_values[-1],
            'Total_Return': (portfolio_values[-1] - self.initial_capital) / self.initial_capital,
            'Number_of_Trades': len(trades)
        }
    
    def mean_reversion_strategy(self,
                               predictions: np.ndarray,
                               actual_prices: np.ndarray,
                               window: int = 20,
                               threshold: float = 2.0) -> Dict[str, Union[float, List]]:
        """
        Mean reversion trading strategy using Bollinger Bands.
        
        Args:
            predictions: Model predictions
            actual_prices: Actual price data
            window: Moving average window
            threshold: Standard deviation threshold
            
        Returns:
            Backtest results dictionary
        """
        portfolio_values = [self.initial_capital]
        trades = []
        cash = self.initial_capital
        shares = 0
        
        for i in range(window, len(predictions)):
            current_price = actual_prices[i]
            
            # Calculate Bollinger Bands
            moving_avg = np.mean(actual_prices[i-window:i])
            std_dev = np.std(actual_prices[i-window:i])
            upper_band = moving_avg + threshold * std_dev
            lower_band = moving_avg - threshold * std_dev
            
            # Generate signals based on mean reversion
            if current_price < lower_band and predictions[i] > current_price:  # Oversold, predict recovery
                if shares == 0:  # Buy
                    shares = cash / (current_price * (1 + self.transaction_cost + self.slippage))
                    cash = 0
                    trades.append(('BUY', current_price, shares, i))
            elif current_price > upper_band and predictions[i] < current_price:  # Overbought, predict decline
                if shares > 0:  # Sell
                    cash = shares * current_price * (1 - self.transaction_cost - self.slippage)
                    shares = 0
                    trades.append(('SELL', current_price, cash/current_price, i))
            
            # Calculate portfolio value
            portfolio_value = cash + shares * current_price
            portfolio_values.append(portfolio_value)
        
        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        return {
            'Portfolio_Values': portfolio_values,
            'Returns': returns,
            'Trades': trades,
            'Final_Value': portfolio_values[-1],
            'Total_Return': (portfolio_values[-1] - self.initial_capital) / self.initial_capital,
            'Number_of_Trades': len(trades)
        }


class ModelComplexityAnalyzer:
    """
    Analyze model complexity and efficiency metrics.
    """
    
    @staticmethod
    def analyze_model_complexity(model: nn.Module) -> Dict[str, Union[int, float]]:
        """
        Analyze model complexity and parameter efficiency.
        
        Args:
            model: PyTorch model to analyze
            
        Returns:
            Dictionary of complexity metrics
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Memory usage estimation (approximate)
        param_memory = total_params * 4  # 4 bytes per float32 parameter
        
        # Model depth (number of layers)
        num_layers = len(list(model.modules())) - 1  # Subtract 1 for the model itself
        
        # FLOPs estimation (simplified)
        def estimate_flops(model):
            flops = 0
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    flops += module.in_features * module.out_features
                elif isinstance(module, (nn.LSTM, nn.GRU)):
                    # Simplified FLOP estimation for RNNs
                    hidden_size = module.hidden_size
                    input_size = module.input_size
                    num_layers = module.num_layers
                    flops += (input_size + hidden_size + 1) * hidden_size * 4 * num_layers  # LSTM gates
            return flops
        
        estimated_flops = estimate_flops(model)
        
        return {
            'Total_Parameters': total_params,
            'Trainable_Parameters': trainable_params,
            'Parameter_Memory_MB': param_memory / (1024 * 1024),
            'Number_of_Layers': num_layers,
            'Estimated_FLOPs': estimated_flops,
            'Parameters_per_Layer': total_params / num_layers if num_layers > 0 else 0
        }
    
    @staticmethod
    def efficiency_score(model: nn.Module, 
                        performance_metric: float,
                        metric_type: str = 'r2') -> float:
        """
        Calculate model efficiency score (performance per parameter).
        
        Args:
            model: PyTorch model
            performance_metric: Performance metric value (R², accuracy, etc.)
            metric_type: Type of performance metric
            
        Returns:
            Efficiency score
        """
        total_params = sum(p.numel() for p in model.parameters())
        
        # Normalize performance metric
        if metric_type.lower() in ['r2', 'accuracy']:
            # Higher is better, normalize to [0, 1]
            normalized_performance = max(0, min(1, performance_metric))
        elif metric_type.lower() in ['mae', 'rmse', 'mse']:
            # Lower is better, invert and normalize
            normalized_performance = 1 / (1 + performance_metric)
        else:
            normalized_performance = performance_metric
        
        # Calculate efficiency: performance per million parameters
        efficiency = (normalized_performance * 1e6) / total_params
        
        return efficiency


class TimeSeriesCV:
    """
    Time series cross-validation for robust model evaluation.
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 test_size: Optional[int] = None,
                 gap: int = 0):
        """
        Initialize time series cross-validator.
        
        Args:
            n_splits: Number of splits
            test_size: Size of test set in each split
            gap: Gap between train and test sets
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
    
    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate time series cross-validation splits.
        
        Args:
            X: Input data array
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(X)
        
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        splits = []
        
        for i in range(self.n_splits):
            # Calculate split points
            test_start = n_samples - (self.n_splits - i) * test_size
            test_end = test_start + test_size
            train_end = test_start - self.gap
            
            if train_end <= 0:
                continue
                
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, min(test_end, n_samples))
            
            splits.append((train_indices, test_indices))
        
        return splits


def comprehensive_evaluation(model: nn.Module,
                           predictions: np.ndarray,
                           targets: np.ndarray,
                           prices: np.ndarray,
                           risk_free_rate: float = 0.02) -> Dict[str, Union[float, Dict]]:
    """
    Comprehensive model evaluation with all industrial-grade metrics.
    
    Args:
        model: PyTorch model to evaluate
        predictions: Model predictions
        targets: Ground truth values
        prices: Actual price data
        risk_free_rate: Risk-free rate for financial metrics
        
    Returns:
        Comprehensive evaluation results
    """
    metrics = AdvancedMetrics()
    backtester = AdvancedBacktester()
    complexity_analyzer = ModelComplexityAnalyzer()
    
    # Regression metrics
    regression_metrics = metrics.calculate_regression_metrics(predictions, targets)
    
    # Directional metrics
    directional_metrics = metrics.calculate_directional_metrics(predictions, targets)
    
    # Backtesting with momentum strategy
    momentum_backtest = backtester.simple_momentum_strategy(predictions, prices)
    momentum_financial_metrics = metrics.calculate_financial_metrics(
        momentum_backtest['Returns'], risk_free_rate
    )
    
    # Backtesting with mean reversion strategy
    mean_reversion_backtest = backtester.mean_reversion_strategy(predictions, prices)
    mean_reversion_financial_metrics = metrics.calculate_financial_metrics(
        mean_reversion_backtest['Returns'], risk_free_rate
    )
    
    # Model complexity analysis
    complexity_metrics = complexity_analyzer.analyze_model_complexity(model)
    efficiency = complexity_analyzer.efficiency_score(
        model, regression_metrics['R2'], 'r2'
    )
    
    return {
        'Regression_Metrics': regression_metrics,
        'Directional_Metrics': directional_metrics,
        'Momentum_Strategy': {
            'Backtest_Results': momentum_backtest,
            'Financial_Metrics': momentum_financial_metrics
        },
        'Mean_Reversion_Strategy': {
            'Backtest_Results': mean_reversion_backtest,
            'Financial_Metrics': mean_reversion_financial_metrics
        },
        'Model_Complexity': complexity_metrics,
        'Efficiency_Score': efficiency
    }
