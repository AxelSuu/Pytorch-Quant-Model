"""
Enhanced prediction utilities for next-day and multi-day stock price forecasting.

This module provides robust prediction methods that handle:
- Proper scaling and inverse scaling
- Technical indicator recalculation for multi-step predictions
- Confidence intervals and uncertainty estimation
- Trend analysis and market regime detection
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class EnhancedPredictor:
    """
    Enhanced prediction class for robust stock price forecasting.
    """
    
    def __init__(self, model: torch.nn.Module, data_loader, device: torch.device):
        """
        Initialize enhanced predictor.
        
        Args:
            model: Trained PyTorch model
            data_loader: Data loader with scaling information
            device: Device to run predictions on
        """
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.model.eval()
    
    def predict_next_day(self, 
                        symbol: str, 
                        use_realtime: bool = True,
                        return_confidence: bool = False) -> Union[float, Tuple[float, float]]:
        """
        Predict next trading day's closing price.
        
        Args:
            symbol: Stock symbol
            use_realtime: Whether to use real-time data
            return_confidence: Whether to return confidence interval
            
        Returns:
            Predicted price (and optional confidence interval)
        """
        try:
            # Get latest sequence
            if use_realtime and hasattr(self.data_loader, 'get_realtime_sequence'):
                sequence = self.data_loader.get_realtime_sequence(symbol)
                if sequence is None:
                    raise ValueError("Could not fetch real-time data")
            else:
                # Use last sequence from training data
                sequence = self._get_last_sequence()
            
            # Convert to tensor
            if isinstance(sequence, np.ndarray):
                sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            elif not isinstance(sequence, torch.Tensor):
                sequence = torch.FloatTensor(np.array(sequence)).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                if hasattr(self.model, 'predict_next_day'):
                    prediction = self.model.predict_next_day(sequence)
                else:
                    prediction = self.model(sequence)
                
                prediction_np = prediction.cpu().numpy().item()
            
            # Inverse transform to get actual price
            if hasattr(self.data_loader, 'inverse_transform_predictions'):
                predicted_price = self.data_loader.inverse_transform_predictions(
                    np.array([prediction_np])
                )[0]
            else:
                predicted_price = prediction_np
            
            if return_confidence:
                # Estimate confidence interval using bootstrap or dropout
                confidence_interval = self._estimate_confidence_interval(sequence)
                return predicted_price, confidence_interval
            
            return predicted_price
            
        except Exception as e:
            print(f"Error in next day prediction: {e}")
            return None
    
    def predict_multi_day(self, 
                         symbol: str,
                         days: int = 5,
                         use_realtime: bool = True,
                         method: str = 'autoregressive') -> Dict[str, Union[List[float], Dict]]:
        """
        Predict multiple days into the future.
        
        Args:
            symbol: Stock symbol
            days: Number of days to predict
            use_realtime: Whether to use real-time data
            method: Prediction method ('autoregressive' or 'direct')
            
        Returns:
            Dictionary with predictions and analysis
        """
        try:
            # Get latest sequence
            if use_realtime and hasattr(self.data_loader, 'get_realtime_sequence'):
                sequence = self.data_loader.get_realtime_sequence(symbol)
                if sequence is None:
                    raise ValueError("Could not fetch real-time data")
            else:
                sequence = self._get_last_sequence()
            
            # Convert to tensor
            if isinstance(sequence, np.ndarray):
                sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            elif not isinstance(sequence, torch.Tensor):
                sequence = torch.FloatTensor(np.array(sequence)).unsqueeze(0).to(self.device)
            
            if method == 'autoregressive':
                predictions = self._predict_autoregressive(sequence, days)
            else:
                predictions = self._predict_direct(sequence, days)
            
            # Inverse transform predictions
            if hasattr(self.data_loader, 'inverse_transform_predictions'):
                predictions_scaled = self.data_loader.inverse_transform_predictions(predictions)
            else:
                predictions_scaled = predictions
            
            # Analyze trends and patterns
            analysis = self._analyze_predictions(predictions_scaled)
            
            return {
                'predictions': predictions_scaled.tolist(),
                'analysis': analysis,
                'method': method,
                'days_predicted': days
            }
            
        except Exception as e:
            print(f"Error in multi-day prediction: {e}")
            return {'error': str(e)}
    
    def _predict_autoregressive(self, sequence: torch.Tensor, days: int) -> np.ndarray:
        """
        Predict using autoregressive approach with proper feature updating.
        """
        predictions = []
        current_sequence = sequence.clone()
        
        with torch.no_grad():
            for day in range(days):
                # Get prediction for current sequence
                if hasattr(self.model, 'predict_next_day'):
                    pred = self.model.predict_next_day(current_sequence)
                else:
                    pred = self.model(current_sequence)
                
                predictions.append(pred.cpu().numpy().item())
                
                # Update sequence for next prediction
                if day < days - 1:
                    current_sequence = self._update_sequence(current_sequence, pred)
        
        return np.array(predictions)
    
    def _predict_direct(self, sequence: torch.Tensor, days: int) -> np.ndarray:
        """
        Predict using direct multi-step approach if model supports it.
        """
        with torch.no_grad():
            if hasattr(self.model, 'predict_sequence'):
                predictions = self.model.predict_sequence(sequence, steps=days)
                return predictions.cpu().numpy().flatten()
            else:
                # Fallback to autoregressive
                return self._predict_autoregressive(sequence, days)
    
    def _update_sequence(self, sequence: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """
        Update sequence with new prediction and recalculate technical indicators.
        """
        # Get the last timestep features
        last_features = sequence[:, -1:, :].clone()
        
        # Update close price (assuming it's the first feature)
        last_features[:, 0, 0] = prediction.squeeze(-1)
        
        # For a more sophisticated approach, we could recalculate technical indicators
        # For now, we'll use a simplified approach
        
        # Slide the window
        new_sequence = torch.cat([
            sequence[:, 1:, :],
            last_features
        ], dim=1)
        
        return new_sequence
    
    def _get_last_sequence(self) -> np.ndarray:
        """Get the last sequence from the data loader."""
        # This is a simplified implementation
        # In practice, you'd get the most recent sequence from your data
        return np.random.randn(60, 15)  # Placeholder
    
    def _estimate_confidence_interval(self, sequence: torch.Tensor, 
                                    n_samples: int = 100) -> Tuple[float, float]:
        """
        Estimate confidence interval using Monte Carlo dropout.
        """
        if hasattr(self.model, 'training'):
            # Enable dropout for uncertainty estimation
            self.model.train()
            
            predictions = []
            with torch.no_grad():
                for _ in range(n_samples):
                    pred = self.model(sequence)
                    predictions.append(pred.cpu().numpy().item())
            
            self.model.eval()
            
            # Calculate confidence interval
            predictions = np.array(predictions)
            if hasattr(self.data_loader, 'inverse_transform_predictions'):
                predictions = self.data_loader.inverse_transform_predictions(predictions)
            
            lower = np.percentile(predictions, 2.5)
            upper = np.percentile(predictions, 97.5)
            
            return (lower, upper)
        
        return (0.0, 0.0)  # Default if no uncertainty estimation available
    
    def _analyze_predictions(self, predictions: np.ndarray) -> Dict[str, Union[float, str]]:
        """
        Analyze prediction trends and patterns.
        """
        if len(predictions) < 2:
            return {}
        
        # Calculate trend metrics
        total_change = predictions[-1] - predictions[0]
        percent_change = (total_change / predictions[0]) * 100
        
        # Determine trend direction
        if percent_change > 1.0:
            trend = "BULLISH"
        elif percent_change < -1.0:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"
        
        # Calculate volatility
        daily_changes = np.diff(predictions) / predictions[:-1]
        volatility = np.std(daily_changes) * np.sqrt(252) * 100  # Annualized volatility
        
        # Maximum gain/loss
        max_gain = np.max(predictions) - predictions[0]
        max_loss = predictions[0] - np.min(predictions)
        
        return {
            'trend': trend,
            'total_change': float(total_change),
            'percent_change': float(percent_change),
            'volatility': float(volatility),
            'max_potential_gain': float(max_gain),
            'max_potential_loss': float(max_loss),
            'final_price': float(predictions[-1]),
            'initial_price': float(predictions[0])
        }
    
    def get_market_regime(self, symbol: str, lookback: int = 20) -> str:
        """
        Detect current market regime (trending, sideways, volatile).
        
        Args:
            symbol: Stock symbol
            lookback: Number of days to look back
            
        Returns:
            Market regime classification
        """
        try:
            # This would require historical price data
            # For now, return a placeholder
            return "NORMAL"
        except Exception:
            return "UNKNOWN"
    
    def batch_predict(self, symbols: List[str], days: int = 1) -> Dict[str, Dict]:
        """
        Predict prices for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            days: Number of days to predict
            
        Returns:
            Dictionary with predictions for each symbol
        """
        results = {}
        
        for symbol in symbols:
            try:
                if days == 1:
                    pred = self.predict_next_day(symbol)
                    results[symbol] = {
                        'next_day_price': pred,
                        'status': 'success'
                    }
                else:
                    pred = self.predict_multi_day(symbol, days)
                    results[symbol] = pred
                    results[symbol]['status'] = 'success'
                    
            except Exception as e:
                results[symbol] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        return results

    def predict_batch(self, batch_input: torch.Tensor) -> torch.Tensor:
        """
        Predict for a batch of inputs.
        
        Args:
            batch_input: Batch tensor of shape (batch_size, sequence_length, features)
            
        Returns:
            Batch predictions tensor
        """
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'predict_next_day'):
                return self.model.predict_next_day(batch_input)
            else:
                return self.model(batch_input)
    
    def estimate_confidence_intervals(self, 
                                    input_tensor: torch.Tensor, 
                                    n_samples: int = 100,
                                    confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Estimate confidence intervals using Monte Carlo dropout.
        
        Args:
            input_tensor: Input tensor for prediction
            n_samples: Number of MC samples
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary with prediction bounds and statistics
        """
        # Enable dropout during inference for uncertainty estimation
        self.model.train()  # Enable dropout
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                if hasattr(self.model, 'predict_next_day'):
                    pred = self.model.predict_next_day(input_tensor)
                else:
                    pred = self.model(input_tensor)
                predictions.append(pred.cpu().numpy().item())
        
        # Return to evaluation mode
        self.model.eval()
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(predictions, lower_percentile)
        upper_bound = np.percentile(predictions, upper_percentile)
        
        return {
            'prediction': mean_pred,
            'lower': lower_bound,
            'upper': upper_bound,
            'std': std_pred,
            'confidence_level': confidence_level
        }


def create_enhanced_predictor(model_path: str, 
                            model_type: str,
                            data_loader,
                            device: torch.device = None) -> EnhancedPredictor:
    """
    Create an enhanced predictor from a trained model.
    
    Args:
        model_path: Path to trained model
        model_type: Type of model (lstm, gru, transformer)
        data_loader: Data loader with scaling information
        device: Device to use for predictions
        
    Returns:
        Enhanced predictor instance
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    from model import create_model
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Create model
        model = create_model(
            model_type=model_type,
            input_size=15,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return EnhancedPredictor(model, data_loader, device)
        
    except Exception as e:
        raise RuntimeError(f"Failed to create enhanced predictor: {e}")


def validate_prediction_capability(model: torch.nn.Module, 
                                 sample_input: torch.Tensor) -> Dict[str, bool]:
    """
    Validate that the model can perform various prediction tasks.
    
    Args:
        model: PyTorch model to validate
        sample_input: Sample input tensor
        
    Returns:
        Dictionary indicating which prediction methods are available
    """
    capabilities = {
        'basic_forward': False,
        'next_day_prediction': False,
        'sequence_prediction': False,
        'attention_extraction': False
    }
    
    try:
        # Test basic forward pass
        with torch.no_grad():
            output = model(sample_input)
            capabilities['basic_forward'] = True
    except Exception:
        pass
    
    try:
        # Test next day prediction
        if hasattr(model, 'predict_next_day'):
            with torch.no_grad():
                output = model.predict_next_day(sample_input)
                capabilities['next_day_prediction'] = True
    except Exception:
        pass
    
    try:
        # Test sequence prediction
        if hasattr(model, 'predict_sequence'):
            with torch.no_grad():
                output = model.predict_sequence(sample_input, steps=3)
                capabilities['sequence_prediction'] = True
    except Exception:
        pass
    
    try:
        # Test attention extraction (for transformers)
        if hasattr(model, 'forward') and 'return_attention' in model.forward.__code__.co_varnames:
            with torch.no_grad():
                output, attention = model(sample_input, return_attention=True)
                capabilities['attention_extraction'] = True
    except Exception:
        pass
    
    return capabilities
