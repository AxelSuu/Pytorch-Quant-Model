"""
Model Interpretability and Explainability Tools for Stock Prediction Models.

This module provides advanced interpretability methods including:
- Feature importance analysis
- Attention visualization for Transformer models
- SHAP-like attribution methods
- Gradient-based saliency analysis
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance using various methods.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def permutation_importance(self,
                             X: torch.Tensor,
                             y: torch.Tensor,
                             feature_names: List[str],
                             n_repeats: int = 10) -> Dict[str, float]:
        """
        Calculate permutation feature importance.
        
        Args:
            X: Input features
            y: Target values
            feature_names: Names of features
            n_repeats: Number of permutation repeats
            
        Returns:
            Dictionary of feature importance scores
        """
        # Get baseline performance
        with torch.no_grad():
            baseline_pred = self.model(X)
            baseline_loss = nn.MSELoss()(baseline_pred, y).item()
        
        importance_scores = {}
        
        for i, feature_name in enumerate(feature_names):
            scores = []
            
            for _ in range(n_repeats):
                # Create a copy and permute the feature
                X_permuted = X.clone()
                perm_indices = torch.randperm(X.size(0))
                X_permuted[:, :, i] = X_permuted[perm_indices, :, i]
                
                # Calculate performance with permuted feature
                with torch.no_grad():
                    permuted_pred = self.model(X_permuted)
                    permuted_loss = nn.MSELoss()(permuted_pred, y).item()
                
                # Importance is the increase in loss
                scores.append(permuted_loss - baseline_loss)
            
            importance_scores[feature_name] = np.mean(scores)
        
        return importance_scores
    
    def gradient_importance(self,
                          X: torch.Tensor,
                          feature_names: List[str]) -> Dict[str, float]:
        """
        Calculate feature importance using gradient magnitudes.
        
        Args:
            X: Input features
            feature_names: Names of features
            
        Returns:
            Dictionary of gradient-based importance scores
        """
        X.requires_grad_(True)
        
        # Forward pass
        output = self.model(X)
        
        # Calculate gradients
        grad = torch.autograd.grad(
            outputs=output.sum(),
            inputs=X,
            create_graph=False,
            retain_graph=False
        )[0]
        
        # Calculate importance as mean absolute gradient
        importance_scores = {}
        for i, feature_name in enumerate(feature_names):
            importance_scores[feature_name] = grad[:, :, i].abs().mean().item()
        
        return importance_scores
    
    def integrated_gradients(self,
                           X: torch.Tensor,
                           feature_names: List[str],
                           baseline: Optional[torch.Tensor] = None,
                           steps: int = 50) -> Dict[str, float]:
        """
        Calculate feature importance using Integrated Gradients.
        
        Args:
            X: Input features
            feature_names: Names of features
            baseline: Baseline input (defaults to zeros)
            steps: Number of integration steps
            
        Returns:
            Dictionary of integrated gradient importance scores
        """
        if baseline is None:
            baseline = torch.zeros_like(X)
        
        # Generate path from baseline to input
        alphas = torch.linspace(0, 1, steps).to(self.device)
        
        gradients = []
        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated = baseline + alpha * (X - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            output = self.model(interpolated)
            
            # Calculate gradients
            grad = torch.autograd.grad(
                outputs=output.sum(),
                inputs=interpolated,
                create_graph=False,
                retain_graph=False
            )[0]
            
            gradients.append(grad)
        
        # Average gradients and multiply by input difference
        avg_gradients = torch.stack(gradients).mean(dim=0)
        integrated_gradients = (X - baseline) * avg_gradients
        
        # Calculate importance scores
        importance_scores = {}
        for i, feature_name in enumerate(feature_names):
            importance_scores[feature_name] = integrated_gradients[:, :, i].abs().mean().item()
        
        return importance_scores


class AttentionVisualizer:
    """
    Visualize attention patterns in Transformer models.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        if hasattr(model, 'transformer_encoder'):
            self.is_transformer = True
        else:
            self.is_transformer = False
            print("Warning: Model does not appear to be a Transformer. Attention visualization may not work.")
    
    def extract_attention_weights(self, X: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights from the model.
        
        Args:
            X: Input tensor
            
        Returns:
            Attention weights tensor
        """
        if not self.is_transformer:
            return None
        
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'forward') and 'return_attention' in self.model.forward.__code__.co_varnames:
                _, attention_weights = self.model(X, return_attention=True)
                return attention_weights
            else:
                # Fallback: try to extract from model attributes
                _ = self.model(X)
                if hasattr(self.model, 'attention_weights'):
                    return self.model.attention_weights
        
        return None
    
    def plot_attention_heatmap(self,
                             attention_weights: torch.Tensor,
                             feature_names: List[str],
                             time_steps: Optional[List[int]] = None,
                             save_path: Optional[str] = None):
        """
        Plot attention weights as a heatmap.
        
        Args:
            attention_weights: Attention weights tensor
            feature_names: Names of input features
            time_steps: Time step labels
            save_path: Path to save the plot
        """
        if attention_weights is None:
            print("No attention weights available for visualization.")
            return
        
        # Average attention across batch and heads if necessary
        if attention_weights.dim() == 4:  # [batch, heads, seq, seq]
            attention_weights = attention_weights.mean(dim=[0, 1])
        elif attention_weights.dim() == 3:  # [batch, seq, seq]
            attention_weights = attention_weights.mean(dim=0)
        
        attention_np = attention_weights.cpu().numpy()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            attention_np,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            xticklabels=time_steps or range(attention_np.shape[1]),
            yticklabels=time_steps or range(attention_np.shape[0])
        )
        plt.title('Attention Weights Heatmap')
        plt.xlabel('Key Positions')
        plt.ylabel('Query Positions')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class SaliencyAnalyzer:
    """
    Analyze input saliency for model predictions.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
    
    def vanilla_gradients(self, X: torch.Tensor, target_class: int = 0) -> torch.Tensor:
        """
        Calculate vanilla gradients for saliency analysis.
        
        Args:
            X: Input tensor
            target_class: Target output dimension
            
        Returns:
            Gradient tensor
        """
        X.requires_grad_(True)
        
        output = self.model(X)
        
        # Create target for gradient calculation
        target = torch.zeros_like(output)
        target[:, target_class] = 1
        
        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=output,
            inputs=X,
            grad_outputs=target,
            create_graph=False,
            retain_graph=False
        )[0]
        
        return gradients
    
    def smooth_gradients(self,
                        X: torch.Tensor,
                        target_class: int = 0,
                        noise_level: float = 0.1,
                        n_samples: int = 50) -> torch.Tensor:
        """
        Calculate SmoothGrad for reduced noise in saliency maps.
        
        Args:
            X: Input tensor
            target_class: Target output dimension
            noise_level: Standard deviation of noise to add
            n_samples: Number of noisy samples to average
            
        Returns:
            Smoothed gradient tensor
        """
        gradients = []
        
        for _ in range(n_samples):
            # Add noise to input
            noise = torch.normal(0, noise_level, size=X.shape).to(self.device)
            noisy_X = X + noise
            
            # Calculate gradients
            grad = self.vanilla_gradients(noisy_X, target_class)
            gradients.append(grad)
        
        # Average gradients
        smooth_grad = torch.stack(gradients).mean(dim=0)
        
        return smooth_grad
    
    def plot_saliency_map(self,
                         saliency: torch.Tensor,
                         feature_names: List[str],
                         time_steps: Optional[List[int]] = None,
                         save_path: Optional[str] = None):
        """
        Plot saliency map.
        
        Args:
            saliency: Saliency tensor
            feature_names: Names of input features
            time_steps: Time step labels
            save_path: Path to save the plot
        """
        # Average across batch dimension
        if saliency.dim() == 3:
            saliency = saliency.mean(dim=0)
        
        saliency_np = saliency.abs().cpu().numpy()
        
        plt.figure(figsize=(15, 8))
        sns.heatmap(
            saliency_np.T,
            annot=True,
            fmt='.3f',
            cmap='Reds',
            xticklabels=time_steps or range(saliency_np.shape[0]),
            yticklabels=feature_names
        )
        plt.title('Input Saliency Map')
        plt.xlabel('Time Steps')
        plt.ylabel('Features')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ModelExplainer:
    """
    Comprehensive model explanation combining multiple interpretability methods.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.feature_analyzer = FeatureImportanceAnalyzer(model, device)
        self.attention_viz = AttentionVisualizer(model)
        self.saliency_analyzer = SaliencyAnalyzer(model, device)
    
    def explain_prediction(self,
                          X: torch.Tensor,
                          feature_names: List[str],
                          sample_idx: int = 0) -> Dict[str, Union[Dict, torch.Tensor]]:
        """
        Provide comprehensive explanation for a single prediction.
        
        Args:
            X: Input tensor
            feature_names: Names of input features
            sample_idx: Index of sample to explain
            
        Returns:
            Dictionary containing various explanation methods
        """
        # Select single sample
        x_sample = X[sample_idx:sample_idx+1]
        
        explanations = {}
        
        # Feature importance
        try:
            explanations['permutation_importance'] = self.feature_analyzer.permutation_importance(
                x_sample, torch.zeros(1, 1).to(self.device), feature_names
            )
        except Exception as e:
            print(f"Permutation importance failed: {e}")
        
        try:
            explanations['gradient_importance'] = self.feature_analyzer.gradient_importance(
                x_sample, feature_names
            )
        except Exception as e:
            print(f"Gradient importance failed: {e}")
        
        # Attention weights (for Transformer models)
        try:
            explanations['attention_weights'] = self.attention_viz.extract_attention_weights(x_sample)
        except Exception as e:
            print(f"Attention extraction failed: {e}")
        
        # Saliency analysis
        try:
            explanations['vanilla_gradients'] = self.saliency_analyzer.vanilla_gradients(x_sample)
        except Exception as e:
            print(f"Saliency analysis failed: {e}")
        
        try:
            explanations['smooth_gradients'] = self.saliency_analyzer.smooth_gradients(x_sample)
        except Exception as e:
            print(f"Smooth gradients failed: {e}")
        
        return explanations
    
    def plot_comprehensive_explanation(self,
                                     explanations: Dict,
                                     feature_names: List[str],
                                     save_path: Optional[str] = None):
        """
        Create comprehensive visualization of model explanations.
        
        Args:
            explanations: Explanation results from explain_prediction
            feature_names: Names of input features
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Comprehensive Model Explanation', fontsize=16)
        
        # Plot 1: Feature Importance Comparison
        if 'permutation_importance' in explanations and 'gradient_importance' in explanations:
            perm_scores = list(explanations['permutation_importance'].values())
            grad_scores = list(explanations['gradient_importance'].values())
            
            x_pos = np.arange(len(feature_names))
            width = 0.35
            
            axes[0, 0].bar(x_pos - width/2, perm_scores, width, label='Permutation', alpha=0.8)
            axes[0, 0].bar(x_pos + width/2, grad_scores, width, label='Gradient', alpha=0.8)
            axes[0, 0].set_xlabel('Features')
            axes[0, 0].set_ylabel('Importance Score')
            axes[0, 0].set_title('Feature Importance Comparison')
            axes[0, 0].set_xticks(x_pos)
            axes[0, 0].set_xticklabels(feature_names, rotation=45)
            axes[0, 0].legend()
        
        # Plot 2: Attention Heatmap
        if 'attention_weights' in explanations and explanations['attention_weights'] is not None:
            attention = explanations['attention_weights'][0].cpu().numpy()
            if attention.ndim > 2:
                attention = attention.mean(axis=0)
            
            im = axes[0, 1].imshow(attention, cmap='Blues', aspect='auto')
            axes[0, 1].set_title('Attention Weights')
            axes[0, 1].set_xlabel('Key Positions')
            axes[0, 1].set_ylabel('Query Positions')
            plt.colorbar(im, ax=axes[0, 1])
        else:
            axes[0, 1].text(0.5, 0.5, 'No Attention Weights\n(Not a Transformer model)', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Attention Weights')
        
        # Plot 3: Vanilla Gradients Saliency
        if 'vanilla_gradients' in explanations:
            saliency = explanations['vanilla_gradients'][0].abs().cpu().numpy()
            im = axes[1, 0].imshow(saliency.T, cmap='Reds', aspect='auto')
            axes[1, 0].set_title('Vanilla Gradients Saliency')
            axes[1, 0].set_xlabel('Time Steps')
            axes[1, 0].set_ylabel('Features')
            axes[1, 0].set_yticks(range(len(feature_names)))
            axes[1, 0].set_yticklabels(feature_names)
            plt.colorbar(im, ax=axes[1, 0])
        
        # Plot 4: Smooth Gradients Saliency
        if 'smooth_gradients' in explanations:
            smooth_saliency = explanations['smooth_gradients'][0].abs().cpu().numpy()
            im = axes[1, 1].imshow(smooth_saliency.T, cmap='Reds', aspect='auto')
            axes[1, 1].set_title('Smooth Gradients Saliency')
            axes[1, 1].set_xlabel('Time Steps')
            axes[1, 1].set_ylabel('Features')
            axes[1, 1].set_yticks(range(len(feature_names)))
            axes[1, 1].set_yticklabels(feature_names)
            plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def analyze_feature_interactions(model: nn.Module,
                               X: torch.Tensor,
                               feature_names: List[str],
                               device: torch.device) -> pd.DataFrame:
    """
    Analyze pairwise feature interactions using gradients.
    
    Args:
        model: PyTorch model
        X: Input tensor
        feature_names: Names of features
        device: Device to run analysis on
        
    Returns:
        DataFrame containing interaction scores
    """
    model.eval()
    n_features = len(feature_names)
    interaction_matrix = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(i, n_features):
            # Calculate second-order gradients
            X_copy = X.clone().requires_grad_(True)
            
            output = model(X_copy)
            
            # First-order gradient w.r.t. feature i
            grad_i = torch.autograd.grad(
                outputs=output.sum(),
                inputs=X_copy,
                create_graph=True
            )[0][:, :, i]
            
            # Second-order gradient w.r.t. feature j
            grad_ij = torch.autograd.grad(
                outputs=grad_i.sum(),
                inputs=X_copy,
                create_graph=False
            )[0][:, :, j]
            
            interaction_score = grad_ij.abs().mean().item()
            interaction_matrix[i, j] = interaction_score
            interaction_matrix[j, i] = interaction_score
    
    return pd.DataFrame(interaction_matrix, 
                       index=feature_names, 
                       columns=feature_names)
