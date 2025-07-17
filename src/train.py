"""
Training module for stock price prediction models.

This module contains training loops, evaluation functions, and utilities
for training PyTorch models on stock price data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os
from tqdm import tqdm
import time
import json


class StockTrainer:
    """
    Trainer class for stock price prediction models.
    
    Args:
        model: PyTorch model to train
        device: Device to run training on
        model_name: Name of the model for saving
        save_dir: Directory to save model checkpoints
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 model_name: str = "stock_model",
                 save_dir: str = "checkpoints",
                 model_type: str = "lstm"):
        self.model = model
        self.device = device
        self.model_name = model_name
        self.save_dir = save_dir
        self.model_type = model_type
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Move model to device
        self.model.to(device)
        
    def train_epoch(self, 
                   train_loader: DataLoader,
                   optimizer: optim.Optimizer,
                   criterion: nn.Module) -> float:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        with tqdm(train_loader, desc="Training", leave=False) as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update parameters
                optimizer.step()
                
                # Update statistics
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.6f}',
                    'Batch': f'{batch_idx + 1}/{num_batches}'
                })
                
        return total_loss / num_batches
    
    def validate_epoch(self, 
                      val_loader: DataLoader,
                      criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model for one epoch.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Average validation loss and metrics
        """
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            with tqdm(val_loader, desc="Validation", leave=False) as pbar:
                for data, target in pbar:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Forward pass
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    # Update statistics
                    total_loss += loss.item()
                    predictions.extend(output.cpu().numpy())
                    targets.extend(target.cpu().numpy())
                    
                    # Update progress bar
                    pbar.set_postfix({'Val Loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        metrics = self.calculate_metrics(predictions, targets)
        
        return avg_loss, metrics
    
    def calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of metrics including regression, directional, and financial metrics
        """
        from advanced_metrics import AdvancedMetrics
        
        metrics = AdvancedMetrics()
        
        # Regression metrics
        regression_metrics = metrics.calculate_regression_metrics(predictions, targets)
        
        # Directional metrics
        directional_metrics = metrics.calculate_directional_metrics(predictions, targets)
        
        # Combine all metrics
        all_metrics = {**regression_metrics, **directional_metrics}
        
        return all_metrics
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 100,
              lr: float = 0.001,
              weight_decay: float = 1e-5,
              patience: int = 15,
              scheduler_step: int = 10,
              scheduler_gamma: float = 0.5) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: Weight decay for regularization
            patience: Early stopping patience
            scheduler_step: Step size for learning rate scheduler
            scheduler_gamma: Gamma for learning rate scheduler
            
        Returns:
            Training history
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Initialize optimizer and scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        criterion = nn.MSELoss()
        
        # Early stopping
        early_stopping_counter = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Training phase
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            
            # Validation phase
            val_loss, metrics = self.validate_epoch(val_loader, criterion)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Metrics: {metrics}")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, metrics, is_best=True)
                early_stopping_counter = 0
                print("New best model saved!")
            else:
                early_stopping_counter += 1
                
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_loss, metrics, is_best=False)
                
            # Early stopping
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
                
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
    
    def save_checkpoint(self, 
                       epoch: int,
                       val_loss: float,
                       metrics: Dict[str, float],
                       is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            metrics: Evaluation metrics
            is_best: Whether this is the best model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'model_type': self.model_type,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if is_best:
            checkpoint_path = os.path.join(self.save_dir, f"{self.model_name}_best.pth")
        else:
            checkpoint_path = os.path.join(self.save_dir, f"{self.model_name}_epoch_{epoch}.pth")
            
        torch.save(checkpoint, checkpoint_path)
        
        # Save metrics as JSON
        metrics_path = os.path.join(self.save_dir, f"{self.model_name}_metrics.json")
        # Convert numpy types to Python types for JSON serialization
        serializable_metrics = {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()}
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training history
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Validation loss: {checkpoint['val_loss']:.6f}")
        
        return checkpoint
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history.
        
        Args:
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Model Loss (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                epochs: int = 100,
                lr: float = 0.001,
                device: torch.device = None,
                model_name: str = "stock_model",
                save_dir: str = "checkpoints",
                **kwargs) -> Dict[str, List[float]]:
    """
    Convenience function to train a model.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to run training on
        model_name: Name of the model for saving
        save_dir: Directory to save model checkpoints
        **kwargs: Additional arguments for trainer
        
    Returns:
        Training history
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trainer = StockTrainer(model, device, model_name, save_dir)
    history = trainer.train(train_loader, val_loader, epochs, lr, **kwargs)
    
    return history


# Alias for backward compatibility
train = train_model