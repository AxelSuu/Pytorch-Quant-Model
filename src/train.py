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
    
    def train_epoch_with_accumulation(self, 
                                     train_loader: DataLoader,
                                     optimizer: optim.Optimizer,
                                     criterion: nn.Module,
                                     accumulation_steps: int = 1) -> float:
        """
        Train the model for one epoch with gradient accumulation.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            accumulation_steps: Number of steps to accumulate gradients
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        optimizer.zero_grad()
        
        # Disable colored output for Windows compatibility
        with tqdm(train_loader, desc="Training", leave=False) as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = criterion(output, target)
                
                # Scale loss by accumulation steps
                loss = loss / accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update parameters every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Update statistics
                total_loss += loss.item() * accumulation_steps
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
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            with tqdm(val_loader, desc="Validation", leave=False) as pbar:
                for data, target in pbar:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Forward pass
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    # Update statistics
                    total_loss += loss.item()
                    all_predictions.extend(output.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                    
                    # Update progress bar
                    pbar.set_postfix({'Val Loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)

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
        try:
            from advanced.advanced_metrics import AdvancedMetrics
            metrics = AdvancedMetrics()
            
            # Regression metrics
            regression_metrics = metrics.calculate_regression_metrics(predictions, targets)
            
            # Directional metrics
            directional_metrics = metrics.calculate_directional_metrics(predictions, targets)
            
            # Combine all metrics
            return {**regression_metrics, **directional_metrics}
        except ImportError:
            # Fall back to basic metrics if advanced_metrics not available
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            import numpy as np
            
            mse = mean_squared_error(targets, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(targets, predictions)
            r2 = r2_score(targets, predictions)
            
            # Calculate directional accuracy
            pred_direction = np.sign(np.diff(predictions, prepend=predictions[0]))
            true_direction = np.sign(np.diff(targets, prepend=targets[0]))
            direction_accuracy = np.mean(pred_direction == true_direction) * 100
            
            return {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'direction_accuracy': direction_accuracy
            }
        all_metrics = {**regression_metrics, **directional_metrics}
        
        return all_metrics
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 100,
              lr: float = 0.001,
              weight_decay: float = 1e-5,
              patience: int = 15,
              scheduler_type: str = "cosine",
              warmup_epochs: int = 5,
              gradient_accumulation_steps: int = 1) -> Dict[str, List[float]]:
        """
        Train the model with enhanced features.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: Weight decay for regularization
            patience: Early stopping patience
            scheduler_type: Type of scheduler ('cosine', 'step', 'plateau')
            warmup_epochs: Number of warmup epochs for learning rate
            gradient_accumulation_steps: Steps for gradient accumulation
            
        Returns:
            Training history
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Scheduler: {scheduler_type}, Warmup: {warmup_epochs} epochs")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        
        # Initialize optimizer with gradient clipping
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Initialize scheduler based on type
        if scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=lr * 0.01
            )
        elif scheduler_type == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        else:  # step
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        # Warmup scheduler
        if warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_epochs
            )
        
        criterion = nn.MSELoss()
        
        # Early stopping
        early_stopping_counter = 0
        best_metrics = {}
        
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Training phase with gradient accumulation
            train_loss = self.train_epoch_with_accumulation(
                train_loader, optimizer, criterion, gradient_accumulation_steps
            )
            self.train_losses.append(train_loss)
            
            # Validation phase
            val_loss, metrics = self.validate_epoch(val_loader, criterion)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            if epoch < warmup_epochs and warmup_epochs > 0:
                warmup_scheduler.step()
            elif scheduler_type == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Learning Rate: {current_lr:.2e}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_metrics = metrics.copy()
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