#!/usr/bin/env python3
"""
Model training script for hand gesture detection system.
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import json
from datetime import datetime
import wandb
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hand_gesture_detection.utils.config import ConfigManager
from hand_gesture_detection.utils.logger import Logger
from hand_gesture_detection.models.enhanced_hand_gesture_net import EnhancedHandGestureNet, AdvancedHandGestureNet
from hand_gesture_detection.data.enhanced_data_processor import EnhancedDataProcessor


class ModelTrainer:
    """Professional model trainer with comprehensive monitoring."""
    
    def __init__(self, config, logger, use_wandb=True):
        """Initialize the trainer."""
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_wandb = use_wandb
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_confidences = []
        self.learning_rates = []
        
        # Best model tracking
        self.best_val_accuracy = 0.0
        self.best_val_confidence = 0.0
        self.best_model_state = None
        self.patience_counter = 0
        self.best_epoch = 0
        
        # Initialize wandb if enabled
        if self.use_wandb and config.get('monitoring', {}).get('use_wandb', False):
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize Weights & Biases for experiment tracking."""
        wandb.init(
            project="hand-gesture-detection",
            name=f"mlp_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model": self.config.get('model', {}),
                "hyperparameters": self.config.get('hyperparameters', {}),
                "optimizer": self.config.get('optimizer', {}),
                "loss": self.config.get('loss', {}),
                "data": self.config.get('data', {}),
                "augmentation": self.config.get('augmentation', {})
            },
            tags=["mlp", "hand-gesture", "computer-vision"]
        )
        
        # Log model architecture
        self.logger.info("Initialized wandb for experiment tracking")
    
    def train(self, train_loader, val_loader, model, num_classes, class_weights=None):
        """Train the enhanced model."""
        # Setup optimizer and scheduler
        optimizer = self._setup_optimizer(model)
        scheduler = self._setup_scheduler(optimizer)
        criterion = self._setup_criterion(class_weights)
        
        # Move model to device
        model = model.to(self.device)
        
        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training loop
        for epoch in range(self.config.get('hyperparameters', {}).get('epochs', 200)):
            # Training phase
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
            
            # Validation phase
            val_loss, val_accuracy, val_confidence = self._validate_epoch(model, val_loader, criterion)
            
            # Update scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Log metrics
            current_lr = optimizer.param_groups[0]['lr']
            self.logger.log_training_epoch(epoch + 1, train_loss, val_loss, val_accuracy)
            
            # Update history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            self.val_confidences.append(val_confidence)
            self.learning_rates.append(current_lr)
            
            # Log to wandb
            if self.use_wandb and self.config.get('monitoring', {}).get('use_wandb', False):
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "val_confidence": val_confidence,
                    "learning_rate": current_lr,
                    "best_val_accuracy": self.best_val_accuracy,
                    "best_val_confidence": self.best_val_confidence
                })
            
            # Check for best model (prioritize confidence)
            combined_score = val_accuracy * 0.7 + val_confidence * 0.3
            best_combined_score = self.best_val_accuracy * 0.7 + self.best_val_confidence * 0.3
            
            if combined_score > best_combined_score:
                self.best_val_accuracy = val_accuracy
                self.best_val_confidence = val_confidence
                self.best_model_state = model.state_dict().copy()
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                
                # Log best model achievement to wandb
                if self.use_wandb and self.config.get('monitoring', {}).get('use_wandb', False):
                    wandb.log({
                        "best_val_accuracy": self.best_val_accuracy,
                        "best_epoch": self.best_epoch
                    })
                
                # Save best model
                self._save_checkpoint(model, epoch, val_accuracy, is_best=True)
                self.logger.info(f"🎯 NEW BEST MODEL! Validation accuracy: {val_accuracy:.4f} at epoch {epoch + 1}")
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= self.config.get('hyperparameters', {}).get('early_stopping_patience', 100):
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break
                
            # Regular checkpoint
            if (epoch + 1) % self.config.get('monitoring', {}).get('save_interval', 50) == 0:
                self._save_checkpoint(model, epoch, val_accuracy, is_best=False)
                
        # Load best model
        if self.best_model_state:
            model.load_state_dict(self.best_model_state)
            self.logger.info(f"Loaded best model with validation accuracy: {self.best_val_accuracy:.4f}")
            
        # Log final training summary to wandb
        if self.use_wandb and self.config.get('monitoring', {}).get('use_wandb', False):
            self._log_training_summary()
            
        return model
        
    def _train_epoch(self, model, train_loader, optimizer, criterion):
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
        
    def _validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch with confidence monitoring."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        total_confidence = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Calculate confidence
                probabilities = torch.softmax(output, dim=1)
                confidence, _ = torch.max(probabilities, dim=1)
                total_confidence += confidence.sum().item()
                
        accuracy = correct / total
        avg_loss = total_loss / len(val_loader)
        avg_confidence = total_confidence / total
        
        return avg_loss, accuracy, avg_confidence
    
    def _log_training_summary(self):
        """Log comprehensive training summary to wandb."""
        # Create training curves plot
        fig = self._create_training_curves_plot()
        
        # Log the plot to wandb
        wandb.log({"training_curves": wandb.Image(fig)})
        
        # Log final metrics
        wandb.log({
            "final_train_loss": self.train_losses[-1] if self.train_losses else 0,
            "final_val_loss": self.val_losses[-1] if self.val_losses else 0,
            "final_val_accuracy": self.val_accuracies[-1] if self.val_accuracies else 0,
            "total_epochs": len(self.train_losses),
            "best_epoch": self.best_epoch
        })
        
        # Log model performance summary
        wandb.summary.update({
            "best_val_accuracy": self.best_val_accuracy,
            "best_epoch": self.best_epoch,
            "total_epochs": len(self.train_losses),
            "final_train_loss": self.train_losses[-1] if self.train_losses else 0,
            "final_val_loss": self.val_losses[-1] if self.val_losses else 0
        })
        
        plt.close(fig)
        
    def _create_training_curves_plot(self):
        """Create training curves plot for wandb."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curve
        axes[0, 1].plot(self.val_accuracies, label='Validation Accuracy', color='green')
        axes[0, 1].axhline(y=self.best_val_accuracy, color='red', linestyle='--', 
                          label=f'Best: {self.best_val_accuracy:.4f}')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate curve
        axes[1, 0].plot(self.learning_rates, label='Learning Rate', color='orange')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Combined plot
        ax2 = axes[1, 1].twinx()
        axes[1, 1].plot(self.val_losses, label='Val Loss', color='blue')
        ax2.plot(self.val_accuracies, label='Val Accuracy', color='green')
        axes[1, 1].set_title('Validation Loss and Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss', color='blue')
        ax2.set_ylabel('Accuracy', color='green')
        axes[1, 1].legend(loc='upper left')
        ax2.legend(loc='upper right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    def _setup_optimizer(self, model):
        """Setup optimizer."""
        optimizer_config = self.config.get('optimizer', {})
        
        if optimizer_config.get('type', 'Adam') == 'Adam':
            return optim.Adam(
                model.parameters(),
                lr=float(self.config.get('hyperparameters', {}).get('learning_rate', 0.001)),
                weight_decay=float(self.config.get('hyperparameters', {}).get('weight_decay', 1e-4)),
                betas=list(optimizer_config.get('betas', [0.9, 0.999])),
                eps=float(optimizer_config.get('eps', 1e-8))
            )
        elif optimizer_config.get('type') == 'AdamW':
            return optim.AdamW(
                model.parameters(),
                lr=float(self.config.get('hyperparameters', {}).get('learning_rate', 0.001)),
                weight_decay=float(optimizer_config.get('weight_decay', self.config.get('hyperparameters', {}).get('weight_decay', 1e-4))),
                betas=list(optimizer_config.get('betas', [0.9, 0.999])),
                eps=float(optimizer_config.get('eps', 1e-8))
            )
        elif optimizer_config.get('type') == 'SGD':
            return optim.SGD(
                model.parameters(),
                lr=float(self.config.get('hyperparameters', {}).get('learning_rate', 0.001)),
                weight_decay=float(self.config.get('hyperparameters', {}).get('weight_decay', 1e-4)),
                momentum=float(optimizer_config.get('momentum', 0.9))
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config.get('type', 'Adam')}")
            
    def _setup_scheduler(self, optimizer):
        """Setup learning rate scheduler."""
        scheduler_config = self.config.get('hyperparameters', {}).get('lr_scheduler', {})
        
        if scheduler_config.get('type', 'StepLR') == 'StepLR':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(scheduler_config.get('step_size', 100)),
                gamma=float(scheduler_config.get('gamma', 0.1))
            )
        elif scheduler_config.get('type') == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=float(scheduler_config.get('factor', 0.5)),
                patience=int(scheduler_config.get('patience', 10))
            )
        elif scheduler_config.get('type') == 'CosineAnnealingWarmRestarts':
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=int(scheduler_config.get('T_0', 50)),
                T_mult=int(scheduler_config.get('T_mult', 2)),
                eta_min=float(scheduler_config.get('eta_min', 1e-6))
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_config.get('type', 'StepLR')}")
            
    def _setup_criterion(self, class_weights=None):
        """Setup loss criterion."""
        loss_config = self.config.get('loss', {})
        
        if loss_config.get('type', 'CrossEntropyLoss') == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=float(loss_config.get('label_smoothing', 0.0))
            )
        else:
            raise ValueError(f"Unknown loss: {loss_config.get('type', 'CrossEntropyLoss')}")
            
    def _save_checkpoint(self, model, epoch, val_accuracy, is_best=False):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.get('paths', {}).get('checkpoint_path', 'models/checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_accuracy': val_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'config': self.config
        }
        
        if is_best:
            checkpoint_path = checkpoint_dir / 'best_model.pth'
        else:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            
        torch.save(checkpoint, checkpoint_path)
        
    def plot_training_curves(self, save_path=None):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curve
        axes[0, 1].plot(self.val_accuracies, label='Validation Accuracy', color='green')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate curve
        axes[1, 0].plot(self.learning_rates, label='Learning Rate', color='red')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Combined plot
        ax2 = axes[1, 1].twinx()
        axes[1, 1].plot(self.val_losses, label='Val Loss', color='blue')
        ax2.plot(self.val_accuracies, label='Val Accuracy', color='green')
        axes[1, 1].set_title('Validation Loss and Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss', color='blue')
        ax2.set_ylabel('Accuracy', color='green')
        axes[1, 1].legend(loc='upper left')
        ax2.legend(loc='upper right')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training curves saved to: {save_path}")
        else:
            plt.show()
    
    def evaluate_model(self, model, test_loader, class_names):
        """Evaluate model and log results to wandb."""
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                probabilities = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1, keepdim=True)
                
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
        
        # Create confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Log to wandb
        if self.use_wandb and self.config.get('monitoring', {}).get('use_wandb', False):
            wandb.log({
                "test_accuracy": accuracy,
                "confusion_matrix": wandb.Image(plt.gcf())
            })
            
            # Log classification report
            report = classification_report(all_targets, all_predictions, 
                                         target_names=class_names, output_dict=True)
            wandb.log({"classification_report": report})
        
        plt.close()
        
        return accuracy, cm, all_predictions, all_targets


def main():
    """Main function for model training."""
    parser = argparse.ArgumentParser(description="Train hand gesture detection model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/training.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to processed data"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Name for the trained model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for training"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        help="Number of epochs to wait before early stopping"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    config_name = Path(args.config).stem
    config = config_manager.load_config(config_name)
    
    # Handle nested config structure
    if 'training' in config:
        config = config['training']
    
    # Override config with command line arguments
    if args.data_path:
        config.setdefault('paths', {})['data_path'] = args.data_path
    if args.epochs:
        config.setdefault('hyperparameters', {})['epochs'] = args.epochs
    if args.batch_size:
        config.setdefault('hyperparameters', {})['batch_size'] = args.batch_size
    if args.early_stopping_patience:
        config.setdefault('hyperparameters', {})['early_stopping_patience'] = args.early_stopping_patience
        
    # Initialize logger
    logger = Logger("model_training")
    logger.info("Starting model training")
    logger.log_config(config)
    
    try:
        # Load and process data with enhanced processor
        logger.info("Loading and processing data with enhanced processor...")
        # Create data processor config
        data_config = {
            'raw_data_path': 'data/raw',
            'processed_data_path': config.get('paths', {}).get('data_path', 'data/processed'),
            'train_split': config.get('data', {}).get('train_split', 0.7),
            'val_split': config.get('data', {}).get('val_split', 0.15),
            'test_split': config.get('data', {}).get('test_split', 0.15),
            'both_hands': config.get('data', {}).get('both_hands', False)
        }
        data_processor = EnhancedDataProcessor(data_config, logger)
        X_train, X_val, X_test, y_train, y_val, y_test, class_weights = data_processor.load_enhanced_processed_data()
        
        logger.info(f"Data loaded: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Create data loaders with class balancing
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                    torch.tensor(y_train, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                  torch.tensor(y_val, dtype=torch.long))
        
        # Use weighted sampling if class weights are available
        if class_weights is not None:
            from torch.utils.data import WeightedRandomSampler
            sample_weights = class_weights[y_train]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            train_loader = DataLoader(train_dataset, 
                                    batch_size=config.get('hyperparameters', {}).get('batch_size', 32),
                                    sampler=sampler)
        else:
            train_loader = DataLoader(train_dataset, 
                                    batch_size=config.get('hyperparameters', {}).get('batch_size', 32),
                                    shuffle=True)
            
        val_loader = DataLoader(val_dataset,
                              batch_size=config.get('hyperparameters', {}).get('batch_size', 32),
                              shuffle=False)
        
        # Create advanced model
        model_config = config.get('model', {})
        model_name = model_config.get('name', 'EnhancedHandGestureNet')
        
        if model_name == 'AdvancedHandGestureNet':
            model = AdvancedHandGestureNet(
                num_classes=model_config.get('num_classes', 10),
                input_size=X_train.shape[1],
                hidden_sizes=model_config.get('hidden_sizes', [256, 512, 1024, 512, 256, 128, 64]),
                dropout_rate=model_config.get('dropout_rate', 0.15),
                use_batch_norm=model_config.get('use_batch_norm', True),
                use_residual=model_config.get('use_residual', True),
                activation=model_config.get('activation', 'swish'),
                use_attention=True,
                attention_heads=4
            )
        else:
            model = EnhancedHandGestureNet(
                num_classes=model_config.get('num_classes', 10),
                input_size=X_train.shape[1],
                hidden_sizes=model_config.get('hidden_sizes', [128, 256, 128, 64]),
                dropout_rate=model_config.get('dropout_rate', 0.2),
                use_batch_norm=True,
                use_residual=True
            )
        
        logger.info(f"Model created: {model.summary()}")
        
        # Train model
        trainer = ModelTrainer(config, logger, use_wandb=True)
        trained_model = trainer.train(train_loader, val_loader, model, model_config.get('num_classes', 10), class_weights)
        
        # Save final model
        model_save_path = Path(config.get('paths', {}).get('model_save_path', 'models/trained'))
        model_save_path.mkdir(parents=True, exist_ok=True)
        
        model_name = args.model_name or f"hand_gesture_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        final_model_path = model_save_path / f"{model_name}.pth"
        
        trained_model.save_model(str(final_model_path), {
            'training_config': config,
            'best_val_accuracy': trainer.best_val_accuracy,
            'training_history': {
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses,
                'val_accuracies': trainer.val_accuracies
            }
        })
        
        logger.info(f"Model saved to: {final_model_path}")
        
        # Plot training curves
        curves_path = model_save_path / f"{model_name}_training_curves.png"
        trainer.plot_training_curves(str(curves_path))
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                   torch.tensor(y_test, dtype=torch.long))
        test_loader = DataLoader(test_dataset, batch_size=config.get('hyperparameters', {}).get('batch_size', 32))
        
        # Load class names
        class_names = [
            'closed_hand', 'fist', 'grip', 'index_left', 'index_upward',
            'palm_upward', 'thumb_down', 'thumb_up', 'two_finger_up', 'vertical_fingers'
        ]
        
        # Evaluate model with comprehensive metrics
        test_accuracy, confusion_matrix, predictions, targets = trainer.evaluate_model(
            trained_model, test_loader, class_names
        )
        
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        logger.info(f"Best validation accuracy: {trainer.best_val_accuracy:.4f} at epoch {trainer.best_epoch}")
        
        # Finish wandb run
        if config.get('monitoring', {}).get('use_wandb', False):
            wandb.finish()
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
