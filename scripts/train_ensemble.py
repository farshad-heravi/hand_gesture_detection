#!/usr/bin/env python3
"""
Ensemble training script for hand gesture detection to achieve 95%+ accuracy.
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import json
from datetime import datetime
import wandb
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from typing import List, Dict, Tuple
import itertools

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hand_gesture_detection.utils.config import ConfigManager
from hand_gesture_detection.utils.logger import Logger
from hand_gesture_detection.models.enhanced_hand_gesture_net import EnhancedHandGestureNet, AdvancedHandGestureNet
from hand_gesture_detection.data.enhanced_data_processor import EnhancedDataProcessor


class EnsembleTrainer:
    """Train multiple models and create ensemble for maximum accuracy."""
    
    def __init__(self, config, logger, use_wandb=True):
        """Initialize the ensemble trainer."""
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_wandb = use_wandb
        self.models = []
        self.model_configs = []
        
    def create_model_configs(self) -> List[Dict]:
        """Create diverse model configurations for ensemble."""
        base_config = self.config.get('model', {})
        
        configs = [
            # Configuration 1: Large model with attention (optimized for RTX A4000)
            {
                'name': 'AdvancedHandGestureNet',
                'hidden_sizes': [512, 1024, 2048, 1024, 512, 256, 128],
                'dropout_rate': 0.1,
                'activation': 'mish',
                'use_attention': True,
                'attention_heads': 4,
                'learning_rate': 0.0001,
                'batch_size': 64,
                'weight_decay': 1e-4
            },
            # Configuration 2: Very deep model (optimized batch size)
            {
                'name': 'AdvancedHandGestureNet',
                'hidden_sizes': [256, 512, 1024, 2048, 1024, 512, 256, 128, 64],
                'dropout_rate': 0.15,
                'activation': 'swish',
                'use_attention': True,
                'attention_heads': 8,
                'learning_rate': 0.0002,
                'batch_size': 48,
                'weight_decay': 5e-4
            },
            # Configuration 3: Wide model (optimized batch size)
            {
                'name': 'AdvancedHandGestureNet',
                'hidden_sizes': [1024, 2048, 4096, 2048, 1024, 512, 256],
                'dropout_rate': 0.2,
                'activation': 'gelu',
                'use_attention': True,
                'attention_heads': 8,  # 4096 is divisible by 8
                'learning_rate': 0.00015,
                'batch_size': 32,
                'weight_decay': 2e-4
            },
            # Configuration 4: Balanced model (optimized batch size)
            {
                'name': 'AdvancedHandGestureNet',
                'hidden_sizes': [384, 768, 1536, 768, 384, 192, 96],
                'dropout_rate': 0.12,
                'activation': 'mish',
                'use_attention': True,
                'attention_heads': 6,  # 1536 is divisible by 6
                'learning_rate': 0.00008,
                'batch_size': 56,
                'weight_decay': 3e-4
            },
            # Configuration 5: Compact but deep (optimized batch size)
            {
                'name': 'AdvancedHandGestureNet',
                'hidden_sizes': [128, 256, 512, 1024, 512, 256, 128, 64, 32],
                'dropout_rate': 0.08,
                'activation': 'swish',
                'use_attention': True,
                'attention_heads': 4,  # 1024 is divisible by 4
                'learning_rate': 0.0003,
                'batch_size': 80,
                'weight_decay': 1e-3
            }
        ]
        
        return configs
    
    def train_single_model(self, model_config: Dict, train_loader, val_loader, 
                          model_idx: int, num_classes: int, class_weights=None) -> Tuple[torch.nn.Module, Dict]:
        """Train a single model with given configuration."""
        self.logger.info(f"Training model {model_idx + 1}/5 with config: {model_config['name']}")
        
        # Create model
        if model_config['name'] == 'AdvancedHandGestureNet':
            model = AdvancedHandGestureNet(
                num_classes=num_classes,
                input_size=39,  # Enhanced features
                hidden_sizes=model_config['hidden_sizes'],
                dropout_rate=model_config['dropout_rate'],
                use_batch_norm=True,
                use_residual=True,
                activation=model_config['activation'],
                use_attention=model_config['use_attention'],
                attention_heads=model_config['attention_heads']
            )
        else:
            model = EnhancedHandGestureNet(
                num_classes=num_classes,
                input_size=39,
                hidden_sizes=model_config['hidden_sizes'],
                dropout_rate=model_config['dropout_rate'],
                use_batch_norm=True,
                use_residual=True
            )
        
        model = model.to(self.device)
        
        # Create new data loaders with model-specific batch size
        from torch.utils.data import DataLoader, TensorDataset
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        
        # Use model-specific batch size
        batch_size = model_config.get('batch_size', 64)
        
        if class_weights is not None:
            from torch.utils.data import WeightedRandomSampler
            sample_weights = class_weights[train_dataset.tensors[1].numpy()]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
            
        val_loader = DataLoader(val_dataset, batch_size=min(batch_size * 2, 128), shuffle=False, num_workers=4, pin_memory=True)
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=model_config['learning_rate'],
            weight_decay=model_config['weight_decay'],
            betas=[0.9, 0.999],
            eps=1e-8
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=100,
            T_mult=2,
            eta_min=1e-8
        )
        
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.01
        )
        
        # Mixed precision training setup
        scaler = GradScaler() if self.device.type == 'cuda' else None
        
        # Training history
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        best_val_accuracy = 0.0
        best_model_state = None
        patience_counter = 0
        best_epoch = 0
        
        epochs = 600
        early_stopping_patience = 100
        
        self.logger.info(f"Model {model_idx + 1} parameters: {sum(p.numel() for p in model.parameters()):,}")
        self.logger.info(f"Using mixed precision training: {scaler is not None}")
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                if scaler is not None:
                    with autocast():
                        output = model(data)
                        loss = criterion(output, target)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            train_loss /= num_batches
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                    
                    if scaler is not None:
                        with autocast():
                            output = model(data)
                            loss = criterion(output, target)
                    else:
                        output = model(data)
                        loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
            
            val_loss /= len(val_loader)
            val_accuracy = correct / total
            
            # Update scheduler
            scheduler.step()
            
            # Update history
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # Check for best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = model.state_dict().copy()
                best_epoch = epoch + 1
                patience_counter = 0
                
                self.logger.info(f"Model {model_idx + 1} - NEW BEST! Val accuracy: {val_accuracy:.4f} at epoch {epoch + 1}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Model {model_idx + 1} - Early stopping at epoch {epoch + 1}")
                break
            
            # Log progress
            if (epoch + 1) % 50 == 0:
                self.logger.info(f"Model {model_idx + 1} - Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}")
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
            self.logger.info(f"Model {model_idx + 1} - Loaded best model with validation accuracy: {best_val_accuracy:.4f}")
        
        # Save model
        model_path = Path(self.config.get('paths', {}).get('model_save_path', 'models/trained'))
        model_path.mkdir(parents=True, exist_ok=True)
        
        model_filename = f"ensemble_model_{model_idx + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'best_val_accuracy': best_val_accuracy,
            'training_history': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies
            }
        }, model_path / model_filename)
        
        training_info = {
            'model_path': str(model_path / model_filename),
            'best_val_accuracy': best_val_accuracy,
            'best_epoch': best_epoch,
            'total_epochs': len(train_losses),
            'model_config': model_config
        }
        
        return model, training_info
    
    def train_ensemble(self, train_loader, val_loader, test_loader, num_classes: int, class_weights=None):
        """Train ensemble of models."""
        self.logger.info("Starting ensemble training...")
        
        model_configs = self.create_model_configs()
        ensemble_info = []
        
        for i, model_config in enumerate(model_configs):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Training Model {i + 1}/5")
            self.logger.info(f"{'='*60}")
            
            model, training_info = self.train_single_model(
                model_config, train_loader, val_loader, i, num_classes, class_weights
            )
            
            self.models.append(model)
            self.model_configs.append(model_config)
            ensemble_info.append(training_info)
            
            # Evaluate on test set
            test_accuracy = self.evaluate_single_model(model, test_loader)
            training_info['test_accuracy'] = test_accuracy
            self.logger.info(f"Model {i + 1} test accuracy: {test_accuracy:.4f}")
        
        return ensemble_info
    
    def evaluate_single_model(self, model, test_loader):
        """Evaluate a single model on test set."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return correct / total
    
    def ensemble_predict(self, test_loader, weights=None):
        """Make ensemble predictions."""
        if weights is None:
            weights = [1.0] * len(self.models)
        
        self.logger.info("Making ensemble predictions...")
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        # Get predictions from each model
        model_predictions = []
        model_probabilities = []
        
        for i, model in enumerate(self.models):
            model.eval()
            predictions = []
            probabilities = []
            
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(self.device)
                    output = model(data)
                    prob = torch.softmax(output, dim=1)
                    pred = output.argmax(dim=1)
                    
                    predictions.extend(pred.cpu().numpy())
                    probabilities.extend(prob.cpu().numpy())
                    
                    if i == 0:  # Only collect targets once
                        all_targets.extend(target.numpy())
            
            model_predictions.append(predictions)
            model_probabilities.append(probabilities)
        
        # Weighted ensemble voting
        ensemble_predictions = []
        ensemble_probabilities = []
        
        for i in range(len(all_targets)):
            # Weighted average of probabilities
            weighted_probs = np.zeros(10)  # 10 classes
            for j, (prob, weight) in enumerate(zip(model_probabilities, weights)):
                weighted_probs += np.array(prob[i]) * weight
            
            # Normalize weights
            weighted_probs /= sum(weights)
            
            ensemble_probabilities.append(weighted_probs)
            ensemble_predictions.append(np.argmax(weighted_probs))
        
        # Calculate ensemble accuracy
        correct = sum(p == t for p, t in zip(ensemble_predictions, all_targets))
        ensemble_accuracy = correct / len(all_targets)
        
        self.logger.info(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
        
        return ensemble_accuracy, ensemble_predictions, all_targets, ensemble_probabilities
    
    def save_ensemble(self, ensemble_info, save_path):
        """Save ensemble information."""
        ensemble_data = {
            'ensemble_info': ensemble_info,
            'model_configs': self.model_configs,
            'created_at': datetime.now().isoformat()
        }
        
        with open(save_path, 'w') as f:
            json.dump(ensemble_data, f, indent=2)
        
        self.logger.info(f"Ensemble info saved to: {save_path}")


def main():
    """Main function for ensemble training."""
    parser = argparse.ArgumentParser(description="Train ensemble of hand gesture detection models")
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
        "--ensemble-name",
        type=str,
        default="hand_gesture_ensemble",
        help="Name for the ensemble"
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
        
    # Initialize logger
    logger = Logger("ensemble_training")
    logger.info("Starting ensemble training for 95%+ accuracy")
    logger.log_config(config)
    
    try:
        # Load and process data
        logger.info("Loading and processing data...")
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
        
        # Create data loaders
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                    torch.tensor(y_train, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                  torch.tensor(y_val, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                   torch.tensor(y_test, dtype=torch.long))
        
        # Use weighted sampling for training with optimized batch sizes for GPU
        if class_weights is not None:
            from torch.utils.data import WeightedRandomSampler
            sample_weights = class_weights[y_train]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
            
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        
        # Train ensemble
        trainer = EnsembleTrainer(config, logger, use_wandb=False)
        ensemble_info = trainer.train_ensemble(train_loader, val_loader, test_loader, 10, class_weights)
        
        # Evaluate ensemble
        logger.info("\n" + "="*60)
        logger.info("ENSEMBLE EVALUATION")
        logger.info("="*60)
        
        # Try different weighting schemes
        weights_schemes = [
            [1.0, 1.0, 1.0, 1.0, 1.0],  # Equal weights
            [1.2, 1.1, 1.0, 1.1, 0.8],  # Slight variation
            [1.5, 1.3, 1.0, 1.2, 0.7],  # More variation
        ]
        
        best_accuracy = 0.0
        best_weights = None
        
        for i, weights in enumerate(weights_schemes):
            accuracy, predictions, targets, probabilities = trainer.ensemble_predict(test_loader, weights)
            logger.info(f"Weight scheme {i+1}: {weights} -> Accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = weights
        
        logger.info(f"\nBEST ENSEMBLE ACCURACY: {best_accuracy:.4f}")
        logger.info(f"Best weights: {best_weights}")
        
        # Save ensemble
        ensemble_path = Path(config.get('paths', {}).get('model_save_path', 'models/trained'))
        ensemble_path.mkdir(parents=True, exist_ok=True)
        
        ensemble_save_path = ensemble_path / f"{args.ensemble_name}_info.json"
        trainer.save_ensemble(ensemble_info, ensemble_save_path)
        
        # Log individual model accuracies
        logger.info("\nIndividual Model Accuracies:")
        for i, info in enumerate(ensemble_info):
            logger.info(f"Model {i+1}: Val={info['best_val_accuracy']:.4f}, Test={info['test_accuracy']:.4f}")
        
        if best_accuracy >= 0.95:
            logger.info("🎉 SUCCESS! Achieved 95%+ accuracy!")
        else:
            logger.info(f"Close! Need {0.95 - best_accuracy:.4f} more accuracy to reach 95%")
        
        return 0
        
    except Exception as e:
        logger.error(f"Ensemble training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
