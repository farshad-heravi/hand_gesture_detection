#!/usr/bin/env python3
"""
Model evaluation script for hand gesture detection system.
"""

import sys
import os
import argparse
import torch
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hand_gesture_detection.utils.config import ConfigManager
from hand_gesture_detection.utils.logger import Logger
from hand_gesture_detection.models.hand_gesture_net import HandGestureNet
from hand_gesture_detection.data.data_processor import DataProcessor
from hand_gesture_detection.utils.visualization import VisualizationUtils


class ModelEvaluator:
    """Professional model evaluation system."""
    
    def __init__(self, config, logger):
        """Initialize the evaluator."""
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visualizer = VisualizationUtils()
        
    def evaluate_model(self, model, test_loader, class_names):
        """Comprehensive model evaluation."""
        model.eval()
        model = model.to(self.device)
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        self.logger.info("Starting model evaluation...")
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = model(data)
                probabilities = torch.softmax(output, dim=1)
                predictions = torch.argmax(output, dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    self.logger.info(f"Evaluated batch {batch_idx}/{len(test_loader)}")
                    
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Generate classification report
        report = classification_report(
            all_targets, all_predictions,
            target_names=class_names,
            output_dict=True
        )
        
        self.logger.info(f"Test Accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
        
    def generate_evaluation_report(self, results, class_names, save_path=None):
        """Generate comprehensive evaluation report."""
        report = {
            'model_evaluation': {
                'accuracy': float(results['accuracy']),
                'classification_report': results['classification_report']
            },
            'class_performance': {},
            'confusion_matrix': results['confusion_matrix'].tolist()
        }
        
        # Per-class performance
        for i, class_name in enumerate(class_names):
            if str(i) in results['classification_report']:
                class_metrics = results['classification_report'][str(i)]
                report['class_performance'][class_name] = {
                    'precision': float(class_metrics['precision']),
                    'recall': float(class_metrics['recall']),
                    'f1_score': float(class_metrics['f1-score']),
                    'support': int(class_metrics['support'])
                }
                
        # Save report
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Evaluation report saved to: {save_path}")
            
        return report
        
    def plot_evaluation_results(self, results, class_names, save_dir=None):
        """Plot evaluation results."""
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
        # Confusion matrix
        cm_path = save_dir / "confusion_matrix.png" if save_dir else None
        self.visualizer.plot_confusion_matrix(
            results['confusion_matrix'],
            class_names,
            save_path=str(cm_path) if cm_path else None,
            title="Hand Gesture Classification Confusion Matrix"
        )
        
        # Per-class performance
        class_names_list = list(class_names)
        precisions = [results['classification_report'][class_name]['precision'] for class_name in class_names_list]
        recalls = [results['classification_report'][class_name]['recall'] for class_name in class_names_list]
        f1_scores = [results['classification_report'][class_name]['f1-score'] for class_name in class_names_list]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Precision
        axes[0].bar(class_names_list, precisions, color='skyblue')
        axes[0].set_title('Precision by Class')
        axes[0].set_ylabel('Precision')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Recall
        axes[1].bar(class_names_list, recalls, color='lightgreen')
        axes[1].set_title('Recall by Class')
        axes[1].set_ylabel('Recall')
        axes[1].tick_params(axis='x', rotation=45)
        
        # F1 Score
        axes[2].bar(class_names_list, f1_scores, color='lightcoral')
        axes[2].set_title('F1-Score by Class')
        axes[2].set_ylabel('F1-Score')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(save_dir / "class_performance.png", dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()


def main():
    """Main function for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate hand gesture detection model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to processed data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save evaluation plots"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config("training")  # Use training config for data loading
    
    # Handle nested config structure
    if 'training' in config:
        config = config['training']
    
    # Override config with command line arguments
    if args.data_path:
        config['paths']['data_path'] = args.data_path
    if args.batch_size:
        config['hyperparameters']['batch_size'] = args.batch_size
        
    # Initialize logger
    logger = Logger("model_evaluation")
    logger.info("Starting model evaluation")
    
    try:
        # Load model
        model_path = args.model_path
        if not Path(model_path).exists():
            logger.error(f"Model file not found: {model_path}")
            return 1
            
        logger.info(f"Loading model from: {model_path}")
        model = HandGestureNet.load_model(model_path)
        
        # Load test data
        logger.info("Loading test data...")
        # Create data processor config
        data_config = {
            'raw_data_path': 'data/raw',
            'processed_data_path': config['paths']['data_path'],
            'train_split': config['data']['train_split'],
            'val_split': config['data']['val_split'],
            'test_split': config['data']['test_split']
        }
        data_processor = DataProcessor(data_config, logger)
        X_train, X_val, X_test, y_train, y_val, y_test = data_processor.load_processed_data()
        
        # Create test data loader
        from torch.utils.data import DataLoader, TensorDataset
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['hyperparameters']['batch_size'],
            shuffle=False
        )
        
        # Get class names
        class_names = [
            "palm_upward", "index_upward", "fist", "thumb_up", "thumb_down",
            "two_finger_up", "grip", "closed_hand", "vertical_fingers", "index_left"
        ]
        
        logger.info(f"Test dataset size: {len(X_test)}")
        logger.info(f"Number of classes: {len(class_names)}")
        
        # Evaluate model
        evaluator = ModelEvaluator(config, logger)
        results = evaluator.evaluate_model(model, test_loader, class_names)
        
        # Generate report
        output_dir = Path(args.output_dir) if args.output_dir else Path("evaluation_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / "evaluation_report.json"
        report = evaluator.generate_evaluation_report(results, class_names, str(report_path))
        
        # Print summary
        logger.info("Evaluation Summary:")
        logger.info(f"Overall Accuracy: {results['accuracy']:.4f}")
        
        # Print per-class metrics
        logger.info("Per-class Performance:")
        for class_name in class_names:
            if class_name in report['class_performance']:
                metrics = report['class_performance'][class_name]
                logger.info(f"  {class_name}: P={metrics['precision']:.3f}, "
                          f"R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
                          
        # Generate plots
        if args.save_plots:
            logger.info("Generating evaluation plots...")
            evaluator.plot_evaluation_results(results, class_names, output_dir)
            
        logger.info(f"Evaluation completed. Results saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
