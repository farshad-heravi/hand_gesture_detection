"""
Visualization utilities for hand gesture detection.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class VisualizationUtils:
    """Professional visualization utilities for hand gesture detection."""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        """
        Initialize visualization utilities.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        sns.set_palette("husl")
        
    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        val_accuracies: List[float],
        learning_rates: Optional[List[float]] = None,
        save_path: Optional[str] = None,
        title: str = "Training Progress"
    ) -> None:
        """
        Plot comprehensive training curves.
        
        Args:
            train_losses: Training loss values
            val_losses: Validation loss values
            val_accuracies: Validation accuracy values
            learning_rates: Learning rate values (optional)
            save_path: Path to save the plot
            title: Plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, train_losses, label='Training Loss', color='blue')
        axes[0, 0].plot(epochs, val_losses, label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curve
        axes[0, 1].plot(epochs, val_accuracies, label='Validation Accuracy', color='green')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate curve
        if learning_rates:
            axes[1, 0].plot(epochs, learning_rates, label='Learning Rate', color='orange')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].axis('off')
            
        # Combined plot
        ax2 = axes[1, 1].twinx()
        axes[1, 1].plot(epochs, val_losses, label='Val Loss', color='red', alpha=0.7)
        ax2.plot(epochs, val_accuracies, label='Val Accuracy', color='green', alpha=0.7)
        axes[1, 1].set_title('Validation Loss and Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss', color='red')
        ax2.set_ylabel('Accuracy', color='green')
        axes[1, 1].legend(loc='upper left')
        ax2.legend(loc='upper right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        save_path: Optional[str] = None,
        title: str = "Confusion Matrix"
    ) -> None:
        """
        Plot confusion matrix with annotations.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            save_path: Path to save the plot
            title: Plot title
        """
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Normalized Count'}
        )
        
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
    def plot_feature_importance(
        self,
        feature_importance: np.ndarray,
        feature_names: List[str],
        save_path: Optional[str] = None,
        title: str = "Feature Importance"
    ) -> None:
        """
        Plot feature importance.
        
        Args:
            feature_importance: Feature importance values
            feature_names: List of feature names
            save_path: Path to save the plot
            title: Plot title
        """
        # Sort features by importance
        sorted_indices = np.argsort(feature_importance)[::-1]
        sorted_importance = feature_importance[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(sorted_importance)), sorted_importance)
        
        # Color bars by importance
        colors = plt.cm.viridis(sorted_importance / np.max(sorted_importance))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            
        plt.title(title)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, sorted_importance)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
                    
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
    def plot_dataset_distribution(
        self,
        gesture_counts: Dict[str, int],
        save_path: Optional[str] = None,
        title: str = "Dataset Distribution"
    ) -> None:
        """
        Plot dataset distribution by gesture.
        
        Args:
            gesture_counts: Dictionary mapping gesture names to counts
            save_path: Path to save the plot
            title: Plot title
        """
        gestures = list(gesture_counts.keys())
        counts = list(gesture_counts.values())
        
        plt.figure(figsize=(12, 6))
        
        # Bar plot
        bars = plt.bar(gestures, counts)
        
        # Color bars
        colors = plt.cm.Set3(np.linspace(0, 1, len(gestures)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            
        plt.title(title)
        plt.xlabel('Gesture')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom')
                    
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
    def plot_performance_metrics(
        self,
        metrics: Dict[str, List[float]],
        save_path: Optional[str] = None,
        title: str = "Performance Metrics"
    ) -> None:
        """
        Plot real-time performance metrics.
        
        Args:
            metrics: Dictionary of metric names to time series data
            save_path: Path to save the plot
            title: Plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            if i >= 4:  # Only plot first 4 metrics
                break
                
            axes[i].plot(values, label=metric_name)
            axes[i].set_title(f'{metric_name}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel(metric_name)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
        # Hide unused subplots
        for i in range(len(metrics), 4):
            axes[i].axis('off')
            
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
    def create_interactive_dashboard(
        self,
        training_data: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """
        Create interactive dashboard using Plotly.
        
        Args:
            training_data: Dictionary containing training metrics
            save_path: Path to save the HTML file
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Validation Accuracy', 
                          'Learning Rate', 'Feature Importance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Training loss
        if 'train_losses' in training_data and 'val_losses' in training_data:
            epochs = list(range(1, len(training_data['train_losses']) + 1))
            fig.add_trace(
                go.Scatter(x=epochs, y=training_data['train_losses'], 
                          name='Train Loss', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=epochs, y=training_data['val_losses'], 
                          name='Val Loss', line=dict(color='red')),
                row=1, col=1
            )
            
        # Validation accuracy
        if 'val_accuracies' in training_data:
            epochs = list(range(1, len(training_data['val_accuracies']) + 1))
            fig.add_trace(
                go.Scatter(x=epochs, y=training_data['val_accuracies'], 
                          name='Val Accuracy', line=dict(color='green')),
                row=1, col=2
            )
            
        # Learning rate
        if 'learning_rates' in training_data:
            epochs = list(range(1, len(training_data['learning_rates']) + 1))
            fig.add_trace(
                go.Scatter(x=epochs, y=training_data['learning_rates'], 
                          name='Learning Rate', line=dict(color='orange')),
                row=2, col=1
            )
            
        # Feature importance
        if 'feature_importance' in training_data and 'feature_names' in training_data:
            fig.add_trace(
                go.Bar(x=training_data['feature_names'], 
                      y=training_data['feature_importance'],
                      name='Feature Importance'),
                row=2, col=2
            )
            
        # Update layout
        fig.update_layout(
            title_text="Hand Gesture Detection Training Dashboard",
            showlegend=True,
            height=800
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_yaxes(title_text="Learning Rate", row=2, col=1)
        fig.update_xaxes(title_text="Features", row=2, col=2)
        fig.update_yaxes(title_text="Importance", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
            
    def draw_landmarks_on_image(
        self,
        image: np.ndarray,
        landmarks: List[Tuple[float, float, float]],
        connections: Optional[List[Tuple[int, int]]] = None,
        landmark_color: Tuple[int, int, int] = (0, 255, 0),
        connection_color: Tuple[int, int, int] = (255, 0, 0),
        landmark_radius: int = 3,
        connection_thickness: int = 2
    ) -> np.ndarray:
        """
        Draw hand landmarks on an image.
        
        Args:
            image: Input image
            landmarks: List of landmark coordinates (x, y, z)
            connections: List of landmark connections (optional)
            landmark_color: Color for landmarks (BGR)
            connection_color: Color for connections (BGR)
            landmark_radius: Radius of landmark circles
            connection_thickness: Thickness of connection lines
            
        Returns:
            Image with drawn landmarks
        """
        annotated_image = image.copy()
        h, w = image.shape[:2]
        
        # Draw connections first (so they appear behind landmarks)
        if connections:
            for start_idx, end_idx in connections:
                if 0 <= start_idx < len(landmarks) and 0 <= end_idx < len(landmarks):
                    start_point = (
                        int(landmarks[start_idx][0] * w),
                        int(landmarks[start_idx][1] * h)
                    )
                    end_point = (
                        int(landmarks[end_idx][0] * w),
                        int(landmarks[end_idx][1] * h)
                    )
                    cv2.line(annotated_image, start_point, end_point, 
                            connection_color, connection_thickness)
                            
        # Draw landmarks
        for i, (x, y, z) in enumerate(landmarks):
            point = (int(x * w), int(y * h))
            cv2.circle(annotated_image, point, landmark_radius, landmark_color, -1)
            
            # Add landmark index
            cv2.putText(annotated_image, str(i), 
                       (point[0] + 5, point[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, landmark_color, 1)
                       
        return annotated_image
