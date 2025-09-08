#!/usr/bin/env python3
"""
Script to visualize hand gesture predictions on random images from train/val sets.
Shows landmark detection, predictions, confidence scores, and other important data.
"""

import argparse
import sys
import os
import random
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
import torch
import yaml
from collections import defaultdict

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hand_gesture_detection.core.landmark_extractor import HandLandmarkExtractor
from hand_gesture_detection.core.feature_extractor_v2 import FeatureExtractorV2
from hand_gesture_detection.models.enhanced_hand_gesture_net import EnhancedHandGestureNet
from hand_gesture_detection.core.detector import draw_landmarks, draw_hand_info
from hand_gesture_detection.utils.logger import Logger


class PredictionVisualizer:
    """Visualizes hand gesture predictions on random dataset samples."""
    
    def __init__(self, model_path: str, config_path: str = None):
        """
        Initialize the prediction visualizer.
        
        Args:
            model_path: Path to the trained model
            config_path: Path to inference config file
        """
        self.model_path = Path(model_path)
        self.logger = Logger("prediction_visualizer")
        
        # Load config
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
            
        # Initialize components
        self._load_model()
        self._load_data()
        self._setup_visualization()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'mediapipe': {
                'mode': 'image',
                'max_num_hands': 1,
                'min_detection_confidence': 0.5,
                'min_tracking_confidence': 0.5
            },
            'display': {
                'landmark_color': [0, 255, 0],
                'connection_color': [255, 0, 0],
                'landmark_radius': 3,
                'connection_thickness': 2,
                'font_scale': 0.8,
                'font_thickness': 2
            }
        }
        
    def _load_model(self):
        """Load the trained model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        self.logger.info(f"Loading model from: {self.model_path}")
        
        # Load model checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        model_info = checkpoint.get('model_info', {})
        
        # Create model with correct architecture
        self.model = EnhancedHandGestureNet(
            num_classes=model_info.get('num_classes', 10),
            input_size=model_info.get('input_size', 42),
            hidden_sizes=model_info.get('hidden_sizes', [128, 256, 128, 64]),
            dropout_rate=model_info.get('dropout_rate', 0.2),
            use_batch_norm=True,
            use_residual=True
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load gesture mapping
        self._load_gesture_mapping()
        
        self.logger.info("Model loaded successfully")
        
    def _load_gesture_mapping(self):
        """Load gesture class mapping."""
        # Create mapping from string labels to display names
        self.string_to_display = {
            'closed_hand': 'Closed Hand',
            'fist': 'Fist',
            'grip': 'Grip',
            'index_left': 'Index Left',
            'index_upward': 'Index Upward',
            'palm_upward': 'Palm Upward',
            'thumb_down': 'Thumbs Down',
            'thumb_up': 'Thumbs Up',
            'two_finger_up': 'Two Fingers Up',
            'vertical_fingers': 'Vertical Fingers'
        }
        
        # Create mapping from string labels to integer indices for model prediction
        self.string_to_int = {
            'closed_hand': 0,
            'fist': 1,
            'grip': 2,
            'index_left': 3,
            'index_upward': 4,
            'palm_upward': 5,
            'thumb_down': 6,
            'thumb_up': 7,
            'two_finger_up': 8,
            'vertical_fingers': 9
        }
        
        # Create reverse mapping from integer indices to display names
        self.int_to_display = {self.string_to_int[k]: v for k, v in self.string_to_display.items()}
        
        self.logger.info(f"Loaded gesture mapping: {self.string_to_display}")
        
    def _load_data(self):
        """Load processed data and create train/val splits."""
        processed_data_path = Path("data/processed")
        
        # Load processed data
        processed_file = processed_data_path / "enhanced_processed_data_right_hand_only.pkl"
        if not processed_file.exists():
            processed_file = processed_data_path / "processed_data_both_hands.pkl"
            
        if not processed_file.exists():
            raise FileNotFoundError("No processed data found. Please run data processing first.")
            
        self.logger.info(f"Loading processed data from: {processed_file}")
        
        with open(processed_file, 'rb') as f:
            processed_data = pickle.load(f)
            
        if len(processed_data) == 3:
            X, y, metadata = processed_data
            # Check if metadata is a list or dict
            if isinstance(metadata, dict):
                # If metadata is a dict, create empty list
                metadata = [{}] * len(X)
        else:
            X, y = processed_data
            metadata = [{}] * len(X)
            
        # Load split information
        split_info_path = processed_data_path / "enhanced_split_info.json"
        if not split_info_path.exists():
            split_info_path = processed_data_path / "split_info.json"
            
        if split_info_path.exists():
            with open(split_info_path, 'r') as f:
                split_info = json.load(f)
            train_indices = split_info.get('train_indices', [])
            val_indices = split_info.get('val_indices', [])
            test_indices = split_info.get('test_indices', [])
            
            # If indices are not in the split info, create them based on sample counts
            if not train_indices and not val_indices and not test_indices:
                train_samples = split_info.get('train_samples', 0)
                val_samples = split_info.get('val_samples', 0)
                test_samples = split_info.get('test_samples', 0)
                
                if train_samples > 0 or val_samples > 0 or test_samples > 0:
                    # Create indices based on sample counts
                    indices = list(range(len(X)))
                    random.shuffle(indices)
                    
                    train_indices = indices[:train_samples] if train_samples > 0 else []
                    val_start = train_samples
                    val_indices = indices[val_start:val_start + val_samples] if val_samples > 0 else []
                    test_start = val_start + val_samples
                    test_indices = indices[test_start:test_start + test_samples] if test_samples > 0 else []
                else:
                    # Fallback to random split
                    self.logger.warning("No valid split info found, creating random 80/20 split")
                    indices = list(range(len(X)))
                    random.shuffle(indices)
                    split_idx = int(0.8 * len(indices))
                    train_indices = indices[:split_idx]
                    val_indices = indices[split_idx:]
                    test_indices = []
        else:
            # Create random split if no split info
            self.logger.warning("No split info found, creating random 80/20 split")
            indices = list(range(len(X)))
            random.shuffle(indices)
            split_idx = int(0.8 * len(indices))
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            test_indices = []
            
        # Create datasets
        self.train_data = {
            'X': X[train_indices] if len(train_indices) > 0 else np.array([]),
            'y': y[train_indices] if len(train_indices) > 0 else np.array([]),
            'metadata': [metadata[i] for i in train_indices] if len(train_indices) > 0 else []
        }
        
        self.val_data = {
            'X': X[val_indices] if len(val_indices) > 0 else np.array([]),
            'y': y[val_indices] if len(val_indices) > 0 else np.array([]),
            'metadata': [metadata[i] for i in val_indices] if len(val_indices) > 0 else []
        }
        
        self.logger.info(f"Loaded data - Train: {len(self.train_data['X'])}, Val: {len(self.val_data['X'])}")
        
    def _setup_visualization(self):
        """Setup visualization components."""
        # Initialize landmark extractor
        self.landmark_extractor = HandLandmarkExtractor(
            mode=self.config['mediapipe']['mode'],
            max_num_hands=self.config['mediapipe']['max_num_hands'],
            min_detection_confidence=self.config['mediapipe']['min_detection_confidence'],
            min_tracking_confidence=self.config['mediapipe']['min_tracking_confidence']
        )
        
        # Initialize feature extractor (match training configuration)
        self.feature_extractor = FeatureExtractorV2(
            normalize_features=True, 
            include_handedness=False  # Match training data
        )
        
        # Load original images for visualization
        self._load_original_images()
        
    def _load_original_images(self):
        """Load original images from raw data for visualization."""
        self.original_images = {}
        raw_data_path = Path("data/raw")
        
        # Map processed data indices to original images
        for split_name, split_data in [('train', self.train_data), ('val', self.val_data)]:
            self.original_images[split_name] = []
            
            for i, metadata in enumerate(split_data['metadata']):
                image_path = None
                
                # Try to find image in metadata first
                if 'image_path' in metadata:
                    image_path = Path(metadata['image_path'])
                    if not image_path.exists():
                        image_path = None
                
                # If no image found, try to find a representative image for this gesture
                if image_path is None:
                    gesture_name = split_data['y'][i]  # This is now a string
                    gesture_dir = raw_data_path / gesture_name
                    
                    if gesture_dir.exists():
                        # Find a random image from this gesture
                        image_files = list(gesture_dir.glob("*.png"))
                        if image_files:
                            image_path = random.choice(image_files)
                        else:
                            # Try other image formats
                            for ext in ['*.jpg', '*.jpeg', '*.bmp']:
                                image_files = list(gesture_dir.glob(ext))
                                if image_files:
                                    image_path = random.choice(image_files)
                                    break
                
                self.original_images[split_name].append(str(image_path) if image_path else None)
                        
    def _predict_gesture(self, features: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """Make gesture prediction from features."""
        # Convert to tensor
        feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(feature_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get prediction
            confidence, predicted_class = torch.max(probabilities, 1)
            confidence = confidence.item()
            predicted_class = predicted_class.item()
            
            # Get all probabilities
            all_probs = probabilities.squeeze().numpy()
            
        return predicted_class, confidence, all_probs
        
    def _create_placeholder_image(self, true_label: str, split_name: str, data_idx: int) -> np.ndarray:
        """Create a placeholder image when original image is not available."""
        # Create a dark background
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add title
        cv2.putText(image, "Hand Gesture Prediction", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add sample info
        cv2.putText(image, f"Sample: {split_name.capitalize()} #{data_idx + 1}", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # Add gesture info
        gesture_name = self.string_to_display.get(true_label, true_label.replace('_', ' ').title())
        cv2.putText(image, f"True Gesture: {gesture_name}", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Add note about image
        cv2.putText(image, "Original image not available", (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
        cv2.putText(image, "Using feature-based prediction only", (50, 230), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
        
        # Draw a simple hand outline as placeholder
        center = (320, 300)
        cv2.circle(image, center, 80, (100, 100, 100), 2)  # Palm
        cv2.circle(image, center, 5, (255, 255, 255), -1)  # Center
        
        # Draw finger lines
        for i, angle in enumerate([-60, -30, 0, 30, 60]):
            x = int(center[0] + 60 * np.cos(np.radians(angle)))
            y = int(center[1] + 60 * np.sin(np.radians(angle)))
            cv2.line(image, center, (x, y), (100, 100, 100), 2)
        
        return image
        
    def _draw_prediction_info(self, image: np.ndarray, prediction: int, confidence: float, 
                            all_probs: np.ndarray, true_label: str, 
                            landmarks_detected: bool = True) -> np.ndarray:
        """Draw prediction information on image."""
        # Get colors
        landmark_color = tuple(self.config['display']['landmark_color'])
        connection_color = tuple(self.config['display']['connection_color'])
        font_scale = self.config['display']['font_scale']
        font_thickness = self.config['display']['font_thickness']
        
        # Get gesture names
        predicted_name = self.int_to_display.get(prediction, f"Unknown_{prediction}")
        true_name = self.string_to_display.get(true_label, true_label.replace('_', ' ').title())
        
        # Determine text color based on correctness
        # Convert true_label to int for comparison
        true_label_int = self.string_to_int.get(true_label, -1)
        is_correct = prediction == true_label_int
        text_color = (0, 255, 0) if is_correct else (0, 0, 255)  # Green if correct, red if wrong
        
        # Draw main prediction info
        y_offset = 30
        cv2.putText(image, f"Predicted: {predicted_name}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
        y_offset += 30
        
        cv2.putText(image, f"True Label: {true_name}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        y_offset += 30
        
        cv2.putText(image, f"Confidence: {confidence:.3f}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        y_offset += 30
        
        cv2.putText(image, f"Correct: {'Yes' if is_correct else 'No'}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
        y_offset += 30
        
        cv2.putText(image, f"Landmarks: {'Detected' if landmarks_detected else 'Not Detected'}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                   (0, 255, 0) if landmarks_detected else (0, 0, 255), font_thickness)
        y_offset += 30
        
        # Draw top 3 predictions
        top3_indices = np.argsort(all_probs)[-3:][::-1]
        cv2.putText(image, "Top 3 Predictions:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        y_offset += 25
        
        for i, idx in enumerate(top3_indices):
            gesture_name = self.int_to_display.get(idx, f"Unknown_{idx}")
            prob = all_probs[idx]
            cv2.putText(image, f"{i+1}. {gesture_name}: {prob:.3f}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (255, 255, 255), font_thickness)
            y_offset += 20
            
        return image
        
    def visualize_random_samples(self, num_samples: int = 10, split: str = 'both', 
                               show_landmarks: bool = True) -> None:
        """
        Visualize random samples with predictions.
        
        Args:
            num_samples: Number of samples to visualize
            split: Which split to use ('train', 'val', 'both')
            show_landmarks: Whether to show hand landmarks
        """
        if split == 'both':
            splits = ['train', 'val']
        else:
            splits = [split]
            
        samples_per_split = num_samples // len(splits)
        
        for split_name in splits:
            split_data = getattr(self, f'{split_name}_data')
            if len(split_data['X']) == 0:
                self.logger.warning(f"No data available for {split_name} split")
                continue
                
            # Get random samples
            num_available = len(split_data['X'])
            num_samples_split = min(samples_per_split, num_available)
            sample_indices = random.sample(range(num_available), num_samples_split)
            
            self.logger.info(f"Visualizing {num_samples_split} samples from {split_name} split")
            
            for i, idx in enumerate(sample_indices):
                self._visualize_single_sample(split_name, idx, i, show_landmarks)
                
    def _visualize_single_sample(self, split_name: str, data_idx: int, 
                               sample_num: int, show_landmarks: bool) -> None:
        """Visualize a single sample."""
        split_data = getattr(self, f'{split_name}_data')
        
        # Get data
        features = split_data['X'][data_idx]
        true_label = split_data['y'][data_idx]  # This is now a string
        metadata = split_data['metadata'][data_idx] if data_idx < len(split_data['metadata']) else {}
        
        # Get original image
        image_path = self.original_images[split_name][data_idx] if data_idx < len(self.original_images[split_name]) else None
        
        if image_path and Path(image_path).exists():
            image = cv2.imread(image_path)
            if image is None:
                # If image failed to load, create placeholder
                image = self._create_placeholder_image(true_label, split_name, data_idx)
        else:
            # Create a placeholder image if no original image available
            image = self._create_placeholder_image(true_label, split_name, data_idx)
        
        # Make prediction
        prediction, confidence, all_probs = self._predict_gesture(features)
        
        # Try to detect landmarks on the image for visualization
        landmarks_detected = False
        if image_path and Path(image_path).exists():
            landmarks_list = self.landmark_extractor.extract_landmarks(image)
            if landmarks_list:
                landmarks_detected = True
                if show_landmarks:
                    # Draw landmarks on the image
                    image = draw_landmarks(
                        image, landmarks_list[0],
                        landmark_color=tuple(self.config['display']['landmark_color']),
                        connection_color=tuple(self.config['display']['connection_color']),
                        landmark_radius=self.config['display']['landmark_radius'],
                        connection_thickness=self.config['display']['connection_thickness']
                    )
        
        # Draw prediction information
        image = self._draw_prediction_info(
            image, prediction, confidence, all_probs, true_label, landmarks_detected
        )
        
        # Add sample info
        cv2.putText(image, f"Sample {sample_num + 1} ({split_name})", 
                   (image.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   self.config['display']['font_scale'], (255, 255, 255), 
                   self.config['display']['font_thickness'])
        
        # Show image
        window_name = f"Hand Gesture Prediction - {split_name.capitalize()} Sample {sample_num + 1}"
        cv2.imshow(window_name, image)
        
        # Wait for key press
        print(f"\nShowing {split_name} sample {sample_num + 1}")
        print(f"True Label: {self.string_to_display.get(true_label, true_label.replace('_', ' ').title())}")
        print(f"Predicted: {self.int_to_display.get(prediction, f'Unknown_{prediction}')}")
        print(f"Confidence: {confidence:.3f}")
        true_label_int = self.string_to_int.get(true_label, -1)
        print(f"Correct: {'Yes' if prediction == true_label_int else 'No'}")
        print(f"Landmarks Detected: {'Yes' if landmarks_detected else 'No'}")
        print("Press any key to continue, 'q' to quit...")
        
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(window_name)
        
        if key == ord('q'):
            return False
            
        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Visualize hand gesture predictions on random dataset samples")
    parser.add_argument("model_path", help="Path to the trained model file")
    parser.add_argument("--config", help="Path to inference config file", default=None)
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--split", choices=['train', 'val', 'both'], default='both', 
                       help="Which data split to use")
    parser.add_argument("--no_landmarks", action='store_true', help="Don't show hand landmarks")
    
    args = parser.parse_args()
    
    try:
        # Create visualizer
        visualizer = PredictionVisualizer(args.model_path, args.config)
        
        # Visualize samples
        visualizer.visualize_random_samples(
            num_samples=args.num_samples,
            split=args.split,
            show_landmarks=not args.no_landmarks
        )
        
        print("\nVisualization complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    finally:
        cv2.destroyAllWindows()
        
    return 0


if __name__ == "__main__":
    sys.exit(main())


