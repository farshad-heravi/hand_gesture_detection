"""
Enhanced data processing with better augmentation and confidence-focused improvements.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
import json
import torch

from ..core.feature_extractor_v2 import FeatureExtractorV2, HandFeatures
from ..utils.logger import Logger


class EnhancedDataProcessor:
    """Enhanced data processing pipeline with confidence-focused improvements."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[Logger] = None):
        """
        Initialize the enhanced data processor.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or Logger("enhanced_data_processor")
        
        # Setup paths
        self.raw_data_path = Path(config.get('raw_data_path', 'data/raw'))
        self.processed_data_path = Path(config.get('processed_data_path', 'data/processed'))
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Data splits
        self.train_split = config.get('train_split', 0.7)
        self.val_split = config.get('val_split', 0.15)
        self.test_split = config.get('test_split', 0.15)
        
        # Hand filtering
        self.both_hands = config.get('both_hands', True)
        
        # Adjust feature extractor based on hand filtering
        include_handedness = self.both_hands
        self.feature_extractor = FeatureExtractorV2(normalize_features=True, include_handedness=include_handedness)
        
        # Enhanced preprocessing
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.label_encoder = LabelEncoder()
        
        # Class balancing
        self.class_weights = None
        
    def process_raw_data_with_augmentation(self, force_reprocess: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process raw data with enhanced augmentation for better confidence.
        
        Args:
            force_reprocess: Force reprocessing even if processed data exists
            
        Returns:
            Tuple of (features, labels, class_weights)
        """
        hand_suffix = "both_hands" if self.both_hands else "right_hand_only"
        processed_file = self.processed_data_path / f"enhanced_processed_data_{hand_suffix}.pkl"
        
        if processed_file.exists() and not force_reprocess:
            self.logger.info("Loading existing enhanced processed data...")
            with open(processed_file, 'rb') as f:
                data = pickle.load(f)
                if len(data) == 3:
                    # Old format without scaler
                    self.logger.warning("Loaded data without scaler - this may cause inference issues")
                    return data[0], data[1], data[2]
                elif len(data) == 4:
                    # New format with scaler
                    X, y, class_weight_dict, self.scaler = data
                    self.logger.info("Loaded data with scaler")
                    return X, y, class_weight_dict
                else:
                    raise ValueError(f"Unexpected data format with {len(data)} items")
                
        self.logger.info("Processing raw data with enhanced augmentation...")
        
        # Load raw data
        features_list = []
        labels_list = []
        metadata_list = []
        
        gesture_dirs = [d for d in self.raw_data_path.iterdir() if d.is_dir()]
        
        for gesture_dir in gesture_dirs:
            gesture_name = gesture_dir.name
            self.logger.info(f"Processing gesture: {gesture_name}")
            
            # Load landmark files
            landmark_files = list(gesture_dir.glob("*_landmarks.pkl"))
            
            for landmark_file in landmark_files:
                try:
                    # Load landmarks
                    with open(landmark_file, 'rb') as f:
                        hand_landmarks = pickle.load(f)
                        
                    # Load metadata
                    metadata_file = landmark_file.parent / f"{landmark_file.stem.replace('_landmarks', '')}_metadata.json"
                    metadata = {}
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        # Filter by handedness if both_hands is False
                        if not self.both_hands:
                            handedness = metadata.get('handedness', '').lower()
                            if handedness != 'right':
                                continue
                    else:
                        if not self.both_hands:
                            continue
                    
                    # Extract features
                    features = self.feature_extractor.extract_features(hand_landmarks)
                    
                    if features is None or not self.feature_extractor.validate_features(features):
                        continue
                        
                    # Add original features
                    features_list.append(features.feature_vector)
                    labels_list.append(gesture_name)
                    metadata_list.append(metadata)
                    
                    # Add augmented features for underrepresented classes
                    if self._should_augment(gesture_name, len(landmark_files)):
                        augmented_features = self._augment_features(features.feature_vector)
                        features_list.append(augmented_features)
                        labels_list.append(gesture_name)
                        metadata_list.append(metadata)
                        
                except Exception as e:
                    self.logger.error(f"Error processing {landmark_file}: {e}")
                    continue
                    
        if not features_list:
            raise ValueError("No valid features found in raw data")
            
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Compute class weights for balanced training
        unique_classes = np.unique(y)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y
        )
        class_weight_dict = dict(zip(unique_classes, class_weights))
        
        # Fit the scaler on the data
        self.scaler.fit(X)
        self.logger.info("Fitted scaler on training data")
        
        self.logger.info(f"Processed {len(X)} samples with {X.shape[1]} features")
        self.logger.info(f"Class weights: {class_weight_dict}")
        
        # Save processed data with fitted scaler
        processed_data = (X, y, class_weight_dict, self.scaler)
        with open(processed_file, 'wb') as f:
            pickle.dump(processed_data, f)
            
        # Save processing statistics
        self._save_enhanced_processing_stats(X, y, metadata_list, class_weight_dict)
        
        return X, y, class_weight_dict
        
    def _should_augment(self, gesture_name: str, total_samples: int) -> bool:
        """Determine if a gesture should be augmented based on sample count."""
        # Augment if gesture has fewer than 60 samples
        return total_samples < 60
        
    def _augment_features(self, features: np.ndarray) -> np.ndarray:
        """Apply feature augmentation to increase dataset diversity."""
        augmented = features.copy()
        
        # Add small amount of noise
        noise_std = 0.01
        noise = np.random.normal(0, noise_std, features.shape)
        augmented += noise
        
        # Scale features slightly
        scale_factor = np.random.uniform(0.95, 1.05)
        augmented *= scale_factor
        
        # Randomly zero out some features (feature dropout)
        dropout_prob = 0.1
        mask = np.random.random(features.shape) > dropout_prob
        augmented *= mask
        
        return augmented
        
    def load_enhanced_processed_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Load enhanced processed data with class weights.
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, class_weights)
        """
        # Load processed data
        X, y, class_weights = self.process_raw_data_with_augmentation()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data with stratification
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, 
            test_size=self.test_split, 
            random_state=42, 
            stratify=y_encoded
        )
        
        val_size = self.val_split / (self.train_split + self.val_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=42,
            stratify=y_temp
        )
        
        # Enhanced normalization
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        self.logger.info(f"Enhanced data splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Convert class weights to tensor format
        class_weight_tensor = torch.zeros(len(self.label_encoder.classes_))
        for class_name, weight in class_weights.items():
            class_idx = self.label_encoder.transform([class_name])[0]
            class_weight_tensor[class_idx] = weight
            
        # Save enhanced split information
        self._save_enhanced_split_info(X_train, X_val, X_test, y_train, y_val, y_test, class_weights)
        
        return X_train, X_val, X_test, y_train, y_val, y_test, class_weight_tensor
        
    def _save_enhanced_processing_stats(self, X: np.ndarray, y: np.ndarray, metadata_list: List[Dict], class_weights: Dict) -> None:
        """Save enhanced processing statistics."""
        stats = {
            'total_samples': len(X),
            'num_features': X.shape[1],
            'gesture_counts': {},
            'class_weights': class_weights,
            'hand_filtering': {
                'both_hands': self.both_hands,
                'right_hand_only': not self.both_hands
            },
            'feature_statistics': {
                'mean': np.mean(X, axis=0).tolist(),
                'std': np.std(X, axis=0).tolist(),
                'min': np.min(X, axis=0).tolist(),
                'max': np.max(X, axis=0).tolist(),
                'robust_mean': np.median(X, axis=0).tolist(),
                'robust_std': np.median(np.abs(X - np.median(X, axis=0)), axis=0).tolist()
            }
        }
        
        # Count samples per gesture
        unique_gestures, counts = np.unique(y, return_counts=True)
        for gesture, count in zip(unique_gestures, counts):
            stats['gesture_counts'][gesture] = int(count)
            
        # Save statistics
        stats_file = self.processed_data_path / "enhanced_processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
        self.logger.info(f"Enhanced processing statistics saved to: {stats_file}")
        
    def _save_enhanced_split_info(self, X_train, X_val, X_test, y_train, y_val, y_test, class_weights) -> None:
        """Save enhanced data split information."""
        split_info = {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'train_split': self.train_split,
            'val_split': self.val_split,
            'test_split': self.test_split,
            'feature_names': self.feature_extractor.get_feature_names(),
            'label_mapping': dict(enumerate(self.label_encoder.classes_)),
            'class_weights': class_weights,
            'scaler_type': 'RobustScaler'
        }
        
        split_file = self.processed_data_path / "enhanced_split_info.json"
        with open(split_file, 'w') as f:
            json.dump(split_info, f, indent=2)
            
        self.logger.info(f"Enhanced split information saved to: {split_file}")
        
    def get_confidence_analysis(self) -> Dict[str, Any]:
        """Analyze data for confidence-related issues."""
        try:
            X, y, class_weights = self.process_raw_data_with_augmentation()
            
            analysis = {
                'dataset_balance': self._analyze_dataset_balance(y),
                'feature_quality': self._analyze_feature_quality(X),
                'class_separability': self._analyze_class_separability(X, y),
                'recommendations': []
            }
            
            # Generate recommendations
            if analysis['dataset_balance']['balance_score'] < 0.7:
                analysis['recommendations'].append("Dataset is imbalanced - consider data augmentation")
                
            if analysis['feature_quality']['low_variance_features'] > 5:
                analysis['recommendations'].append("Many features have low variance - consider feature selection")
                
            if analysis['class_separability']['avg_separability'] < 0.5:
                analysis['recommendations'].append("Classes are not well separated - consider feature engineering")
                
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in confidence analysis: {e}")
            return {}
            
    def _analyze_dataset_balance(self, y: np.ndarray) -> Dict[str, Any]:
        """Analyze dataset balance."""
        unique_classes, counts = np.unique(y, return_counts=True)
        balance_score = 1.0 / (1.0 + np.std(counts) / np.mean(counts))
        
        return {
            'balance_score': float(balance_score),
            'min_samples': int(np.min(counts)),
            'max_samples': int(np.max(counts)),
            'class_counts': dict(zip(unique_classes, counts.tolist()))
        }
        
    def _analyze_feature_quality(self, X: np.ndarray) -> Dict[str, Any]:
        """Analyze feature quality."""
        feature_vars = np.var(X, axis=0)
        low_variance_features = np.sum(feature_vars < 1e-6)
        
        return {
            'low_variance_features': int(low_variance_features),
            'avg_variance': float(np.mean(feature_vars)),
            'feature_correlation': float(np.mean(np.corrcoef(X.T)))
        }
        
    def _analyze_class_separability(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze class separability."""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        
        try:
            lda = LinearDiscriminantAnalysis()
            lda.fit(X, y)
            separability_score = np.mean(lda.explained_variance_ratio_)
            
            return {
                'avg_separability': float(separability_score),
                'lda_components': int(len(lda.explained_variance_ratio_))
            }
        except:
            return {
                'avg_separability': 0.0,
                'lda_components': 0
            }
