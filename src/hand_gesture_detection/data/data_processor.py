"""
Data processing utilities for hand gesture detection.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json

from ..core.feature_extractor_v2 import FeatureExtractorV2, HandFeatures
from ..utils.logger import Logger


class DataProcessor:
    """Professional data processing pipeline for hand gesture detection."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[Logger] = None):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or Logger("data_processor")
        
        # Setup paths
        self.raw_data_path = Path(config.get('raw_data_path', 'data/raw'))
        self.processed_data_path = Path(config.get('processed_data_path', 'data/processed'))
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Data splits
        self.train_split = config.get('train_split', 0.7)
        self.val_split = config.get('val_split', 0.15)
        self.test_split = config.get('test_split', 0.15)
        
        # Hand filtering
        self.both_hands = config.get('both_hands', True)  # Default to True for backward compatibility
        
        # Adjust feature extractor based on hand filtering
        include_handedness = self.both_hands  # Only include handedness features if using both hands
        self.feature_extractor = FeatureExtractorV2(normalize_features=True, include_handedness=include_handedness)
        
        # Validate splits
        if abs(self.train_split + self.val_split + self.test_split - 1.0) > 1e-6:
            raise ValueError("Data splits must sum to 1.0")
            
        # Preprocessing
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def process_raw_data(self, force_reprocess: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process raw data from collected samples.
        
        Args:
            force_reprocess: Force reprocessing even if processed data exists
            
        Returns:
            Tuple of (features, labels)
        """
        # Include hand filtering in cache file name
        hand_suffix = "both_hands" if self.both_hands else "right_hand_only"
        processed_file = self.processed_data_path / f"processed_data_{hand_suffix}.pkl"
        
        if processed_file.exists() and not force_reprocess:
            self.logger.info("Loading existing processed data...")
            with open(processed_file, 'rb') as f:
                data = pickle.load(f)
                # Handle both old format (X, y) and new format (X, y, metadata)
                if len(data) == 3:
                    return data[0], data[1]  # Return only X, y
                else:
                    return data
                
        self.logger.info("Processing raw data...")
        
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
                        
                    # Load metadata first to check handedness before processing
                    metadata_file = landmark_file.parent / f"{landmark_file.stem.replace('_landmarks', '')}_metadata.json"
                    metadata = {}
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        # Filter by handedness if both_hands is False
                        if not self.both_hands:
                            handedness = metadata.get('handedness', '').lower()
                            if handedness != 'right':
                                self.logger.debug(f"Skipping {landmark_file} - not right hand (handedness: {handedness})")
                                continue
                    else:
                        # If no metadata, skip if both_hands is False (we can't determine handedness)
                        if not self.both_hands:
                            self.logger.warning(f"No metadata found for {landmark_file}, skipping due to both_hands=False")
                            continue
                    
                    # Extract features using the new feature extractor
                    features = self.feature_extractor.extract_features(hand_landmarks)
                    
                    if features is None:
                        self.logger.warning(f"Failed to extract features from: {landmark_file}")
                        continue
                        
                    # Validate features
                    if not self.feature_extractor.validate_features(features):
                        self.logger.warning(f"Invalid features: {landmark_file}")
                        continue
                        
                    # Add features, labels, and metadata only after all checks pass
                    features_list.append(features.feature_vector)
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
        
        self.logger.info(f"Processed {len(X)} samples with {X.shape[1]} features")
        
        # Log hand filtering information
        if not self.both_hands:
            right_hand_count = sum(1 for meta in metadata_list if meta.get('handedness', '').lower() == 'right')
            self.logger.info(f"Hand filtering enabled: Using only right hand data ({right_hand_count} samples)")
        else:
            self.logger.info("Hand filtering disabled: Using both left and right hand data")
        
        # Save processed data
        processed_data = (X, y, metadata_list)
        with open(processed_file, 'wb') as f:
            pickle.dump(processed_data, f)
            
        # Save processing statistics
        self._save_processing_stats(X, y, metadata_list)
        
        return X, y
        
    def load_processed_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load processed data and split into train/val/test sets.
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Load processed data
        X, y = self.process_raw_data()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
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
        
        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        self.logger.info(f"Data splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Save split information
        self._save_split_info(X_train, X_val, X_test, y_train, y_val, y_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def _save_processing_stats(self, X: np.ndarray, y: np.ndarray, metadata_list: List[Dict]) -> None:
        """Save processing statistics."""
        stats = {
            'total_samples': len(X),
            'num_features': X.shape[1],
            'gesture_counts': {},
            'hand_filtering': {
                'both_hands': self.both_hands,
                'right_hand_only': not self.both_hands
            },
            'feature_statistics': {
                'mean': np.mean(X, axis=0).tolist(),
                'std': np.std(X, axis=0).tolist(),
                'min': np.min(X, axis=0).tolist(),
                'max': np.max(X, axis=0).tolist()
            }
        }
        
        # Count samples per gesture
        unique_gestures, counts = np.unique(y, return_counts=True)
        for gesture, count in zip(unique_gestures, counts):
            stats['gesture_counts'][gesture] = int(count)
            
        # Save statistics
        stats_file = self.processed_data_path / "processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
        self.logger.info(f"Processing statistics saved to: {stats_file}")
        
    def _save_split_info(self, X_train, X_val, X_test, y_train, y_val, y_test) -> None:
        """Save data split information."""
        split_info = {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'train_split': self.train_split,
            'val_split': self.val_split,
            'test_split': self.test_split,
            'feature_names': self.feature_extractor.get_feature_names(),
            'label_mapping': dict(enumerate(self.label_encoder.classes_))
        }
        
        split_file = self.processed_data_path / "split_info.json"
        with open(split_file, 'w') as f:
            json.dump(split_info, f, indent=2)
            
        self.logger.info(f"Split information saved to: {split_file}")
        
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get comprehensive data statistics."""
        try:
            # Load processed data
            X, y = self.process_raw_data()
            
            stats = {
                'total_samples': len(X),
                'num_features': X.shape[1],
                'gesture_distribution': {},
                'feature_statistics': {},
                'data_quality': {}
            }
            
            # Gesture distribution
            unique_gestures, counts = np.unique(y, return_counts=True)
            for gesture, count in zip(unique_gestures, counts):
                stats['gesture_distribution'][gesture] = {
                    'count': int(count),
                    'percentage': float(count / len(y) * 100)
                }
                
            # Feature statistics
            stats['feature_statistics'] = {
                'mean': np.mean(X, axis=0).tolist(),
                'std': np.std(X, axis=0).tolist(),
                'min': np.min(X, axis=0).tolist(),
                'max': np.max(X, axis=0).tolist(),
                'has_nan': bool(np.any(np.isnan(X))),
                'has_inf': bool(np.any(np.isinf(X)))
            }
            
            # Data quality metrics
            stats['data_quality'] = {
                'completeness': 1.0 - np.isnan(X).sum() / X.size,
                'consistency': self._calculate_consistency_score(X),
                'balance_score': self._calculate_balance_score(counts)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")
            return {}
            
    def _calculate_consistency_score(self, X: np.ndarray) -> float:
        """Calculate data consistency score."""
        # Simple consistency metric based on feature variance
        feature_vars = np.var(X, axis=0)
        consistency_score = 1.0 / (1.0 + np.mean(feature_vars))
        return float(consistency_score)
        
    def _calculate_balance_score(self, counts: np.ndarray) -> float:
        """Calculate dataset balance score."""
        if len(counts) == 0:
            return 0.0
            
        # Balance score based on coefficient of variation
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        balance_score = 1.0 / (1.0 + std_count / mean_count)
        return float(balance_score)
        
    def validate_data_quality(self) -> Dict[str, Any]:
        """Validate data quality and return issues."""
        issues = []
        warnings = []
        
        try:
            # Load processed data
            X, y = self.process_raw_data()
            
            # Check for missing values
            if np.any(np.isnan(X)):
                issues.append("Dataset contains NaN values")
                
            if np.any(np.isinf(X)):
                issues.append("Dataset contains infinite values")
                
            # Check gesture balance
            unique_gestures, counts = np.unique(y, return_counts=True)
            min_count = np.min(counts)
            max_count = np.max(counts)
            
            if max_count / min_count > 3:
                warnings.append(f"Imbalanced dataset: min={min_count}, max={max_count}")
                
            # Check minimum samples per gesture
            min_samples = 50  # Minimum samples per gesture
            for gesture, count in zip(unique_gestures, counts):
                if count < min_samples:
                    warnings.append(f"Insufficient samples for {gesture}: {count} < {min_samples}")
                    
            # Check feature variance
            feature_vars = np.var(X, axis=0)
            low_var_features = np.sum(feature_vars < 1e-6)
            if low_var_features > 0:
                warnings.append(f"{low_var_features} features have very low variance")
                
        except Exception as e:
            issues.append(f"Error validating data: {e}")
            
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
        
    def export_data_summary(self, output_path: str) -> None:
        """Export comprehensive data summary."""
        stats = self.get_data_statistics()
        quality = self.validate_data_quality()
        
        summary = {
            'data_statistics': stats,
            'quality_validation': quality,
            'processing_info': {
                'raw_data_path': str(self.raw_data_path),
                'processed_data_path': str(self.processed_data_path),
                'train_split': self.train_split,
                'val_split': self.val_split,
                'test_split': self.test_split
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"Data summary exported to: {output_path}")
