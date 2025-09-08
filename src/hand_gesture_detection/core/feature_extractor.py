"""
Feature extraction from hand landmarks for gesture recognition.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from .landmark_extractor import HandLandmarks


@dataclass
class HandFeatures:
    """Container for extracted hand features."""
    distances: np.ndarray
    angles: np.ndarray
    finger_lengths: np.ndarray
    hand_orientation: float
    handedness: np.ndarray
    feature_vector: np.ndarray
    confidence: float


class FeatureExtractor:
    """Extract meaningful features from hand landmarks for gesture recognition."""
    
    def __init__(self, normalize_features: bool = True):
        """
        Initialize the feature extractor.
        
        Args:
            normalize_features: Whether to normalize the feature vector
        """
        self.normalize_features = normalize_features
        
        # Define key landmark indices for feature extraction
        self.landmark_indices = {
            'wrist': 0,
            'thumb_tip': 4,
            'thumb_ip': 3,
            'thumb_mcp': 2,
            'index_tip': 8,
            'index_pip': 6,
            'index_mcp': 5,
            'middle_tip': 12,
            'middle_pip': 10,
            'middle_mcp': 9,
            'ring_tip': 16,
            'ring_pip': 14,
            'ring_mcp': 13,
            'pinky_tip': 20,
            'pinky_pip': 18,
            'pinky_mcp': 17
        }
        
        # Define feature extraction pairs
        self.distance_pairs = [
            (self.landmark_indices['thumb_tip'], self.landmark_indices['index_tip']),
            (self.landmark_indices['thumb_tip'], self.landmark_indices['middle_tip']),
            (self.landmark_indices['thumb_tip'], self.landmark_indices['ring_tip']),
            (self.landmark_indices['thumb_tip'], self.landmark_indices['pinky_tip']),
            (self.landmark_indices['wrist'], self.landmark_indices['thumb_tip']),
            (self.landmark_indices['wrist'], self.landmark_indices['index_tip']),
            (self.landmark_indices['wrist'], self.landmark_indices['middle_tip']),
            (self.landmark_indices['wrist'], self.landmark_indices['ring_tip']),
            (self.landmark_indices['wrist'], self.landmark_indices['pinky_tip'])
        ]
        
        self.angle_triplets = [
            (self.landmark_indices['wrist'], self.landmark_indices['index_mcp'], self.landmark_indices['index_tip']),
            (self.landmark_indices['wrist'], self.landmark_indices['middle_mcp'], self.landmark_indices['middle_tip']),
            (self.landmark_indices['wrist'], self.landmark_indices['ring_mcp'], self.landmark_indices['ring_tip']),
            (self.landmark_indices['wrist'], self.landmark_indices['pinky_mcp'], self.landmark_indices['pinky_tip'])
        ]
        
    def extract_features(self, hand_landmarks: HandLandmarks) -> Optional[HandFeatures]:
        """
        Extract features from hand landmarks.
        
        Args:
            hand_landmarks: HandLandmarks object
            
        Returns:
            HandFeatures object or None if extraction fails
        """
        if not hand_landmarks or len(hand_landmarks.landmarks) < 21:
            return None
            
        try:
            # Extract different types of features
            distances = self._extract_distances(hand_landmarks.landmarks)
            angles = self._extract_angles(hand_landmarks.landmarks)
            finger_lengths = self._extract_finger_lengths(hand_landmarks.landmarks)
            hand_orientation = self._extract_hand_orientation(hand_landmarks.landmarks)
            
            # Extract handedness feature
            handedness_feature = self._extract_handedness(hand_landmarks.handedness)
            
            # Combine all features (excluding handedness to match training data)
            feature_vector = np.concatenate([distances, angles, finger_lengths, [hand_orientation]])
            
            # Normalize if requested
            if self.normalize_features:
                feature_vector = self._normalize_features(feature_vector)
                
            return HandFeatures(
                distances=distances,
                angles=angles,
                finger_lengths=finger_lengths,
                hand_orientation=hand_orientation,
                handedness=handedness_feature,
                feature_vector=feature_vector,
                confidence=hand_landmarks.confidence
            )
            
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            return None
            
    def _extract_distances(self, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """Extract distance features between key landmarks."""
        distances = []
        
        for p1_idx, p2_idx in self.distance_pairs:
            p1 = np.array(landmarks[p1_idx])
            p2 = np.array(landmarks[p2_idx])
            distance = np.linalg.norm(p1 - p2)
            distances.append(distance)
            
        return np.array(distances)
        
    def _extract_angles(self, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """Extract angle features between finger joints."""
        angles = []
        
        for p1_idx, p2_idx, p3_idx in self.angle_triplets:
            p1 = np.array(landmarks[p1_idx])
            p2 = np.array(landmarks[p2_idx])
            p3 = np.array(landmarks[p3_idx])
            
            # Calculate angle between vectors p1-p2 and p3-p2
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate angle in radians
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
            angle = np.arccos(cos_angle)
            
            angles.append(angle)
            
        return np.array(angles)
        
    def _extract_finger_lengths(self, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """Extract normalized finger lengths."""
        finger_lengths = []
        
        # Calculate palm size for normalization
        wrist = np.array(landmarks[self.landmark_indices['wrist']])
        middle_mcp = np.array(landmarks[self.landmark_indices['middle_mcp']])
        palm_size = np.linalg.norm(wrist - middle_mcp)
        
        # Extract finger lengths (wrist to tip)
        finger_tips = [
            self.landmark_indices['index_tip'],
            self.landmark_indices['middle_tip'],
            self.landmark_indices['ring_tip'],
            self.landmark_indices['pinky_tip']
        ]
        
        for tip_idx in finger_tips:
            tip = np.array(landmarks[tip_idx])
            length = np.linalg.norm(wrist - tip)
            normalized_length = length / (palm_size + 1e-8)
            finger_lengths.append(normalized_length)
            
        return np.array(finger_lengths)
        
    def _extract_hand_orientation(self, landmarks: List[Tuple[float, float, float]]) -> float:
        """Extract hand orientation angle."""
        wrist = np.array(landmarks[self.landmark_indices['wrist']])
        middle_mcp = np.array(landmarks[self.landmark_indices['middle_mcp']])
        
        # Calculate hand vector
        hand_vector = middle_mcp - wrist
        
        # Calculate angle in degrees
        angle_rad = np.arctan2(hand_vector[1], hand_vector[0])
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
        
    def _extract_handedness(self, handedness: str) -> np.ndarray:
        """Extract handedness as one-hot encoded feature."""
        # One-hot encoding: [is_left, is_right, is_unknown]
        if handedness.lower() == 'left':
            return np.array([1.0, 0.0, 0.0])
        elif handedness.lower() == 'right':
            return np.array([0.0, 1.0, 0.0])
        else:  # unknown or other
            return np.array([0.0, 0.0, 1.0])
        
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize feature vector."""
        # L2 normalization
        norm = np.linalg.norm(features) + 1e-8
        return features / norm
        
    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        names = []
        
        # Distance feature names
        for i, (p1_idx, p2_idx) in enumerate(self.distance_pairs):
            p1_name = list(self.landmark_indices.keys())[list(self.landmark_indices.values()).index(p1_idx)]
            p2_name = list(self.landmark_indices.keys())[list(self.landmark_indices.values()).index(p2_idx)]
            names.append(f"distance_{p1_name}_{p2_name}")
            
        # Angle feature names
        for i, (p1_idx, p2_idx, p3_idx) in enumerate(self.angle_triplets):
            p2_name = list(self.landmark_indices.keys())[list(self.landmark_indices.values()).index(p2_idx)]
            names.append(f"angle_{p2_name}_finger")
            
        # Finger length names
        finger_names = ['index', 'middle', 'ring', 'pinky']
        for name in finger_names:
            names.append(f"finger_length_{name}")
            
        # Hand orientation
        names.append("hand_orientation")
        
        return names
        
    def get_feature_statistics(self, features_list: List[HandFeatures]) -> Dict[str, Any]:
        """Calculate statistics for a list of features."""
        if not features_list:
            return {}
            
        feature_vectors = np.array([f.feature_vector for f in features_list])
        
        stats = {
            'mean': np.mean(feature_vectors, axis=0),
            'std': np.std(feature_vectors, axis=0),
            'min': np.min(feature_vectors, axis=0),
            'max': np.max(feature_vectors, axis=0),
            'count': len(features_list)
        }
        
        return stats
        
    def validate_features(self, features: HandFeatures) -> bool:
        """Validate extracted features."""
        if not features:
            return False
            
        # Check for NaN or infinite values
        if np.any(np.isnan(features.feature_vector)) or np.any(np.isinf(features.feature_vector)):
            return False
            
        # Check feature vector length
        expected_length = len(self.distance_pairs) + len(self.angle_triplets) + 4 + 1  # distances + angles + finger_lengths + orientation
        if len(features.feature_vector) != expected_length:
            return False
            
        return True
