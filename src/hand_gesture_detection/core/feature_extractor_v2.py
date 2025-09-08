"""
Enhanced feature extraction from hand landmarks for gesture recognition.
Based on the implementation from https://github.com/farshad-heravi/hand_gesture_detection
with additional handedness support and improved feature engineering.
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
    palm_size: float
    finger_ratios: np.ndarray
    joint_angles: np.ndarray


class FeatureExtractorV2:
    """Enhanced feature extractor with comprehensive hand gesture features."""
    
    def __init__(self, normalize_features: bool = True, include_handedness: bool = True):
        """
        Initialize the enhanced feature extractor.
        
        Args:
            normalize_features: Whether to normalize the feature vector
            include_handedness: Whether to include handedness features
        """
        self.normalize_features = normalize_features
        self.include_handedness = include_handedness
        
        # MediaPipe hand landmark indices (21 landmarks per hand)
        self.landmark_indices = {
            'wrist': 0,
            'thumb_cmc': 1, 'thumb_mcp': 2, 'thumb_ip': 3, 'thumb_tip': 4,
            'index_mcp': 5, 'index_pip': 6, 'index_dip': 7, 'index_tip': 8,
            'middle_mcp': 9, 'middle_pip': 10, 'middle_dip': 11, 'middle_tip': 12,
            'ring_mcp': 13, 'ring_pip': 14, 'ring_dip': 15, 'ring_tip': 16,
            'pinky_mcp': 17, 'pinky_pip': 18, 'pinky_dip': 19, 'pinky_tip': 20
        }
        
        # Define comprehensive distance pairs for better feature extraction
        self.distance_pairs = [
            # Thumb to other fingers
            (self.landmark_indices['thumb_tip'], self.landmark_indices['index_tip']),
            (self.landmark_indices['thumb_tip'], self.landmark_indices['middle_tip']),
            (self.landmark_indices['thumb_tip'], self.landmark_indices['ring_tip']),
            (self.landmark_indices['thumb_tip'], self.landmark_indices['pinky_tip']),
            
            # Wrist to finger tips
            (self.landmark_indices['wrist'], self.landmark_indices['thumb_tip']),
            (self.landmark_indices['wrist'], self.landmark_indices['index_tip']),
            (self.landmark_indices['wrist'], self.landmark_indices['middle_tip']),
            (self.landmark_indices['wrist'], self.landmark_indices['ring_tip']),
            (self.landmark_indices['wrist'], self.landmark_indices['pinky_tip']),
            
            # Inter-finger distances
            (self.landmark_indices['index_tip'], self.landmark_indices['middle_tip']),
            (self.landmark_indices['middle_tip'], self.landmark_indices['ring_tip']),
            (self.landmark_indices['ring_tip'], self.landmark_indices['pinky_tip']),
            
            # Palm width and height
            (self.landmark_indices['index_mcp'], self.landmark_indices['pinky_mcp']),
            (self.landmark_indices['wrist'], self.landmark_indices['middle_mcp']),
        ]
        
        # Define angle triplets for finger joint angles
        self.angle_triplets = [
            # Finger extension angles (wrist -> mcp -> tip)
            (self.landmark_indices['wrist'], self.landmark_indices['thumb_mcp'], self.landmark_indices['thumb_tip']),
            (self.landmark_indices['wrist'], self.landmark_indices['index_mcp'], self.landmark_indices['index_tip']),
            (self.landmark_indices['wrist'], self.landmark_indices['middle_mcp'], self.landmark_indices['middle_tip']),
            (self.landmark_indices['wrist'], self.landmark_indices['ring_mcp'], self.landmark_indices['ring_tip']),
            (self.landmark_indices['wrist'], self.landmark_indices['pinky_mcp'], self.landmark_indices['pinky_tip']),
            
            # Finger joint angles (mcp -> pip -> tip)
            (self.landmark_indices['thumb_mcp'], self.landmark_indices['thumb_ip'], self.landmark_indices['thumb_tip']),
            (self.landmark_indices['index_mcp'], self.landmark_indices['index_pip'], self.landmark_indices['index_tip']),
            (self.landmark_indices['middle_mcp'], self.landmark_indices['middle_pip'], self.landmark_indices['middle_tip']),
            (self.landmark_indices['ring_mcp'], self.landmark_indices['ring_pip'], self.landmark_indices['ring_tip']),
            (self.landmark_indices['pinky_mcp'], self.landmark_indices['pinky_pip'], self.landmark_indices['pinky_tip']),
        ]
        
    def extract_features(self, hand_landmarks: HandLandmarks) -> Optional[HandFeatures]:
        """
        Extract comprehensive features from hand landmarks.
        
        Args:
            hand_landmarks: HandLandmarks object
            
        Returns:
            HandFeatures object or None if extraction fails
        """
        if not hand_landmarks or len(hand_landmarks.landmarks) < 21:
            return None
            
        try:
            landmarks = hand_landmarks.landmarks
            
            # Extract different types of features
            distances = self._extract_distances(landmarks)
            angles = self._extract_angles(landmarks)
            finger_lengths = self._extract_finger_lengths(landmarks)
            hand_orientation = self._extract_hand_orientation(landmarks)
            palm_size = self._extract_palm_size(landmarks)
            finger_ratios = self._extract_finger_ratios(landmarks)
            joint_angles = self._extract_joint_angles(landmarks)
            
            # Extract handedness feature
            handedness_feature = self._extract_handedness(hand_landmarks.handedness)
            
            # Combine all features
            feature_components = [distances, angles, finger_lengths, finger_ratios, joint_angles, [hand_orientation]]
            
            if self.include_handedness:
                feature_components.append(handedness_feature)
                
            feature_vector = np.concatenate(feature_components)
            
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
                confidence=hand_landmarks.confidence,
                palm_size=palm_size,
                finger_ratios=finger_ratios,
                joint_angles=joint_angles
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
        palm_size = self._extract_palm_size(landmarks)
        
        # Extract finger lengths (wrist to tip)
        finger_tips = [
            self.landmark_indices['thumb_tip'],
            self.landmark_indices['index_tip'],
            self.landmark_indices['middle_tip'],
            self.landmark_indices['ring_tip'],
            self.landmark_indices['pinky_tip']
        ]
        
        wrist = np.array(landmarks[self.landmark_indices['wrist']])
        
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
        
    def _extract_palm_size(self, landmarks: List[Tuple[float, float, float]]) -> float:
        """Extract palm size for normalization."""
        wrist = np.array(landmarks[self.landmark_indices['wrist']])
        middle_mcp = np.array(landmarks[self.landmark_indices['middle_mcp']])
        palm_size = np.linalg.norm(wrist - middle_mcp)
        return palm_size
        
    def _extract_finger_ratios(self, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """Extract finger length ratios for better gesture discrimination."""
        ratios = []
        
        # Get finger lengths
        finger_lengths = self._extract_finger_lengths(landmarks)
        
        # Calculate ratios between different fingers
        if len(finger_lengths) >= 4:
            # Index to middle ratio
            ratios.append(finger_lengths[1] / (finger_lengths[2] + 1e-8))
            # Middle to ring ratio
            ratios.append(finger_lengths[2] / (finger_lengths[3] + 1e-8))
            # Ring to pinky ratio
            ratios.append(finger_lengths[3] / (finger_lengths[4] + 1e-8))
            # Thumb to index ratio
            ratios.append(finger_lengths[0] / (finger_lengths[1] + 1e-8))
            
        return np.array(ratios)
        
    def _extract_joint_angles(self, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """Extract joint angles for finger bending detection."""
        joint_angles = []
        
        # Define joint angle triplets (pip -> dip -> tip)
        joint_triplets = [
            (self.landmark_indices['thumb_ip'], self.landmark_indices['thumb_tip']),
            (self.landmark_indices['index_pip'], self.landmark_indices['index_dip'], self.landmark_indices['index_tip']),
            (self.landmark_indices['middle_pip'], self.landmark_indices['middle_dip'], self.landmark_indices['middle_tip']),
            (self.landmark_indices['ring_pip'], self.landmark_indices['ring_dip'], self.landmark_indices['ring_tip']),
            (self.landmark_indices['pinky_pip'], self.landmark_indices['pinky_dip'], self.landmark_indices['pinky_tip']),
        ]
        
        for triplet in joint_triplets:
            if len(triplet) == 3:
                p1, p2, p3 = triplet
                point1 = np.array(landmarks[p1])
                point2 = np.array(landmarks[p2])
                point3 = np.array(landmarks[p3])
                
                # Calculate angle
                v1 = point1 - point2
                v2 = point3 - point2
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                joint_angles.append(angle)
            elif len(triplet) == 2:
                # Special case for thumb (only 2 points)
                p1, p2 = triplet
                point1 = np.array(landmarks[p1])
                point2 = np.array(landmarks[p2])
                
                # Use wrist as reference point for thumb
                wrist = np.array(landmarks[self.landmark_indices['wrist']])
                v1 = wrist - point1
                v2 = point2 - point1
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                joint_angles.append(angle)
                
        return np.array(joint_angles)
        
    def _extract_handedness(self, handedness: str) -> np.ndarray:
        """Extract handedness as one-hot encoded feature."""
        if not self.include_handedness:
            return np.array([])
            
        # One-hot encoding: [is_left, is_right, is_unknown]
        if handedness.lower() == 'left':
            return np.array([1.0, 0.0, 0.0])
        elif handedness.lower() == 'right':
            return np.array([0.0, 1.0, 0.0])
        else:  # unknown or other
            return np.array([0.0, 0.0, 1.0])
        
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize feature vector using L2 normalization."""
        norm = np.linalg.norm(features) + 1e-8
        return features / norm
        
    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        names = []
        
        # Distance feature names
        distance_names = [
            "thumb_to_index", "thumb_to_middle", "thumb_to_ring", "thumb_to_pinky",
            "wrist_to_thumb", "wrist_to_index", "wrist_to_middle", "wrist_to_ring", "wrist_to_pinky",
            "index_to_middle", "middle_to_ring", "ring_to_pinky",
            "palm_width", "palm_height"
        ]
        names.extend(distance_names)
            
        # Angle feature names
        angle_names = [
            "thumb_extension", "index_extension", "middle_extension", "ring_extension", "pinky_extension",
            "thumb_joint", "index_joint", "middle_joint", "ring_joint", "pinky_joint"
        ]
        names.extend(angle_names)
            
        # Finger length names
        finger_names = ['thumb_length', 'index_length', 'middle_length', 'ring_length', 'pinky_length']
        names.extend(finger_names)
        
        # Finger ratio names
        ratio_names = ['index_middle_ratio', 'middle_ring_ratio', 'ring_pinky_ratio', 'thumb_index_ratio']
        names.extend(ratio_names)
        
        # Joint angle names
        joint_names = ['thumb_joint_angle', 'index_joint_angle', 'middle_joint_angle', 'ring_joint_angle', 'pinky_joint_angle']
        names.extend(joint_names)
            
        # Hand orientation
        names.append("hand_orientation")
        
        # Handedness features
        if self.include_handedness:
            names.extend(["is_left_hand", "is_right_hand", "is_unknown_hand"])
        
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
            'count': len(features_list),
            'feature_names': self.get_feature_names()
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
        expected_length = (len(self.distance_pairs) + len(self.angle_triplets) + 
                          5 + 4 + 5 + 1)  # distances + angles + finger_lengths + ratios + joint_angles + orientation
        if self.include_handedness:
            expected_length += 3  # handedness features
            
        if len(features.feature_vector) != expected_length:
            return False
            
        return True
        
    def get_feature_vector_size(self) -> int:
        """Get the expected size of the feature vector."""
        expected_length = (len(self.distance_pairs) + len(self.angle_triplets) + 
                          5 + 4 + 5 + 1)  # distances + angles + finger_lengths + ratios + joint_angles + orientation
        if self.include_handedness:
            expected_length += 3  # handedness features
        return expected_length
