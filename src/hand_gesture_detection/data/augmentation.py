"""
Data augmentation pipeline for hand gesture detection.
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
import random
from dataclasses import dataclass

from ..core.feature_extractor_v2 import HandFeatures
from ..utils.logger import Logger


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    rotation_range: float = 15.0
    scale_range: float = 0.1
    translation_range: float = 0.1
    brightness_range: float = 0.2
    contrast_range: float = 0.2
    noise_std: float = 0.01
    blur_kernel_size: int = 3
    landmark_noise_std: float = 0.01
    temporal_jitter_range: float = 0.05
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0


class AugmentationPipeline:
    """Professional data augmentation pipeline for hand gesture data."""
    
    def __init__(
        self,
        config: Optional[AugmentationConfig] = None,
        logger: Optional[Logger] = None
    ):
        """
        Initialize the augmentation pipeline.
        
        Args:
            config: Augmentation configuration
            logger: Logger instance
        """
        self.config = config or AugmentationConfig()
        self.logger = logger or Logger("augmentation")
        
        # Augmentation methods
        self.geometric_methods = [
            self._apply_rotation,
            self._apply_scaling,
            self._apply_translation
        ]
        
        self.photometric_methods = [
            self._apply_brightness,
            self._apply_contrast,
            self._apply_noise,
            self._apply_blur
        ]
        
        self.feature_methods = [
            self._apply_landmark_noise,
            self._apply_temporal_jitter
        ]
        
    def augment_features(
        self,
        features: HandFeatures,
        num_augmentations: int = 1,
        preserve_original: bool = True
    ) -> List[HandFeatures]:
        """
        Augment hand features.
        
        Args:
            features: Original features
            num_augmentations: Number of augmentations to generate
            preserve_original: Whether to include original in output
            
        Returns:
            List of augmented features
        """
        augmented_features = []
        
        if preserve_original:
            augmented_features.append(features)
            
        for _ in range(num_augmentations):
            augmented_feature = self._augment_single_feature(features)
            if augmented_feature:
                augmented_features.append(augmented_feature)
                
        return augmented_features
        
    def _augment_single_feature(self, features: HandFeatures) -> Optional[HandFeatures]:
        """Apply random augmentations to a single feature."""
        try:
            # Start with original feature vector
            augmented_vector = features.feature_vector.copy()
            
            # Apply feature-level augmentations
            if random.random() < 0.3:  # 30% chance
                augmented_vector = self._apply_landmark_noise(augmented_vector)
                
            if random.random() < 0.2:  # 20% chance
                augmented_vector = self._apply_temporal_jitter(augmented_vector)
                
            # Create new HandFeatures object
            augmented_features = HandFeatures(
                distances=features.distances.copy(),
                angles=features.angles.copy(),
                finger_lengths=features.finger_lengths.copy(),
                hand_orientation=features.hand_orientation,
                handedness=features.handedness.copy(),
                feature_vector=augmented_vector,
                confidence=features.confidence,
                palm_size=features.palm_size,
                finger_ratios=features.finger_ratios.copy(),
                joint_angles=features.joint_angles.copy()
            )
            
            return augmented_features
            
        except Exception as e:
            self.logger.error(f"Feature augmentation failed: {e}")
            return None
            
    def augment_image(
        self,
        image: np.ndarray,
        landmarks: Optional[List[Tuple[float, float, float]]] = None,
        num_augmentations: int = 1
    ) -> List[Tuple[np.ndarray, Optional[List[Tuple[float, float, float]]]]]:
        """
        Augment image and corresponding landmarks.
        
        Args:
            image: Input image
            landmarks: Hand landmarks (optional)
            num_augmentations: Number of augmentations to generate
            
        Returns:
            List of (augmented_image, augmented_landmarks) tuples
        """
        augmented_samples = []
        
        for _ in range(num_augmentations):
            augmented_image = image.copy()
            augmented_landmarks = landmarks.copy() if landmarks else None
            
            # Apply geometric augmentations
            if random.random() < 0.5:  # 50% chance
                augmented_image, augmented_landmarks = self._apply_geometric_augmentation(
                    augmented_image, augmented_landmarks
                )
                
            # Apply photometric augmentations
            if random.random() < 0.5:  # 50% chance
                augmented_image = self._apply_photometric_augmentation(augmented_image)
                
            augmented_samples.append((augmented_image, augmented_landmarks))
            
        return augmented_samples
        
    def _apply_geometric_augmentation(
        self,
        image: np.ndarray,
        landmarks: Optional[List[Tuple[float, float, float]]]
    ) -> Tuple[np.ndarray, Optional[List[Tuple[float, float, float]]]]:
        """Apply geometric augmentations."""
        h, w = image.shape[:2]
        
        # Random rotation
        if random.random() < 0.3:
            angle = random.uniform(-self.config.rotation_range, self.config.rotation_range)
            image, landmarks = self._apply_rotation(image, landmarks, angle)
            
        # Random scaling
        if random.random() < 0.3:
            scale = random.uniform(1 - self.config.scale_range, 1 + self.config.scale_range)
            image, landmarks = self._apply_scaling(image, landmarks, scale)
            
        # Random translation
        if random.random() < 0.3:
            tx = random.uniform(-self.config.translation_range, self.config.translation_range) * w
            ty = random.uniform(-self.config.translation_range, self.config.translation_range) * h
            image, landmarks = self._apply_translation(image, landmarks, tx, ty)
            
        return image, landmarks
        
    def _apply_photometric_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply photometric augmentations."""
        # Random brightness
        if random.random() < 0.3:
            image = self._apply_brightness(image)
            
        # Random contrast
        if random.random() < 0.3:
            image = self._apply_contrast(image)
            
        # Random noise
        if random.random() < 0.2:
            image = self._apply_noise(image)
            
        # Random blur
        if random.random() < 0.1:
            image = self._apply_blur(image)
            
        return image
        
    def _apply_rotation(
        self,
        image: np.ndarray,
        landmarks: Optional[List[Tuple[float, float, float]]],
        angle: float
    ) -> Tuple[np.ndarray, Optional[List[Tuple[float, float, float]]]]:
        """Apply rotation to image and landmarks."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Rotate image
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        
        # Rotate landmarks
        rotated_landmarks = None
        if landmarks:
            rotated_landmarks = []
            for x, y, z in landmarks:
                # Convert to pixel coordinates
                px, py = int(x * w), int(y * h)
                
                # Apply rotation
                rotated_point = np.dot(rotation_matrix, [px, py, 1])
                rx, ry = rotated_point[0] / w, rotated_point[1] / h
                
                rotated_landmarks.append((rx, ry, z))
                
        return rotated_image, rotated_landmarks
        
    def _apply_scaling(
        self,
        image: np.ndarray,
        landmarks: Optional[List[Tuple[float, float, float]]],
        scale: float
    ) -> Tuple[np.ndarray, Optional[List[Tuple[float, float, float]]]]:
        """Apply scaling to image and landmarks."""
        h, w = image.shape[:2]
        
        # Scale image
        new_h, new_w = int(h * scale), int(w * scale)
        scaled_image = cv2.resize(image, (new_w, new_h))
        
        # Resize back to original size
        if scale != 1.0:
            scaled_image = cv2.resize(scaled_image, (w, h))
            
        # Scale landmarks (no change needed for normalized coordinates)
        return scaled_image, landmarks
        
    def _apply_translation(
        self,
        image: np.ndarray,
        landmarks: Optional[List[Tuple[float, float, float]]],
        tx: float,
        ty: float
    ) -> Tuple[np.ndarray, Optional[List[Tuple[float, float, float]]]]:
        """Apply translation to image and landmarks."""
        h, w = image.shape[:2]
        
        # Translate image
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_image = cv2.warpAffine(image, translation_matrix, (w, h))
        
        # Translate landmarks
        translated_landmarks = None
        if landmarks:
            translated_landmarks = []
            for x, y, z in landmarks:
                # Convert to pixel coordinates, translate, convert back
                px, py = x * w + tx, y * h + ty
                nx, ny = px / w, py / h
                translated_landmarks.append((nx, ny, z))
                
        return translated_image, translated_landmarks
        
    def _apply_brightness(self, image: np.ndarray) -> np.ndarray:
        """Apply brightness adjustment."""
        brightness_factor = random.uniform(
            1 - self.config.brightness_range,
            1 + self.config.brightness_range
        )
        
        # Convert to float, apply brightness, convert back
        bright_image = image.astype(np.float32) * brightness_factor
        bright_image = np.clip(bright_image, 0, 255).astype(np.uint8)
        
        return bright_image
        
    def _apply_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply contrast adjustment."""
        contrast_factor = random.uniform(
            1 - self.config.contrast_range,
            1 + self.config.contrast_range
        )
        
        # Apply contrast
        contrasted_image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
        
        return contrasted_image
        
    def _apply_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian noise."""
        noise = np.random.normal(0, self.config.noise_std * 255, image.shape)
        noisy_image = image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return noisy_image
        
    def _apply_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur."""
        kernel_size = random.choice([3, 5, 7])
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        return blurred_image
        
    def _apply_landmark_noise(self, feature_vector: np.ndarray) -> np.ndarray:
        """Apply noise to landmark-based features."""
        noise = np.random.normal(0, self.config.landmark_noise_std, feature_vector.shape)
        noisy_features = feature_vector + noise
        
        # Ensure features remain in valid range
        noisy_features = np.clip(noisy_features, -1, 1)
        
        return noisy_features
        
    def _apply_temporal_jitter(self, feature_vector: np.ndarray) -> np.ndarray:
        """Apply temporal jitter to features."""
        jitter = np.random.uniform(
            -self.config.temporal_jitter_range,
            self.config.temporal_jitter_range,
            feature_vector.shape
        )
        
        jittered_features = feature_vector + jitter
        
        # Ensure features remain in valid range
        jittered_features = np.clip(jittered_features, -1, 1)
        
        return jittered_features
        
    def mixup(
        self,
        features1: HandFeatures,
        features2: HandFeatures,
        alpha: Optional[float] = None
    ) -> HandFeatures:
        """
        Apply mixup augmentation between two feature sets.
        
        Args:
            features1: First feature set
            features2: Second feature set
            alpha: Mixup parameter (optional)
            
        Returns:
            Mixed features
        """
        if alpha is None:
            alpha = self.config.mixup_alpha
            
        # Generate mixing weight
        lam = np.random.beta(alpha, alpha)
        
        # Mix feature vectors
        mixed_vector = lam * features1.feature_vector + (1 - lam) * features2.feature_vector
        
        # Mix other components
        mixed_distances = lam * features1.distances + (1 - lam) * features2.distances
        mixed_angles = lam * features1.angles + (1 - lam) * features2.angles
        mixed_finger_lengths = lam * features1.finger_lengths + (1 - lam) * features2.finger_lengths
        mixed_orientation = lam * features1.hand_orientation + (1 - lam) * features2.hand_orientation
        mixed_handedness = lam * features1.handedness + (1 - lam) * features2.handedness
        mixed_palm_size = lam * features1.palm_size + (1 - lam) * features2.palm_size
        mixed_finger_ratios = lam * features1.finger_ratios + (1 - lam) * features2.finger_ratios
        mixed_joint_angles = lam * features1.joint_angles + (1 - lam) * features2.joint_angles
        mixed_confidence = lam * features1.confidence + (1 - lam) * features2.confidence
        
        return HandFeatures(
            distances=mixed_distances,
            angles=mixed_angles,
            finger_lengths=mixed_finger_lengths,
            hand_orientation=mixed_orientation,
            handedness=mixed_handedness,
            feature_vector=mixed_vector,
            confidence=mixed_confidence,
            palm_size=mixed_palm_size,
            finger_ratios=mixed_finger_ratios,
            joint_angles=mixed_joint_angles
        )
        
    def get_augmentation_statistics(self, original_count: int, augmented_count: int) -> Dict[str, Any]:
        """Get augmentation statistics."""
        return {
            'original_samples': original_count,
            'augmented_samples': augmented_count,
            'total_samples': original_count + augmented_count,
            'augmentation_ratio': augmented_count / original_count if original_count > 0 else 0,
            'augmentation_factor': (original_count + augmented_count) / original_count if original_count > 0 else 1
        }
