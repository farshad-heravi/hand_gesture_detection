"""
Hand landmark extraction using MediaPipe.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class HandLandmarks:
    """Container for hand landmark data."""
    landmarks: List[Tuple[float, float, float]]
    handedness: str
    confidence: float
    timestamp: float


class HandLandmarkExtractor:
    """Professional hand landmark extraction using MediaPipe."""
    
    def __init__(
        self,
        mode: str = "image",
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.8,
        min_tracking_confidence: float = 0.9
    ):
        """
        Initialize the hand landmark extractor.
        
        Args:
            mode: MediaPipe mode ("image" or "video")
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.mode = mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe components
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create hands solution
        self.hands = self.mp_hands.Hands(
            static_image_mode=(mode == "image"),
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Hand landmark connections for drawing
        self.hand_connections = self.mp_hands.HAND_CONNECTIONS
        
    def extract_landmarks(self, image: np.ndarray) -> Optional[List[HandLandmarks]]:
        """
        Extract hand landmarks from an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of HandLandmarks objects or None if no hands detected
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(rgb_image)
        
        if not results.multi_hand_landmarks:
            return None
            
        hand_landmarks_list = []
        
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Extract landmark coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y, landmark.z))
                
            # Get handedness if available
            handedness = "unknown"
            confidence = 0.0
            
            if results.multi_handedness:
                handedness_info = results.multi_handedness[idx]
                handedness = handedness_info.classification[0].label
                confidence = handedness_info.classification[0].score
                
            # Create HandLandmarks object
            hand_landmarks_obj = HandLandmarks(
                landmarks=landmarks,
                handedness=handedness,
                confidence=confidence,
                timestamp=0.0  # Will be set by caller if needed
            )
            
            hand_landmarks_list.append(hand_landmarks_obj)
            
        return hand_landmarks_list
        
    def draw_landmarks(
        self, 
        image: np.ndarray, 
        hand_landmarks_list: List[HandLandmarks],
        draw_connections: bool = True
    ) -> np.ndarray:
        """
        Draw hand landmarks on an image.
        
        Args:
            image: Input image
            hand_landmarks_list: List of HandLandmarks objects
            draw_connections: Whether to draw connections between landmarks
            
        Returns:
            Image with drawn landmarks
        """
        annotated_image = image.copy()
        h, w = image.shape[:2]
        
        for hand_landmarks in hand_landmarks_list:
            # Draw landmarks as circles
            for i, (x, y, z) in enumerate(hand_landmarks.landmarks):
                # Convert normalized coordinates to pixel coordinates
                px = int(x * w)
                py = int(y * h)
                
                # Draw landmark point
                cv2.circle(annotated_image, (px, py), 3, (0, 255, 0), -1)
                
                # Draw landmark number
                cv2.putText(annotated_image, str(i), (px + 5, py - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Draw connections if requested
            if draw_connections and hasattr(self, 'hand_connections'):
                for connection in self.hand_connections:
                    start_idx, end_idx = connection
                    if start_idx < len(hand_landmarks.landmarks) and end_idx < len(hand_landmarks.landmarks):
                        start_x, start_y, _ = hand_landmarks.landmarks[start_idx]
                        end_x, end_y, _ = hand_landmarks.landmarks[end_idx]
                        
                        # Convert to pixel coordinates
                        start_px = int(start_x * w)
                        start_py = int(start_y * h)
                        end_px = int(end_x * w)
                        end_py = int(end_y * h)
                        
                        # Draw connection line
                        cv2.line(annotated_image, (start_px, start_py), (end_px, end_py), (0, 255, 0), 1)
            
        return annotated_image
        
    def get_landmark_coordinates(
        self, 
        hand_landmarks: HandLandmarks, 
        landmark_indices: List[int]
    ) -> List[Tuple[float, float, float]]:
        """
        Get specific landmark coordinates by indices.
        
        Args:
            hand_landmarks: HandLandmarks object
            landmark_indices: List of landmark indices to extract
            
        Returns:
            List of (x, y, z) coordinates
        """
        coordinates = []
        for idx in landmark_indices:
            if 0 <= idx < len(hand_landmarks.landmarks):
                coordinates.append(hand_landmarks.landmarks[idx])
            else:
                raise ValueError(f"Invalid landmark index: {idx}")
                
        return coordinates
        
    def get_hand_bounding_box(
        self, 
        hand_landmarks: HandLandmarks, 
        image_shape: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Get bounding box for the hand.
        
        Args:
            hand_landmarks: HandLandmarks object
            image_shape: (height, width) of the image
            
        Returns:
            (x_min, y_min, x_max, y_max) bounding box coordinates
        """
        height, width = image_shape[:2]
        
        x_coords = [lm[0] for lm in hand_landmarks.landmarks]
        y_coords = [lm[1] for lm in hand_landmarks.landmarks]
        
        x_min = int(min(x_coords) * width)
        y_min = int(min(y_coords) * height)
        x_max = int(max(x_coords) * width)
        y_max = int(max(y_coords) * height)
        
        return (x_min, y_min, x_max, y_max)
        
    def crop_hand_roi(
        self, 
        image: np.ndarray, 
        hand_landmarks: HandLandmarks,
        padding: float = 0.2
    ) -> np.ndarray:
        """
        Crop region of interest around the hand.
        
        Args:
            image: Input image
            hand_landmarks: HandLandmarks object
            padding: Padding around the hand (as fraction of bounding box)
            
        Returns:
            Cropped image
        """
        height, width = image.shape[:2]
        
        # Get bounding box
        x_min, y_min, x_max, y_max = self.get_hand_bounding_box(
            hand_landmarks, (height, width)
        )
        
        # Add padding
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        
        padding_x = int(bbox_width * padding)
        padding_y = int(bbox_height * padding)
        
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(width, x_max + padding_x)
        y_max = min(height, y_max + padding_y)
        
        # Crop the image
        cropped = image[y_min:y_max, x_min:x_max]
        
        return cropped
        
    def close(self) -> None:
        """Close the MediaPipe hands solution."""
        if hasattr(self, 'hands'):
            self.hands.close()
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
