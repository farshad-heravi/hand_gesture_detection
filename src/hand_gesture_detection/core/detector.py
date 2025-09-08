"""
Main hand gesture detector for real-time inference.
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import time

from .landmark_extractor import HandLandmarkExtractor, HandLandmarks
from .feature_extractor_v2 import FeatureExtractorV2, HandFeatures
from ..models.base_model import BaseModel
from ..utils.logger import Logger


# MediaPipe hand landmark connections for drawing
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17)
]


def draw_landmarks(frame: np.ndarray, landmarks: HandLandmarks, 
                  show_connections: bool = True, 
                  landmark_color: Tuple[int, int, int] = (0, 255, 0),
                  connection_color: Tuple[int, int, int] = (255, 0, 0),
                  landmark_radius: int = 3,
                  connection_thickness: int = 2) -> np.ndarray:
    """
    Draw hand landmarks and connections on frame.
    
    Args:
        frame: Input frame
        landmarks: HandLandmarks object
        show_connections: Whether to draw connections between landmarks
        landmark_color: Color for landmarks (BGR)
        connection_color: Color for connections (BGR)
        landmark_radius: Radius of landmark circles
        connection_thickness: Thickness of connection lines
        
    Returns:
        Frame with landmarks drawn
    """
    if not landmarks or len(landmarks.landmarks) < 21:
        return frame
        
    h, w = frame.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    pixel_landmarks = []
    for landmark in landmarks.landmarks:
        x = int(landmark[0] * w)
        y = int(landmark[1] * h)
        pixel_landmarks.append((x, y))
    
    # Draw connections
    if show_connections:
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(pixel_landmarks) and end_idx < len(pixel_landmarks):
                start_point = pixel_landmarks[start_idx]
                end_point = pixel_landmarks[end_idx]
                cv2.line(frame, start_point, end_point, connection_color, connection_thickness)
    
    # Draw landmarks
    for i, (x, y) in enumerate(pixel_landmarks):
        # Use different colors for different landmark types
        if i == 0:  # Wrist
            color = (0, 0, 255)  # Red
        elif i in [4, 8, 12, 16, 20]:  # Fingertips
            color = (0, 255, 255)  # Yellow
        else:  # Other landmarks
            color = landmark_color
            
        cv2.circle(frame, (x, y), landmark_radius, color, -1)
        
        # Add landmark index for debugging (optional)
        # cv2.putText(frame, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    return frame


def draw_hand_info(frame: np.ndarray, landmarks: HandLandmarks, 
                  confidence: float, handedness: str,
                  position: Tuple[int, int] = (10, 100)) -> np.ndarray:
    """
    Draw hand information (confidence, handedness) on frame.
    
    Args:
        frame: Input frame
        landmarks: HandLandmarks object
        confidence: Detection confidence
        handedness: Hand handedness
        position: Position to draw text (x, y)
        
    Returns:
        Frame with hand info drawn
    """
    x, y = position
    
    # Draw confidence
    conf_text = f"Confidence: {confidence:.3f}"
    cv2.putText(frame, conf_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw handedness
    hand_text = f"Hand: {handedness}"
    cv2.putText(frame, hand_text, (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame


@dataclass
class GesturePrediction:
    """Container for gesture prediction results."""
    gesture_name: str
    confidence: float
    class_id: int
    timestamp: float
    features: Optional[HandFeatures] = None
    landmarks: Optional[HandLandmarks] = None


@dataclass
class DetectionStats:
    """Container for detection statistics."""
    fps: float
    avg_confidence: float
    total_predictions: int
    successful_detections: int
    failed_detections: int


class HandGestureDetector:
    """Professional hand gesture detector for real-time inference."""
    
    def __init__(
        self,
        model: BaseModel,
        config: Dict[str, Any],
        gesture_mapping: Dict[int, str],
        logger: Optional[Logger] = None,
        scaler: Optional[Any] = None,
        print_instructions: bool = True
    ):
        """
        Initialize the hand gesture detector.
        
        Args:
            model: Trained model for gesture classification
            config: Configuration dictionary
            gesture_mapping: Mapping from class IDs to gesture names
            logger: Logger instance
            scaler: Optional scaler for feature normalization
            print_instructions: Whether to print instructions
        """
        self.model = model
        self.config = config
        self.gesture_mapping = gesture_mapping
        self.scaler = scaler
        self.logger = logger or Logger("gesture_detector")
        self.print_instructions = print_instructions
        
        # Initialize components
        self.landmark_extractor = HandLandmarkExtractor(
            mode=config['mediapipe']['mode'],
            max_num_hands=config['mediapipe']['max_num_hands'],
            min_detection_confidence=config['mediapipe']['min_detection_confidence'],
            min_tracking_confidence=config['mediapipe']['min_tracking_confidence']
        )
        
        # Use feature extractor that matches training data (39 features, no handedness)
        self.feature_extractor = FeatureExtractorV2(normalize_features=True, include_handedness=False)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Performance tracking
        self.stats = DetectionStats(
            fps=0.0,
            avg_confidence=0.0,
            total_predictions=0,
            successful_detections=0,
            failed_detections=0
        )
        
        # Temporal smoothing
        self.confidence_threshold = config['processing']['confidence_threshold']
        self.smoothing_window = config['processing']['smoothing_window']
        self.temporal_consistency = config['processing']['temporal_consistency']
        
        # Prediction history for smoothing
        self.prediction_history = deque(maxlen=self.smoothing_window)
        self.confidence_history = deque(maxlen=self.smoothing_window)
        
        # Performance monitoring
        self.frame_times = deque(maxlen=30)  # Last 30 frames
        self.last_frame_time = time.time()
        
    def detect_gesture(self, frame: np.ndarray) -> Optional[GesturePrediction]:
        """
        Detect gesture in a single frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            GesturePrediction object or None if no gesture detected
        """
        start_time = time.time()
        
        try:
            # Extract hand landmarks
            landmarks_list = self.landmark_extractor.extract_landmarks(frame)
            
            if not landmarks_list:
                self.stats.failed_detections += 1
                return None
                
            # Use first detected hand
            landmarks = landmarks_list[0]
            
            # Extract features
            features = self.feature_extractor.extract_features(landmarks)
            
            if not features:
                self.stats.failed_detections += 1
                return None
                
            # Make prediction
            prediction = self._predict_gesture(features, landmarks)
            
            if prediction:
                # Apply temporal smoothing
                if self.temporal_consistency:
                    prediction = self._apply_temporal_smoothing(prediction)
                    
                # Update statistics
                self._update_stats(prediction, start_time)
                
            return prediction
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            self.stats.failed_detections += 1
            return None
            
    def _predict_gesture(self, features: HandFeatures, landmarks: HandLandmarks) -> Optional[GesturePrediction]:
        """Make gesture prediction from features."""
        try:
            # Scale features if scaler is available
            feature_vector = features.feature_vector
            if self.scaler is not None:
                feature_vector = self.scaler.transform(feature_vector.reshape(1, -1)).flatten()
                self.logger.debug("Applied feature scaling")
            
            # Convert features to tensor
            feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
            
            # Move to device if model is on GPU
            if next(self.model.parameters()).is_cuda:
                feature_tensor = feature_tensor.cuda()
                
            # Make prediction
            with torch.no_grad():
                outputs = self.model(feature_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get prediction
                confidence, predicted_class = torch.max(probabilities, 1)
                confidence = confidence.item()
                predicted_class = predicted_class.item()
                
                # Check confidence threshold
                if confidence < self.confidence_threshold:
                    return None
                    
                # Get gesture name
                gesture_name = self.gesture_mapping.get(predicted_class, f"Unknown_{predicted_class}")
                
                return GesturePrediction(
                    gesture_name=gesture_name,
                    confidence=confidence,
                    class_id=predicted_class,
                    timestamp=time.time(),
                    features=features,
                    landmarks=landmarks
                )
                
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return None
            
    def _apply_temporal_smoothing(self, prediction: GesturePrediction) -> GesturePrediction:
        """Apply temporal smoothing to prediction."""
        # Add to history
        self.prediction_history.append(prediction.class_id)
        self.confidence_history.append(prediction.confidence)
        
        if len(self.prediction_history) < self.smoothing_window:
            return prediction
            
        # Calculate smoothed prediction
        # Use most common class in window
        from collections import Counter
        class_counts = Counter(self.prediction_history)
        most_common_class = class_counts.most_common(1)[0][0]
        
        # Calculate average confidence for the most common class
        class_confidences = [
            conf for conf, cls in zip(self.confidence_history, self.prediction_history)
            if cls == most_common_class
        ]
        avg_confidence = np.mean(class_confidences) if class_confidences else prediction.confidence
        
        # Update prediction
        prediction.class_id = most_common_class
        prediction.gesture_name = self.gesture_mapping.get(most_common_class, f"Unknown_{most_common_class}")
        prediction.confidence = avg_confidence
        
        return prediction
        
    def _update_stats(self, prediction: GesturePrediction, start_time: float) -> None:
        """Update detection statistics."""
        # Update frame timing
        current_time = time.time()
        frame_time = current_time - start_time
        self.frame_times.append(frame_time)
        
        # Calculate FPS
        if len(self.frame_times) > 1:
            self.stats.fps = 1.0 / np.mean(self.frame_times)
            
        # Update prediction counts
        self.stats.total_predictions += 1
        self.stats.successful_detections += 1
        
        # Update average confidence
        if self.stats.total_predictions == 1:
            self.stats.avg_confidence = prediction.confidence
        else:
            # Running average
            alpha = 0.1  # Smoothing factor
            self.stats.avg_confidence = (
                alpha * prediction.confidence + 
                (1 - alpha) * self.stats.avg_confidence
            )
            
    def run_realtime_detection(self, camera_id: int = 0) -> None:
        """Run real-time gesture detection from camera."""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")
            
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
        cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
        
        self.logger.info("Starting real-time gesture detection")
        self.logger.info("Press 'q' to quit, 's' to save video, 'r' to reset stats")
        
        # Video recording
        video_writer = None
        recording = False
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect gesture
                prediction = self.detect_gesture(frame)
                
                # Draw results
                frame = self._draw_detection_results(frame, prediction)
                
                # Record video if enabled
                if recording and video_writer:
                    video_writer.write(frame)
                    
                # Display frame
                cv2.imshow("Hand Gesture Detection", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if not recording:
                        recording = True
                        video_writer = self._start_video_recording(frame.shape)
                        self.logger.info("Started video recording")
                    else:
                        recording = False
                        if video_writer:
                            video_writer.release()
                            video_writer = None
                        self.logger.info("Stopped video recording")
                elif key == ord('r'):
                    self._reset_stats()
                    self.logger.info("Reset detection statistics")
                    
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            
    def _draw_detection_results(
        self, 
        frame: np.ndarray, 
        prediction: Optional[GesturePrediction]
    ) -> np.ndarray:
        """Draw detection results on frame."""
        h, w = frame.shape[:2]
        
        # Draw landmarks if enabled and we have a prediction with landmarks
        if (self.config['display']['show_landmarks'] and 
            prediction and prediction.landmarks):
            
            # Draw landmarks and connections
            display_config = self.config.get('display', {})
            frame = draw_landmarks(
                frame, 
                prediction.landmarks,
                show_connections=True,
                landmark_color=tuple(display_config.get('landmark_color', [0, 255, 0])),
                connection_color=tuple(display_config.get('connection_color', [255, 0, 0])),
                landmark_radius=display_config.get('landmark_radius', 3),
                connection_thickness=display_config.get('connection_thickness', 2)
            )
            
            # Draw hand info if confidence is enabled
            if self.config['display']['show_confidence']:
                frame = draw_hand_info(
                    frame,
                    prediction.landmarks,
                    prediction.confidence,
                    prediction.landmarks.handedness,
                    position=(10, 100)
                )
        
        if prediction:
            # Draw prediction
            if self.print_instructions:
                text = f"{prediction.gesture_name}: {prediction.confidence:.3f}"
            else:
                text = f"{prediction.gesture_name}"
            color = (30, 255, 30) if prediction.confidence > 0.8 else (0, 255, 255)
            
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Draw confidence bar
            bar_width = 200
            bar_height = 20
            bar_x = 10
            bar_y = 60
            
            # Background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            
            # Confidence fill
            fill_width = int(bar_width * prediction.confidence)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
            
        else:
            # No detection
            cv2.putText(frame, "No gesture detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                       
        # Draw statistics
        if self.config['display']['show_fps']:
            fps_text = f"FPS: {self.stats.fps:.1f}"
            cv2.putText(frame, fps_text, (w - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                       
        # Draw instructions
        instructions = [
            "Q - Quit",
            "S - Toggle recording", 
            "R - Reset stats"
        ]
        
        if self.print_instructions:
            y_offset = h - 80
            for i, instruction in enumerate(instructions):
                cv2.putText(frame, instruction, (10, y_offset + i * 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
        return frame
        
    def _start_video_recording(self, frame_shape: Tuple[int, int, int]) -> cv2.VideoWriter:
        """Start video recording."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"gesture_detection_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.config['camera']['fps']
        
        return cv2.VideoWriter(filename, fourcc, fps, (frame_shape[1], frame_shape[0]))
        
    def _reset_stats(self) -> None:
        """Reset detection statistics."""
        self.stats = DetectionStats(
            fps=0.0,
            avg_confidence=0.0,
            total_predictions=0,
            successful_detections=0,
            failed_detections=0
        )
        self.frame_times.clear()
        self.prediction_history.clear()
        self.confidence_history.clear()
        
    def get_detection_stats(self) -> DetectionStats:
        """Get current detection statistics."""
        return self.stats
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        success_rate = (
            self.stats.successful_detections / max(self.stats.total_predictions, 1)
        )
        
        return {
            'fps': self.stats.fps,
            'avg_confidence': self.stats.avg_confidence,
            'success_rate': success_rate,
            'total_predictions': self.stats.total_predictions,
            'avg_frame_time': np.mean(self.frame_times) if self.frame_times else 0.0
        }
