"""
Professional data collection system for hand gesture dataset.
"""

import cv2
import numpy as np
import os
import pickle
import uuid
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from ..core.landmark_extractor import HandLandmarkExtractor, HandLandmarks
from ..core.feature_extractor_v2 import FeatureExtractorV2, HandFeatures
from ..utils.logger import Logger


@dataclass
class CollectionSession:
    """Container for data collection session information."""
    session_id: str
    gesture_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    samples_collected: int = 0
    quality_score: float = 0.0


class DataCollector:
    """Professional data collection system for hand gesture recognition."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[Logger] = None
    ):
        """
        Initialize the data collector.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or Logger("data_collector")
        
        # Initialize components
        self.landmark_extractor = HandLandmarkExtractor(
            mode=config['mediapipe']['mode'],
            max_num_hands=config['mediapipe']['max_num_hands'],
            min_detection_confidence=config['mediapipe']['min_detection_confidence'],
            min_tracking_confidence=config['mediapipe']['min_tracking_confidence']
        )
        
        self.feature_extractor = FeatureExtractorV2(normalize_features=True, include_handedness=True)
        
        # Setup paths
        self.base_path = Path(config['dataset']['base_path'])
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Collection state
        self.current_session: Optional[CollectionSession] = None
        self.collection_stats = {
            'total_sessions': 0,
            'total_samples': 0,
            'gesture_counts': {gesture: 0 for gesture in config['dataset']['gestures']}
        }
        
        # UI state
        self.show_ui = True
        self.current_gesture = None
        self.collection_mode = "single"  # "single" or "continuous"
        
        # Continuous mode state
        self.last_capture_time = 0.0
        self.capture_interval = 1.0 / config['collection'].get('continuous_fps', 2.0)  # Default 2 FPS
        self.consecutive_good_frames = 0
        self.min_consecutive_frames = config['collection'].get('min_consecutive_frames', 3)
        
    def start_collection_session(self, gesture_name: str) -> str:
        """
        Start a new data collection session.
        
        Args:
            gesture_name: Name of the gesture to collect
            
        Returns:
            Session ID
        """
        if gesture_name not in self.config['dataset']['gestures']:
            raise ValueError(f"Unknown gesture: {gesture_name}")
            
        session_id = str(uuid.uuid4())
        self.current_session = CollectionSession(
            session_id=session_id,
            gesture_name=gesture_name,
            start_time=datetime.now()
        )
        
        self.current_gesture = gesture_name
        self.logger.info(f"Started collection session for gesture: {gesture_name}")
        
        return session_id
        
    def end_collection_session(self) -> CollectionSession:
        """
        End the current collection session.
        
        Returns:
            Completed session information
        """
        if not self.current_session:
            raise RuntimeError("No active collection session")
            
        self.current_session.end_time = datetime.now()
        self.current_session.quality_score = self._calculate_quality_score()
        
        # Update statistics
        self.collection_stats['total_sessions'] += 1
        self.collection_stats['total_samples'] += self.current_session.samples_collected
        self.collection_stats['gesture_counts'][self.current_session.gesture_name] += self.current_session.samples_collected
        
        self.logger.info(f"Ended collection session: {self.current_session.samples_collected} samples collected")
        
        session = self.current_session
        self.current_session = None
        self.current_gesture = None
        
        return session
        
    def collect_data_from_camera(self) -> None:
        """Collect data from camera with interactive UI."""
        cap = cv2.VideoCapture(self.config['camera']['device_id'])
        
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")
            
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
        cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
        
        self.logger.info("Starting data collection from camera")
        self.logger.info("Controls: 'q'=quit, 'c'=change gesture, 'f'=capture, 'v'=toggle mode")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                processed_frame = self._process_frame(frame)
                
                # Handle continuous mode automatic capture
                if self.collection_mode == "continuous" and self.current_gesture:
                    self._handle_continuous_capture(frame)
                
                # Display frame
                cv2.imshow("Hand Gesture Data Collection", processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self._change_gesture()
                elif key == ord('f'):
                    self._capture_sample(frame)
                elif key == ord('v'):
                    self._toggle_collection_mode()
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame for display."""
        # Extract landmarks
        landmarks_list = self.landmark_extractor.extract_landmarks(frame)
        
        # Draw landmarks if detected
        if landmarks_list:
            frame = self.landmark_extractor.draw_landmarks(frame, landmarks_list)
            
            # Draw ROI
            if self.config['collection']['roi_size']:
                roi_size = self.config['collection']['roi_size']
                h, w = frame.shape[:2]
                x1 = (w - roi_size[0]) // 2
                y1 = (h - roi_size[1]) // 2
                x2 = x1 + roi_size[0]
                y2 = y1 + roi_size[1]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
        # Add UI information
        self._draw_ui_info(frame)
        
        return frame
        
    def _draw_ui_info(self, frame: np.ndarray) -> None:
        """Draw UI information on frame."""
        h, w = frame.shape[:2]
        
        # Current gesture
        gesture_text = f"Gesture: {self.current_gesture or 'None'}"
        cv2.putText(frame, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Collection mode with color coding
        mode_text = f"Mode: {self.collection_mode.upper()}"
        mode_color = (0, 255, 255) if self.collection_mode == "continuous" else (0, 255, 0)
        cv2.putText(frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Session info
        if self.current_session:
            session_text = f"Samples: {self.current_session.samples_collected}"
            cv2.putText(frame, session_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # Continuous mode status
        if self.collection_mode == "continuous" and self.current_gesture:
            # Show consecutive frame count
            consecutive_text = f"Good frames: {self.consecutive_good_frames}/{self.min_consecutive_frames}"
            consecutive_color = (0, 255, 0) if self.consecutive_good_frames >= self.min_consecutive_frames else (0, 255, 255)
            cv2.putText(frame, consecutive_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, consecutive_color, 2)
            
            # Show capture rate
            rate_text = f"Capture rate: {1.0/self.capture_interval:.1f} FPS"
            cv2.putText(frame, rate_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        # Instructions
        instructions = [
            "Controls:",
            "Q - Quit",
            "C - Change gesture", 
            "F - Capture sample (single mode)",
            "V - Toggle mode"
        ]
        
        y_offset = h - 120
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, y_offset + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
    def _change_gesture(self) -> None:
        """Change the current gesture."""
        print("\nAvailable gestures:")
        for i, gesture in enumerate(self.config['dataset']['gestures']):
            print(f"{i}: {gesture}")
            
        try:
            choice = input("Select gesture (number): ")
            gesture_idx = int(choice)
            if 0 <= gesture_idx < len(self.config['dataset']['gestures']):
                self.current_gesture = self.config['dataset']['gestures'][gesture_idx]
                print(f"Selected gesture: {self.current_gesture}")
            else:
                print("Invalid choice")
        except (ValueError, KeyboardInterrupt):
            print("Invalid input")
            
    def _capture_sample(self, frame: np.ndarray) -> None:
        """Capture a sample from the current frame."""
        if not self.current_gesture:
            print("Please select a gesture first (press 'c')")
            return
            
        # Extract landmarks
        landmarks_list = self.landmark_extractor.extract_landmarks(frame)
        
        if not landmarks_list:
            print("No hand detected")
            return
            
        # Use first detected hand
        landmarks = landmarks_list[0]
        
        # Check confidence threshold
        if landmarks.confidence < self.config['collection']['min_confidence_threshold']:
            print(f"Low confidence: {landmarks.confidence:.3f}")
            return
            
        # Extract features
        features = self.feature_extractor.extract_features(landmarks)
        
        if not features:
            print("Feature extraction failed")
            return
            
        # Save sample
        self._save_sample(frame, landmarks, features)
        
        # Update session
        if self.current_session:
            self.current_session.samples_collected += 1
            
        print(f"Captured sample for {self.current_gesture}")
        
    def _save_sample(
        self, 
        frame: np.ndarray, 
        landmarks: HandLandmarks, 
        features: HandFeatures
    ) -> None:
        """Save a collected sample."""
        # Create gesture directory
        gesture_dir = self.base_path / self.current_gesture
        gesture_dir.mkdir(exist_ok=True)
        
        # Generate unique filename
        sample_id = str(uuid.uuid4())
        
        # Save image
        if self.config['collection']['save_images']:
            image_path = gesture_dir / f"{sample_id}.png"
            cv2.imwrite(str(image_path), frame)
            
        # Save annotated image
        if self.config['collection']['save_annotated']:
            annotated_frame = self.landmark_extractor.draw_landmarks(frame, [landmarks])
            annotated_path = gesture_dir / f"{sample_id}_annotated.png"
            cv2.imwrite(str(annotated_path), annotated_frame)
            
        # Save landmarks
        if self.config['collection']['save_keypoints']:
            landmarks_path = gesture_dir / f"{sample_id}_landmarks.pkl"
            with open(landmarks_path, 'wb') as f:
                pickle.dump(landmarks, f)
                
        # Save features
        features_path = gesture_dir / f"{sample_id}_features.pkl"
        with open(features_path, 'wb') as f:
            pickle.dump(features, f)
            
        # Save metadata
        metadata = {
            'sample_id': sample_id,
            'gesture': self.current_gesture,
            'timestamp': datetime.now().isoformat(),
            'landmarks_confidence': landmarks.confidence,
            'features_confidence': features.confidence,
            'handedness': landmarks.handedness
        }
        
        metadata_path = gesture_dir / f"{sample_id}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def _handle_continuous_capture(self, frame: np.ndarray) -> None:
        """Handle automatic capture in continuous mode."""
        current_time = time.time()
        
        # Check if enough time has passed since last capture
        if current_time - self.last_capture_time < self.capture_interval:
            return
            
        # Extract landmarks
        landmarks_list = self.landmark_extractor.extract_landmarks(frame)
        
        if not landmarks_list:
            # Reset consecutive frame counter if no hand detected
            self.consecutive_good_frames = 0
            return
            
        # Use first detected hand
        landmarks = landmarks_list[0]
        
        # Check confidence threshold
        if landmarks.confidence < self.config['collection']['min_confidence_threshold']:
            self.consecutive_good_frames = 0
            return
            
        # Increment consecutive good frames counter
        self.consecutive_good_frames += 1
        
        # Only capture if we have enough consecutive good frames
        if self.consecutive_good_frames >= self.min_consecutive_frames:
            # Extract features
            features = self.feature_extractor.extract_features(landmarks)
            
            if features:
                # Save sample
                self._save_sample(frame, landmarks, features)
                
                # Update session
                if self.current_session:
                    self.current_session.samples_collected += 1
                    
                # Update capture time
                self.last_capture_time = current_time
                
                # Reset consecutive frame counter
                self.consecutive_good_frames = 0
                
                print(f"Auto-captured sample for {self.current_gesture} (confidence: {landmarks.confidence:.3f})")
    
    def _toggle_collection_mode(self) -> None:
        """Toggle between single and continuous collection modes."""
        self.collection_mode = "continuous" if self.collection_mode == "single" else "single"
        print(f"Collection mode: {self.collection_mode}")
        
        # Reset continuous mode state when switching modes
        if self.collection_mode == "continuous":
            self.consecutive_good_frames = 0
            self.last_capture_time = 0.0
        
    def _calculate_quality_score(self) -> float:
        """Calculate quality score for the collection session."""
        if not self.current_session or self.current_session.samples_collected == 0:
            return 0.0
            
        # Simple quality metric based on sample count
        target_samples = self.config['collection']['samples_per_gesture']
        sample_ratio = min(self.current_session.samples_collected / target_samples, 1.0)
        
        return sample_ratio
        
    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            'total_sessions': self.collection_stats['total_sessions'],
            'total_samples': self.collection_stats['total_samples'],
            'gesture_counts': self.collection_stats['gesture_counts'].copy(),
            'average_samples_per_gesture': (
                self.collection_stats['total_samples'] / len(self.config['dataset']['gestures'])
                if self.config['dataset']['gestures'] else 0
            )
        }
        
    def validate_dataset(self) -> Dict[str, Any]:
        """Validate the collected dataset."""
        validation_results = {
            'valid': True,
            'issues': [],
            'gesture_stats': {},
            'total_samples': 0
        }
        
        for gesture in self.config['dataset']['gestures']:
            gesture_dir = self.base_path / gesture
            
            if not gesture_dir.exists():
                validation_results['issues'].append(f"Missing directory for gesture: {gesture}")
                validation_results['valid'] = False
                continue
                
            # Count samples
            sample_files = list(gesture_dir.glob("*_features.pkl"))
            sample_count = len(sample_files)
            
            validation_results['gesture_stats'][gesture] = {
                'sample_count': sample_count,
                'has_images': len(list(gesture_dir.glob("*.png"))) > 0,
                'has_landmarks': len(list(gesture_dir.glob("*_landmarks.pkl"))) > 0,
                'has_features': sample_count > 0
            }
            
            validation_results['total_samples'] += sample_count
            
            # Check minimum samples
            min_samples = self.config['collection']['samples_per_gesture'] // 10
            if sample_count < min_samples:
                validation_results['issues'].append(
                    f"Insufficient samples for {gesture}: {sample_count} < {min_samples}"
                )
                validation_results['valid'] = False
                
        return validation_results
