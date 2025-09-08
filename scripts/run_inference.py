#!/usr/bin/env python3
"""
Real-time inference script for hand gesture detection system with enhanced FPS display.
"""

import sys
import os
import argparse
import torch
import numpy as np
import cv2
import time
from pathlib import Path
from collections import deque

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hand_gesture_detection.utils.config import ConfigManager
from hand_gesture_detection.utils.logger import Logger
from hand_gesture_detection.models.enhanced_hand_gesture_net import EnhancedHandGestureNet, AdvancedHandGestureNet
from hand_gesture_detection.core.detector import HandGestureDetector


class EnhancedRealtimeDetector:
    """Enhanced real-time detector with improved FPS tracking and display."""
    
    def __init__(self, detector, config, logger):
        self.detector = detector
        self.config = config
        self.logger = logger
        
        # Enhanced FPS tracking
        self.fps_history = deque(maxlen=30)  # Last 30 frames for smooth FPS
        self.frame_times = deque(maxlen=30)
        self.last_fps_update = time.time()
        self.current_fps = 0.0
        self.target_fps = config.get('camera', {}).get('fps', 30)
        
        # Performance metrics
        self.total_frames = 0
        self.detection_frames = 0
        self.start_time = time.time()
        
    def run_realtime_detection(self, camera_id=0, input_video=None, output_video=None, clean_output=False):
        """Run enhanced real-time detection with improved FPS display."""
        if input_video:
            cap = cv2.VideoCapture(input_video)
            self.logger.info(f"Processing video file: {input_video}")
        else:
            cap = cv2.VideoCapture(camera_id)
            self.logger.info(f"Using camera {camera_id}")
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {'video file' if input_video else 'camera'} {input_video or camera_id}")
            
        # Set camera properties (only for live camera)
        if not input_video:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
            cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if input_video else -1
        
        self.logger.info(f"Video properties: {width}x{height} @ {fps:.1f} FPS")
        if total_frames > 0:
            self.logger.info(f"Total frames: {total_frames}")
        
        # Initialize video writer if output video is specified
        video_writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            if not video_writer.isOpened():
                raise RuntimeError(f"Could not initialize video writer for {output_video}")
            self.logger.info(f"Output video will be saved to: {output_video}")
        
        self.logger.info("Starting enhanced real-time gesture detection")
        if clean_output and output_video:
            self.logger.info("Clean output mode: Video will be saved without UI overlays (only gesture detection results)")
        if input_video:
            self.logger.info("Press 'q' to quit, 'p' to pause/resume, 'f' to toggle FPS display")
        else:
            self.logger.info("Press 'q' to quit, 's' to save video, 'r' to reset stats, 'f' to toggle FPS display")
        
        # Video recording (for live camera mode)
        recording = False
        
        # Performance monitoring
        last_stats_time = time.time()
        paused = False
        current_frame = 0
        
        try:
            while True:
                frame_start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    if input_video:
                        self.logger.info("End of video file reached")
                    else:
                        self.logger.warning("Failed to read frame from camera")
                    break
                    
                # Flip frame horizontally for mirror effect (only for live camera)
                if not input_video:
                    frame = cv2.flip(frame, 1)
                
                self.total_frames += 1
                current_frame += 1
                
                # Detect gesture
                prediction = self.detector.detect_gesture(frame)
                if prediction:
                    self.detection_frames += 1
                
                # Update FPS calculation
                self._update_fps(frame_start_time)
                
                # Create clean frame for output (gesture detection results + FPS) - use original frame
                clean_frame = frame.copy()
                if prediction:
                    # Draw the gesture detection results
                    clean_frame = self.detector._draw_detection_results(clean_frame, prediction)
                
                # Add FPS display to clean frame
                if self.config['display']['show_fps']:
                    self._draw_fps_only(clean_frame)
                
                # Draw enhanced results for display (with all UI overlays) - use a fresh copy
                display_frame = frame.copy()
                display_frame = self._draw_enhanced_results(display_frame, prediction, input_video, current_frame, total_frames)
                
                # Write to output video if specified
                if video_writer:
                    if clean_output:
                        video_writer.write(clean_frame)  # Write clean frame without UI overlays
                    else:
                        video_writer.write(display_frame)  # Write frame with all overlays
                
                # Record video if enabled (for live camera mode)
                if recording and not input_video:
                    # This would be for live recording, but we already have video_writer for output
                    pass
                    
                # Display frame
                window_title = "Hand Gesture Detection - Video" if input_video else "Hand Gesture Detection - Real-time"
                cv2.imshow(window_title, display_frame)
                
                # Handle key presses
                wait_time = 1 if not input_video else int(1000 / fps)  # Use video FPS for video mode
                key = cv2.waitKey(wait_time) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p') and input_video:
                    paused = not paused
                    self.logger.info(f"Video {'paused' if paused else 'resumed'}")
                elif key == ord('s') and not input_video:
                    if not recording:
                        recording = True
                        temp_video_writer = self._start_video_recording(frame.shape)
                        self.logger.info("Started video recording")
                    else:
                        recording = False
                        if temp_video_writer:
                            temp_video_writer.release()
                            temp_video_writer = None
                        self.logger.info("Stopped video recording")
                elif key == ord('r') and not input_video:
                    self._reset_stats()
                    self.logger.info("Reset detection statistics")
                elif key == ord('f'):
                    self.config['display']['show_fps'] = not self.config['display']['show_fps']
                    self.logger.info(f"FPS display: {'ON' if self.config['display']['show_fps'] else 'OFF'}")
                
                # Skip processing if paused
                if paused:
                    continue
                
                # Log performance stats every 5 seconds
                current_time = time.time()
                if current_time - last_stats_time >= 5.0:
                    self._log_performance_stats()
                    last_stats_time = current_time
                    
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            
    def _update_fps(self, frame_start_time):
        """Update FPS calculation with smoothing."""
        current_time = time.time()
        frame_time = current_time - frame_start_time
        self.frame_times.append(frame_time)
        
        # Calculate FPS every few frames for stability
        if len(self.frame_times) >= 5:
            avg_frame_time = np.mean(self.frame_times)
            if avg_frame_time > 0:
                fps = 1.0 / avg_frame_time
                self.fps_history.append(fps)
                
                # Smooth FPS calculation
                if len(self.fps_history) >= 3:
                    self.current_fps = np.mean(self.fps_history)
                    
    def _draw_enhanced_results(self, frame, prediction, input_video=None, current_frame=0, total_frames=0):
        """Draw enhanced detection results with improved FPS display."""
        h, w = frame.shape[:2]
        
        # Draw prediction results (using detector's method)
        frame = self.detector._draw_detection_results(frame, prediction)
        
        # Enhanced FPS and performance display
        if self.config['display']['show_fps']:
            self._draw_performance_overlay(frame)
            
        # Draw detection status indicator
        self._draw_detection_status(frame, prediction)
        
        # Draw video progress if processing video file
        if input_video and total_frames > 0:
            self._draw_video_progress(frame, current_frame, total_frames)
        
        # Draw instructions
        self._draw_instructions(frame, input_video)
        
        return frame
        
    def _draw_fps_only(self, frame):
        """Draw only FPS display for clean output."""
        h, w = frame.shape[:2]
        
        # FPS display with color coding
        fps_color = (0, 255, 0) if self.current_fps >= self.target_fps * 0.8 else (0, 255, 255) if self.current_fps >= self.target_fps * 0.6 else (0, 0, 255)
        
        # Main FPS display (top-right)
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(frame, fps_text, (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        
        # Target FPS indicator
        target_text = f"Target: {self.target_fps}"
        cv2.putText(frame, target_text, (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _draw_performance_overlay(self, frame):
        """Draw comprehensive performance overlay."""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # FPS display with color coding
        fps_color = (0, 255, 0) if self.current_fps >= self.target_fps * 0.8 else (0, 255, 255) if self.current_fps >= self.target_fps * 0.6 else (0, 0, 255)
        
        # Main FPS display (top-right)
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(frame, fps_text, (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, fps_color, 3)
        
        # Target FPS indicator
        target_text = f"Target: {self.target_fps}"
        cv2.putText(frame, target_text, (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Performance bar
        bar_width = 150
        bar_height = 15
        bar_x = w - 200
        bar_y = 80
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Performance fill
        performance_ratio = min(self.current_fps / self.target_fps, 1.0)
        fill_width = int(bar_width * performance_ratio)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), fps_color, -1)
        
        # Detection rate
        detection_rate = (self.detection_frames / max(self.total_frames, 1)) * 100
        detection_text = f"Detection: {detection_rate:.1f}%"
        cv2.putText(frame, detection_text, (w - 200, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Runtime
        runtime = time.time() - self.start_time
        runtime_text = f"Runtime: {runtime:.0f}s"
        cv2.putText(frame, runtime_text, (w - 200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    def _draw_detection_status(self, frame, prediction):
        """Draw detection status indicator."""
        h, w = frame.shape[:2]
        
        # Status indicator (bottom-right corner)
        status_size = 20
        status_x = w - status_size - 10
        status_y = h - status_size - 10
        
        if prediction:
            # Green circle for successful detection
            cv2.circle(frame, (status_x + status_size//2, status_y + status_size//2), 
                      status_size//2, (0, 255, 0), -1)
            cv2.circle(frame, (status_x + status_size//2, status_y + status_size//2), 
                      status_size//2, (255, 255, 255), 2)
        else:
            # Red circle for no detection
            cv2.circle(frame, (status_x + status_size//2, status_y + status_size//2), 
                      status_size//2, (0, 0, 255), -1)
            cv2.circle(frame, (status_x + status_size//2, status_y + status_size//2), 
                      status_size//2, (255, 255, 255), 2)
                      
    def _draw_video_progress(self, frame, current_frame, total_frames):
        """Draw video progress bar and frame information."""
        h, w = frame.shape[:2]
        
        # Progress bar
        bar_width = w - 40
        bar_height = 20
        bar_x = 20
        bar_y = h - 60
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Progress fill
        progress = current_frame / total_frames
        fill_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 255, 0), -1)
        
        # Frame counter
        frame_text = f"Frame: {current_frame}/{total_frames} ({progress*100:.1f}%)"
        cv2.putText(frame, frame_text, (bar_x, bar_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _draw_instructions(self, frame, input_video=None):
        """Draw control instructions."""
        h, w = frame.shape[:2]
        
        if input_video:
            instructions = [
                "Q - Quit",
                "P - Pause/Resume", 
                "F - Toggle FPS display"
            ]
        else:
            instructions = [
                "Q - Quit",
                "S - Toggle recording", 
                "R - Reset stats",
                "F - Toggle FPS display"
            ]
        
        y_offset = h - 100
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, y_offset + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                       
    def _start_video_recording(self, frame_shape):
        """Start video recording."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"gesture_detection_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.config['camera']['fps']
        
        return cv2.VideoWriter(filename, fourcc, fps, (frame_shape[1], frame_shape[0]))
        
    def _reset_stats(self):
        """Reset all statistics."""
        self.fps_history.clear()
        self.frame_times.clear()
        self.current_fps = 0.0
        self.total_frames = 0
        self.detection_frames = 0
        self.start_time = time.time()
        self.detector._reset_stats()
        
    def _log_performance_stats(self):
        """Log current performance statistics."""
        detection_rate = (self.detection_frames / max(self.total_frames, 1)) * 100
        runtime = time.time() - self.start_time
        
        self.logger.info(f"Performance Stats - FPS: {self.current_fps:.1f}, "
                        f"Detection Rate: {detection_rate:.1f}%, "
                        f"Runtime: {runtime:.0f}s, "
                        f"Total Frames: {self.total_frames}")


def main():
    """Main function for real-time inference."""
    parser = argparse.ArgumentParser(description="Run real-time hand gesture detection with enhanced FPS display")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/inference.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to trained model"
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera device ID"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        help="Confidence threshold for predictions"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save video output"
    )
    parser.add_argument(
        "--input-video",
        type=str,
        help="Path to input video file for processing"
    )
    parser.add_argument(
        "--output-video",
        type=str,
        help="Path to output video file for saving results"
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Save clean video output without UI overlays (only gesture detection results)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    # Extract config name from path (e.g., "config/inference.yaml" -> "inference")
    config_name = Path(args.config).stem
    config = config_manager.load_config(config_name)
    
    # Handle nested config structure
    if 'inference' in config:
        config = config['inference']
    
    # Override config with command line arguments
    if args.model_path:
        config.setdefault('model', {})['path'] = args.model_path
    if args.camera_id is not None:
        config.setdefault('camera', {})['device_id'] = args.camera_id
    if args.confidence_threshold is not None:
        config.setdefault('processing', {})['confidence_threshold'] = args.confidence_threshold
    if args.save_video:
        config.setdefault('output', {})['save_video'] = True
        
    # Initialize logger
    logger = Logger("inference")
    logger.info("Starting enhanced real-time hand gesture detection")
    logger.log_config(config)
    
    try:
        # Load enhanced model
        model_path = config.get('model', {}).get('path', 'models/trained/hand_gesture_model_20250905_034110.pth')
        if not Path(model_path).exists():
            logger.error(f"Model file not found: {model_path}")
            return 1
            
        logger.info(f"Loading enhanced model from: {model_path}")
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model_info = checkpoint.get('model_info', {})
        model_config = checkpoint.get('model_config', {})
        
        # Determine model architecture and create appropriate model
        if model_config.get('name') == 'AdvancedHandGestureNet':
            # Use AdvancedHandGestureNet for ensemble models
            model = AdvancedHandGestureNet(
                num_classes=model_config.get('num_classes', 10),
                input_size=model_config.get('input_size', 39),
                hidden_sizes=model_config.get('hidden_sizes', [128, 256, 512, 1024, 512, 256, 128, 64]),
                dropout_rate=model_config.get('dropout_rate', 0.4),
                use_batch_norm=True,
                use_residual=True,
                activation=model_config.get('activation', 'swish'),
                use_attention=model_config.get('use_attention', True),
                attention_heads=model_config.get('attention_heads', 8)
            )
        else:
            # Use EnhancedHandGestureNet for other models
            model = EnhancedHandGestureNet(
                num_classes=model_info.get('num_classes', 10),
                input_size=model_info.get('input_size', 42),
                hidden_sizes=model_info.get('hidden_sizes', [128, 256, 128, 64]),
                dropout_rate=model_info.get('dropout_rate', 0.2),
                use_batch_norm=True,
                use_residual=True
            )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load gesture mapping and scaler from processed data
        import pickle
        processed_data_path = Path("data/processed/enhanced_processed_data_right_hand_only.pkl")
        scaler = None
        
        if processed_data_path.exists():
            with open(processed_data_path, 'rb') as f:
                data = pickle.load(f)
            
            if len(data) >= 4:
                # New format with scaler
                X, y, class_weight_dict, scaler = data[:4]
                logger.info("Loaded scaler from processed data")
            else:
                # Old format without scaler
                logger.warning("No scaler found in processed data - this may cause accuracy issues")
            
            # Get class mapping
            class_names = list(class_weight_dict.keys())
            gesture_mapping = config.get('gesture_mapping', {})
            class_to_gesture = {}
            for i, class_name in enumerate(class_names):
                display_name = gesture_mapping.get(class_name, class_name.replace('_', ' ').title())
                class_to_gesture[i] = display_name
        else:
            # Fallback to config mapping
            logger.warning("Processed data not found - using fallback mapping")
            gesture_mapping = config.get('gesture_mapping', {})
            class_to_gesture = {i: gesture_mapping[gesture] for i, gesture in enumerate(gesture_mapping.keys())}
        
        logger.info(f"Loaded model with {model.num_classes} classes")
        logger.info(f"Gesture mapping: {class_to_gesture}")
        
        # Initialize detector with scaler
        logger.info(f"Initializing detector with config keys: {list(config.keys())}")
        detector = HandGestureDetector(model, config, class_to_gesture, logger, scaler, print_instructions=False)
        
        # Initialize enhanced real-time detector
        enhanced_detector = EnhancedRealtimeDetector(detector, config, logger)
        
        # Run enhanced real-time detection
        logger.info("Starting enhanced real-time detection...")
        enhanced_detector.run_realtime_detection(
            camera_id=config.get('camera', {}).get('device_id', 0),
            input_video=args.input_video,
            output_video=args.output_video,
            clean_output=args.clean_output
        )
        
        # Print final statistics
        stats = detector.get_detection_stats()
        metrics = detector.get_performance_metrics()
        
        logger.info("Final Detection Statistics:")
        logger.info(f"Total predictions: {stats.total_predictions}")
        logger.info(f"Successful detections: {stats.successful_detections}")
        logger.info(f"Failed detections: {stats.failed_detections}")
        logger.info(f"Average FPS: {metrics['fps']:.1f}")
        logger.info(f"Average confidence: {metrics['avg_confidence']:.3f}")
        logger.info(f"Success rate: {metrics['success_rate']:.3f}")
        logger.info(f"Total frames processed: {enhanced_detector.total_frames}")
        logger.info(f"Detection rate: {(enhanced_detector.detection_frames / max(enhanced_detector.total_frames, 1)) * 100:.1f}%")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Inference interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
