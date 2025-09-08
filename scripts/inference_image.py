#!/usr/bin/env python3
"""
Single image inference script for hand gesture detection system.
"""

import sys
import os
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hand_gesture_detection.utils.config import ConfigManager
from hand_gesture_detection.utils.logger import Logger
from hand_gesture_detection.models.enhanced_hand_gesture_net import EnhancedHandGestureNet, AdvancedHandGestureNet
from hand_gesture_detection.core.detector import HandGestureDetector


def run_inference_on_image(image_path: str, model_path: str, config_path: str = "config/inference.yaml", 
                          output_path: str = None, confidence_threshold: float = None) -> dict:
    """
    Run hand gesture inference on a single image.
    
    Args:
        image_path: Path to input image
        model_path: Path to trained model
        config_path: Path to configuration file
        output_path: Path to save output image (optional)
        confidence_threshold: Override confidence threshold (optional)
        
    Returns:
        Dictionary with prediction results
    """
    # Load configuration
    config_manager = ConfigManager()
    config_name = Path(config_path).stem
    config = config_manager.load_config(config_name)
    
    # Handle nested config structure
    if 'inference' in config:
        config = config['inference']
    
    # Override model path and confidence threshold if provided
    config.setdefault('model', {})['path'] = model_path
    if confidence_threshold is not None:
        config.setdefault('processing', {})['confidence_threshold'] = confidence_threshold
    
    # Initialize logger
    logger = Logger("image_inference")
    logger.info(f"Running inference on image: {image_path}")
    
    # Load model
    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        return {"error": "Model file not found"}
        
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
    
    logger.info(f"Loaded enhanced model with {model.num_classes} classes")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Gesture mapping: {class_to_gesture}")
    
    # Initialize detector with scaler
    detector = HandGestureDetector(model, config, class_to_gesture, logger, scaler)
    
    # Load and process image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not load image: {image_path}")
        return {"error": "Could not load image"}
    
    logger.info(f"Image loaded: {image.shape}")
    
    # Detect gesture
    prediction = detector.detect_gesture(image)
    
    # Prepare results
    results = {
        "image_path": image_path,
        "image_shape": image.shape,
        "prediction": None,
        "success": False
    }
    
    if prediction:
        results["prediction"] = {
            "gesture_name": prediction.gesture_name,
            "confidence": float(prediction.confidence),
            "class_id": int(prediction.class_id),
            "timestamp": prediction.timestamp
        }
        results["success"] = True
        logger.info(f"Prediction: {prediction.gesture_name} (confidence: {prediction.confidence:.3f})")
    else:
        logger.info("No gesture detected in image")
    
    # Draw results on image if output path is specified
    if output_path:
        # Draw detection results
        output_image = detector._draw_detection_results(image, prediction)
        
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save output image
        cv2.imwrite(output_path, output_image)
        logger.info(f"Output image saved to: {output_path}")
        results["output_path"] = output_path
    
    return results


def main():
    """Main function for image inference."""
    parser = argparse.ArgumentParser(description="Run hand gesture detection on a single image")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/trained/hand_gesture_model_20250905_034110.pth",
        help="Path to trained model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/inference.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save output image with annotations"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        help="Confidence threshold for predictions"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        return 1
    
    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        return 1
    
    try:
        # Run inference
        results = run_inference_on_image(
            image_path=args.image,
            model_path=args.model_path,
            config_path=args.config,
            output_path=args.output,
            confidence_threshold=args.confidence_threshold
        )
        
        # Print results
        if "error" in results:
            print(f"Error: {results['error']}")
            return 1
        
        print(f"\nImage: {results['image_path']}")
        print(f"Shape: {results['image_shape']}")
        
        if results["success"]:
            pred = results["prediction"]
            print(f"Gesture: {pred['gesture_name']}")
            print(f"Confidence: {pred['confidence']:.3f}")
            print(f"Class ID: {pred['class_id']}")
        else:
            print("No gesture detected")
        
        if "output_path" in results:
            print(f"Output saved to: {results['output_path']}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
