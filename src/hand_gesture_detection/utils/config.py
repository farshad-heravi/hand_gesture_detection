"""
Configuration management utilities.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from omegaconf import OmegaConf


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self._configs = {}
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a configuration file.
        
        Args:
            config_name: Name of the configuration file (without .yaml extension)
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If configuration file is invalid
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Use OmegaConf for advanced configuration features
            config = OmegaConf.create(config)
            self._configs[config_name] = config
            
            return config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in {config_path}: {e}")
            
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """
        Get a previously loaded configuration.
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            Configuration dictionary
            
        Raises:
            KeyError: If configuration hasn't been loaded
        """
        if config_name not in self._configs:
            raise KeyError(f"Configuration '{config_name}' not loaded. Call load_config() first.")
            
        return self._configs[config_name]
        
    def save_config(self, config: Dict[str, Any], config_name: str) -> None:
        """
        Save a configuration to file.
        
        Args:
            config: Configuration dictionary
            config_name: Name for the configuration file
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
            
    def merge_configs(self, base_config: str, override_config: str) -> Dict[str, Any]:
        """
        Merge two configurations with override taking precedence.
        
        Args:
            base_config: Base configuration name
            override_config: Override configuration name
            
        Returns:
            Merged configuration dictionary
        """
        base = self.get_config(base_config)
        override = self.get_config(override_config)
        
        # Use OmegaConf merge
        merged = OmegaConf.merge(base, override)
        return merged
        
    def validate_config(self, config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Validate configuration against a schema.
        
        Args:
            config: Configuration to validate
            schema: Schema definition
            
        Returns:
            True if valid, False otherwise
        """
        # Basic validation - can be extended with more sophisticated validation
        try:
            for key, value in schema.items():
                if key not in config:
                    return False
                    
                if isinstance(value, dict) and isinstance(config[key], dict):
                    if not self.validate_config(config[key], value):
                        return False
                        
            return True
            
        except Exception:
            return False
            
    def get_default_config(self, config_type: str) -> Dict[str, Any]:
        """
        Get default configuration for a specific type.
        
        Args:
            config_type: Type of configuration (data_collection, training, inference)
            
        Returns:
            Default configuration dictionary
        """
        defaults = {
            "data_collection": {
                "dataset": {
                    "name": "hand_gesture_dataset",
                    "base_path": "data/raw",
                    "gestures": [
                        "palm_upward", "index_upward", "fist", "thumb_up", 
                        "thumb_down", "two_finger_up", "grip", "closed_hand",
                        "vertical_fingers", "index_left"
                    ]
                },
                "camera": {
                    "device_id": 0,
                    "width": 640,
                    "height": 480,
                    "fps": 30
                },
                "mediapipe": {
                    "mode": "image",
                    "max_num_hands": 1,
                    "min_detection_confidence": 0.8,
                    "min_tracking_confidence": 0.9
                }
            },
            "training": {
                "model": {
                    "name": "HandGestureNet",
                    "architecture": "mlp",
                    "num_classes": 10,
                    "dropout_rate": 0.2
                },
                "hyperparameters": {
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "epochs": 200
                }
            },
            "inference": {
                "model": {
                    "path": "models/trained/hand_gesture_model.pth",
                    "num_classes": 10,
                    "device": "auto"
                },
                "processing": {
                    "confidence_threshold": 0.7,
                    "roi_size": [320, 320]
                }
            }
        }
        
        return defaults.get(config_type, {})
