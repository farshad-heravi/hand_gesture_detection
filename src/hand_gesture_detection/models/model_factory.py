"""
Model factory for creating different types of hand gesture detection models.
"""

import torch
from typing import Dict, Any, Type, Optional
from .base_model import BaseModel
from .hand_gesture_net import HandGestureNet, HandGestureNetV2, HandGestureCNN


class ModelFactory:
    """Factory class for creating hand gesture detection models."""
    
    _model_registry = {
        'HandGestureNet': HandGestureNet,
        'HandGestureNetV2': HandGestureNetV2,
        'HandGestureCNN': HandGestureCNN,
        'mlp': HandGestureNet,
        'mlp_v2': HandGestureNetV2,
        'cnn': HandGestureCNN
    }
    
    @classmethod
    def create_model(
        cls,
        model_type: str,
        num_classes: int = 10,
        input_size: int = 18,
        **kwargs
    ) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model to create
            num_classes: Number of gesture classes
            input_size: Size of input feature vector
            **kwargs: Additional model parameters
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model type is not supported
        """
        if model_type not in cls._model_registry:
            available_models = list(cls._model_registry.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available_models}")
            
        model_class = cls._model_registry[model_type]
        
        # Create model with appropriate parameters
        if model_type in ['HandGestureCNN', 'cnn']:
            # CNN models need different parameters
            return model_class(
                num_classes=num_classes,
                input_channels=kwargs.get('input_channels', 3),
                image_size=kwargs.get('image_size', 64)
            )
        else:
            # MLP models
            return model_class(
                num_classes=num_classes,
                input_size=input_size,
                hidden_sizes=kwargs.get('hidden_sizes', [64, 128, 128, 64, 32]),
                dropout_rate=kwargs.get('dropout_rate', 0.2),
                activation=kwargs.get('activation', 'relu'),
                use_batch_norm=kwargs.get('use_batch_norm', True),
                use_residual=kwargs.get('use_residual', True)
            )
            
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseModel]) -> None:
        """
        Register a new model type.
        
        Args:
            name: Name for the model type
            model_class: Model class to register
        """
        if not issubclass(model_class, BaseModel):
            raise ValueError("Model class must inherit from BaseModel")
            
        cls._model_registry[name] = model_class
        
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model types."""
        return list(cls._model_registry.keys())
        
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """
        Get information about a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary with model information
        """
        if model_type not in cls._model_registry:
            raise ValueError(f"Unknown model type: {model_type}")
            
        model_class = cls._model_registry[model_type]
        
        # Create a temporary instance to get model info
        if model_type in ['HandGestureCNN', 'cnn']:
            temp_model = model_class(num_classes=10, input_channels=3, image_size=64)
        else:
            temp_model = model_class(num_classes=10, input_size=18)
            
        return {
            'name': model_class.__name__,
            'type': model_type,
            'description': model_class.__doc__ or "No description available",
            'parameters': temp_model.count_parameters(),
            'model_size_mb': temp_model.get_model_size_mb(),
            'input_size': temp_model.input_size,
            'num_classes': temp_model.num_classes
        }
        
    @classmethod
    def compare_models(cls, model_types: list, num_classes: int = 10, input_size: int = 18) -> Dict[str, Any]:
        """
        Compare multiple model types.
        
        Args:
            model_types: List of model types to compare
            num_classes: Number of classes
            input_size: Input size for MLP models
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {}
        
        for model_type in model_types:
            try:
                model_info = cls.get_model_info(model_type)
                comparison[model_type] = model_info
            except Exception as e:
                comparison[model_type] = {'error': str(e)}
                
        return comparison
        
    @classmethod
    def create_model_from_config(cls, config: Dict[str, Any]) -> BaseModel:
        """
        Create model from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Model instance
        """
        model_config = config.get('model', {})
        
        model_type = model_config.get('name', 'HandGestureNet')
        num_classes = model_config.get('num_classes', 10)
        input_size = model_config.get('input_size', 18)
        
        # Extract model-specific parameters
        model_kwargs = {}
        
        if 'hidden_sizes' in model_config:
            model_kwargs['hidden_sizes'] = model_config['hidden_sizes']
        if 'dropout_rate' in model_config:
            model_kwargs['dropout_rate'] = model_config['dropout_rate']
        if 'activation' in model_config:
            model_kwargs['activation'] = model_config['activation']
        if 'use_batch_norm' in model_config:
            model_kwargs['use_batch_norm'] = model_config['use_batch_norm']
        if 'use_residual' in model_config:
            model_kwargs['use_residual'] = model_config['use_residual']
        if 'input_channels' in model_config:
            model_kwargs['input_channels'] = model_config['input_channels']
        if 'image_size' in model_config:
            model_kwargs['image_size'] = model_config['image_size']
            
        return cls.create_model(model_type, num_classes, input_size, **model_kwargs)
        
    @classmethod
    def optimize_model_for_device(cls, model: BaseModel, device: str = 'auto') -> BaseModel:
        """
        Optimize model for specific device.
        
        Args:
            model: Model to optimize
            device: Target device ('cpu', 'cuda', 'auto')
            
        Returns:
            Optimized model
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        # Move model to device
        model = model.to(device)
        
        # Optimize for inference if on GPU
        if device == 'cuda':
            model.eval()
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            
        return model
        
    @classmethod
    def get_recommended_model(cls, requirements: Dict[str, Any]) -> str:
        """
        Get recommended model type based on requirements.
        
        Args:
            requirements: Dictionary with requirements
            
        Returns:
            Recommended model type
        """
        accuracy_requirement = requirements.get('accuracy', 0.9)
        speed_requirement = requirements.get('speed', 'medium')  # low, medium, high
        memory_requirement = requirements.get('memory', 'medium')  # low, medium, high
        input_type = requirements.get('input_type', 'features')  # features, images
        
        if input_type == 'images':
            return 'HandGestureCNN'
            
        if speed_requirement == 'high':
            return 'HandGestureNet'  # Lightweight MLP
        elif accuracy_requirement > 0.95:
            return 'HandGestureNetV2'  # Enhanced MLP with batch norm
        else:
            return 'HandGestureNet'  # Standard MLP
