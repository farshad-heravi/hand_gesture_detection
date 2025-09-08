"""
Base model class for hand gesture detection models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import json
from pathlib import Path


class BaseModel(nn.Module, ABC):
    """Base class for all hand gesture detection models."""
    
    def __init__(self, num_classes: int, input_size: int):
        """
        Initialize the base model.
        
        Args:
            num_classes: Number of gesture classes
            input_size: Size of input feature vector
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.model_name = self.__class__.__name__
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'input_size': self.input_size,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        
    def save_model(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save model to file.
        
        Args:
            path: Path to save the model
            metadata: Additional metadata to save
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info(),
            'model_class': self.__class__.__name__
        }
        
        if metadata:
            save_dict['metadata'] = metadata
            
        torch.save(save_dict, path)
        
    @classmethod
    def load_model(cls, path: str, **kwargs) -> 'BaseModel':
        """
        Load model from file.
        
        Args:
            path: Path to the saved model
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        # Get model info
        model_info = checkpoint.get('model_info', {})
        num_classes = model_info.get('num_classes', kwargs.get('num_classes', 10))
        input_size = model_info.get('input_size', kwargs.get('input_size', 18))
        
        # Create model instance
        model = cls(num_classes=num_classes, input_size=input_size, **kwargs)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
        
    def predict(self, x: torch.Tensor, return_probabilities: bool = False) -> torch.Tensor:
        """
        Make predictions on input data.
        
        Args:
            x: Input tensor
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Predictions (class indices or probabilities)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            if return_probabilities:
                return torch.softmax(outputs, dim=1)
            else:
                return torch.argmax(outputs, dim=1)
                
    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature importance using gradient-based method.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature importance scores
        """
        self.eval()
        x.requires_grad_(True)
        
        outputs = self.forward(x)
        predicted_class = torch.argmax(outputs, dim=1)
        
        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs[0, predicted_class[0]], x, retain_graph=True
        )[0]
        
        # Return absolute gradients as importance scores
        return torch.abs(gradients)
        
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }
        
    def get_model_size_mb(self) -> float:
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
            
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
        
    def summary(self) -> str:
        """Get model summary as string."""
        info = self.get_model_info()
        param_counts = self.count_parameters()
        model_size = self.get_model_size_mb()
        
        summary = f"""
Model: {info['model_name']}
Input Size: {info['input_size']}
Number of Classes: {info['num_classes']}
Total Parameters: {param_counts['total_parameters']:,}
Trainable Parameters: {param_counts['trainable_parameters']:,}
Model Size: {model_size:.2f} MB
        """.strip()
        
        return summary
