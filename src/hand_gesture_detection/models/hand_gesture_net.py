"""
Hand gesture recognition neural network models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
from .base_model import BaseModel


class HandGestureNet(BaseModel):
    """Multi-layer perceptron for hand gesture recognition."""
    
    def __init__(
        self,
        num_classes: int = 10,
        input_size: int = 18,
        hidden_sizes: List[int] = [64, 128, 128, 64, 32],
        dropout_rate: float = 0.2,
        activation: str = "relu"
    ):
        """
        Initialize the HandGestureNet model.
        
        Args:
            num_classes: Number of gesture classes
            input_size: Size of input feature vector
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            activation: Activation function ("relu", "gelu", "swish")
        """
        super().__init__(num_classes, input_size)
        
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.activation = activation
        
        # Build the network
        self.layers = nn.ModuleList()
        
        # Input layer
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
            
        # Output layer
        self.output_layer = nn.Linear(prev_size, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
    def _get_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation function."""
        if self.activation == "relu":
            return F.relu(x)
        elif self.activation == "gelu":
            return F.gelu(x)
        elif self.activation == "swish":
            return x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        # Pass through hidden layers
        for i in range(0, len(self.layers), 2):  # Skip dropout layers in iteration
            x = self.layers[i](x)  # Linear layer
            x = self._get_activation(x)
            if i + 1 < len(self.layers):
                x = self.layers[i + 1](x)  # Dropout layer
                
        # Output layer
        x = self.output_layer(x)
        
        return x
        
    def get_layer_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get outputs from each layer for analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary mapping layer names to outputs
        """
        outputs = {}
        
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        current_x = x
        layer_idx = 0
        
        for i in range(0, len(self.layers), 2):
            # Linear layer
            current_x = self.layers[i](current_x)
            outputs[f"linear_{layer_idx}"] = current_x.clone()
            
            # Activation
            current_x = self._get_activation(current_x)
            outputs[f"activation_{layer_idx}"] = current_x.clone()
            
            # Dropout (if exists)
            if i + 1 < len(self.layers):
                current_x = self.layers[i + 1](current_x)
                outputs[f"dropout_{layer_idx}"] = current_x.clone()
                
            layer_idx += 1
            
        # Output layer
        current_x = self.output_layer(current_x)
        outputs["output"] = current_x.clone()
        
        return outputs

    @classmethod
    def load_model(cls, path: str, **kwargs) -> 'HandGestureNet':
        """
        Load model from file with proper architecture handling.
        
        Args:
            path: Path to the saved model
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        # Get model info from checkpoint
        model_info = checkpoint.get('model_info', {})
        num_classes = model_info.get('num_classes', kwargs.get('num_classes', 10))
        
        # If model_info doesn't exist, infer from state_dict
        state_dict = checkpoint['model_state_dict']
        if 'input_size' not in model_info:
            # Infer input_size from first layer weight shape
            first_layer_weight = state_dict.get('layers.0.weight')
            if first_layer_weight is not None:
                input_size = first_layer_weight.shape[1]  # Second dimension is input size
            else:
                input_size = kwargs.get('input_size', 18)
        else:
            input_size = model_info.get('input_size', kwargs.get('input_size', 18))
        
        # Extract hidden_sizes from the saved model architecture
        # We need to infer this from the state_dict since it's not saved in model_info
        
        # Infer hidden_sizes from the layer weights
        hidden_sizes = []
        layer_idx = 0
        prev_size = input_size
        
        while f'layers.{layer_idx}.weight' in state_dict:
            weight_shape = state_dict[f'layers.{layer_idx}.weight'].shape
            hidden_size = weight_shape[0]
            hidden_sizes.append(hidden_size)
            prev_size = hidden_size
            layer_idx += 2  # Skip dropout layers
            
        # Create model with correct architecture
        model = cls(
            num_classes=num_classes,
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            dropout_rate=kwargs.get('dropout_rate', 0.2),
            activation=kwargs.get('activation', 'relu')
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model


class HandGestureNetV2(BaseModel):
    """Enhanced version with residual connections and batch normalization."""
    
    def __init__(
        self,
        num_classes: int = 10,
        input_size: int = 18,
        hidden_sizes: List[int] = [64, 128, 128, 64, 32],
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        use_residual: bool = True
    ):
        """
        Initialize the enhanced HandGestureNet model.
        
        Args:
            num_classes: Number of gesture classes
            input_size: Size of input feature vector
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
        """
        super().__init__(num_classes, input_size)
        
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        # Build the network
        self.layers = nn.ModuleList()
        
        # Input layer
        prev_size = input_size
        
        # Hidden layers with residual connections
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            self.layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_size))
                
            # Dropout
            self.layers.append(nn.Dropout(dropout_rate))
            
            # Residual connection (if sizes match)
            if use_residual and prev_size == hidden_size:
                self.layers.append(nn.Identity())  # Placeholder for residual
            else:
                self.layers.append(None)
                
            prev_size = hidden_size
            
        # Output layer
        self.output_layer = nn.Linear(prev_size, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the enhanced model.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        # Pass through hidden layers
        layer_idx = 0
        for i in range(0, len(self.layers), 4):  # 4 components per layer: linear, bn, dropout, residual
            # Linear layer
            x = self.layers[i](x)
            
            # Batch normalization
            if self.use_batch_norm and i + 1 < len(self.layers):
                x = self.layers[i + 1](x)
                
            # Activation
            x = F.relu(x)
            
            # Dropout
            if i + 2 < len(self.layers):
                x = self.layers[i + 2](x)
                
            # Residual connection
            if self.use_residual and i + 3 < len(self.layers) and self.layers[i + 3] is not None:
                x = x + self.layers[i + 3](x)
                
            layer_idx += 1
            
        # Output layer
        x = self.output_layer(x)
        
        return x


class HandGestureCNN(BaseModel):
    """Convolutional neural network for hand gesture recognition from images."""
    
    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        image_size: int = 64
    ):
        """
        Initialize the CNN model.
        
        Args:
            num_classes: Number of gesture classes
            input_channels: Number of input channels
            image_size: Size of input image (assumed square)
        """
        super().__init__(num_classes, image_size * image_size * input_channels)
        
        self.input_channels = input_channels
        self.image_size = image_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size
        self.flattened_size = self._calculate_flattened_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def _calculate_flattened_size(self) -> int:
        """Calculate the size after convolutional layers."""
        # Simulate forward pass to calculate size
        x = torch.zeros(1, self.input_channels, self.image_size, self.image_size)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x.numel()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Reshape if input is flattened
        if x.dim() == 2:
            x = x.view(x.size(0), self.input_channels, self.image_size, self.image_size)
            
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
