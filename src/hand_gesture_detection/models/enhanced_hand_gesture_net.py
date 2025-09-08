"""
Enhanced hand gesture recognition neural network models with improved confidence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
from .base_model import BaseModel


class EnhancedHandGestureNet(BaseModel):
    """Enhanced MLP with batch normalization and residual connections for better confidence."""
    
    def __init__(
        self,
        num_classes: int = 10,
        input_size: int = 42,
        hidden_sizes: List[int] = [128, 256, 128, 64],
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        use_residual: bool = True,
        activation: str = "relu"
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
            activation: Activation function
        """
        super().__init__(num_classes, input_size)
        
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.activation = activation
        
        # Build the network
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        prev_size = input_size
        
        # Hidden layers
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            self.layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            else:
                self.batch_norms.append(nn.Identity())
                
            # Dropout
            self.dropouts.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
            
        # Output layer with temperature scaling for better confidence
        self.output_layer = nn.Linear(prev_size, num_classes)
        self.temperature = nn.Parameter(torch.ones(1))  # Learnable temperature scaling
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights with better initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                
    def _get_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation function."""
        if self.activation == "relu":
            return F.relu(x)
        elif self.activation == "gelu":
            return F.gelu(x)
        elif self.activation == "swish":
            return x * torch.sigmoid(x)
        elif self.activation == "leaky_relu":
            return F.leaky_relu(x, 0.1)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
            
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
            
        # Store input for residual connection
        residual = x
        
        # Pass through hidden layers
        for i in range(len(self.layers)):
            # Linear layer
            x = self.layers[i](x)
            
            # Batch normalization
            x = self.batch_norms[i](x)
            
            # Activation
            x = self._get_activation(x)
            
            # Residual connection (if sizes match and residual is enabled)
            if (self.use_residual and 
                i > 0 and 
                residual.size(1) == x.size(1)):
                x = x + residual
                
            # Dropout
            x = self.dropouts[i](x)
            
            # Update residual for next layer
            residual = x
            
        # Output layer with temperature scaling
        x = self.output_layer(x)
        x = x / self.temperature  # Temperature scaling for better confidence
        
        return x
        
    def get_confidence_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get confidence scores for predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Confidence scores tensor
        """
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = F.softmax(outputs, dim=1)
            confidence, _ = torch.max(probabilities, dim=1)
            return confidence
            
    def predict_with_confidence(self, x: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Make predictions with confidence filtering.
        
        Args:
            x: Input tensor
            threshold: Minimum confidence threshold
            
        Returns:
            Dictionary with predictions, confidences, and valid indices
        """
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
            
            # Filter by confidence threshold
            valid_mask = confidence >= threshold
            
            return {
                'predictions': predicted_class,
                'confidences': confidence,
                'probabilities': probabilities,
                'valid_mask': valid_mask,
                'valid_predictions': predicted_class[valid_mask],
                'valid_confidences': confidence[valid_mask]
            }


class ConfidenceCalibratedNet(EnhancedHandGestureNet):
    """Model with additional confidence calibration."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Additional calibration layers
        self.confidence_head = nn.Sequential(
            nn.Linear(self.hidden_sizes[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with confidence calibration."""
        # Get base features
        features = self._forward_features(x)
        
        # Classification head
        classification_output = self.output_layer(features)
        classification_output = classification_output / self.temperature
        
        return classification_output
        
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to get features."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        residual = x
        
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.batch_norms[i](x)
            x = self._get_activation(x)
            
            if (self.use_residual and 
                i > 0 and 
                residual.size(1) == x.size(1)):
                x = x + residual
                
            x = self.dropouts[i](x)
            residual = x
            
        return x
        
    def get_calibrated_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """Get calibrated confidence scores."""
        with torch.no_grad():
            features = self._forward_features(x)
            confidence = self.confidence_head(features)
            return confidence.squeeze(-1)


class AdvancedHandGestureNet(EnhancedHandGestureNet):
    """Advanced model with attention mechanisms and improved architecture."""
    
    def __init__(
        self,
        num_classes: int = 10,
        input_size: int = 39,
        hidden_sizes: List[int] = [256, 512, 1024, 512, 256, 128, 64],
        dropout_rate: float = 0.15,
        use_batch_norm: bool = True,
        use_residual: bool = True,
        activation: str = "swish",
        use_attention: bool = True,
        attention_heads: int = 4
    ):
        """
        Initialize the advanced HandGestureNet model.
        
        Args:
            num_classes: Number of gesture classes
            input_size: Size of input feature vector
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
            activation: Activation function
            use_attention: Whether to use attention mechanisms
            attention_heads: Number of attention heads
        """
        super().__init__(
            num_classes=num_classes,
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_residual=use_residual,
            activation=activation
        )
        
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        
        if use_attention:
            # Multi-head self-attention
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_sizes[2],  # Use the largest hidden size
                num_heads=attention_heads,
                dropout=dropout_rate,
                batch_first=True
            )
            
            # Layer normalization for attention
            self.attention_norm = nn.LayerNorm(hidden_sizes[2])
            
            # Feature projection for attention
            self.feature_projection = nn.Linear(hidden_sizes[2], hidden_sizes[2])
            
        # Advanced regularization
        self.stochastic_depth = nn.Dropout2d(p=0.1)
        
        # Ensemble-like multiple heads
        self.auxiliary_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_sizes[-1], 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, num_classes)
            ) for _ in range(3)
        ])
        
        # Temperature scaling for better calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def _get_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation function."""
        if self.activation == "relu":
            return F.relu(x)
        elif self.activation == "gelu":
            return F.gelu(x)
        elif self.activation == "swish":
            return x * torch.sigmoid(x)
        elif self.activation == "leaky_relu":
            return F.leaky_relu(x, 0.1)
        elif self.activation == "mish":
            return x * torch.tanh(F.softplus(x))
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the advanced model.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        # Store input for residual connection
        residual = x
        
        # Pass through initial layers
        for i in range(min(3, len(self.layers))):
            x = self.layers[i](x)
            x = self.batch_norms[i](x)
            x = self._get_activation(x)
            
            if (self.use_residual and 
                i > 0 and 
                residual.size(1) == x.size(1)):
                x = x + residual
                
            x = self.dropouts[i](x)
            residual = x
            
        # Apply attention mechanism if enabled
        if self.use_attention and len(self.layers) > 2:
            # Project features for attention
            attn_features = self.feature_projection(x)
            
            # Reshape for attention (batch_size, 1, feature_dim)
            attn_input = attn_features.unsqueeze(1)
            
            # Apply self-attention
            attn_output, _ = self.attention(attn_input, attn_input, attn_input)
            
            # Residual connection and layer norm
            attn_output = self.attention_norm(attn_output + attn_input)
            
            # Flatten back
            x = attn_output.squeeze(1)
            
        # Pass through remaining layers
        for i in range(3, len(self.layers)):
            x = self.layers[i](x)
            x = self.batch_norms[i](x)
            x = self._get_activation(x)
            
            if (self.use_residual and 
                i > 0 and 
                residual.size(1) == x.size(1)):
                x = x + residual
                
            x = self.dropouts[i](x)
            residual = x
            
        # Apply stochastic depth
        x = self.stochastic_depth(x.unsqueeze(-1)).squeeze(-1)
        
        # Main output layer with temperature scaling
        main_output = self.output_layer(x)
        main_output = main_output / self.temperature
        
        # Auxiliary outputs for ensemble-like behavior
        aux_outputs = [head(x) for head in self.auxiliary_heads]
        
        # Combine outputs (weighted average)
        combined_output = main_output * 0.7 + sum(aux_outputs) * 0.1
        
        return combined_output
        
    def get_ensemble_predictions(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get predictions from all heads for ensemble analysis."""
        with torch.no_grad():
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
                
            # Get features
            residual = x
            for i in range(len(self.layers)):
                x = self.layers[i](x)
                x = self.batch_norms[i](x)
                x = self._get_activation(x)
                
                if (self.use_residual and 
                    i > 0 and 
                    residual.size(1) == x.size(1)):
                    x = x + residual
                    
                x = self.dropouts[i](x)
                residual = x
                
            # Get all predictions
            main_pred = self.output_layer(x) / self.temperature
            aux_preds = [head(x) for head in self.auxiliary_heads]
            
            return {
                'main': main_pred,
                'auxiliary': aux_preds,
                'ensemble': main_pred * 0.7 + sum(aux_preds) * 0.1
            }

