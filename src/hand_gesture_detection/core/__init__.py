"""
Core modules for hand gesture detection.
"""

from .landmark_extractor import HandLandmarkExtractor
from .feature_extractor import FeatureExtractor
from .feature_extractor_v2 import FeatureExtractorV2
# Note: HandGestureDetector import removed to avoid PyTorch dependency at package level
# Import it directly when needed: from .detector import HandGestureDetector

__all__ = [
    "HandLandmarkExtractor",
    "FeatureExtractor",
    "FeatureExtractorV2", 
    # "HandGestureDetector",  # Removed to avoid PyTorch dependency
]
