"""
Professional Hand Gesture Detection System

A comprehensive, production-ready hand gesture recognition system built with 
MediaPipe, PyTorch, and modern software engineering practices.
"""

__version__ = "2.0.0"
__author__ = "Farshad Nozad Heravi"
__email__ = "f.n.heravi@gmail.com"

from .core import HandLandmarkExtractor
# Note: PyTorch-dependent imports removed to avoid import issues at package level
# Import them directly when needed:
# from .core import HandGestureDetector
# from .models import HandGestureNet, ModelFactory
from .data import DataCollector, DataProcessor, DatasetManager
from .utils import ConfigManager, Logger, PerformanceMonitor

__all__ = [
    # "HandGestureDetector",  # Removed to avoid PyTorch dependency
    "HandLandmarkExtractor", 
    # "HandGestureNet",  # Removed to avoid PyTorch dependency
    # "ModelFactory",  # Removed to avoid PyTorch dependency
    "DataCollector",
    "DataProcessor",
    "DatasetManager",
    "ConfigManager",
    "Logger",
    "PerformanceMonitor",
]
