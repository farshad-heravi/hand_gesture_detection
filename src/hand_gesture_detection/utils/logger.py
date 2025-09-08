"""
Logging utilities for the hand gesture detection system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


class Logger:
    """Professional logging system with multiple outputs and formatting."""
    
    def __init__(
        self,
        name: str = "hand_gesture_detection",
        log_dir: str = "logs",
        level: str = "INFO",
        console_output: bool = True,
        file_output: bool = True,
        json_output: bool = False
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console_output: Enable console output
            file_output: Enable file output
            json_output: Enable JSON formatted output
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        self.console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        self.file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self.console_formatter)
            self.logger.addHandler(console_handler)
            
        # File handler
        if file_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"{name}_{timestamp}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(self.file_formatter)
            self.logger.addHandler(file_handler)
            
        # JSON handler for structured logging
        if json_output:
            json_file = self.log_dir / f"{name}_{timestamp}.json"
            self.json_handler = JsonFileHandler(json_file)
            self.logger.addHandler(self.json_handler)
            
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)
        
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, extra=kwargs)
        
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)
        
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.logger.error(message, extra=kwargs)
        
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(message, extra=kwargs)
        
    def log_metric(self, metric_name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value."""
        metric_data = {
            "metric_name": metric_name,
            "value": value,
            "step": step,
            "timestamp": datetime.now().isoformat()
        }
        self.info(f"METRIC: {metric_name} = {value}", **metric_data)
        
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration parameters."""
        self.info("Configuration loaded", config=config)
        
    def log_training_start(self, model_name: str, dataset_size: int) -> None:
        """Log training start."""
        self.info(f"Training started: {model_name} on {dataset_size} samples")
        
    def log_training_epoch(
        self, 
        epoch: int, 
        train_loss: float, 
        val_loss: float, 
        val_accuracy: float
    ) -> None:
        """Log training epoch results."""
        self.info(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}"
        )
        
    def log_inference_stats(
        self, 
        fps: float, 
        avg_confidence: float, 
        total_predictions: int
    ) -> None:
        """Log inference statistics."""
        self.info(
            f"Inference stats: FPS={fps:.1f}, "
            f"avg_confidence={avg_confidence:.3f}, "
            f"total_predictions={total_predictions}"
        )


class JsonFileHandler(logging.Handler):
    """Custom handler for JSON formatted logs."""
    
    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename
        
    def emit(self, record):
        """Emit a log record in JSON format."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in log_entry:
                log_entry[key] = value
                
        # Write to file
        with open(self.filename, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


class PerformanceLogger:
    """Specialized logger for performance metrics."""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        self.metrics = {}
        
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.metrics[operation] = {"start_time": datetime.now()}
        
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation not in self.metrics:
            raise ValueError(f"Timer for '{operation}' was not started")
            
        end_time = datetime.now()
        duration = (end_time - self.metrics[operation]["start_time"]).total_seconds()
        
        self.logger.log_metric(f"{operation}_duration", duration)
        del self.metrics[operation]
        
        return duration
        
    def log_memory_usage(self) -> None:
        """Log current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            self.logger.log_metric("memory_usage_mb", memory_mb)
        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")
            
    def log_cpu_usage(self) -> None:
        """Log current CPU usage."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent()
            self.logger.log_metric("cpu_usage_percent", cpu_percent)
        except ImportError:
            self.logger.warning("psutil not available for CPU monitoring")
