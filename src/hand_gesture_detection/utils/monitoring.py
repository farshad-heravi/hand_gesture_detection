"""
Performance monitoring utilities.
"""

import time
import psutil
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from collections import deque
import threading


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    fps: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    frame_time: float
    inference_time: float
    total_time: float


class PerformanceMonitor:
    """Professional performance monitoring system."""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize the performance monitor.
        
        Args:
            window_size: Size of the sliding window for metrics
        """
        self.window_size = window_size
        
        # Metrics storage
        self.fps_history = deque(maxlen=window_size)
        self.cpu_history = deque(maxlen=window_size)
        self.memory_history = deque(maxlen=window_size)
        self.frame_time_history = deque(maxlen=window_size)
        self.inference_time_history = deque(maxlen=window_size)
        
        # Timing
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.start_time = time.time()
        
        # GPU monitoring
        self.gpu_available = self._check_gpu_availability()
        
        # Background monitoring
        self.monitoring = False
        self.monitor_thread = None
        
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
            
    def start_monitoring(self) -> None:
        """Start background monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring:
            try:
                # CPU and memory usage
                cpu_percent = psutil.cpu_percent()
                memory_info = psutil.virtual_memory()
                memory_percent = memory_info.percent
                
                self.cpu_history.append(cpu_percent)
                self.memory_history.append(memory_percent)
                
                time.sleep(0.1)  # Monitor every 100ms
                
            except Exception:
                break
                
    def update_frame_metrics(self, inference_time: float = 0.0) -> PerformanceMetrics:
        """
        Update frame-based metrics.
        
        Args:
            inference_time: Time taken for inference in seconds
            
        Returns:
            Current performance metrics
        """
        current_time = time.time()
        
        # Frame timing
        frame_time = current_time - self.last_frame_time
        self.frame_time_history.append(frame_time)
        
        # FPS calculation
        self.frame_count += 1
        elapsed_time = current_time - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0.0
        self.fps_history.append(fps)
        
        # Inference time
        self.inference_time_history.append(inference_time)
        
        # Update last frame time
        self.last_frame_time = current_time
        
        # Get current system metrics
        cpu_usage = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        
        # GPU usage
        gpu_usage = self._get_gpu_usage() if self.gpu_available else None
        
        return PerformanceMetrics(
            fps=fps,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            frame_time=frame_time,
            inference_time=inference_time,
            total_time=frame_time
        )
        
    def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage percentage."""
        try:
            import torch
            if torch.cuda.is_available():
                # Get GPU memory usage
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                return gpu_memory * 100
        except Exception:
            pass
        return None
        
    def get_average_metrics(self) -> PerformanceMetrics:
        """Get average performance metrics over the monitoring window."""
        return PerformanceMetrics(
            fps=np.mean(self.fps_history) if self.fps_history else 0.0,
            cpu_usage=np.mean(self.cpu_history) if self.cpu_history else 0.0,
            memory_usage=np.mean(self.memory_history) if self.memory_history else 0.0,
            gpu_usage=np.mean([g for g in [self._get_gpu_usage()] if g is not None]) if self.gpu_available else None,
            frame_time=np.mean(self.frame_time_history) if self.frame_time_history else 0.0,
            inference_time=np.mean(self.inference_time_history) if self.inference_time_history else 0.0,
            total_time=np.mean(self.frame_time_history) if self.frame_time_history else 0.0
        )
        
    def get_peak_metrics(self) -> PerformanceMetrics:
        """Get peak performance metrics."""
        return PerformanceMetrics(
            fps=np.max(self.fps_history) if self.fps_history else 0.0,
            cpu_usage=np.max(self.cpu_history) if self.cpu_history else 0.0,
            memory_usage=np.max(self.memory_history) if self.memory_history else 0.0,
            gpu_usage=None,  # Peak GPU usage not easily available
            frame_time=np.min(self.frame_time_history) if self.frame_time_history else 0.0,
            inference_time=np.min(self.inference_time_history) if self.inference_time_history else 0.0,
            total_time=np.min(self.frame_time_history) if self.frame_time_history else 0.0
        )
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        avg_metrics = self.get_average_metrics()
        peak_metrics = self.get_peak_metrics()
        
        return {
            'average': {
                'fps': avg_metrics.fps,
                'cpu_usage': avg_metrics.cpu_usage,
                'memory_usage': avg_metrics.memory_usage,
                'gpu_usage': avg_metrics.gpu_usage,
                'frame_time_ms': avg_metrics.frame_time * 1000,
                'inference_time_ms': avg_metrics.inference_time * 1000
            },
            'peak': {
                'fps': peak_metrics.fps,
                'cpu_usage': peak_metrics.cpu_usage,
                'memory_usage': peak_metrics.memory_usage,
                'frame_time_ms': peak_metrics.frame_time * 1000,
                'inference_time_ms': peak_metrics.inference_time * 1000
            },
            'statistics': {
                'total_frames': self.frame_count,
                'monitoring_duration': time.time() - self.start_time,
                'window_size': self.window_size
            }
        }
        
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.fps_history.clear()
        self.cpu_history.clear()
        self.memory_history.clear()
        self.frame_time_history.clear()
        self.inference_time_history.clear()
        
        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame_time = time.time()
        
    def is_performance_acceptable(self, target_fps: float = 30.0, max_cpu: float = 80.0) -> bool:
        """
        Check if current performance meets requirements.
        
        Args:
            target_fps: Target FPS
            max_cpu: Maximum acceptable CPU usage
            
        Returns:
            True if performance is acceptable
        """
        if not self.fps_history or not self.cpu_history:
            return True  # Not enough data
            
        avg_fps = np.mean(list(self.fps_history)[-10:])  # Last 10 frames
        avg_cpu = np.mean(list(self.cpu_history)[-10:])  # Last 10 measurements
        
        return avg_fps >= target_fps and avg_cpu <= max_cpu
        
    def get_performance_warnings(self) -> List[str]:
        """Get performance warnings based on current metrics."""
        warnings = []
        
        if not self.fps_history or not self.cpu_history:
            return warnings
            
        # Check FPS
        recent_fps = list(self.fps_history)[-10:]
        if recent_fps and np.mean(recent_fps) < 15.0:
            warnings.append(f"Low FPS: {np.mean(recent_fps):.1f}")
            
        # Check CPU usage
        recent_cpu = list(self.cpu_history)[-10:]
        if recent_cpu and np.mean(recent_cpu) > 90.0:
            warnings.append(f"High CPU usage: {np.mean(recent_cpu):.1f}%")
            
        # Check memory usage
        recent_memory = list(self.memory_history)[-10:]
        if recent_memory and np.mean(recent_memory) > 90.0:
            warnings.append(f"High memory usage: {np.mean(recent_memory):.1f}%")
            
        # Check frame time consistency
        if self.frame_time_history:
            frame_times = list(self.frame_time_history)[-20:]
            if len(frame_times) > 5:
                frame_time_std = np.std(frame_times)
                if frame_time_std > 0.05:  # 50ms standard deviation
                    warnings.append(f"Inconsistent frame times: std={frame_time_std:.3f}s")
                    
        return warnings
