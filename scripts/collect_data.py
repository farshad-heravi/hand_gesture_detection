#!/usr/bin/env python3
"""
Data collection script for hand gesture detection system.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hand_gesture_detection.utils.config import ConfigManager
from hand_gesture_detection.utils.logger import Logger
from hand_gesture_detection.data.data_collector import DataCollector


def main():
    """Main function for data collection."""
    parser = argparse.ArgumentParser(description="Collect hand gesture data")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/data_collection.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--gesture",
        type=str,
        help="Specific gesture to collect (optional)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Number of samples to collect per gesture"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for collected data"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    config_name = args.config.replace('.yaml', '').replace('.yml', '')
    if '/' in config_name:
        # If path includes directory, extract just the filename
        config_name = config_name.split('/')[-1]
    config = config_manager.load_config(config_name)
    
    # Override config with command line arguments
    if args.samples:
        config['data_collection']['collection']['samples_per_gesture'] = args.samples
    if args.output_dir:
        config['data_collection']['dataset']['base_path'] = args.output_dir
        
    # Initialize logger
    logger = Logger("data_collection")
    logger.info("Starting hand gesture data collection")
    logger.log_config(config)
    
    try:
        # Initialize data collector with the correct config structure
        # The DataCollector expects the config to be at the root level, not under 'data_collection'
        data_collection_config = config['data_collection']
        collector = DataCollector(data_collection_config, logger)
        
        if args.gesture:
            # Collect specific gesture
            if args.gesture not in config['data_collection']['dataset']['gestures']:
                logger.error(f"Unknown gesture: {args.gesture}")
                return 1
                
            logger.info(f"Collecting data for gesture: {args.gesture}")
            collector.start_collection_session(args.gesture)
            collector.collect_data_from_camera()
            session = collector.end_collection_session()
            
            logger.info(f"Collection completed: {session.samples_collected} samples")
            
        else:
            # Collect all gestures
            logger.info("Collecting data for all gestures")
            
            for gesture in config['data_collection']['dataset']['gestures']:
                logger.info(f"Starting collection for gesture: {gesture}")
                
                # Start session
                session_id = collector.start_collection_session(gesture)
                
                # Collect data
                collector.collect_data_from_camera()
                
                # End session
                session = collector.end_collection_session()
                logger.info(f"Completed {gesture}: {session.samples_collected} samples")
                
                # Ask if user wants to continue
                if gesture != config['data_collection']['dataset']['gestures'][-1]:
                    continue_collection = input(f"Continue to next gesture? (y/n): ").lower()
                    if continue_collection != 'y':
                        break
                        
        # Print final statistics
        stats = collector.get_collection_statistics()
        logger.info("Collection Statistics:")
        logger.info(f"Total sessions: {stats['total_sessions']}")
        logger.info(f"Total samples: {stats['total_samples']}")
        logger.info(f"Average samples per gesture: {stats['average_samples_per_gesture']:.1f}")
        
        # Validate dataset
        validation = collector.validate_dataset()
        if validation['valid']:
            logger.info("Dataset validation passed")
        else:
            logger.warning("Dataset validation failed:")
            for issue in validation['issues']:
                logger.warning(f"  - {issue}")
                
        return 0
        
    except KeyboardInterrupt:
        logger.info("Data collection interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
