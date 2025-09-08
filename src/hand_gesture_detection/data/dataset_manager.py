"""
Dataset management utilities for hand gesture detection.
"""

import os
import json
import pickle
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib

from ..utils.logger import Logger


class DatasetManager:
    """Professional dataset management system."""
    
    def __init__(self, base_path: str = "data", logger: Optional[Logger] = None):
        """
        Initialize the dataset manager.
        
        Args:
            base_path: Base path for dataset storage
            logger: Logger instance
        """
        self.base_path = Path(base_path)
        self.logger = logger or Logger("dataset_manager")
        
        # Create directory structure
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        self.backup_path = self.base_path / "backups"
        self.metadata_path = self.base_path / "metadata"
        
        for path in [self.raw_path, self.processed_path, self.backup_path, self.metadata_path]:
            path.mkdir(parents=True, exist_ok=True)
            
    def create_dataset(
        self,
        dataset_name: str,
        description: str = "",
        version: str = "1.0.0",
        gestures: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a new dataset.
        
        Args:
            dataset_name: Name of the dataset
            description: Dataset description
            version: Dataset version
            gestures: List of gesture names
            
        Returns:
            Dataset metadata
        """
        dataset_id = self._generate_dataset_id(dataset_name, version)
        dataset_path = self.raw_path / dataset_id
        
        # Create dataset directory
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Create gesture directories
        if gestures:
            for gesture in gestures:
                (dataset_path / gesture).mkdir(exist_ok=True)
                
        # Create metadata
        metadata = {
            'dataset_id': dataset_id,
            'name': dataset_name,
            'description': description,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'gestures': gestures or [],
            'path': str(dataset_path),
            'status': 'created',
            'statistics': {
                'total_samples': 0,
                'gesture_counts': {},
                'last_updated': datetime.now().isoformat()
            }
        }
        
        # Save metadata
        self._save_dataset_metadata(dataset_id, metadata)
        
        self.logger.info(f"Created dataset: {dataset_name} (ID: {dataset_id})")
        
        return metadata
        
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets."""
        datasets = []
        
        for metadata_file in self.metadata_path.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                datasets.append(metadata)
            except Exception as e:
                self.logger.error(f"Error loading metadata {metadata_file}: {e}")
                
        return sorted(datasets, key=lambda x: x['created_at'], reverse=True)
        
    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get dataset metadata.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            Dataset metadata or None if not found
        """
        metadata_file = self.metadata_path / f"{dataset_id}.json"
        
        if not metadata_file.exists():
            return None
            
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading dataset {dataset_id}: {e}")
            return None
            
    def update_dataset_statistics(self, dataset_id: str) -> Dict[str, Any]:
        """
        Update dataset statistics.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            Updated statistics
        """
        metadata = self.get_dataset(dataset_id)
        if not metadata:
            raise ValueError(f"Dataset {dataset_id} not found")
            
        dataset_path = Path(metadata['path'])
        
        # Count samples
        total_samples = 0
        gesture_counts = {}
        
        for gesture_dir in dataset_path.iterdir():
            if gesture_dir.is_dir():
                gesture_name = gesture_dir.name
                sample_count = len(list(gesture_dir.glob("*_features.pkl")))
                gesture_counts[gesture_name] = sample_count
                total_samples += sample_count
                
        # Update metadata
        metadata['statistics'] = {
            'total_samples': total_samples,
            'gesture_counts': gesture_counts,
            'last_updated': datetime.now().isoformat()
        }
        metadata['updated_at'] = datetime.now().isoformat()
        
        # Save updated metadata
        self._save_dataset_metadata(dataset_id, metadata)
        
        self.logger.info(f"Updated statistics for dataset {dataset_id}: {total_samples} samples")
        
        return metadata['statistics']
        
    def backup_dataset(self, dataset_id: str, backup_name: Optional[str] = None) -> str:
        """
        Create a backup of a dataset.
        
        Args:
            dataset_id: Dataset ID
            backup_name: Name for the backup (optional)
            
        Returns:
            Backup path
        """
        metadata = self.get_dataset(dataset_id)
        if not metadata:
            raise ValueError(f"Dataset {dataset_id} not found")
            
        if not backup_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{dataset_id}_backup_{timestamp}"
            
        backup_path = self.backup_path / backup_name
        
        # Copy dataset directory
        dataset_path = Path(metadata['path'])
        shutil.copytree(dataset_path, backup_path)
        
        # Copy metadata
        metadata_backup_path = backup_path / "metadata.json"
        with open(metadata_backup_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.logger.info(f"Created backup: {backup_name}")
        
        return str(backup_path)
        
    def restore_dataset(self, backup_path: str, new_dataset_name: Optional[str] = None) -> str:
        """
        Restore a dataset from backup.
        
        Args:
            backup_path: Path to backup directory
            new_dataset_name: New name for restored dataset (optional)
            
        Returns:
            Dataset ID of restored dataset
        """
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            raise ValueError(f"Backup path does not exist: {backup_path}")
            
        # Load metadata
        metadata_file = backup_path / "metadata.json"
        if not metadata_file.exists():
            raise ValueError("Backup metadata not found")
            
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        # Generate new dataset ID
        if new_dataset_name:
            dataset_name = new_dataset_name
        else:
            dataset_name = f"{metadata['name']}_restored"
            
        dataset_id = self._generate_dataset_id(dataset_name, metadata['version'])
        
        # Create new dataset directory
        new_dataset_path = self.raw_path / dataset_id
        shutil.copytree(backup_path, new_dataset_path)
        
        # Update metadata
        metadata['dataset_id'] = dataset_id
        metadata['name'] = dataset_name
        metadata['created_at'] = datetime.now().isoformat()
        metadata['updated_at'] = datetime.now().isoformat()
        metadata['path'] = str(new_dataset_path)
        metadata['status'] = 'restored'
        
        # Save metadata
        self._save_dataset_metadata(dataset_id, metadata)
        
        self.logger.info(f"Restored dataset: {dataset_name} (ID: {dataset_id})")
        
        return dataset_id
        
    def delete_dataset(self, dataset_id: str, confirm: bool = False) -> bool:
        """
        Delete a dataset.
        
        Args:
            dataset_id: Dataset ID
            confirm: Confirmation flag
            
        Returns:
            True if deleted successfully
        """
        if not confirm:
            self.logger.warning("Dataset deletion requires confirmation")
            return False
            
        metadata = self.get_dataset(dataset_id)
        if not metadata:
            self.logger.error(f"Dataset {dataset_id} not found")
            return False
            
        # Delete dataset directory
        dataset_path = Path(metadata['path'])
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
            
        # Delete metadata
        metadata_file = self.metadata_path / f"{dataset_id}.json"
        if metadata_file.exists():
            metadata_file.unlink()
            
        self.logger.info(f"Deleted dataset: {dataset_id}")
        
        return True
        
    def export_dataset(self, dataset_id: str, export_path: str, format: str = "zip") -> str:
        """
        Export dataset to external format.
        
        Args:
            dataset_id: Dataset ID
            export_path: Export destination path
            format: Export format ("zip", "tar")
            
        Returns:
            Export file path
        """
        metadata = self.get_dataset(dataset_id)
        if not metadata:
            raise ValueError(f"Dataset {dataset_id} not found")
            
        dataset_path = Path(metadata['path'])
        
        if format == "zip":
            import zipfile
            export_file = f"{export_path}.zip"
            with zipfile.ZipFile(export_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in dataset_path.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(dataset_path.parent)
                        zipf.write(file_path, arcname)
        elif format == "tar":
            import tarfile
            export_file = f"{export_path}.tar.gz"
            with tarfile.open(export_file, 'w:gz') as tarf:
                tarf.add(dataset_path, arcname=dataset_path.name)
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
        self.logger.info(f"Exported dataset {dataset_id} to {export_file}")
        
        return export_file
        
    def import_dataset(self, import_path: str, dataset_name: str) -> str:
        """
        Import dataset from external format.
        
        Args:
            import_path: Path to import file
            dataset_name: Name for imported dataset
            
        Returns:
            Dataset ID of imported dataset
        """
        import_path = Path(import_path)
        
        if not import_path.exists():
            raise ValueError(f"Import file does not exist: {import_path}")
            
        # Generate dataset ID
        dataset_id = self._generate_dataset_id(dataset_name, "1.0.0")
        dataset_path = self.raw_path / dataset_id
        
        # Extract based on file extension
        if import_path.suffix == '.zip':
            import zipfile
            with zipfile.ZipFile(import_path, 'r') as zipf:
                zipf.extractall(dataset_path.parent)
        elif import_path.suffix in ['.tar', '.gz']:
            import tarfile
            with tarfile.open(import_path, 'r:*') as tarf:
                tarf.extractall(dataset_path.parent)
        else:
            raise ValueError(f"Unsupported import format: {import_path.suffix}")
            
        # Create metadata
        metadata = {
            'dataset_id': dataset_id,
            'name': dataset_name,
            'description': f"Imported from {import_path.name}",
            'version': "1.0.0",
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'gestures': [],
            'path': str(dataset_path),
            'status': 'imported',
            'statistics': {
                'total_samples': 0,
                'gesture_counts': {},
                'last_updated': datetime.now().isoformat()
            }
        }
        
        # Update statistics
        self._save_dataset_metadata(dataset_id, metadata)
        self.update_dataset_statistics(dataset_id)
        
        self.logger.info(f"Imported dataset: {dataset_name} (ID: {dataset_id})")
        
        return dataset_id
        
    def _generate_dataset_id(self, name: str, version: str) -> str:
        """Generate unique dataset ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content = f"{name}_{version}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
        
    def _save_dataset_metadata(self, dataset_id: str, metadata: Dict[str, Any]) -> None:
        """Save dataset metadata."""
        metadata_file = self.metadata_path / f"{dataset_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get summary of all datasets."""
        datasets = self.list_datasets()
        
        summary = {
            'total_datasets': len(datasets),
            'total_samples': sum(d['statistics']['total_samples'] for d in datasets),
            'datasets': []
        }
        
        for dataset in datasets:
            summary['datasets'].append({
                'id': dataset['dataset_id'],
                'name': dataset['name'],
                'version': dataset['version'],
                'samples': dataset['statistics']['total_samples'],
                'gestures': len(dataset['gestures']),
                'created': dataset['created_at'],
                'status': dataset['status']
            })
            
        return summary
