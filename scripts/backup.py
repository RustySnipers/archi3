#!/usr/bin/env python3
"""
Backup script for Archie
Handles automated backups of configuration, data, and logs
"""

import os
import sys
import json
import tarfile
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ArchieBackup:
    def __init__(self):
        self.project_dir = Path(__file__).parent.parent
        self.data_dir = self.project_dir / "data"
        self.backup_dir = self.data_dir / "backups"
        self.config_dir = self.project_dir / "configs"
        self.logger = self._setup_logging()
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger('archie_backup')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _get_backup_paths(self) -> dict:
        """Get dictionary of paths to backup"""
        return {
            "configs": self.config_dir,
            "data": {
                "chroma": self.data_dir / "chroma",
                "logs": self.data_dir / "logs",
                "models": self.data_dir / "models",
                "homeassistant": self.data_dir / "homeassistant",
                "n8n": self.data_dir / "n8n",
                "mosquitto": self.data_dir / "mosquitto",
                "redis": self.data_dir / "redis"
            },
            "project_files": [
                self.project_dir / "docker-compose.yml",
                self.project_dir / "requirements.txt",
                self.project_dir / ".env",
                self.project_dir / "project_plan.json"
            ]
        }
    
    def create_backup(self, backup_type: str = "full") -> dict:
        """Create a backup archive"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"archie_backup_{backup_type}_{timestamp}.tar.gz"
        backup_path = self.backup_dir / backup_name
        
        self.logger.info(f"Creating {backup_type} backup: {backup_name}")
        
        try:
            with tarfile.open(backup_path, "w:gz") as tar:
                backup_paths = self._get_backup_paths()
                
                # Add configuration files
                if backup_paths["configs"].exists():
                    tar.add(backup_paths["configs"], arcname="configs")
                    self.logger.info("Added configuration files to backup")
                
                # Add data directories
                for data_name, data_path in backup_paths["data"].items():
                    if data_path.exists():
                        tar.add(data_path, arcname=f"data/{data_name}")
                        self.logger.info(f"Added {data_name} data to backup")
                
                # Add project files
                for project_file in backup_paths["project_files"]:
                    if project_file.exists():
                        tar.add(project_file, arcname=f"project/{project_file.name}")
                        self.logger.info(f"Added {project_file.name} to backup")
                
                # Add metadata
                metadata = {
                    "backup_type": backup_type,
                    "timestamp": timestamp,
                    "version": "1.0.0",
                    "created_by": "archie_backup",
                    "paths_included": [str(p) for p in backup_paths["project_files"] if p.exists()]
                }
                
                # Create temporary metadata file
                metadata_path = self.backup_dir / f"metadata_{timestamp}.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                
                tar.add(metadata_path, arcname="metadata.json")
                metadata_path.unlink()  # Remove temporary file
            
            # Calculate backup checksum
            checksum = self._calculate_checksum(backup_path)
            
            # Create backup info
            backup_info = {
                "name": backup_name,
                "path": str(backup_path),
                "size_bytes": backup_path.stat().st_size,
                "size_mb": round(backup_path.stat().st_size / (1024 * 1024), 2),
                "checksum": checksum,
                "created_at": datetime.now().isoformat(),
                "type": backup_type,
                "status": "completed"
            }
            
            self.logger.info(f"Backup created successfully: {backup_name} ({backup_info['size_mb']} MB)")
            return backup_info
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            # Clean up partial backup
            if backup_path.exists():
                backup_path.unlink()
            return {
                "name": backup_name,
                "status": "failed",
                "error": str(e),
                "created_at": datetime.now().isoformat()
            }
    
    def list_backups(self) -> list:
        """List all available backups"""
        backups = []
        
        for backup_file in self.backup_dir.glob("archie_backup_*.tar.gz"):
            try:
                stat = backup_file.stat()
                backups.append({
                    "name": backup_file.name,
                    "path": str(backup_file),
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "age_days": (datetime.now() - datetime.fromtimestamp(stat.st_mtime)).days
                })
            except Exception as e:
                self.logger.warning(f"Could not read backup info for {backup_file}: {e}")
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x["created_at"], reverse=True)
        return backups
    
    def cleanup_old_backups(self, retention_days: int = 30) -> dict:
        """Remove backups older than retention period"""
        self.logger.info(f"Cleaning up backups older than {retention_days} days")
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        removed_count = 0
        removed_size = 0
        
        for backup_file in self.backup_dir.glob("archie_backup_*.tar.gz"):
            try:
                file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                if file_time < cutoff_date:
                    file_size = backup_file.stat().st_size
                    backup_file.unlink()
                    removed_count += 1
                    removed_size += file_size
                    self.logger.info(f"Removed old backup: {backup_file.name}")
            except Exception as e:
                self.logger.warning(f"Could not remove backup {backup_file}: {e}")
        
        cleanup_info = {
            "removed_count": removed_count,
            "removed_size_mb": round(removed_size / (1024 * 1024), 2),
            "retention_days": retention_days,
            "cleanup_date": datetime.now().isoformat()
        }
        
        self.logger.info(f"Cleanup completed: {removed_count} backups removed ({cleanup_info['removed_size_mb']} MB freed)")
        return cleanup_info
    
    def verify_backup(self, backup_name: str) -> dict:
        """Verify backup integrity"""
        backup_path = self.backup_dir / backup_name
        
        if not backup_path.exists():
            return {
                "status": "failed",
                "error": "Backup file not found"
            }
        
        try:
            # Test if archive can be opened
            with tarfile.open(backup_path, "r:gz") as tar:
                members = tar.getmembers()
                
                # Check if metadata exists
                has_metadata = any(member.name == "metadata.json" for member in members)
                
                # Basic integrity check
                verification_info = {
                    "status": "valid",
                    "file_count": len(members),
                    "has_metadata": has_metadata,
                    "size_mb": round(backup_path.stat().st_size / (1024 * 1024), 2),
                    "verified_at": datetime.now().isoformat()
                }
                
                self.logger.info(f"Backup verification passed: {backup_name}")
                return verification_info
                
        except Exception as e:
            verification_info = {
                "status": "corrupted",
                "error": str(e),
                "verified_at": datetime.now().isoformat()
            }
            self.logger.error(f"Backup verification failed: {backup_name} - {e}")
            return verification_info
    
    def restore_backup(self, backup_name: str, restore_path: str = None) -> dict:
        """Restore from backup"""
        backup_path = self.backup_dir / backup_name
        
        if not backup_path.exists():
            return {
                "status": "failed",
                "error": "Backup file not found"
            }
        
        if restore_path is None:
            restore_path = self.project_dir / "restored"
        else:
            restore_path = Path(restore_path)
        
        try:
            # Create restore directory
            restore_path.mkdir(parents=True, exist_ok=True)
            
            # Extract backup
            with tarfile.open(backup_path, "r:gz") as tar:
                tar.extractall(path=restore_path)
            
            restore_info = {
                "status": "completed",
                "backup_name": backup_name,
                "restore_path": str(restore_path),
                "restored_at": datetime.now().isoformat()
            }
            
            self.logger.info(f"Backup restored successfully: {backup_name} to {restore_path}")
            return restore_info
            
        except Exception as e:
            restore_info = {
                "status": "failed",
                "error": str(e),
                "backup_name": backup_name,
                "attempted_at": datetime.now().isoformat()
            }
            self.logger.error(f"Backup restore failed: {backup_name} - {e}")
            return restore_info

async def main():
    """Main entry point for backup script"""
    backup = ArchieBackup()
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python backup.py <command> [options]")
        print("Commands: create, list, cleanup, verify, restore")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "create":
        backup_type = sys.argv[2] if len(sys.argv) > 2 else "full"
        result = backup.create_backup(backup_type)
        print(json.dumps(result, indent=2))
        
    elif command == "list":
        backups = backup.list_backups()
        print(json.dumps(backups, indent=2))
        
    elif command == "cleanup":
        retention_days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        result = backup.cleanup_old_backups(retention_days)
        print(json.dumps(result, indent=2))
        
    elif command == "verify":
        if len(sys.argv) < 3:
            print("Usage: python backup.py verify <backup_name>")
            sys.exit(1)
        backup_name = sys.argv[2]
        result = backup.verify_backup(backup_name)
        print(json.dumps(result, indent=2))
        
    elif command == "restore":
        if len(sys.argv) < 3:
            print("Usage: python backup.py restore <backup_name> [restore_path]")
            sys.exit(1)
        backup_name = sys.argv[2]
        restore_path = sys.argv[3] if len(sys.argv) > 3 else None
        result = backup.restore_backup(backup_name, restore_path)
        print(json.dumps(result, indent=2))
        
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())