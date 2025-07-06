#!/usr/bin/env python3
"""
Health check script for Archie
Monitors system health and reports status
"""

import os
import sys
import time
import json
import logging
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class HealthChecker:
    def __init__(self):
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.services = {
            'localai': {'url': 'http://localhost:8080/readyz', 'timeout': 10},
            'homeassistant': {'url': 'http://localhost:8123', 'timeout': 10},
            'n8n': {'url': 'http://localhost:5678/healthz', 'timeout': 10},
            'chromadb': {'url': 'http://localhost:8000/api/v1/heartbeat', 'timeout': 5},
            'mosquitto': {'host': 'localhost', 'port': 1883, 'timeout': 5},
            'redis': {'host': 'localhost', 'port': 6379, 'timeout': 5},
            'prometheus': {'url': 'http://localhost:9090/-/healthy', 'timeout': 5},
            'grafana': {'url': 'http://localhost:3000/api/health', 'timeout': 5}
        }
        self.health_status = {}
        
    def _load_config(self) -> Dict:
        """Load configuration from environment or config file"""
        config_path = os.getenv('ARCHIE_CONFIG_PATH', '/opt/archie/configs')
        config_file = os.path.join(config_path, 'archie_config.yaml')
        
        if os.path.exists(config_file):
            import yaml
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger('health_check')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def check_http_service(self, name: str, url: str, timeout: int = 10) -> Dict:
        """Check HTTP service health"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return {
                            'name': name,
                            'status': 'healthy',
                            'response_time': response.headers.get('X-Response-Time', 'N/A'),
                            'status_code': response.status
                        }
                    else:
                        return {
                            'name': name,
                            'status': 'unhealthy',
                            'error': f'HTTP {response.status}',
                            'status_code': response.status
                        }
        except Exception as e:
            return {
                'name': name,
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def check_tcp_service(self, name: str, host: str, port: int, timeout: int = 5) -> Dict:
        """Check TCP service health"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=timeout
            )
            writer.close()
            await writer.wait_closed()
            return {
                'name': name,
                'status': 'healthy',
                'host': host,
                'port': port
            }
        except Exception as e:
            return {
                'name': name,
                'status': 'unhealthy',
                'error': str(e),
                'host': host,
                'port': port
            }
    
    async def check_disk_space(self) -> Dict:
        """Check available disk space"""
        try:
            data_path = os.getenv('ARCHIE_DATA_PATH', '/opt/archie/data')
            statvfs = os.statvfs(data_path)
            
            # Calculate space in GB
            total_space = (statvfs.f_frsize * statvfs.f_blocks) / (1024**3)
            free_space = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
            used_space = total_space - free_space
            usage_percent = (used_space / total_space) * 100
            
            status = 'healthy'
            if usage_percent > 90:
                status = 'critical'
            elif usage_percent > 80:
                status = 'warning'
            
            return {
                'name': 'disk_space',
                'status': status,
                'total_gb': round(total_space, 2),
                'free_gb': round(free_space, 2),
                'used_gb': round(used_space, 2),
                'usage_percent': round(usage_percent, 2)
            }
        except Exception as e:
            return {
                'name': 'disk_space',
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def check_memory_usage(self) -> Dict:
        """Check memory usage"""
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            
            # Parse memory information
            lines = meminfo.strip().split('\n')
            mem_data = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    mem_data[key.strip()] = value.strip()
            
            # Calculate usage
            total_kb = int(mem_data['MemTotal'].split()[0])
            available_kb = int(mem_data['MemAvailable'].split()[0])
            used_kb = total_kb - available_kb
            usage_percent = (used_kb / total_kb) * 100
            
            status = 'healthy'
            if usage_percent > 90:
                status = 'critical'
            elif usage_percent > 80:
                status = 'warning'
            
            return {
                'name': 'memory_usage',
                'status': status,
                'total_mb': round(total_kb / 1024, 2),
                'used_mb': round(used_kb / 1024, 2),
                'available_mb': round(available_kb / 1024, 2),
                'usage_percent': round(usage_percent, 2)
            }
        except Exception as e:
            return {
                'name': 'memory_usage',
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def check_all_services(self) -> Dict:
        """Check all services and system health"""
        checks = []
        
        # Check HTTP services
        for service, config in self.services.items():
            if 'url' in config:
                checks.append(
                    self.check_http_service(service, config['url'], config['timeout'])
                )
        
        # Check TCP services
        for service, config in self.services.items():
            if 'host' in config and 'port' in config:
                checks.append(
                    self.check_tcp_service(service, config['host'], config['port'], config['timeout'])
                )
        
        # Check system resources
        checks.extend([
            self.check_disk_space(),
            self.check_memory_usage()
        ])
        
        # Execute all checks concurrently
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        # Process results
        healthy_count = 0
        total_count = len(results)
        service_status = {}
        
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Health check failed: {result}")
                continue
            
            service_name = result['name']
            service_status[service_name] = result
            
            if result['status'] == 'healthy':
                healthy_count += 1
                self.logger.info(f"Service {service_name}: healthy")
            else:
                self.logger.warning(f"Service {service_name}: {result['status']} - {result.get('error', 'Unknown error')}")
        
        # Determine overall health
        overall_status = 'healthy' if healthy_count == total_count else 'unhealthy'
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': overall_status,
            'healthy_services': healthy_count,
            'total_services': total_count,
            'services': service_status
        }
    
    async def run_health_check(self) -> int:
        """Run health check and return exit code"""
        try:
            self.logger.info("Starting health check...")
            health_report = await self.check_all_services()
            
            # Log summary
            self.logger.info(f"Health check completed. Status: {health_report['overall_status']}")
            self.logger.info(f"Healthy services: {health_report['healthy_services']}/{health_report['total_services']}")
            
            # Save health report
            data_path = os.getenv('ARCHIE_DATA_PATH', '/opt/archie/data')
            health_file = os.path.join(data_path, 'health_status.json')
            
            os.makedirs(os.path.dirname(health_file), exist_ok=True)
            with open(health_file, 'w') as f:
                json.dump(health_report, f, indent=2)
            
            # Return appropriate exit code
            return 0 if health_report['overall_status'] == 'healthy' else 1
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return 1

def main():
    """Main entry point"""
    checker = HealthChecker()
    exit_code = asyncio.run(checker.run_health_check())
    sys.exit(exit_code)

if __name__ == "__main__":
    main()