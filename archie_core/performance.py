"""
Performance Optimization System for Archie
Monitors and optimizes system performance, resource usage, and response times
"""

import os
import psutil
import logging
import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import threading
import gc
import weakref

logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """Performance optimization levels"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

class ResourceType(Enum):
    """System resource types"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"

@dataclass
class ResourceMetrics:
    """System resource metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    open_files: int
    threads_count: int
    process_count: int

@dataclass
class PerformanceAlert:
    """Performance alert when thresholds exceeded"""
    id: str
    alert_type: str
    resource: ResourceType
    threshold: float
    current_value: float
    severity: str
    timestamp: datetime
    resolved: bool = False

@dataclass
class OptimizationAction:
    """Action taken to optimize performance"""
    id: str
    action_type: str
    description: str
    target_component: str
    parameters: Dict[str, Any]
    expected_impact: str
    timestamp: datetime
    success: bool = False
    actual_impact: Optional[str] = None

class CacheManager:
    """Intelligent cache management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.caches = {}
        self.cache_stats = defaultdict(lambda: {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "memory_usage": 0
        })
        self.max_total_memory = config.get('max_cache_memory_mb', 512) * 1024 * 1024  # bytes
        
    def register_cache(self, name: str, max_size: int, ttl_seconds: int = 3600):
        """Register a new cache"""
        self.caches[name] = {
            "data": {},
            "access_times": {},
            "max_size": max_size,
            "ttl": ttl_seconds,
            "created_at": time.time()
        }
        logger.info(f"Registered cache '{name}' with max size {max_size}")
    
    def get(self, cache_name: str, key: str) -> Any:
        """Get item from cache"""
        if cache_name not in self.caches:
            return None
        
        cache = self.caches[cache_name]
        current_time = time.time()
        
        # Check if key exists and is not expired
        if key in cache["data"]:
            created_time = cache["access_times"].get(key, 0)
            if current_time - created_time < cache["ttl"]:
                # Update access time
                cache["access_times"][key] = current_time
                self.cache_stats[cache_name]["hits"] += 1
                return cache["data"][key]
            else:
                # Expired, remove
                self._remove_from_cache(cache_name, key)
        
        self.cache_stats[cache_name]["misses"] += 1
        return None
    
    def put(self, cache_name: str, key: str, value: Any):
        """Put item in cache"""
        if cache_name not in self.caches:
            return False
        
        cache = self.caches[cache_name]
        current_time = time.time()
        
        # Check if we need to evict items
        if len(cache["data"]) >= cache["max_size"]:
            self._evict_lru(cache_name)
        
        # Store item
        cache["data"][key] = value
        cache["access_times"][key] = current_time
        
        return True
    
    def _remove_from_cache(self, cache_name: str, key: str):
        """Remove item from cache"""
        cache = self.caches[cache_name]
        if key in cache["data"]:
            del cache["data"][key]
            del cache["access_times"][key]
    
    def _evict_lru(self, cache_name: str):
        """Evict least recently used item"""
        cache = self.caches[cache_name]
        if not cache["access_times"]:
            return
        
        # Find LRU item
        lru_key = min(cache["access_times"], key=cache["access_times"].get)
        self._remove_from_cache(cache_name, lru_key)
        self.cache_stats[cache_name]["evictions"] += 1
    
    def clear_cache(self, cache_name: str):
        """Clear specific cache"""
        if cache_name in self.caches:
            self.caches[cache_name]["data"].clear()
            self.caches[cache_name]["access_times"].clear()
            logger.info(f"Cleared cache '{cache_name}'")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {}
        total_memory = 0
        
        for name, cache in self.caches.items():
            cache_size = len(cache["data"])
            hit_ratio = 0
            if self.cache_stats[name]["hits"] + self.cache_stats[name]["misses"] > 0:
                hit_ratio = self.cache_stats[name]["hits"] / (
                    self.cache_stats[name]["hits"] + self.cache_stats[name]["misses"]
                )
            
            # Estimate memory usage
            memory_usage = cache_size * 1024  # Rough estimate
            total_memory += memory_usage
            
            stats[name] = {
                "size": cache_size,
                "max_size": cache["max_size"],
                "hit_ratio": hit_ratio,
                "memory_usage_kb": memory_usage // 1024,
                **self.cache_stats[name]
            }
        
        stats["total_memory_mb"] = total_memory // (1024 * 1024)
        stats["memory_limit_mb"] = self.max_total_memory // (1024 * 1024)
        return stats

class ResourceMonitor:
    """System resource monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitoring = False
        self.metrics_history = deque(maxlen=1000)
        
        # Thresholds for alerts
        self.thresholds = {
            ResourceType.CPU: config.get('cpu_threshold', 80.0),
            ResourceType.MEMORY: config.get('memory_threshold', 85.0),
            ResourceType.DISK: config.get('disk_threshold', 90.0),
            ResourceType.NETWORK: config.get('network_threshold', 100.0)  # MB/s
        }
        
        # Alert management
        self.active_alerts = {}
        self.alert_history = deque(maxlen=100)
        
        # Monitoring task
        self.monitor_task = None
        self.monitor_interval = config.get('monitor_interval', 5)  # seconds
        
    async def start_monitoring(self):
        """Start resource monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Resource monitoring started")
    
    async def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Resource monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self.monitoring:
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check for threshold violations
                await self._check_thresholds(metrics)
                
                await asyncio.sleep(self.monitor_interval)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
    
    async def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_io_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
            disk_io_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_sent_mb = net_io.bytes_sent / (1024 * 1024) if net_io else 0
            network_recv_mb = net_io.bytes_recv / (1024 * 1024) if net_io else 0
            
            # Process metrics
            process = psutil.Process()
            open_files = len(process.open_files())
            threads_count = process.num_threads()
            process_count = len(psutil.pids())
            
            return ResourceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                disk_usage_percent=disk_usage_percent,
                disk_io_read_mb=disk_io_read_mb,
                disk_io_write_mb=disk_io_write_mb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                open_files=open_files,
                threads_count=threads_count,
                process_count=process_count
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return ResourceMetrics(
                timestamp=datetime.now(),
                cpu_percent=0, memory_percent=0, memory_used_mb=0,
                disk_usage_percent=0, disk_io_read_mb=0, disk_io_write_mb=0,
                network_sent_mb=0, network_recv_mb=0,
                open_files=0, threads_count=0, process_count=0
            )
    
    async def _check_thresholds(self, metrics: ResourceMetrics):
        """Check if metrics exceed thresholds"""
        current_alerts = set()
        
        # CPU threshold
        if metrics.cpu_percent > self.thresholds[ResourceType.CPU]:
            alert_id = f"cpu_high_{int(metrics.timestamp.timestamp())}"
            alert = PerformanceAlert(
                id=alert_id,
                alert_type="high_cpu",
                resource=ResourceType.CPU,
                threshold=self.thresholds[ResourceType.CPU],
                current_value=metrics.cpu_percent,
                severity="warning" if metrics.cpu_percent < 95 else "critical",
                timestamp=metrics.timestamp
            )
            await self._handle_alert(alert)
            current_alerts.add(ResourceType.CPU)
        
        # Memory threshold
        if metrics.memory_percent > self.thresholds[ResourceType.MEMORY]:
            alert_id = f"memory_high_{int(metrics.timestamp.timestamp())}"
            alert = PerformanceAlert(
                id=alert_id,
                alert_type="high_memory",
                resource=ResourceType.MEMORY,
                threshold=self.thresholds[ResourceType.MEMORY],
                current_value=metrics.memory_percent,
                severity="warning" if metrics.memory_percent < 95 else "critical",
                timestamp=metrics.timestamp
            )
            await self._handle_alert(alert)
            current_alerts.add(ResourceType.MEMORY)
        
        # Disk threshold
        if metrics.disk_usage_percent > self.thresholds[ResourceType.DISK]:
            alert_id = f"disk_high_{int(metrics.timestamp.timestamp())}"
            alert = PerformanceAlert(
                id=alert_id,
                alert_type="high_disk",
                resource=ResourceType.DISK,
                threshold=self.thresholds[ResourceType.DISK],
                current_value=metrics.disk_usage_percent,
                severity="warning" if metrics.disk_usage_percent < 98 else "critical",
                timestamp=metrics.timestamp
            )
            await self._handle_alert(alert)
            current_alerts.add(ResourceType.DISK)
        
        # Resolve alerts that are no longer active
        for resource in list(self.active_alerts.keys()):
            if resource not in current_alerts:
                self.active_alerts[resource].resolved = True
                del self.active_alerts[resource]
    
    async def _handle_alert(self, alert: PerformanceAlert):
        """Handle performance alert"""
        self.active_alerts[alert.resource] = alert
        self.alert_history.append(alert)
        
        logger.warning(f"Performance alert: {alert.alert_type} - {alert.resource.value} "
                      f"at {alert.current_value:.1f}% (threshold: {alert.threshold:.1f}%)")
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get most recent metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, minutes: int = 60) -> List[ResourceMetrics]:
        """Get metrics history for specified duration"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get currently active alerts"""
        return list(self.active_alerts.values())

class PerformanceOptimizer:
    """Main performance optimization system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.optimization_level = OptimizationLevel(
            config.get('optimization_level', 'balanced')
        )
        
        # Components
        self.cache_manager = CacheManager(config.get('cache', {}))
        self.resource_monitor = ResourceMonitor(config.get('monitoring', {}))
        
        # Optimization tracking
        self.optimization_actions = deque(maxlen=1000)
        self.performance_baseline = None
        
        # Optimization parameters
        self.auto_optimize = config.get('auto_optimize', True)
        self.optimization_interval = config.get('optimization_interval', 60)  # seconds
        
        # Weak references to monitored objects
        self.monitored_objects = weakref.WeakSet()
        
        # Statistics
        self.stats = {
            "optimizations_performed": 0,
            "memory_freed_mb": 0,
            "cache_hit_improvement": 0.0,
            "response_time_improvement": 0.0,
            "last_optimization": None
        }
        
        # Optimization task
        self.optimization_task = None
        self.running = False
    
    async def initialize(self):
        """Initialize performance optimization system"""
        try:
            # Start resource monitoring
            await self.resource_monitor.start_monitoring()
            
            # Set up default caches
            await self._setup_default_caches()
            
            # Collect baseline performance
            await self._collect_baseline()
            
            # Start auto-optimization if enabled
            if self.auto_optimize:
                await self.start_auto_optimization()
            
            logger.info(f"Performance optimizer initialized with {self.optimization_level.value} level")
            
        except Exception as e:
            logger.error(f"Failed to initialize performance optimizer: {e}")
            raise
    
    async def _setup_default_caches(self):
        """Set up default caches for common components"""
        # Memory cache
        self.cache_manager.register_cache("memory_results", 1000, 3600)
        
        # LLM response cache
        self.cache_manager.register_cache("llm_responses", 500, 1800)
        
        # Tool result cache
        self.cache_manager.register_cache("tool_results", 200, 600)
        
        # Multi-modal processing cache
        self.cache_manager.register_cache("multimodal_processing", 100, 7200)
        
        logger.info("Default caches configured")
    
    async def _collect_baseline(self):
        """Collect baseline performance metrics"""
        try:
            # Wait for initial metrics
            await asyncio.sleep(10)
            
            metrics = self.resource_monitor.get_current_metrics()
            if metrics:
                self.performance_baseline = {
                    "cpu_percent": metrics.cpu_percent,
                    "memory_percent": metrics.memory_percent,
                    "response_time": 1.0  # Default baseline
                }
                logger.info("Performance baseline established")
            
        except Exception as e:
            logger.error(f"Error collecting baseline: {e}")
    
    async def start_auto_optimization(self):
        """Start automatic optimization"""
        if self.running:
            return
        
        self.running = True
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        logger.info("Auto-optimization started")
    
    async def stop_auto_optimization(self):
        """Stop automatic optimization"""
        self.running = False
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
        logger.info("Auto-optimization stopped")
    
    async def _optimization_loop(self):
        """Main optimization loop"""
        try:
            while self.running:
                await self._perform_optimization_cycle()
                await asyncio.sleep(self.optimization_interval)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in optimization loop: {e}")
    
    async def _perform_optimization_cycle(self):
        """Perform one optimization cycle"""
        try:
            current_metrics = self.resource_monitor.get_current_metrics()
            if not current_metrics:
                return
            
            optimizations_applied = 0
            
            # Memory optimization
            if current_metrics.memory_percent > 75:
                if await self._optimize_memory():
                    optimizations_applied += 1
            
            # Cache optimization
            cache_stats = self.cache_manager.get_cache_stats()
            if cache_stats.get("total_memory_mb", 0) > 100:
                if await self._optimize_caches():
                    optimizations_applied += 1
            
            # Garbage collection optimization
            if current_metrics.memory_percent > 80:
                if await self._optimize_garbage_collection():
                    optimizations_applied += 1
            
            # Update statistics
            if optimizations_applied > 0:
                self.stats["optimizations_performed"] += optimizations_applied
                self.stats["last_optimization"] = datetime.now()
                
                logger.info(f"Applied {optimizations_applied} optimizations")
            
        except Exception as e:
            logger.error(f"Error in optimization cycle: {e}")
    
    async def _optimize_memory(self) -> bool:
        """Optimize memory usage"""
        try:
            initial_memory = psutil.virtual_memory().percent
            
            # Clear object caches
            cleared_objects = 0
            for obj in list(self.monitored_objects):
                if hasattr(obj, 'clear_cache'):
                    obj.clear_cache()
                    cleared_objects += 1
            
            # Force garbage collection
            collected = gc.collect()
            
            # Wait for memory to update
            await asyncio.sleep(2)
            final_memory = psutil.virtual_memory().percent
            
            memory_freed = initial_memory - final_memory
            if memory_freed > 0:
                self.stats["memory_freed_mb"] += memory_freed * psutil.virtual_memory().total / (100 * 1024 * 1024)
                
                action = OptimizationAction(
                    id=f"memory_opt_{int(time.time())}",
                    action_type="memory_optimization",
                    description=f"Cleared {cleared_objects} object caches, collected {collected} objects",
                    target_component="memory_management",
                    parameters={"objects_cleared": cleared_objects, "gc_collected": collected},
                    expected_impact=f"Reduce memory by {memory_freed:.1f}%",
                    timestamp=datetime.now(),
                    success=True,
                    actual_impact=f"Reduced memory by {memory_freed:.1f}%"
                )
                
                self.optimization_actions.append(action)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error optimizing memory: {e}")
            return False
    
    async def _optimize_caches(self) -> bool:
        """Optimize cache usage"""
        try:
            cache_stats = self.cache_manager.get_cache_stats()
            optimized = False
            
            # Clear caches with low hit ratios
            for cache_name, stats in cache_stats.items():
                if isinstance(stats, dict) and stats.get("hit_ratio", 1.0) < 0.3:
                    self.cache_manager.clear_cache(cache_name)
                    optimized = True
                    logger.info(f"Cleared cache '{cache_name}' due to low hit ratio ({stats['hit_ratio']:.2f})")
            
            if optimized:
                action = OptimizationAction(
                    id=f"cache_opt_{int(time.time())}",
                    action_type="cache_optimization",
                    description="Cleared underperforming caches",
                    target_component="cache_management",
                    parameters={"method": "low_hit_ratio_cleanup"},
                    expected_impact="Improve cache efficiency",
                    timestamp=datetime.now(),
                    success=True
                )
                
                self.optimization_actions.append(action)
            
            return optimized
            
        except Exception as e:
            logger.error(f"Error optimizing caches: {e}")
            return False
    
    async def _optimize_garbage_collection(self) -> bool:
        """Optimize garbage collection"""
        try:
            # Get current GC stats
            gc_stats = gc.get_stats()
            
            # Force full garbage collection
            collected = 0
            for generation in range(3):
                collected += gc.collect(generation)
            
            # Disable automatic GC temporarily for performance
            if self.optimization_level in [OptimizationLevel.AGGRESSIVE, OptimizationLevel.MAXIMUM]:
                gc.disable()
                # Re-enable after delay
                asyncio.create_task(self._re_enable_gc(30))
            
            if collected > 0:
                action = OptimizationAction(
                    id=f"gc_opt_{int(time.time())}",
                    action_type="garbage_collection_optimization",
                    description=f"Collected {collected} objects",
                    target_component="garbage_collector",
                    parameters={"objects_collected": collected},
                    expected_impact="Reduce memory fragmentation",
                    timestamp=datetime.now(),
                    success=True
                )
                
                self.optimization_actions.append(action)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error optimizing garbage collection: {e}")
            return False
    
    async def _re_enable_gc(self, delay_seconds: int):
        """Re-enable garbage collection after delay"""
        await asyncio.sleep(delay_seconds)
        gc.enable()
        logger.info("Garbage collection re-enabled")
    
    def register_monitored_object(self, obj: Any):
        """Register object for performance monitoring"""
        self.monitored_objects.add(obj)
    
    def cache_get(self, cache_name: str, key: str) -> Any:
        """Get item from cache"""
        return self.cache_manager.get(cache_name, key)
    
    def cache_put(self, cache_name: str, key: str, value: Any):
        """Put item in cache"""
        return self.cache_manager.put(cache_name, key, value)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        try:
            current_metrics = self.resource_monitor.get_current_metrics()
            cache_stats = self.cache_manager.get_cache_stats()
            active_alerts = self.resource_monitor.get_active_alerts()
            
            # Calculate performance improvements
            improvements = {}
            if self.performance_baseline and current_metrics:
                improvements = {
                    "cpu_change": self.performance_baseline["cpu_percent"] - current_metrics.cpu_percent,
                    "memory_change": self.performance_baseline["memory_percent"] - current_metrics.memory_percent,
                    "memory_freed_total_mb": self.stats["memory_freed_mb"]
                }
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "optimization_level": self.optimization_level.value,
                "current_metrics": asdict(current_metrics) if current_metrics else {},
                "performance_baseline": self.performance_baseline,
                "improvements": improvements,
                "cache_stats": cache_stats,
                "active_alerts": [asdict(alert) for alert in active_alerts],
                "recent_optimizations": [asdict(action) for action in list(self.optimization_actions)[-10:]],
                "system_stats": self.stats,
                "recommendations": self._generate_recommendations()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        try:
            current_metrics = self.resource_monitor.get_current_metrics()
            if not current_metrics:
                return recommendations
            
            # Memory recommendations
            if current_metrics.memory_percent > 85:
                recommendations.append("Consider increasing system memory or reducing memory-intensive operations")
            
            # CPU recommendations
            if current_metrics.cpu_percent > 80:
                recommendations.append("High CPU usage detected - consider optimizing computational tasks")
            
            # Cache recommendations
            cache_stats = self.cache_manager.get_cache_stats()
            total_hit_ratio = 0
            cache_count = 0
            
            for cache_name, stats in cache_stats.items():
                if isinstance(stats, dict) and "hit_ratio" in stats:
                    total_hit_ratio += stats["hit_ratio"]
                    cache_count += 1
            
            if cache_count > 0:
                avg_hit_ratio = total_hit_ratio / cache_count
                if avg_hit_ratio < 0.6:
                    recommendations.append("Cache hit ratio is low - consider adjusting cache sizes or TTL values")
            
            # Optimization level recommendations
            if self.optimization_level == OptimizationLevel.CONSERVATIVE:
                recommendations.append("Consider using 'balanced' optimization level for better performance")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]
    
    async def shutdown(self):
        """Shutdown performance optimization system"""
        try:
            await self.stop_auto_optimization()
            await self.resource_monitor.stop_monitoring()
            
            # Re-enable garbage collection if disabled
            gc.enable()
            
            logger.info("Performance optimization system shutdown")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Factory function
async def create_performance_optimizer(config: Dict[str, Any]) -> PerformanceOptimizer:
    """Create and initialize performance optimizer"""
    optimizer = PerformanceOptimizer(config.get('performance', {}))
    await optimizer.initialize()
    return optimizer

# Decorators for performance monitoring
def monitor_performance(func):
    """Decorator to monitor function performance"""
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"Function {func.__name__} took {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"Function {func.__name__} took {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

def cache_result(cache_name: str, ttl: int = 3600):
    """Decorator to cache function results"""
    def decorator(func):
        async def async_wrapper(self, *args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            if hasattr(self, 'performance_optimizer'):
                cached_result = self.performance_optimizer.cache_get(cache_name, cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Execute function
            result = await func(self, *args, **kwargs)
            
            # Store in cache
            if hasattr(self, 'performance_optimizer'):
                self.performance_optimizer.cache_put(cache_name, cache_key, result)
            
            return result
        
        def sync_wrapper(self, *args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            if hasattr(self, 'performance_optimizer'):
                cached_result = self.performance_optimizer.cache_get(cache_name, cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Execute function
            result = func(self, *args, **kwargs)
            
            # Store in cache
            if hasattr(self, 'performance_optimizer'):
                self.performance_optimizer.cache_put(cache_name, cache_key, result)
            
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator