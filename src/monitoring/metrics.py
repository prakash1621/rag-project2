"""
Metrics collection for monitoring and observability
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Metrics for cache performance"""
    tier: str
    hits: int = 0
    misses: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    hit_rate: float = 0.0
    size: int = 0
    evictions: int = 0
    errors: int = 0
    
    def update(self, hit: bool, latency_ms: float):
        """Update metrics with new request"""
        if hit:
            self.hits += 1
        else:
            self.misses += 1
        
        self.total_latency_ms += latency_ms
        total_requests = self.hits + self.misses
        self.avg_latency_ms = self.total_latency_ms / total_requests if total_requests > 0 else 0
        self.hit_rate = self.hits / total_requests if total_requests > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'tier': self.tier,
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': self.hits + self.misses,
            'hit_rate': self.hit_rate,
            'avg_latency_ms': self.avg_latency_ms,
            'size': self.size,
            'evictions': self.evictions,
            'errors': self.errors
        }


@dataclass
class PipelineMetrics:
    """Metrics for RAG pipeline performance"""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    cache_hit_rate: float = 0.0
    
    def update(self, success: bool, latency_ms: float, from_cache: bool = False):
        """Update metrics with new query"""
        self.total_queries += 1
        
        if success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1
        
        self.total_latency_ms += latency_ms
        self.avg_latency_ms = self.total_latency_ms / self.total_queries
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'total_queries': self.total_queries,
            'successful_queries': self.successful_queries,
            'failed_queries': self.failed_queries,
            'success_rate': self.successful_queries / self.total_queries if self.total_queries > 0 else 0,
            'avg_latency_ms': self.avg_latency_ms,
            'retrieval_latency_ms': self.retrieval_latency_ms,
            'generation_latency_ms': self.generation_latency_ms,
            'cache_hit_rate': self.cache_hit_rate
        }


class MetricsCollector:
    """Collects and aggregates metrics for monitoring"""
    
    def __init__(self):
        self.cache_metrics: Dict[str, CacheMetrics] = {
            'exact': CacheMetrics(tier='exact'),
            'semantic': CacheMetrics(tier='semantic'),
            'retrieval': CacheMetrics(tier='retrieval')
        }
        self.pipeline_metrics = PipelineMetrics()
        self.query_history: List[Dict] = []
        self.max_history = 1000
    
    def record_cache_access(self, tier: str, hit: bool, latency_ms: float):
        """
        Record a cache access.
        
        Args:
            tier: Cache tier ('exact', 'semantic', 'retrieval')
            hit: Whether it was a cache hit
            latency_ms: Latency in milliseconds
        """
        if tier in self.cache_metrics:
            self.cache_metrics[tier].update(hit, latency_ms)
            logger.debug(f"Cache {tier}: {'HIT' if hit else 'MISS'} ({latency_ms:.2f}ms)")
    
    def record_query(self, query: str, success: bool, latency_ms: float, 
                    from_cache: bool = False, cache_tier: Optional[str] = None,
                    error: Optional[str] = None):
        """
        Record a query execution.
        
        Args:
            query: User query
            success: Whether query was successful
            latency_ms: Total latency in milliseconds
            from_cache: Whether response came from cache
            cache_tier: Which cache tier was hit (if any)
            error: Error message (if failed)
        """
        self.pipeline_metrics.update(success, latency_ms, from_cache)
        
        # Add to history
        record = {
            'timestamp': datetime.now().isoformat(),
            'query': query[:100],  # Truncate for privacy
            'success': success,
            'latency_ms': latency_ms,
            'from_cache': from_cache,
            'cache_tier': cache_tier,
            'error': error
        }
        
        self.query_history.append(record)
        
        # Limit history size
        if len(self.query_history) > self.max_history:
            self.query_history = self.query_history[-self.max_history:]
        
        logger.info(f"Query processed: success={success}, latency={latency_ms:.2f}ms, "
                   f"from_cache={from_cache}, tier={cache_tier}")
    
    def get_summary(self) -> Dict:
        """Get summary of all metrics"""
        return {
            'pipeline': self.pipeline_metrics.to_dict(),
            'cache': {
                tier: metrics.to_dict() 
                for tier, metrics in self.cache_metrics.items()
            },
            'overall_cache_hit_rate': self._calculate_overall_cache_hit_rate()
        }
    
    def _calculate_overall_cache_hit_rate(self) -> float:
        """Calculate overall cache hit rate across all tiers"""
        total_hits = sum(m.hits for m in self.cache_metrics.values())
        total_requests = sum(m.hits + m.misses for m in self.cache_metrics.values())
        return total_hits / total_requests if total_requests > 0 else 0
    
    def get_recent_queries(self, limit: int = 10) -> List[Dict]:
        """Get recent query history"""
        return self.query_history[-limit:]
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        try:
            data = {
                'exported_at': datetime.now().isoformat(),
                'summary': self.get_summary(),
                'recent_queries': self.get_recent_queries(100)
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Metrics exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def reset(self):
        """Reset all metrics"""
        self.cache_metrics = {
            'exact': CacheMetrics(tier='exact'),
            'semantic': CacheMetrics(tier='semantic'),
            'retrieval': CacheMetrics(tier='retrieval')
        }
        self.pipeline_metrics = PipelineMetrics()
        self.query_history = []
        logger.info("Metrics reset")


# Global metrics collector instance
_metrics_collector = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
