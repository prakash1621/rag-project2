"""
Tier 1: Exact Cache

Matches queries exactly (after normalization) and returns cached responses immediately.
No retrieval, no LLM call.
"""

import time
from typing import Optional, Dict
from .base_cache import BaseExactCache, CacheOperationError


class ExactCache(BaseExactCache):
    def __init__(self, ttl_seconds: int = 3600, normalize: bool = True):
        """
        Args:
            ttl_seconds: Time-to-live for cache entries in seconds
            normalize: Whether to normalize queries (lowercase, strip whitespace)
        """
        super().__init__(ttl_seconds, normalize)
        self.cache: Dict[str, Dict] = {}
    

    
    def get(self, query: str) -> Optional[str]:
        """
        Get cached response for exact query match.
        
        Returns:
            Cached response if found and not expired, None otherwise
            
        Raises:
            CacheOperationError: If cache operation fails
        """
        try:
            normalized = self.normalize_query(query, self.normalize)
            cache_key = self.generate_cache_key(normalized)
        
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                
                # Check if expired
                if time.time() - entry['timestamp'] < self.ttl_seconds:
                    self._record_hit()
                    entry['access_count'] += 1
                    entry['last_accessed'] = time.time()
                    return entry['response']
                else:
                    # Expired, remove from cache
                    del self.cache[cache_key]
            
            self._record_miss()
            return None
        except Exception as e:
            self._logger.error(f"Cache get operation failed: {e}")
            raise CacheOperationError(f"Failed to get from cache: {e}")
    
    def set(self, query: str, response: str) -> None:
        """
        Cache a query-response pair.
        
        Raises:
            CacheOperationError: If cache operation fails
        """
        try:
            normalized = self.normalize_query(query, self.normalize)
            cache_key = self.generate_cache_key(normalized)
            
            self.cache[cache_key] = {
                'query': query,
                'response': response,
                'timestamp': time.time(),
                'last_accessed': time.time(),
                'access_count': 0
            }
            self._logger.debug(f"Cached response for query: {query[:50]}...")
        except Exception as e:
            self._logger.error(f"Cache set operation failed: {e}")
            raise CacheOperationError(f"Failed to set cache: {e}")
    
    def clear(self) -> None:
        """
        Clear all cache entries.
        
        Raises:
            CacheOperationError: If cache operation fails
        """
        try:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            self._logger.info("Cache cleared successfully")
        except Exception as e:
            self._logger.error(f"Cache clear operation failed: {e}")
            raise CacheOperationError(f"Failed to clear cache: {e}")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        stats = super().get_stats()
        stats.update({
            'backend': 'memory',
            'size': len(self.cache)
        })
        return stats
