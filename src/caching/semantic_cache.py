"""
Tier 2: Semantic Cache

Compares query embeddings using cosine similarity.
If similarity exceeds threshold, returns cached response.
No retrieval, no LLM call.
"""

import time
import numpy as np
from typing import Optional, Dict, List, Tuple
from collections import OrderedDict
from .base_cache import BaseSemanticCache, CacheOperationError


class SemanticCache(BaseSemanticCache):
    def __init__(self, embedder, similarity_threshold: float = 0.95,
                 ttl_seconds: int = 3600, max_cache_size: int = 1000):
        """
        Args:
            embedder: Function that takes text and returns embedding vector
            similarity_threshold: Minimum cosine similarity for cache hit
            ttl_seconds: Time-to-live for cache entries
            max_cache_size: Maximum number of entries (LRU eviction)
        """
        super().__init__(embedder, similarity_threshold, ttl_seconds, max_cache_size)
        
        # Use OrderedDict for LRU behavior
        self.cache: OrderedDict[str, Dict] = OrderedDict()
        self.embeddings: Dict[str, np.ndarray] = {}
    

    
    def _find_similar_query(self, query_embedding: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Find most similar cached query above threshold.
        
        Returns:
            (cache_key, similarity) if found, None otherwise
            
        Raises:
            CacheOperationError: If similarity calculation fails
        """
        try:
            best_match = None
            best_similarity = 0.0
            
            current_time = time.time()
            expired_keys = []
            
            for cache_key, entry in self.cache.items():
                # Check expiration
                if current_time - entry['timestamp'] >= self.ttl_seconds:
                    expired_keys.append(cache_key)
                    continue
                
                cached_embedding = self.embeddings[cache_key]
                similarity = self.cosine_similarity(query_embedding, cached_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = cache_key
            
            # Remove expired entries
            for key in expired_keys:
                del self.cache[key]
                del self.embeddings[key]
            
            if best_match and best_similarity >= self.similarity_threshold:
                return best_match, best_similarity
            
            return None
        except Exception as e:
            self._logger.error(f"Error finding similar query: {e}")
            raise CacheOperationError(f"Failed to find similar query: {e}")
    
    def get(self, query: str) -> Optional[Dict]:
        """
        Get cached response for semantically similar query.
        
        Returns:
            Dict with 'response' and 'similarity' if found, None otherwise
            
        Raises:
            CacheOperationError: If cache operation fails
        """
        try:
            query_embedding = self.embed_query(query)
            if query_embedding is None:
                self._record_miss()
                return None
            
            match = self._find_similar_query(query_embedding)
            
            if match:
                cache_key, similarity = match
                entry = self.cache[cache_key]
                
                # Move to end (most recently used)
                self.cache.move_to_end(cache_key)
                
                self._record_hit()
                entry['access_count'] += 1
                entry['last_accessed'] = time.time()
                
                return {
                    'response': entry['response'],
                    'similarity': similarity,
                    'original_query': entry['query']
                }
            
            self._record_miss()
            return None
        except CacheOperationError:
            raise
        except Exception as e:
            self._logger.error(f"Cache get operation failed: {e}")
            raise CacheOperationError(f"Failed to get from cache: {e}")
    
    def set(self, query: str, response: str) -> None:
        """
        Cache a query-response pair with its embedding.
        
        Raises:
            CacheOperationError: If cache operation fails
        """
        try:
            query_embedding = self.embed_query(query)
            if query_embedding is None:
                return  # Skip caching if embedding fails
            
            # Generate cache key
            cache_key = f"sem_{hash(query)}_{time.time()}"
            
            # LRU eviction if at capacity
            if len(self.cache) >= self.max_cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.embeddings[oldest_key]
                self._logger.debug(f"Evicted oldest entry: {oldest_key}")
            
            self.cache[cache_key] = {
                'query': query,
                'response': response,
                'timestamp': time.time(),
                'last_accessed': time.time(),
                'access_count': 0
            }
            self.embeddings[cache_key] = query_embedding
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
            self.embeddings.clear()
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
