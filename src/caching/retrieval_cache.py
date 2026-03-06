"""
Tier 3: Retrieval Cache

Caches retrieved chunks for similar queries.
Skips vector DB retrieval but still calls LLM for generation.
"""

import time
import numpy as np
from typing import Optional, Dict, List, Any
from collections import OrderedDict


class RetrievalCache:
    def __init__(self, embedder, similarity_threshold: float = 0.90,
                 ttl_seconds: int = 1800, max_cache_size: int = 500):
        """
        Args:
            embedder: Function that takes text and returns embedding vector
            similarity_threshold: Minimum cosine similarity for cache hit
            ttl_seconds: Time-to-live for cache entries
            max_cache_size: Maximum number of entries (LRU eviction)
        """
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_cache_size = max_cache_size
        
        self.cache: OrderedDict[str, Dict] = OrderedDict()
        self.embeddings: Dict[str, np.ndarray] = {}
        
        self.hits = 0
        self.misses = 0
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _find_similar_query(self, query_embedding: np.ndarray) -> Optional[str]:
        """Find most similar cached query above threshold"""
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
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cache_key
        
        # Remove expired entries
        for key in expired_keys:
            del self.cache[key]
            del self.embeddings[key]
        
        if best_match and best_similarity >= self.similarity_threshold:
            return best_match
        
        return None
    
    def get(self, query: str) -> Optional[List[Any]]:
        """
        Get cached retrieved chunks for semantically similar query.
        
        Returns:
            List of retrieved document chunks if found, None otherwise
        """
        try:
            query_embedding = self.embedder(query)
        except Exception:
            self.misses += 1
            return None
        
        cache_key = self._find_similar_query(query_embedding)
        
        if cache_key:
            entry = self.cache[cache_key]
            
            # Move to end (most recently used)
            self.cache.move_to_end(cache_key)
            
            self.hits += 1
            entry['access_count'] += 1
            entry['last_accessed'] = time.time()
            
            return entry['chunks']
        
        self.misses += 1
        return None
    
    def set(self, query: str, chunks: List[Any]):
        """Cache retrieved chunks for a query"""
        try:
            query_embedding = self.embedder(query)
        except Exception:
            return  # Skip caching if embedding fails
        
        cache_key = f"ret_{hash(query)}_{time.time()}"
        
        # LRU eviction if at capacity
        if len(self.cache) >= self.max_cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            del self.embeddings[oldest_key]
        
        self.cache[cache_key] = {
            'query': query,
            'chunks': chunks,
            'timestamp': time.time(),
            'last_accessed': time.time(),
            'access_count': 0
        }
        self.embeddings[cache_key] = query_embedding
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.embeddings.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'type': 'retrieval',
            'size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'similarity_threshold': self.similarity_threshold,
            'ttl_seconds': self.ttl_seconds,
            'max_size': self.max_cache_size
        }
