"""
Abstract Base Classes for Cache Implementations

Defines interfaces and contracts for all cache types.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseCache(ABC):
    """Abstract base class for all cache implementations"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def get(self, query: str) -> Optional[Any]:
        """
        Get cached value for query.
        
        Args:
            query: User query string
            
        Returns:
            Cached value if found and valid, None otherwise
            
        Raises:
            CacheError: If cache operation fails
        """
        pass
    
    @abstractmethod
    def set(self, query: str, value: Any) -> None:
        """
        Cache a value for query.
        
        Args:
            query: User query string
            value: Value to cache
            
        Raises:
            CacheError: If cache operation fails
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """
        Clear all cache entries.
        
        Raises:
            CacheError: If cache operation fails
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total,
            'hit_rate': hit_rate,
            'cache_type': self.__class__.__name__
        }
    
    def _record_hit(self) -> None:
        """Record a cache hit"""
        self.hits += 1
        self._logger.debug(f"Cache HIT - Total hits: {self.hits}")
    
    def _record_miss(self) -> None:
        """Record a cache miss"""
        self.misses += 1
        self._logger.debug(f"Cache MISS - Total misses: {self.misses}")


class BaseExactCache(BaseCache):
    """Abstract base class for exact match caches"""
    
    def __init__(self, ttl_seconds: int = 3600, normalize: bool = True):
        super().__init__()
        self.ttl_seconds = ttl_seconds
        self.normalize = normalize
    
    @staticmethod
    def normalize_query(query: str, should_normalize: bool = True) -> str:
        """
        Normalize query for exact matching.
        
        Args:
            query: Raw query string
            should_normalize: Whether to apply normalization
            
        Returns:
            Normalized query string
        """
        if should_normalize:
            return query.lower().strip()
        return query
    
    @staticmethod
    def generate_cache_key(query: str) -> str:
        """
        Generate cache key from query.
        
        Args:
            query: Query string (should be normalized first)
            
        Returns:
            Cache key (MD5 hash)
        """
        import hashlib
        return hashlib.md5(query.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics with exact cache specific info"""
        stats = super().get_stats()
        stats.update({
            'type': 'exact',
            'ttl_seconds': self.ttl_seconds,
            'normalize': self.normalize
        })
        return stats


class BaseSemanticCache(BaseCache):
    """Abstract base class for semantic similarity caches"""
    
    def __init__(self, embedder, similarity_threshold: float = 0.95,
                 ttl_seconds: int = 3600, max_cache_size: int = 1000):
        super().__init__()
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_cache_size = max_cache_size
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
            
        Raises:
            ValueError: If vectors have different dimensions
        """
        if vec1.shape != vec2.shape:
            raise ValueError(f"Vector dimension mismatch: {vec1.shape} vs {vec2.shape}")
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def embed_query(self, query: str) -> Optional[np.ndarray]:
        """
        Generate embedding for query with error handling.
        
        Args:
            query: Query string
            
        Returns:
            Embedding vector or None if embedding fails
        """
        try:
            embedding = self.embedder(query)
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            return embedding
        except Exception as e:
            self._logger.error(f"Embedding failed for query '{query[:50]}...': {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics with semantic cache specific info"""
        stats = super().get_stats()
        stats.update({
            'type': 'semantic',
            'similarity_threshold': self.similarity_threshold,
            'ttl_seconds': self.ttl_seconds,
            'max_cache_size': self.max_cache_size
        })
        return stats


class BaseRetrievalCache(BaseSemanticCache):
    """Abstract base class for retrieval caches (extends semantic cache)"""
    
    def __init__(self, embedder, similarity_threshold: float = 0.90,
                 ttl_seconds: int = 1800, max_cache_size: int = 500):
        super().__init__(embedder, similarity_threshold, ttl_seconds, max_cache_size)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics with retrieval cache specific info"""
        stats = super().get_stats()
        stats['type'] = 'retrieval'
        return stats


class CacheError(Exception):
    """Base exception for cache-related errors"""
    pass


class CacheConnectionError(CacheError):
    """Exception raised when cache connection fails"""
    pass


class CacheOperationError(CacheError):
    """Exception raised when cache operation fails"""
    pass


class CacheConfigurationError(CacheError):
    """Exception raised when cache configuration is invalid"""
    pass

