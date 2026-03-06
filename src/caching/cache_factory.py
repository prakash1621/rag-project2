"""
Cache Factory for Dependency Injection

Creates cache instances based on configuration.
"""

from typing import Dict, Optional
import logging
from .base_cache import (
    BaseCache, BaseExactCache, BaseSemanticCache, BaseRetrievalCache,
    CacheConfigurationError
)
from .exact_cache import ExactCache
from .semantic_cache import SemanticCache
from .retrieval_cache import RetrievalCache
from .sqlite_cache import SQLiteExactCache, SQLiteSemanticCache, SQLiteRetrievalCache
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class CacheFactory:
    """Factory for creating cache instances"""
    
    @staticmethod
    def create_exact_cache(config: Dict, backend: str = 'memory') -> BaseExactCache:
        """
        Create an exact cache instance.
        
        Args:
            config: Cache configuration dictionary
            backend: 'memory' or 'sqlite'
            
        Returns:
            BaseExactCache instance
            
        Raises:
            CacheConfigurationError: If configuration is invalid
        """
        try:
            ttl_seconds = config.get('ttl_seconds', 3600)
            normalize = config.get('normalize_query', True)
            
            if backend == 'sqlite':
                cache_dir = config.get('cache_dir', 'cache')
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                db_path = os.path.join(cache_dir, 'exact_cache.db')
                
                logger.info(f"Creating SQLite exact cache at {db_path}")
                return SQLiteExactCache(
                    db_path=db_path,
                    ttl_seconds=ttl_seconds,
                    normalize=normalize
                )
            elif backend == 'memory':
                logger.info("Creating in-memory exact cache")
                return ExactCache(
                    ttl_seconds=ttl_seconds,
                    normalize=normalize
                )
            else:
                raise CacheConfigurationError(f"Unknown backend: {backend}")
                
        except Exception as e:
            logger.error(f"Failed to create exact cache: {e}")
            raise CacheConfigurationError(f"Failed to create exact cache: {e}")
    
    @staticmethod
    def create_semantic_cache(embedder, config: Dict, backend: str = 'memory') -> BaseSemanticCache:
        """
        Create a semantic cache instance.
        
        Args:
            embedder: Embedding function
            config: Cache configuration dictionary
            backend: 'memory' or 'sqlite'
            
        Returns:
            BaseSemanticCache instance
            
        Raises:
            CacheConfigurationError: If configuration is invalid
        """
        try:
            similarity_threshold = config.get('similarity_threshold', 0.95)
            ttl_seconds = config.get('ttl_seconds', 3600)
            max_cache_size = config.get('max_cache_size', 1000)
            
            if backend == 'sqlite':
                cache_dir = config.get('cache_dir', 'cache')
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                db_path = os.path.join(cache_dir, 'semantic_cache.db')
                
                logger.info(f"Creating SQLite semantic cache at {db_path}")
                return SQLiteSemanticCache(
                    embedder=embedder,
                    db_path=db_path,
                    similarity_threshold=similarity_threshold,
                    ttl_seconds=ttl_seconds,
                    max_cache_size=max_cache_size
                )
            elif backend == 'memory':
                logger.info("Creating in-memory semantic cache")
                return SemanticCache(
                    embedder=embedder,
                    similarity_threshold=similarity_threshold,
                    ttl_seconds=ttl_seconds,
                    max_cache_size=max_cache_size
                )
            else:
                raise CacheConfigurationError(f"Unknown backend: {backend}")
                
        except Exception as e:
            logger.error(f"Failed to create semantic cache: {e}")
            raise CacheConfigurationError(f"Failed to create semantic cache: {e}")
    
    @staticmethod
    def create_retrieval_cache(embedder, config: Dict, backend: str = 'memory') -> BaseRetrievalCache:
        """
        Create a retrieval cache instance.
        
        Args:
            embedder: Embedding function
            config: Cache configuration dictionary
            backend: 'memory' or 'sqlite'
            
        Returns:
            BaseRetrievalCache instance
            
        Raises:
            CacheConfigurationError: If configuration is invalid
        """
        try:
            similarity_threshold = config.get('similarity_threshold', 0.90)
            ttl_seconds = config.get('ttl_seconds', 1800)
            max_cache_size = config.get('max_cache_size', 500)
            
            if backend == 'sqlite':
                cache_dir = config.get('cache_dir', 'cache')
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                db_path = os.path.join(cache_dir, 'retrieval_cache.db')
                
                logger.info(f"Creating SQLite retrieval cache at {db_path}")
                return SQLiteRetrievalCache(
                    embedder=embedder,
                    db_path=db_path,
                    similarity_threshold=similarity_threshold,
                    ttl_seconds=ttl_seconds,
                    max_cache_size=max_cache_size
                )
            elif backend == 'memory':
                logger.info("Creating in-memory retrieval cache")
                return RetrievalCache(
                    embedder=embedder,
                    similarity_threshold=similarity_threshold,
                    ttl_seconds=ttl_seconds,
                    max_cache_size=max_cache_size
                )
            else:
                raise CacheConfigurationError(f"Unknown backend: {backend}")
                
        except Exception as e:
            logger.error(f"Failed to create retrieval cache: {e}")
            raise CacheConfigurationError(f"Failed to create retrieval cache: {e}")

