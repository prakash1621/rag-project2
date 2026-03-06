"""
Caching module for RAG pipeline

Provides three-tier caching system with abstract base classes.
"""

from .base_cache import (
    BaseCache,
    BaseExactCache,
    BaseSemanticCache,
    BaseRetrievalCache,
    CacheError,
    CacheConnectionError,
    CacheOperationError,
    CacheConfigurationError
)
from .exact_cache import ExactCache
from .semantic_cache import SemanticCache
from .retrieval_cache import RetrievalCache
from .sqlite_cache import SQLiteExactCache, SQLiteSemanticCache, SQLiteRetrievalCache
from .cache_factory import CacheFactory
from .cache_manager import CacheManager

__all__ = [
    'BaseCache',
    'BaseExactCache',
    'BaseSemanticCache',
    'BaseRetrievalCache',
    'CacheError',
    'CacheConnectionError',
    'CacheOperationError',
    'CacheConfigurationError',
    'ExactCache',
    'SemanticCache',
    'RetrievalCache',
    'SQLiteExactCache',
    'SQLiteSemanticCache',
    'SQLiteRetrievalCache',
    'CacheFactory',
    'CacheManager'
]
