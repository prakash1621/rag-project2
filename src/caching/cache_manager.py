"""
Cache Manager - Orchestrates Three-Tier Caching System

Manages the fallthrough logic:
1. Check Exact Cache
2. Check Semantic Cache
3. Check Retrieval Cache
4. Fall through to full pipeline
"""

from typing import Optional, Dict, List, Any, Tuple
import logging
from .base_cache import BaseExactCache, BaseSemanticCache, BaseRetrievalCache, CacheError
from .cache_factory import CacheFactory


logger = logging.getLogger(__name__)


class CacheManager:
    def __init__(self, embedder=None, config: Dict = None, 
                 exact_cache: Optional[BaseExactCache] = None,
                 semantic_cache: Optional[BaseSemanticCache] = None,
                 retrieval_cache: Optional[BaseRetrievalCache] = None):
        """
        Initialize three-tier cache system with dependency injection.
        
        Args:
            embedder: Function that takes text and returns embedding vector
            config: Cache configuration dict (used if caches not provided)
            exact_cache: Pre-configured exact cache instance (optional)
            semantic_cache: Pre-configured semantic cache instance (optional)
            retrieval_cache: Pre-configured retrieval cache instance (optional)
        
        Raises:
            CacheError: If configuration is invalid
        """
        self.config = config or {}
        backend = self.config.get('backend', 'memory')
        
        # Use injected caches or create from factory
        exact_config = self.config.get('exact', {})
        if exact_cache:
            # Use injected cache regardless of config
            self.exact_cache = exact_cache
            logger.info("Using injected exact cache")
        elif exact_config.get('enabled', True):
            # Create from factory
            self.exact_cache = CacheFactory.create_exact_cache(
                config={**exact_config, 'cache_dir': self.config.get('cache_dir', 'cache')},
                backend=backend
            )
        else:
            self.exact_cache = None
            logger.info("Exact cache disabled")
        
        semantic_config = self.config.get('semantic', {})
        if semantic_cache:
            # Use injected cache regardless of config
            self.semantic_cache = semantic_cache
            logger.info("Using injected semantic cache")
        elif semantic_config.get('enabled', True):
            # Create from factory
            if not embedder:
                raise CacheError("Embedder required for semantic cache")
            self.semantic_cache = CacheFactory.create_semantic_cache(
                embedder=embedder,
                config={**semantic_config, 'cache_dir': self.config.get('cache_dir', 'cache')},
                backend=backend
            )
        else:
            self.semantic_cache = None
            logger.info("Semantic cache disabled")
        
        retrieval_config = self.config.get('retrieval', {})
        if retrieval_cache:
            # Use injected cache regardless of config
            self.retrieval_cache = retrieval_cache
            logger.info("Using injected retrieval cache")
        elif retrieval_config.get('enabled', True):
            # Create from factory
            if not embedder:
                raise CacheError("Embedder required for retrieval cache")
            self.retrieval_cache = CacheFactory.create_retrieval_cache(
                embedder=embedder,
                config={**retrieval_config, 'cache_dir': self.config.get('cache_dir', 'cache')},
                backend=backend
            )
        else:
            self.retrieval_cache = None
            logger.info("Retrieval cache disabled")
    
    def get_response(self, query: str) -> Optional[Tuple[str, str]]:
        """
        Check Tier 1 (Exact) and Tier 2 (Semantic) for complete response.
        
        Args:
            query: User query string
            
        Returns:
            (response, cache_tier) if found, None otherwise
            
        Raises:
            CacheError: If cache operation fails
        """
        try:
            # Tier 1: Exact Cache
            if self.exact_cache:
                response = self.exact_cache.get(query)
                if response:
                    logger.info(f"✓ Tier 1 (Exact) cache HIT for query: {query[:50]}...")
                    return response, "exact"
            
            # Tier 2: Semantic Cache
            if self.semantic_cache:
                result = self.semantic_cache.get(query)
                if result:
                    logger.info(f"✓ Tier 2 (Semantic) cache HIT for query: {query[:50]}... "
                              f"(similarity: {result['similarity']:.3f})")
                    return result['response'], "semantic"
            
            return None
        except Exception as e:
            logger.error(f"Error getting cached response: {e}")
            # Don't raise - allow fallthrough to full pipeline
            return None
    
    def get_chunks(self, query: str) -> Optional[Tuple[List[Any], str]]:
        """
        Check Tier 3 (Retrieval) for cached chunks.
        
        Args:
            query: User query string
            
        Returns:
            (chunks, cache_tier) if found, None otherwise
            
        Raises:
            CacheError: If cache operation fails
        """
        try:
            if self.retrieval_cache:
                chunks = self.retrieval_cache.get(query)
                if chunks:
                    logger.info(f"✓ Tier 3 (Retrieval) cache HIT for query: {query[:50]}...")
                    return chunks, "retrieval"
            
            return None
        except Exception as e:
            logger.error(f"Error getting cached chunks: {e}")
            # Don't raise - allow fallthrough to full pipeline
            return None
    
    def cache_response(self, query: str, response: str) -> None:
        """
        Cache complete response in Tier 1 and Tier 2.
        
        Args:
            query: User query string
            response: Generated response
        """
        try:
            if self.exact_cache:
                self.exact_cache.set(query, response)
            
            if self.semantic_cache:
                self.semantic_cache.set(query, response)
        except Exception as e:
            logger.error(f"Error caching response: {e}")
            # Don't raise - caching failure shouldn't break the pipeline
    
    def cache_chunks(self, query: str, chunks: List[Any]) -> None:
        """
        Cache retrieved chunks in Tier 3.
        
        Args:
            query: User query string
            chunks: Retrieved document chunks
        """
        try:
            if self.retrieval_cache:
                self.retrieval_cache.set(query, chunks)
        except Exception as e:
            logger.error(f"Error caching chunks: {e}")
            # Don't raise - caching failure shouldn't break the pipeline
    
    def clear_all(self) -> None:
        """
        Clear all cache tiers.
        
        Raises:
            CacheError: If clear operation fails
        """
        try:
            if self.exact_cache:
                self.exact_cache.clear()
            if self.semantic_cache:
                self.semantic_cache.clear()
            if self.retrieval_cache:
                self.retrieval_cache.clear()
            logger.info("All caches cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing caches: {e}")
            raise CacheError(f"Failed to clear caches: {e}")
    
    def get_all_stats(self) -> Dict:
        """Get statistics from all cache tiers"""
        stats = {}
        
        if self.exact_cache:
            stats['exact'] = self.exact_cache.get_stats()
        
        if self.semantic_cache:
            stats['semantic'] = self.semantic_cache.get_stats()
        
        if self.retrieval_cache:
            stats['retrieval'] = self.retrieval_cache.get_stats()
        
        # Calculate overall stats
        total_hits = sum(s['hits'] for s in stats.values())
        total_misses = sum(s['misses'] for s in stats.values())
        total_requests = total_hits + total_misses
        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0
        
        stats['overall'] = {
            'total_hits': total_hits,
            'total_misses': total_misses,
            'total_requests': total_requests,
            'hit_rate': overall_hit_rate
        }
        
        return stats
