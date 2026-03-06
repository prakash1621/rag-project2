"""
Main RAG Pipeline with Advanced Chunking and Three-Tier Caching
"""

import logging
from typing import List, Dict, Tuple, Optional
from src.utils import load_config, get_embedder, setup_logger
from src.chunking import ParentChildChunker, SemanticChunker
from src.caching import CacheManager

logger = setup_logger("rag_pipeline")


class RAGPipeline:
    def __init__(self, config_path: str = None):
        """
        Initialize RAG pipeline with advanced chunking and caching.
        
        Args:
            config_path: Path to config.yaml
        """
        # Load configuration
        self.config = load_config(config_path)
        logger.info("Configuration loaded")
        
        # Initialize embedder
        self.embedder = get_embedder(self.config)
        logger.info("Embedder initialized")
        
        # Initialize chunking strategies
        pc_config = self.config['chunking']['parent_child']
        self.parent_child_chunker = ParentChildChunker(
            parent_size=pc_config['parent_size'],
            parent_overlap=pc_config['parent_overlap'],
            child_size=pc_config['child_size'],
            child_overlap=pc_config['child_overlap']
        )
        
        sem_config = self.config['chunking']['semantic']
        self.semantic_chunker = SemanticChunker(
            embedder=self.embedder,
            buffer_size=sem_config['buffer_size'],
            breakpoint_threshold_type=sem_config['breakpoint_threshold_type'],
            breakpoint_threshold_amount=sem_config['breakpoint_threshold_amount']
        )
        logger.info("Chunking strategies initialized")
        
        # Initialize cache manager
        self.cache_manager = CacheManager(
            embedder=self.embedder,
            config=self.config['caching']
        )
        logger.info("Three-tier cache system initialized")
        
        self.vectorstore = None
    
    def chunk_document(self, text: str, metadata: Dict, 
                      strategy: str = "parent_child") -> Tuple[List[str], List[Dict]]:
        """
        Chunk document using specified strategy.
        
        Args:
            text: Document text
            metadata: Document metadata
            strategy: 'parent_child' or 'semantic'
        
        Returns:
            (chunks, metadatas)
        """
        if strategy == "parent_child":
            return self.parent_child_chunker.chunk(text, metadata)
        elif strategy == "semantic":
            return self.semantic_chunker.chunk(text, metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def query(self, question: str, vectorstore=None, 
              retriever_func=None, reranker_func=None, 
              generator_func=None) -> Dict:
        """
        Process query through three-tier cache system and RAG pipeline.
        
        Args:
            question: User query
            vectorstore: Vector store for retrieval
            retriever_func: Function(vectorstore, question) -> docs
            reranker_func: Function(question, docs) -> reranked_docs
            generator_func: Function(question, docs) -> answer
        
        Returns:
            Dict with 'answer', 'cache_tier', and 'metadata'
        """
        logger.info(f"Processing query: {question[:100]}...")
        
        # Tier 1 & 2: Check for complete cached response
        cached_response = self.cache_manager.get_response(question)
        if cached_response:
            response, cache_tier = cached_response
            logger.info(f"✓ Cache HIT at tier: {cache_tier}")
            return {
                'answer': response,
                'cache_tier': cache_tier,
                'from_cache': True
            }
        
        # Tier 3: Check for cached chunks
        cached_chunks = self.cache_manager.get_chunks(question)
        if cached_chunks:
            chunks, cache_tier = cached_chunks
            logger.info(f"✓ Retrieval cache HIT - skipping vector DB query")
            
            # Rerank cached chunks
            if reranker_func:
                chunks = reranker_func(question, chunks)
            
            # Generate answer
            if generator_func:
                answer = generator_func(question, chunks)
            else:
                answer = "Generator function not provided"
            
            # Cache the complete response
            self.cache_manager.cache_response(question, answer)
            
            return {
                'answer': answer,
                'cache_tier': cache_tier,
                'from_cache': True,
                'retrieval_skipped': True
            }
        
        # Full pipeline: No cache hit
        logger.info("✗ Cache MISS - executing full pipeline")
        
        if not vectorstore or not retriever_func or not generator_func:
            return {
                'answer': "Pipeline components not initialized",
                'cache_tier': None,
                'from_cache': False
            }
        
        # Retrieve documents
        docs = retriever_func(vectorstore, question)
        logger.info(f"Retrieved {len(docs)} documents")
        
        # Cache retrieved chunks
        self.cache_manager.cache_chunks(question, docs)
        
        # Rerank
        if reranker_func:
            docs = reranker_func(question, docs)
            logger.info(f"Reranked to top {len(docs)} documents")
        
        # Generate answer
        answer = generator_func(question, docs)
        
        # Cache complete response
        self.cache_manager.cache_response(question, answer)
        
        return {
            'answer': answer,
            'cache_tier': None,
            'from_cache': False
        }
    
    def get_cache_stats(self) -> Dict:
        """Get statistics from all cache tiers"""
        return self.cache_manager.get_all_stats()
    
    def clear_caches(self):
        """Clear all cache tiers"""
        self.cache_manager.clear_all()
