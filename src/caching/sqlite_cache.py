"""
SQLite-based cache implementations for persistence
"""

import sqlite3
import json
import time
import pickle
import numpy as np
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SQLiteExactCache:
    """Tier 1: Exact Cache with SQLite backend"""
    
    def __init__(self, db_path: str, ttl_seconds: int = 3600, normalize: bool = True):
        self.db_path = db_path
        self.ttl_seconds = ttl_seconds
        self.normalize = normalize
        self.hits = 0
        self.misses = 0
        
        # Create database and table
        self._init_db()
    
    def _init_db(self):
        """Initialize database and create table"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exact_cache (
                cache_key TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER DEFAULT 0
            )
        """)
        
        # Create index on timestamp for efficient cleanup
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON exact_cache(timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for exact matching"""
        if self.normalize:
            return query.lower().strip()
        return query
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query"""
        import hashlib
        normalized = self._normalize_query(query)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _cleanup_expired(self, conn: sqlite3.Connection):
        """Remove expired entries"""
        cursor = conn.cursor()
        cutoff_time = time.time() - self.ttl_seconds
        cursor.execute("DELETE FROM exact_cache WHERE timestamp < ?", (cutoff_time,))
        conn.commit()
    
    def get(self, query: str) -> Optional[str]:
        """Get cached response for exact query match"""
        cache_key = self._get_cache_key(query)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Cleanup expired entries periodically
        if self.misses % 100 == 0:
            self._cleanup_expired(conn)
        
        cursor.execute("""
            SELECT response, timestamp 
            FROM exact_cache 
            WHERE cache_key = ?
        """, (cache_key,))
        
        result = cursor.fetchone()
        
        if result:
            response, timestamp = result
            
            # Check if expired
            if time.time() - timestamp < self.ttl_seconds:
                # Update access stats
                cursor.execute("""
                    UPDATE exact_cache 
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE cache_key = ?
                """, (time.time(), cache_key))
                conn.commit()
                conn.close()
                
                self.hits += 1
                return response
            else:
                # Expired, delete it
                cursor.execute("DELETE FROM exact_cache WHERE cache_key = ?", (cache_key,))
                conn.commit()
        
        conn.close()
        self.misses += 1
        return None
    
    def set(self, query: str, response: str):
        """Cache a query-response pair"""
        cache_key = self._get_cache_key(query)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO exact_cache 
            (cache_key, query, response, timestamp, last_accessed, access_count)
            VALUES (?, ?, ?, ?, ?, 0)
        """, (cache_key, query, response, time.time(), time.time()))
        
        conn.commit()
        conn.close()
    
    def clear(self):
        """Clear all cache entries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM exact_cache")
        conn.commit()
        conn.close()
        
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM exact_cache")
        size = cursor.fetchone()[0]
        conn.close()
        
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'type': 'exact',
            'backend': 'sqlite',
            'size': size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'ttl_seconds': self.ttl_seconds
        }


class SQLiteSemanticCache:
    """Tier 2: Semantic Cache with SQLite backend"""
    
    def __init__(self, embedder, db_path: str, similarity_threshold: float = 0.95,
                 ttl_seconds: int = 3600, max_cache_size: int = 1000):
        self.embedder = embedder
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_cache_size = max_cache_size
        self.hits = 0
        self.misses = 0
        
        self._init_db()
    
    def _init_db(self):
        """Initialize database and create table"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS semantic_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                embedding BLOB NOT NULL,
                timestamp REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER DEFAULT 0
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_semantic_timestamp 
            ON semantic_cache(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_semantic_last_accessed 
            ON semantic_cache(last_accessed)
        """)
        
        conn.commit()
        conn.close()
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _cleanup_expired(self, conn: sqlite3.Connection):
        """Remove expired entries"""
        cursor = conn.cursor()
        cutoff_time = time.time() - self.ttl_seconds
        cursor.execute("DELETE FROM semantic_cache WHERE timestamp < ?", (cutoff_time,))
        conn.commit()
    
    def _enforce_size_limit(self, conn: sqlite3.Connection):
        """Enforce max cache size using LRU"""
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM semantic_cache")
        size = cursor.fetchone()[0]
        
        if size > self.max_cache_size:
            # Delete oldest accessed entries
            to_delete = size - self.max_cache_size
            cursor.execute("""
                DELETE FROM semantic_cache 
                WHERE id IN (
                    SELECT id FROM semantic_cache 
                    ORDER BY last_accessed ASC 
                    LIMIT ?
                )
            """, (to_delete,))
            conn.commit()
    
    def get(self, query: str) -> Optional[Dict]:
        """Get cached response for semantically similar query"""
        try:
            query_embedding = self.embedder(query)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            self.misses += 1
            return None
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Cleanup periodically
        if self.misses % 50 == 0:
            self._cleanup_expired(conn)
        
        # Get all non-expired entries
        cutoff_time = time.time() - self.ttl_seconds
        cursor.execute("""
            SELECT id, query, response, embedding 
            FROM semantic_cache 
            WHERE timestamp >= ?
        """, (cutoff_time,))
        
        results = cursor.fetchall()
        
        best_match = None
        best_similarity = 0.0
        best_id = None
        
        for row_id, cached_query, cached_response, embedding_blob in results:
            cached_embedding = pickle.loads(embedding_blob)
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (cached_query, cached_response)
                best_id = row_id
        
        if best_match and best_similarity >= self.similarity_threshold:
            # Update access stats
            cursor.execute("""
                UPDATE semantic_cache 
                SET last_accessed = ?, access_count = access_count + 1
                WHERE id = ?
            """, (time.time(), best_id))
            conn.commit()
            conn.close()
            
            self.hits += 1
            return {
                'response': best_match[1],
                'similarity': best_similarity,
                'original_query': best_match[0]
            }
        
        conn.close()
        self.misses += 1
        return None
    
    def set(self, query: str, response: str):
        """Cache a query-response pair with its embedding"""
        try:
            query_embedding = self.embedder(query)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        embedding_blob = pickle.dumps(query_embedding)
        
        cursor.execute("""
            INSERT INTO semantic_cache 
            (query, response, embedding, timestamp, last_accessed, access_count)
            VALUES (?, ?, ?, ?, ?, 0)
        """, (query, response, embedding_blob, time.time(), time.time()))
        
        conn.commit()
        
        # Enforce size limit
        self._enforce_size_limit(conn)
        
        conn.close()
    
    def clear(self):
        """Clear all cache entries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM semantic_cache")
        conn.commit()
        conn.close()
        
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM semantic_cache")
        size = cursor.fetchone()[0]
        conn.close()
        
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'type': 'semantic',
            'backend': 'sqlite',
            'size': size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'similarity_threshold': self.similarity_threshold,
            'ttl_seconds': self.ttl_seconds,
            'max_size': self.max_cache_size
        }


class SQLiteRetrievalCache:
    """Tier 3: Retrieval Cache with SQLite backend"""
    
    def __init__(self, embedder, db_path: str, similarity_threshold: float = 0.90,
                 ttl_seconds: int = 1800, max_cache_size: int = 500):
        self.embedder = embedder
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_cache_size = max_cache_size
        self.hits = 0
        self.misses = 0
        
        self._init_db()
    
    def _init_db(self):
        """Initialize database and create table"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                chunks BLOB NOT NULL,
                embedding BLOB NOT NULL,
                timestamp REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER DEFAULT 0
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_retrieval_timestamp 
            ON retrieval_cache(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_retrieval_last_accessed 
            ON retrieval_cache(last_accessed)
        """)
        
        conn.commit()
        conn.close()
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _cleanup_expired(self, conn: sqlite3.Connection):
        """Remove expired entries"""
        cursor = conn.cursor()
        cutoff_time = time.time() - self.ttl_seconds
        cursor.execute("DELETE FROM retrieval_cache WHERE timestamp < ?", (cutoff_time,))
        conn.commit()
    
    def _enforce_size_limit(self, conn: sqlite3.Connection):
        """Enforce max cache size using LRU"""
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM retrieval_cache")
        size = cursor.fetchone()[0]
        
        if size > self.max_cache_size:
            to_delete = size - self.max_cache_size
            cursor.execute("""
                DELETE FROM retrieval_cache 
                WHERE id IN (
                    SELECT id FROM retrieval_cache 
                    ORDER BY last_accessed ASC 
                    LIMIT ?
                )
            """, (to_delete,))
            conn.commit()
    
    def get(self, query: str) -> Optional[List[Any]]:
        """Get cached retrieved chunks for semantically similar query"""
        try:
            query_embedding = self.embedder(query)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            self.misses += 1
            return None
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Cleanup periodically
        if self.misses % 50 == 0:
            self._cleanup_expired(conn)
        
        # Get all non-expired entries
        cutoff_time = time.time() - self.ttl_seconds
        cursor.execute("""
            SELECT id, chunks, embedding 
            FROM retrieval_cache 
            WHERE timestamp >= ?
        """, (cutoff_time,))
        
        results = cursor.fetchall()
        
        best_match = None
        best_similarity = 0.0
        best_id = None
        
        for row_id, chunks_blob, embedding_blob in results:
            cached_embedding = pickle.loads(embedding_blob)
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = pickle.loads(chunks_blob)
                best_id = row_id
        
        if best_match and best_similarity >= self.similarity_threshold:
            # Update access stats
            cursor.execute("""
                UPDATE retrieval_cache 
                SET last_accessed = ?, access_count = access_count + 1
                WHERE id = ?
            """, (time.time(), best_id))
            conn.commit()
            conn.close()
            
            self.hits += 1
            return best_match
        
        conn.close()
        self.misses += 1
        return None
    
    def set(self, query: str, chunks: List[Any]):
        """Cache retrieved chunks for a query"""
        try:
            query_embedding = self.embedder(query)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        chunks_blob = pickle.dumps(chunks)
        embedding_blob = pickle.dumps(query_embedding)
        
        cursor.execute("""
            INSERT INTO retrieval_cache 
            (query, chunks, embedding, timestamp, last_accessed, access_count)
            VALUES (?, ?, ?, ?, ?, 0)
        """, (query, chunks_blob, embedding_blob, time.time(), time.time()))
        
        conn.commit()
        
        # Enforce size limit
        self._enforce_size_limit(conn)
        
        conn.close()
    
    def clear(self):
        """Clear all cache entries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM retrieval_cache")
        conn.commit()
        conn.close()
        
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM retrieval_cache")
        size = cursor.fetchone()[0]
        conn.close()
        
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'type': 'retrieval',
            'backend': 'sqlite',
            'size': size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'similarity_threshold': self.similarity_threshold,
            'ttl_seconds': self.ttl_seconds,
            'max_size': self.max_cache_size
        }
