"""
Document reranking logic using embedding similarity
"""

import numpy as np
from app.embedding import get_embeddings
from app.config import RERANK_TOP_K


def rerank_documents(question, docs):
    """
    Rerank documents based on embedding similarity.
    
    Args:
        question: User query string
        docs: List of retrieved documents
        
    Returns:
        List of reranked documents (top K)
    """
    embeddings = get_embeddings()
    question_embedding = embeddings.embed_query(question)
    
    scored_docs = []
    for doc in docs:
        doc_embedding = embeddings.embed_query(doc.page_content)
        score = np.dot(question_embedding, doc_embedding)
        scored_docs.append((score, doc))
    
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored_docs[:RERANK_TOP_K]]
