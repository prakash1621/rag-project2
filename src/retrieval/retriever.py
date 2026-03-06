"""
Document retrieval logic with category detection
"""

from app.config import RETRIEVAL_K, CATEGORY_KEYWORDS


def detect_categories(question):
    """
    Detect relevant categories from question keywords.
    
    Args:
        question: User query string
        
    Returns:
        List of detected category names
    """
    question_lower = question.lower()
    detected_categories = []
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in question_lower for keyword in keywords):
            detected_categories.append(category)
    
    return detected_categories


def retrieve_documents(vectorstore, question):
    """
    Retrieve relevant documents from vector store.
    
    Args:
        vectorstore: FAISS vector store instance
        question: User query string
        
    Returns:
        List of retrieved documents
    """
    detected_categories = detect_categories(question)
    
    if detected_categories:
        docs = vectorstore.similarity_search(
            question, 
            k=RETRIEVAL_K,
            filter={"category": {"$in": detected_categories}}
        )
    else:
        docs = vectorstore.similarity_search(question, k=RETRIEVAL_K)
    
    return docs
