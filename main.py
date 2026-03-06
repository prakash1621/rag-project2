"""
Main entry point for RAG Pipeline with Advanced Chunking and Three-Tier Caching
Assignment 2 - Production-Grade RAG Pipeline
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import shutil
from pathlib import Path

# Import original modules
from app.config import VECTOR_STORE_PATH
from app.ingestion import scan_knowledge_base, extract_text_from_file
from app.embedding import create_vector_store, save_vector_store, load_vector_store

# Import new modular components
from src.retrieval import retrieve_documents, rerank_documents
from src.generation import generate_answer
from src.pipeline import RAGPipeline
from src.utils import setup_logger

logger = setup_logger("main")

# Page config
st.set_page_config(
    page_title="RAG Pipeline v2 - Advanced Chunking & Caching",
    page_icon="🚀",
    layout="wide"
)

# Header
st.markdown("""
# 🚀 RAG Pipeline v2 - Production Grade
## Advanced Chunking + Three-Tier Caching System

**New Features:**
- 🔹 Parent-Child Chunking (precision + context)
- 🔹 Semantic Chunking (embedding-based boundaries)
- 🔹 Three-Tier Caching (Exact → Semantic → Retrieval)
""")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pipeline" not in st.session_state:
    try:
        st.session_state.pipeline = RAGPipeline()
        logger.info("Pipeline initialized")
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        st.session_state.pipeline = None

# Sidebar - Configuration
st.sidebar.title("⚙️ Configuration")

chunking_strategy = st.sidebar.selectbox(
    "Chunking Strategy",
    ["parent_child", "semantic"],
    help="Parent-Child: Large parents + small children. Semantic: Embedding-based boundaries."
)

st.sidebar.markdown("### Cache Settings")
show_cache_stats = st.sidebar.checkbox("Show Cache Statistics", value=True)

# Build knowledge base
def build_knowledge_base_v2():
    """Build knowledge base with new chunking strategies"""
    with st.spinner(f"Building knowledge base with {chunking_strategy} chunking..."):
        categories = scan_knowledge_base()
        if not categories:
            st.error("No documents found")
            return
        
        all_chunks = []
        all_metadatas = []
        
        for category, files in categories.items():
            for file_path in files:
                text, _ = extract_text_from_file(file_path)
                
                if text.strip():
                    metadata = {
                        "source": file_path,
                        "category": category,
                        "filename": os.path.basename(file_path)
                    }
                    
                    # Use selected chunking strategy
                    chunks, metadatas = st.session_state.pipeline.chunk_document(
                        text, metadata, strategy=chunking_strategy
                    )
                    
                    all_chunks.extend(chunks)
                    all_metadatas.extend(metadatas)
        
        if all_chunks:
            vectorstore = create_vector_store(all_chunks, all_metadatas)
            save_vector_store(vectorstore)
            st.session_state.vectorstore = vectorstore
            
            # Clear caches when rebuilding
            st.session_state.pipeline.clear_caches()
            
            st.success(f"✅ Knowledge base built with {len(all_chunks)} chunks "
                      f"from {len(categories)} categories using {chunking_strategy} strategy")
            logger.info(f"Built KB: {len(all_chunks)} chunks, strategy={chunking_strategy}")
        else:
            st.warning("No content found to process")

# Load existing vector store
if st.session_state.vectorstore is None:
    st.session_state.vectorstore = load_vector_store()
    if st.session_state.vectorstore:
        st.info("✅ Loaded existing knowledge base")

# Sidebar buttons
if st.sidebar.button("🔄 Rebuild Knowledge Base"):
    build_knowledge_base_v2()

if st.sidebar.button("🗑️ Clear Knowledge Base"):
    if os.path.exists(VECTOR_STORE_PATH):
        shutil.rmtree(VECTOR_STORE_PATH)
    st.session_state.vectorstore = None
    st.session_state.messages = []
    if st.session_state.pipeline:
        st.session_state.pipeline.clear_caches()
    st.sidebar.success("Knowledge base cleared")
    st.rerun()

if st.sidebar.button("🧹 Clear Caches Only"):
    if st.session_state.pipeline:
        st.session_state.pipeline.clear_caches()
        st.sidebar.success("All caches cleared")

# Cache statistics
if show_cache_stats and st.session_state.pipeline:
    st.sidebar.markdown("### 📊 Cache Statistics")
    stats = st.session_state.pipeline.get_cache_stats()
    
    if 'overall' in stats:
        overall = stats['overall']
        st.sidebar.metric("Overall Hit Rate", f"{overall['hit_rate']:.1%}")
        st.sidebar.metric("Total Requests", overall['total_requests'])
        
        with st.sidebar.expander("Detailed Stats"):
            for tier_name, tier_stats in stats.items():
                if tier_name != 'overall':
                    st.write(f"**{tier_name.title()} Cache**")
                    st.write(f"- Hits: {tier_stats['hits']}")
                    st.write(f"- Misses: {tier_stats['misses']}")
                    st.write(f"- Hit Rate: {tier_stats['hit_rate']:.1%}")
                    st.write(f"- Size: {tier_stats['size']}")

# Process question with new pipeline
def process_question_v2(question):
    """Process question through new pipeline with caching"""
    if st.session_state.vectorstore is None or st.session_state.pipeline is None:
        st.warning("Pipeline not ready")
        return
    
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Use new pipeline
    result = st.session_state.pipeline.query(
        question=question,
        vectorstore=st.session_state.vectorstore,
        retriever_func=retrieve_documents,
        reranker_func=rerank_documents,
        generator_func=generate_answer
    )
    
    answer = result['answer']
    
    # Add cache info to answer if from cache
    if result.get('from_cache'):
        cache_tier = result['cache_tier']
        cache_emoji = {"exact": "🎯", "semantic": "🔍", "retrieval": "📦"}.get(cache_tier, "💾")
        answer = f"{cache_emoji} *[Cached from {cache_tier} tier]*\n\n{answer}"
    
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if st.session_state.vectorstore is not None and st.session_state.pipeline is not None:
    question = st.chat_input("Ask a question about your documents...")
    if question:
        process_question_v2(question)
        st.rerun()
else:
    st.warning("⚠️ No knowledge base loaded. Click 'Rebuild Knowledge Base' in the sidebar.")

# Footer
st.markdown("---")
st.caption("🚀 RAG Pipeline v2 | Assignment 2: Production-Grade RAG with Advanced Chunking & Multi-Tier Caching")
