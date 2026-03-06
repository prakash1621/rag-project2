# RAG Pipeline v2 - Production-Grade RAG with Advanced Chunking & Multi-Tier Caching

Assignment 2: Production-Grade RAG Pipeline  
**Student:** Prakash  
**Submission Date:** March 5, 2026

---

## 🚀 Overview

This project implements a production-grade RAG (Retrieval-Augmented Generation) pipeline with two major enhancements over Assignment 1:

1. **Intelligent Document Chunking**
   - Parent-Child Chunking (precision + context)
   - Semantic Chunking (embedding-based boundaries)

2. **Three-Tier Caching System**
   - Tier 1: Exact Cache (string matching)
   - Tier 2: Semantic Cache (embedding similarity)
   - Tier 3: Retrieval Cache (cached chunks)

---

## 📁 Project Structure

```
rag-project2/
├── README_V2.md              # This file
├── config.yaml               # All configurable parameters
├── .env.example              # Environment variables template
├── requirements.txt          # Python dependencies
├── main.py                # Entry point (Streamlit app)
│
├── src/                      # New modular architecture
│   ├── __init__.py
│   ├── pipeline.py           # Main RAG pipeline orchestrator
│   │
│   ├── chunking/             # Chunking strategies
│   │   ├── __init__.py
│   │   ├── parent_child.py   # Parent-child chunking
│   │   └── semantic_chunker.py  # Semantic chunking
│   │
│   ├── caching/              # Three-tier cache system
│   │   ├── __init__.py
│   │   ├── exact_cache.py    # Tier 1: Exact match
│   │   ├── semantic_cache.py # Tier 2: Semantic similarity
│   │   ├── retrieval_cache.py # Tier 3: Chunk cache
│   │   └── cache_manager.py  # Cache orchestrator
│   │
│   └── utils/                # Utilities
│       ├── __init__.py
│       ├── config_loader.py  # YAML config loader
│       ├── embeddings.py     # Bedrock embedder
│       └── logger.py         # Logging setup
│
├── app/                       
│   ├── config.py
│   ├── ingestion.py
│   ├── embedding.py
│   ├── retrieval.py
│   ├── reranker.py
│   ├── generation.py
│   └── api.py
│
├── docs/
│   └── architecture.md       # Detailed architecture document
│
├── knowledge-base/           # Source documents (by category)
│   ├── Teradata/
│   ├── pusa-sell-kb/
│   ├── dot/
│   └── ...
│
└── vector_store/             # FAISS index (generated)
```

---

## 🛠️ Setup Instructions

### 1. Prerequisites

- Python 3.10+
- AWS credentials configured
- Access to AWS Bedrock (Titan Embeddings + Claude)
- SQLite (built into Python, no installation needed)

### 2. Installation

```bash
# Clone repository
cd rag-project2

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env and add your AWS credentials
```

### 3. Configuration

Edit `config.yaml` to customize:
- Chunking parameters (parent/child sizes, semantic thresholds)
- Cache settings (TTL, similarity thresholds, max sizes)
- AWS region and model IDs

### 4. Add Documents

Place your documents in `knowledge-base/` organized by category:

 

Supported formats: PDF, DOCX, HTML, TXT, MD

---

## 🚀 Running the Application

### Option 1: Run   (New Pipeline)

```bash
streamlit run main.py
```

 

### First Time Setup

1. Click **"Rebuild Knowledge Base"** in sidebar
2. Select chunking strategy (parent_child or semantic)
3. Wait for indexing to complete
4. Start asking questions!

---

## 💡 Usage Examples

### Example 1: Exact Cache Hit

```
Query 1: "What is the Teradata data discrepancy process?"
→ Full pipeline (600ms)

Query 2: "What is the Teradata data discrepancy process?"
→ Tier 1 cache HIT (<1ms) 🎯
```

### Example 2: Semantic Cache Hit

```
Query 1: "How do I fix data discrepancies in Teradata?"
→ Full pipeline (600ms)

Query 2: "What's the process for resolving Teradata data issues?"
→ Tier 2 cache HIT (50ms) 🔍
→ Similarity: 0.97
```

### Example 3: Retrieval Cache Hit

```
Query 1: "Explain the Teradata discrepancy workflow"
→ Full pipeline (600ms)

Query 2: "Summarize Teradata discrepancy handling"
→ Tier 3 cache HIT (350ms) 📦
→ Skipped vector DB, only LLM call
```

---

## 📊 Features

### Intelligent Chunking

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Parent-Child** | Large parents (3000 chars) + small children (500 chars). Retrieval on children, LLM sees parents. | Technical docs, manuals, structured content |
| **Semantic** | Embedding-based topic boundaries. Adaptive chunk sizes. | Articles, notes, conversational content |

### Three-Tier Caching

| Tier | Type | Backend | Threshold | TTL | Max Size | Latency Saved |
|------|------|---------|-----------|-----|----------|---------------|
| **Tier 1** | Exact | SQLite | N/A | 3600s | Unlimited | ~600ms |
| **Tier 2** | Semantic | SQLite | 0.95 | 3600s | 1000 | ~550ms |
| **Tier 3** | Retrieval | SQLite | 0.90 | 1800s | 500 | ~150ms |

**Backend:** SQLite (persistent, no external dependencies)  
**Expected Overall Hit Rate:** 60-90%  
**Persistence:** ✅ Cache survives restarts

### Cache Statistics

View real-time cache performance in the sidebar:
- Overall hit rate
- Hits/misses per tier
- Cache sizes
- Cost savings

---

## 🏗️ Architecture Highlights

### Query Flow

```
User Query
    ↓
Tier 1: Exact Cache → HIT? → Return response (1ms)
    ↓ MISS
Tier 2: Semantic Cache → HIT? → Return response (50ms)
    ↓ MISS
Tier 3: Retrieval Cache → HIT? → Skip retrieval, call LLM (350ms)
    ↓ MISS
Full Pipeline:
    → Retrieve from vector DB (150ms)
    → Rerank (50ms)
    → Generate with LLM (300ms)
    → Cache at all tiers
    ↓
Return response (600ms)
```

### Parent-Child Chunking

```
Document
    ↓
Split into Parents (3000 chars)
    ↓
Each Parent → Split into Children (500 chars)
    ↓
Children embedded & stored in vector DB
    ↓
Retrieval: Search children (precision)
    ↓
Generation: Pass parent to LLM (context)
```

---

## 🧪 Testing

### Manual Testing

1. Ask the same question twice → Should hit Tier 1
2. Ask paraphrased question → Should hit Tier 2
3. Ask related question → Should hit Tier 3
4. Check cache stats in sidebar

### Clear Caches

- **Clear Caches Only**: Keeps vector DB, clears caches
- **Clear Knowledge Base**: Deletes everything, rebuild required

---

## 📈 Performance Metrics

### Latency Comparison

| Scenario | Latency | Improvement |
|----------|---------|-------------|
| Tier 1 Hit | <1ms | 99.8% faster |
| Tier 2 Hit | ~50ms | 91% faster |
| Tier 3 Hit | ~350ms | 42% faster |
| Full Pipeline | ~600ms | Baseline |

### Cost Savings

Assuming 1000 queries/day with 70% cache hit rate:
- **Without cache:** $1.00/day (1000 LLM calls)
- **With cache:** $0.30/day (300 LLM calls)
- **Savings:** 70% reduction

---

## 🔧 Configuration

All settings in `config.yaml`:

```yaml
# Chunking
chunking:
  parent_child:
    parent_size: 3000
    child_size: 500
  semantic:
    breakpoint_threshold_amount: 95

# Caching
caching:
  exact:
    ttl_seconds: 3600
  semantic:
    similarity_threshold: 0.95
    max_cache_size: 1000
  retrieval:
    similarity_threshold: 0.90
    max_cache_size: 500
```

---

## 📚 Documentation

See `docs/architecture.md` for:
- Detailed system architecture diagram
- Chunking strategy explanations
- Caching implementation details
- Design decisions & trade-offs
- Query flow walkthroughs
- Performance analysis

---
 
 

 
