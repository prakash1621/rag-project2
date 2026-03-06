# 🔄 Rebuild Knowledge Base - Complete Flow

## What Happens When You Click "Rebuild Knowledge Base"

### 📍 Location in Code
**File:** `rag-project2/main_v2.py`  
**Function:** `build_knowledge_base_v2()` (lines 73-115)

---

## 🔢 Step-by-Step Process

### Step 1: Scan for Documents 📂
**Location:** `rag-project2/app/ingestion.py` → `scan_knowledge_base()`

```
rag-project2/knowledge-base/
├── avaya/          → 9 markdown files
├── bppsl/          → 10 markdown files
├── dot/            → 11 markdown files
├── galaxy/         → 10 markdown files
├── pusa/           → 19 markdown files
├── swav/           → 12 markdown files
├── HR_Policies/    → 1 PDF
├── Engineering_Docs/ → 1 PDF
├── Product_Manuals/ → 1 PDF
└── Internal_Guides/ → 1 PDF
```

**What it does:**
- Scans `knowledge-base/` folder
- Groups files by category (folder name)
- Supports: PDF, DOCX, HTML, TXT, MD

**Output:** Dictionary like:
```python
{
    'avaya': ['path/to/file1.md', 'path/to/file2.md', ...],
    'bppsl': ['path/to/file1.md', ...],
    'pusa': ['path/to/file1.md', ...],
    ...
}
```

---

### Step 2: Extract Text from Each File 📄
**Location:** `rag-project2/app/ingestion.py` → `extract_text_from_file()`

**For each file:**
```python
# Example: avaya/avaya_01_Overview.md
text = """
Avaya QA Validation and Regression Testing

Overview:
This document outlines the comprehensive QA validation 
and regression testing process for the Avaya system...
[continues for ~5000 characters]
"""
```

**What it does:**
- Reads file content based on type (PDF/DOCX/MD/TXT/HTML)
- Extracts plain text
- Extracts hyperlinks (if any)

---

### Step 3: Chunk the Text ✂️
**Location:** `rag-project2/src/pipeline.py` → `chunk_document()`

**You choose the strategy in the UI:**

#### Option A: Parent-Child Chunking
**Location:** `rag-project2/src/chunking/parent_child.py`

```python
# Input: 5000 character document
text = "Avaya QA Validation and Regression Testing..."

# Output: Multiple parent-child pairs
Parent 1 (3000 chars):
  "Avaya QA Validation... [section 1-3]"
  
  Child 1.1 (500 chars): "Avaya QA Validation... [section 1]"
  Child 1.2 (500 chars): "[section 2]"
  Child 1.3 (500 chars): "[section 3]"

Parent 2 (3000 chars):
  "[section 3-5]"
  
  Child 2.1 (500 chars): "[section 3]"
  Child 2.2 (500 chars): "[section 4]"
  Child 2.3 (500 chars): "[section 5]"
```

**Metadata added to each child:**
```python
{
    'source': 'knowledge-base/avaya/avaya_01_Overview.md',
    'category': 'avaya',
    'filename': 'avaya_01_Overview.md',
    'parent_id': 'abc123...',           # ← Links to parent
    'parent_text': '<full parent text>', # ← Full context
    'child_index': 0,
    'parent_index': 0,
    'chunk_type': 'child',
    'chunking_strategy': 'parent_child'
}
```

#### Option B: Semantic Chunking
**Location:** `rag-project2/src/chunking/semantic_chunker.py`

```python
# Input: Same 5000 character document
text = "Avaya QA Validation and Regression Testing..."

# Output: Variable-sized chunks at topic boundaries
Chunk 1 (800 chars): "Overview section..."
Chunk 2 (1200 chars): "Architecture section..."
Chunk 3 (600 chars): "Testing process section..."
Chunk 4 (900 chars): "Results section..."
```

**Metadata added:**
```python
{
    'source': 'knowledge-base/avaya/avaya_01_Overview.md',
    'category': 'avaya',
    'filename': 'avaya_01_Overview.md',
    'chunk_index': 0,
    'semantic_group': 0,
    'chunking_strategy': 'semantic'
}
```

---

### Step 4: Collect All Chunks 📦
**Location:** `rag-project2/main_v2.py` (lines 88-101)

```python
all_chunks = []      # List of text chunks
all_metadatas = []   # List of metadata dicts

# For each category (avaya, bppsl, pusa, etc.)
#   For each file in category
#     Extract text
#     Chunk text (parent-child or semantic)
#     Add chunks to all_chunks
#     Add metadata to all_metadatas

# Result:
all_chunks = [
    "Avaya QA Validation and Regression Testing...",  # Chunk 1
    "The testing process involves...",                 # Chunk 2
    "BPPSL fare calculation logic...",                 # Chunk 3
    ...                                                # ~500-1000 chunks total
]

all_metadatas = [
    {'source': '...', 'category': 'avaya', ...},      # Metadata 1
    {'source': '...', 'category': 'avaya', ...},      # Metadata 2
    {'source': '...', 'category': 'bppsl', ...},      # Metadata 3
    ...
]
```

---

### Step 5: Create Embeddings 🧮
**Location:** `rag-project2/app/embedding.py` → `create_vector_store()`

```python
# For each chunk, create embedding using AWS Bedrock
embeddings = BedrockEmbeddings(model="amazon.titan-embed-text-v1")

# Example:
chunk = "Avaya QA Validation and Regression Testing..."
embedding = embeddings.embed_query(chunk)
# Result: [0.123, -0.456, 0.789, ...] (1536 dimensions)
```

**What happens:**
1. Each chunk text is sent to AWS Bedrock
2. Titan model converts text → 1536-dimensional vector
3. Vector represents semantic meaning of the text

---

### Step 6: Build FAISS Vector Database 🗄️
**Location:** `rag-project2/app/embedding.py` → `create_vector_store()`

```python
from langchain_community.vectorstores import FAISS

# Create FAISS index with all chunks and embeddings
vectorstore = FAISS.from_texts(
    texts=all_chunks,           # ~500-1000 text chunks
    embedding=embeddings,        # Bedrock embedder
    metadatas=all_metadatas     # Metadata for each chunk
)
```

**What FAISS does:**
- Creates an efficient index for similarity search
- Stores embeddings in a way that allows fast nearest-neighbor lookup
- Links each embedding to its text and metadata

---

### Step 7: Save to Disk 💾
**Location:** `rag-project2/app/embedding.py` → `save_vector_store()`

```python
# Save to: rag-project2/vector_store/
vectorstore.save_local("vector_store")
```

**Files created:**

```
rag-project2/vector_store/
├── index.faiss      # FAISS index (embeddings + search structure)
│                    # Size: ~5-50 MB depending on # of chunks
│                    # Binary format, optimized for fast search
│
├── index.pkl        # Pickled metadata and document store
│                    # Contains: chunk texts + metadata
│                    # Size: ~1-10 MB
│
└── metadata.pkl     # File modification timestamps
                     # Used to detect if docs changed
                     # Size: ~1 KB
```

---

### Step 8: Clear Caches 🧹
**Location:** `rag-project2/main_v2.py` (line 109)

```python
st.session_state.pipeline.clear_caches()
```

**What it does:**
- Clears all 3 cache tiers (Exact, Semantic, Retrieval)
- Ensures fresh start with new chunks
- Deletes entries from SQLite databases

---

## 📊 Complete Example

### Input
```
knowledge-base/
└── avaya/
    └── avaya_01_Overview.md (5000 chars)
```

### Processing (Parent-Child Strategy)

**Step 1:** Extract text → 5000 chars  
**Step 2:** Split into parents → 2 parents (3000 chars each)  
**Step 3:** Split parents into children → 6 children (500 chars each)  
**Step 4:** Create embeddings → 6 embeddings (1536 dims each)  
**Step 5:** Store in FAISS → 6 searchable vectors  

### Output Files

```
vector_store/
├── index.faiss      # Contains 6 embeddings
├── index.pkl        # Contains 6 chunks + metadata
└── metadata.pkl     # Contains file timestamp
```

### Metadata Example

```python
# Child chunk 1
{
    'source': 'knowledge-base/avaya/avaya_01_Overview.md',
    'category': 'avaya',
    'filename': 'avaya_01_Overview.md',
    'parent_id': 'a1b2c3d4...',
    'parent_text': '<3000 char parent context>',
    'child_index': 0,
    'parent_index': 0,
    'chunk_type': 'child',
    'chunking_strategy': 'parent_child'
}
```

---

## 🔍 How Search Works After Building

### User Query: "What is Avaya testing process?"

**Step 1:** Embed query
```python
query_embedding = embeddings.embed_query("What is Avaya testing process?")
# Result: [0.234, -0.567, 0.890, ...]
```

**Step 2:** Search FAISS
```python
# Find 10 most similar chunks
docs = vectorstore.similarity_search(query, k=10)
```

**Step 3:** FAISS compares query embedding to all stored embeddings
```
Query:    [0.234, -0.567, 0.890, ...]
Chunk 1:  [0.245, -0.543, 0.876, ...] → Similarity: 0.95 ✓
Chunk 2:  [0.123, -0.234, 0.345, ...] → Similarity: 0.72
Chunk 3:  [0.234, -0.560, 0.885, ...] → Similarity: 0.98 ✓✓
...
```

**Step 4:** Return top 10 chunks with metadata

**Step 5:** If parent-child, extract parent text from metadata

**Step 6:** Pass to LLM for answer generation

---

## 📁 File Locations Summary

| What | Location | Purpose |
|------|----------|---------|
| **Source Documents** | `rag-project2/knowledge-base/` | Your PDFs, DOCX, MD files |
| **Chunking Code** | `rag-project2/src/chunking/` | Parent-child & semantic strategies |
| **Vector Store** | `rag-project2/vector_store/` | FAISS index + embeddings |
| **Cache Databases** | `rag-project2/cache/` | SQLite cache files |
| **Build Function** | `rag-project2/main_v2.py` | `build_knowledge_base_v2()` |
| **Embedding Code** | `rag-project2/app/embedding.py` | FAISS creation & saving |
| **Ingestion Code** | `rag-project2/app/ingestion.py` | File reading & text extraction |

---

## 🎯 Key Takeaways

1. **Source:** Documents in `knowledge-base/` folder
2. **Chunking:** Happens in memory using selected strategy
3. **Embeddings:** Created via AWS Bedrock Titan
4. **Storage:** Saved to `vector_store/` as FAISS index
5. **Metadata:** Includes parent context (parent-child) or semantic info
6. **Search:** FAISS enables fast similarity search
7. **Caches:** Cleared on rebuild to ensure fresh start

---

## 💡 Tips

### View What's in Vector Store
```python
# Load and inspect
from app.embedding import load_vector_store
vectorstore = load_vector_store()

# Get all documents
docs = vectorstore.docstore._dict
print(f"Total chunks: {len(docs)}")

# View first chunk
first_doc = list(docs.values())[0]
print(f"Text: {first_doc.page_content[:100]}...")
print(f"Metadata: {first_doc.metadata}")
```

### Check Vector Store Size
```bash
ls -lh vector_store/
# Shows file sizes
```

### Rebuild When
- ✅ Added new documents to `knowledge-base/`
- ✅ Modified existing documents
- ✅ Want to try different chunking strategy
- ✅ Changed chunk size in config

### Don't Rebuild When
- ❌ Just want to clear cache (use "Clear Caches Only")
- ❌ Testing different queries
- ❌ Adjusting cache thresholds
