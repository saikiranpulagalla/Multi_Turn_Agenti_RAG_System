# 🤖 Agentic RAG System

> **Advanced Retrieval-Augmented Generation (RAG) powered by intelligent multi-agent orchestration, featuring smart routing, persistent vector storage, and complete observability.**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)

---

## 🎯 Overview

**Agentic RAG System** is an enterprise-grade Retrieval-Augmented Generation platform that combines:

- **Intelligent Multi-Agent Orchestration**: Specialized agents for routing, document retrieval, web search, and response synthesis
- **Smart Query Routing**: Automatically decides whether to use document RAG, web search, or hybrid approach
- **Advanced Retrieval Strategies**: Late-chunking, parent-document retrieval, and semantic similarity ranking
- **Persistent Vector Storage**: Integrated Pinecone support with intelligent fallback mechanisms
- **Complete Observability**: LangSmith tracing for every agent decision and LLM call
- **Production-Ready**: Comprehensive error handling, caching, logging, and type safety

Perfect for:
- Document analysis and Q&A systems
- Knowledge base search engines
- Research paper summarization
- Intelligent chatbots with document context
- Enterprise information retrieval systems

---

## ✨ Key Features

### 🧠 Intelligent Agents
| Agent | Role | Capability |
|-------|------|-----------|
| **Router** | Query Analysis | Determines optimal retrieval strategy (RAG/Web/Hybrid) |
| **RAG Agent** | Document Retrieval | Searches uploaded documents with advanced chunking |
| **Web Search** | Internet Search | Real-time web search for current information |
| **Synthesis** | Response Generation | Creates coherent, comprehensive answers |

### 🔍 Advanced Retrieval
- **Late-Chunking**: Preserves context by overlapping document chunks
- **Parent-Document Strategy**: Groups chunks hierarchically for better context
- **Semantic Similarity**: Uses transformer-based embeddings (all-MiniLM-L6-v2)
- **Cosine Similarity Ranking**: Ranks results by relevance
- **Configurable Thresholds**: Fine-tune retrieval precision vs. recall
- **Intent Detection**: Automatically detects "summarize", "explain", and similar queries

### 💾 Vector Storage
- **Pinecone Integration**: Persistent, scalable vector database
- **Intelligent Fallback**: Automatically uses in-memory store if Pinecone unavailable
- **Batch Operations**: Efficient embedding ingestion for large documents
- **Metadata Tracking**: Preserves document context and chunk relationships

### 🔗 LLM Support
- **Multi-Model Fallback**: OpenAI (primary) → Google Gemini (fallback)
- **Configurable Models**: Supports gpt-4o-mini, gpt-4, gemini-pro, etc.
- **Temperature Tuning**: Control response creativity vs. determinism
- **Token Limits**: Configurable max tokens for cost control

### 📊 Observability
- **LangSmith Tracing**: Real-time traces of all agent decisions
- **Structured Logging**: Event-driven logging with context
- **Performance Metrics**: Track latency, token usage, retrieval performance
- **Debug Dashboard**: Built-in Streamlit UI for monitoring

### ⚡ Performance
- **Response Caching**: Cache frequent queries for <10ms retrieval
- **Batch Embeddings**: Process multiple chunks in parallel
- **Async Operations**: Non-blocking API calls
- **Optimized Similarity Search**: Fast cosine similarity computations

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI / API                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Router Agent (Query Analysis)                   │
│  ├─ Intent Detection (summarize, key points, etc.)          │
│  ├─ Document Availability Check                             │
│  └─ Strategy Selection (RAG / Web / Hybrid)                 │
└────────────────┬─────────────────────────────┬──────────────┘
                 │                             │
         RAG Path▼                      Web Search Path▼
┌──────────────────────────────┐  ┌──────────────────────────┐
│    RAG Agent                 │  │  Web Search Agent        │
│  ├─ Query Encoding           │  ├─ Query Enhancement      │
│  ├─ Parent Doc Retrieval     │  ├─ SerpAPI/DuckDuckGo    │
│  ├─ Chunk Assembly           │  ├─ Result Ranking        │
│  ├─ Similarity Filtering     │  └─ Snippet Extraction    │
│  └─ Result Ranking           │                           │
└────────────┬─────────────────┘  └────────────┬────────────┘
             │                                 │
             └─────────────────┬───────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│         Synthesis Agent (Response Generation)                │
│  ├─ Context Merging (RAG + Web results)                     │
│  ├─ LLM Prompt Crafting                                     │
│  ├─ Multi-Model Fallback (OpenAI → Gemini)                │
│  └─ Response Formatting                                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            Storage Layer                                     │
│  ├─ Pinecone (Vector Embeddings)                           │
│  ├─ In-Memory (Chunk Index)                                │
│  ├─ LangSmith (Trace Storage)                              │
│  └─ Response Cache (Recent Queries)                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 4GB minimum (8GB+ recommended for production)
- **Disk**: 2GB for dependencies + document storage

### API Keys (At Least One Required)

| Service | Purpose | Requirement | Free Tier |
|---------|---------|-------------|-----------|
| **OpenAI** | Primary LLM | API Key | Yes, $5 free |
| **Google Gemini** | Fallback LLM | API Key | Yes, Free tier available |
| **SerpAPI** | Web Search | API Key (Optional) | 100 free searches/month |
| **Pinecone** | Vector DB | API Key (Optional) | Free tier: 1 pod, 1M vectors |
| **LangSmith** | Tracing/Monitoring | API Key (Optional) | Free tier available |

### Get API Keys

1. **OpenAI**: https://platform.openai.com/api-keys
2. **Google Gemini**: https://makersuite.google.com/app/apikey
3. **SerpAPI**: https://serpapi.com (sign up for API key)
4. **Pinecone**: https://www.pinecone.io (create free account)
5. **LangSmith**: https://smith.langchain.com

---

## 🔧 Installation

### Step 1: Clone or Navigate to Project

```bash
cd agentic-rag-system
```

### Step 2: Create Virtual Environment

#### Using venv (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### Using Conda
```bash
conda create -n rag-system python=3.10
conda activate rag-system
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: This installs all required packages including:
- `streamlit` - UI framework
- `langchain` - LLM orchestration
- `sentence-transformers` - Embeddings
- `pinecone` - Vector database
- `langsmith` - Observability
- `openai` & `google-generativeai` - LLM providers
- And more...

### Step 4: Configure API Keys

Create a `.env` file in the project root directory:

```bash
touch .env
```

Edit `.env` and add your API keys:

```env
# ========== REQUIRED: LLM Configuration ==========
# Add at least ONE of these
OPENAI_API_KEY=sk-your-openai-key-here
GEMINI_API_KEY=your-gemini-key-here

# ========== OPTIONAL: LLM Model Selection ==========
OPENAI_MODEL=gpt-4o-mini          # Default: gpt-4o-mini
GEMINI_MODEL=gemini-pro            # Default: gemini-pro

# ========== OPTIONAL: Web Search Configuration ==========
SERP_API_KEY=your-serpapi-key-here
WEB_SEARCH_RESULTS=5                # Default: 5

# ========== OPTIONAL: Vector Database (Pinecone) ==========
USE_PINECONE=true                   # Default: true
PINECONE_API_KEY=your-key-here
PINECONE_HOST=https://rag-xxxxx.svc.aped-4627-b74a.pinecone.io
PINECONE_INDEX_NAME=rag             # Default: rag
EMBED_DIM=1024                      # Depends on Pinecone index dimension

# ========== OPTIONAL: Observability (LangSmith) ==========
ENABLE_LANGSMITH=true               # Default: true
LANGSMITH_API_KEY=your-key-here
LANGSMITH_PROJECT=agentic-rag       # Default: agentic-rag

# ========== OPTIONAL: Embedding Model ==========
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2  # Default

# ========== OPTIONAL: Chunking Strategy ==========
CHUNK_SIZE=1000                     # Document chunk size (chars)
CHUNK_OVERLAP=100                   # Overlap between chunks

# ========== OPTIONAL: Retrieval Tuning ==========
TOP_K_PARENTS=3                     # Parent documents to retrieve
TOP_K_CHUNKS=6                      # Chunks per query
RELEVANCE_THRESHOLD=0.15            # Similarity score threshold

# ========== OPTIONAL: LLM Parameters ==========
LLM_TEMPERATURE=0.0                 # 0=deterministic, 1=creative
LLM_MAX_TOKENS=1024                 # Max response length

# ========== OPTIONAL: Performance & Caching ==========
ENABLE_CACHE=true                   # Cache responses
CACHE_TTL_SECONDS=3600              # Cache validity (1 hour)
ENABLE_TRACING=true                 # Enable event tracing
LOG_LEVEL=INFO                      # DEBUG, INFO, WARNING, ERROR
```

### Step 5: Verify Installation

```bash
# Quick test
python -c "from core.utils.config import Config; print(Config.get_summary())"
```

Expected output (example):
```
{
  'openai_configured': True,
  'gemini_configured': False,
  'web_search_configured': False,
  'pinecone_configured': False,
  'langsmith_enabled': False,
  'chunk_size': 1000,
  'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
  'cache_enabled': True,
  'tracing_enabled': True
}
```

---

## 🚀 Quick Start

### Option 1: Streamlit UI (Recommended)

```bash
streamlit run streamlit_app.py
```

This opens the app at `http://localhost:8501`

**Using the UI:**

1. **Upload Document** (Optional)
   - Click "Upload PDF" in the sidebar
   - Select a PDF file
   - Wait for processing confirmation

2. **Enter Query**
   - Type your question in the text area
   - Examples:
     - "Summarize the document"
     - "What are the key points?"
     - "Explain this in simple terms"

3. **View Results**
   - Response appears below
   - Source attribution (RAG/Web)
   - Processing metrics shown

### Option 2: Python API

```python
from core.agents.rag_agent import RAGAgent
from core.agents.router_agent import RouterAgent
from core.agents.web_search_agent import WebSearchAgent
from core.agents.synthesis_agent import SynthesisAgent
from core.memory.memory_service import MemoryService

# Initialize components
memory = MemoryService()
rag = RAGAgent()
web = WebSearchAgent()
synthesis = SynthesisAgent()
router = RouterAgent(rag, web, synthesis)

# Option A: Ingest a document
doc_id = rag.ingest_pdf("path/to/document.pdf")
print(f"Ingested: {doc_id}")

# Option B: Process a query
query = "Summarize the main points"
response = router.route({
    "query_text": query,
    "has_doc": True
})

print("Answer:", response["result"].get("summary"))
print("Source:", response["route"])  # 'rag' or 'web'
```

### Option 3: Command Line

```bash
python main.py --query "Your question here" --pdf path/to/document.pdf
```

---

## ⚙️ Configuration Guide

### Environment Variables Reference

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **LLM Configuration** |
| `OPENAI_API_KEY` | string | None | OpenAI API key |
| `OPENAI_MODEL` | string | gpt-4o-mini | OpenAI model to use |
| `GEMINI_API_KEY` | string | None | Google Gemini API key |
| `GEMINI_MODEL` | string | gemini-pro | Gemini model to use |
| **Embeddings** |
| `EMBED_MODEL_NAME` | string | sentence-transformers/all-MiniLM-L6-v2 | Embedding model |
| `EMBED_DIM` | int | 384 | Embedding dimension |
| **Chunking** |
| `CHUNK_SIZE` | int | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | int | 100 | Overlap between chunks |
| **Retrieval** |
| `TOP_K_PARENTS` | int | 3 | Parent documents to retrieve |
| `TOP_K_CHUNKS` | int | 6 | Chunks per result |
| `RELEVANCE_THRESHOLD` | float | 0.15 | Min similarity score |
| **LLM Sampling** |
| `LLM_TEMPERATURE` | float | 0.0 | Creativity (0-1) |
| `LLM_MAX_TOKENS` | int | 1024 | Max response tokens |
| **Web Search** |
| `SERP_API_KEY` | string | None | SerpAPI key |
| `WEB_SEARCH_RESULTS` | int | 5 | Results per search |
| **Vector Storage** |
| `USE_PINECONE` | bool | true | Use Pinecone? |
| `PINECONE_API_KEY` | string | None | Pinecone API key |
| `PINECONE_HOST` | string | None | Pinecone host URL |
| `PINECONE_INDEX_NAME` | string | rag | Index name |
| **Observability** |
| `ENABLE_LANGSMITH` | bool | true | Enable tracing? |
| `LANGSMITH_API_KEY` | string | None | LangSmith API key |
| `LANGSMITH_PROJECT` | string | agentic-rag | Project name |
| **Performance** |
| `ENABLE_CACHE` | bool | true | Enable caching? |
| `CACHE_TTL_SECONDS` | int | 3600 | Cache lifetime |
| `ENABLE_TRACING` | bool | true | Enable tracing? |
| `LOG_LEVEL` | string | INFO | Log verbosity |

### Configuration Validation

The system validates configuration on startup:

```python
from core.utils.config import Config

# Validate and get warnings
is_valid, warnings = Config.validate()

for warning in warnings:
    print(warning)

# Get configuration summary
summary = Config.get_summary()
print(summary)
```

---

## 🤖 How It Works

### Query Processing Pipeline

```
User Query
    ↓
[1] ROUTING
    ├─ Intent Analysis
    │  └─ Detects: summarize, explain, key points, etc.
    │
    ├─ Document Availability
    │  └─ Checks if PDFs are ingested
    │
    └─ Route Decision
       ├─ RAG Mode (if document-specific query + docs available)
       ├─ Web Mode (for general/trending queries)
       └─ Hybrid (for comparison/comprehensive queries)
    ↓
[2] RETRIEVAL
    ├─ RAG Path:
    │  ├─ Encode query to embeddings
    │  ├─ Find similar parent documents
    │  ├─ Assemble chunks from documents
    │  ├─ Rank by similarity score
    │  └─ Filter by relevance threshold
    │
    └─ Web Path:
       ├─ Enhance query with context
       ├─ Search via SerpAPI/DuckDuckGo
       ├─ Extract snippets
       └─ Rank by relevance
    ↓
[3] SYNTHESIS
    ├─ Merge results (RAG + Web if hybrid)
    ├─ Create LLM prompt with context
    ├─ Call LLM (OpenAI or Gemini)
    ├─ Format response
    └─ Cache for future requests
    ↓
Response
```

### Agent Responsibilities

#### 🔀 Router Agent
- **Input**: Query text, document availability
- **Process**: 
  - Detects intent keywords
  - Analyzes query complexity
  - Checks document context
- **Output**: Route decision (RAG/Web/Hybrid)

#### 📄 RAG Agent
- **Input**: Query, document collection
- **Process**:
  - Ingests PDFs (chunking, embedding, vectorization)
  - Encodes queries to embeddings
  - Retrieves similar chunks
  - Ranks by relevance
- **Output**: Ranked document chunks

#### 🌐 Web Search Agent
- **Input**: Query, search parameters
- **Process**:
  - Enhances query for better results
  - Calls search API
  - Extracts relevant snippets
  - Ranks results
- **Output**: Search results with snippets

#### 🧬 Synthesis Agent
- **Input**: Query, retrieved context, route
- **Process**:
  - Prepares LLM prompt
  - Calls LLM with context
  - Handles multi-model fallback
  - Formats response
- **Output**: Final answer

---

## 📁 Project Structure

```
agentic-rag-system/
│
├── 📂 core/                          # Core system components
│   ├── 📂 agents/                    # Intelligent agents
│   │   ├── router_agent.py           # Query routing logic
│   │   ├── rag_agent.py              # Document retrieval
│   │   ├── web_search_agent.py       # Web search integration
│   │   ├── synthesis_agent.py        # Response generation
│   │   ├── planner_agent.py          # Query planning
│   │   └── ragas_evaluator.py        # RAG evaluation
│   │
│   ├── 📂 retriever/                 # Retrieval strategies
│   │   ├── late_chunker.py           # Chunk assembly logic
│   │   ├── vectorstore_adapter.py    # Vector DB interface
│   │   ├── pinecone_vectorstore.py   # Pinecone integration
│   │   ├── parent_child_retrieval.py # Hierarchical retrieval
│   │   └── raptor_chunker.py         # MMR-based selection
│   │
│   ├── 📂 utils/                     # Utility functions
│   │   ├── config.py                 # Centralized config
│   │   ├── logging_utils.py          # Logging setup
│   │   ├── langsmith_tracer.py       # LangSmith integration
│   │   ├── session_manager.py        # Session handling
│   │   └── __init__.py
│   │
│   ├── 📂 memory/                    # Memory & state
│   │   ├── memory_service.py         # Conversation memory
│   │   └── __init__.py
│   │
│   ├── 📂 tools/                     # Document tools
│   │   ├── pdf_loader.py             # PDF extraction
│   │   ├── url_loader.py             # URL content loading
│   │   ├── doc_loader_tool.py        # General doc loading
│   │   └── web_search_tool.py        # Search wrapper
│   │
│   ├── 📂 evaluation/                # Evaluation metrics
│   │   ├── metrics.py                # Performance metrics
│   │   └── ragas_eval.py             # RAGAS framework
│   │
│   └── __init__.py
│
├── 📂 llm/                           # LLM abstractions
│   ├── gemini_client.py              # Google Gemini client
│   ├── prompts.py                    # Prompt templates
│   └── __init__.py
│
├── 📂 ui/                            # UI components
│   ├── app.py                        # FastAPI app
│   ├── 📂 components/
│   │   ├── chatbox.py                # Chat widget
│   │   ├── sidebar.py                # Sidebar UI
│   │   └── uploader.py               # File upload
│   └── __init__.py
│
├── 📄 streamlit_app.py               # Main Streamlit app ⭐
├── 📄 main.py                        # CLI entry point
├── 📄 requirements.txt                # Python dependencies
├── 📄 .env.example                   # Example env file
├── 📄 README.md                      # This file
├── 📄 SETUP.md                       # Detailed setup guide
├── 📄 ARCHITECTURE.md                # System architecture
└── 📄 PINECONE_LANGSMITH_GUIDE.md   # Advanced setup
```

---

## 💡 Usage Examples

### Example 1: Simple Document Q&A

```python
from core.agents.rag_agent import RAGAgent
from core.agents.synthesis_agent import SynthesisAgent
from core.agents.router_agent import RouterAgent

# Setup
rag = RAGAgent()
synthesis = SynthesisAgent()
router = RouterAgent(rag, None, synthesis)

# Ingest document
rag.ingest_pdf("research_paper.pdf")

# Query
response = router.route({
    "query_text": "What are the main findings?",
    "has_doc": True
})
print(response["result"]["summary"])
```

### Example 2: Multi-Document Analysis

```python
import os
from core.agents.rag_agent import RAGAgent

rag = RAGAgent()

# Ingest multiple documents
docs = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
for doc in docs:
    try:
        doc_id = rag.ingest_pdf(doc)
        print(f"✓ Ingested: {doc_id}")
    except Exception as e:
        print(f"✗ Failed: {e}")

# Query across all documents
chunks = rag.retrieve("key concept to search")
print(f"Found {len(chunks)} relevant chunks")
```

### Example 3: Hybrid RAG + Web Search

```python
from core.agents.router_agent import RouterAgent
from core.agents.rag_agent import RAGAgent
from core.agents.web_search_agent import WebSearchAgent
from core.agents.synthesis_agent import SynthesisAgent

# Full setup
rag = RAGAgent()
web = WebSearchAgent()
synthesis = SynthesisAgent()
router = RouterAgent(rag, web, synthesis)

# Complex query
response = router.route({
    "query_text": "Compare my document's insights with latest market trends",
    "has_doc": True
})

print(f"Route: {response['route']}")  # Should be 'hybrid'
print(f"Answer: {response['result']['summary']}")
```

### Example 4: Using Streamlit UI

```bash
# Terminal
streamlit run streamlit_app.py

# In browser at http://localhost:8501:
# 1. Upload PDF in sidebar
# 2. Type query: "Summarize the document"
# 3. Click "Run Query"
# 4. Read response with source attribution
```

---

## 🎓 Advanced Features

### Pinecone Vector Database Setup

For production with persistent vector storage:

```env
USE_PINECONE=true
PINECONE_API_KEY=pcak_xxxxxxxxxxxxx
PINECONE_HOST=https://rag-xxxxx.svc.aped-4627-b74a.pinecone.io
EMBED_DIM=1024
```

Then restart the app:
```bash
streamlit run streamlit_app.py
```

### LangSmith Observability

Enable complete tracing of all agent decisions:

```env
ENABLE_LANGSMITH=true
LANGSMITH_API_KEY=ls_xxxxxxxxxxxxx
LANGSMITH_PROJECT=agentic-rag
```

View traces at: https://smith.langchain.com

### Custom Chunking Strategy

Adjust retrieval behavior by modifying chunk parameters:

```env
CHUNK_SIZE=1500           # Larger chunks = more context
CHUNK_OVERLAP=200         # More overlap = better continuity
TOP_K_CHUNKS=8            # More chunks = slower but comprehensive
RELEVANCE_THRESHOLD=0.2   # Higher = more strict filtering
```

### Response Caching

Cache frequent queries for instant retrieval:

```env
ENABLE_CACHE=true
CACHE_TTL_SECONDS=7200    # 2 hours
```

### Debug Mode

Enable detailed logging:

```env
LOG_LEVEL=DEBUG
```

Then check console output for detailed trace information.

---

## 🐛 Troubleshooting

### Issue 1: "Neither OPENAI_API_KEY nor GEMINI_API_KEY is set"

**Cause**: No LLM API keys configured

**Solution**:
```bash
# Create .env file with at least one API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Or use Gemini
echo "GEMINI_API_KEY=your-key-here" > .env
```

### Issue 2: PDF Extraction Fails

**Error**: `Failed to read PDF` or `PDF contains no extractable text`

**Solutions**:
- Ensure PDF is not password-protected
- Try a different PDF file
- Check file is valid: `file document.pdf`
- Some PDFs have images instead of text (OCR not supported)

### Issue 3: Pinecone Connection Error

**Error**: `(404) Not Found` when initializing Pinecone

**Causes & Solutions**:
- Incorrect `PINECONE_HOST`: Copy exact URL from Pinecone console
- Invalid `PINECONE_API_KEY`: Regenerate in Pinecone dashboard
- Vector dimension mismatch: Ensure `EMBED_DIM` matches your index
- Fallback: System automatically uses in-memory storage

### Issue 4: Web Search Returns No Results

**Error**: "Web search returning stub responses"

**Solution**:
```bash
# Get SerpAPI key from https://serpapi.com
echo "SERP_API_KEY=your-key-here" >> .env

# Restart app
streamlit run streamlit_app.py
```

### Issue 5: Slow Response Times

**Issue**: Synthesis takes 10+ seconds

**Solutions**:
```env
# Option 1: Reduce response length
LLM_MAX_TOKENS=512

# Option 2: Enable caching
ENABLE_CACHE=true

# Option 3: Use faster model
OPENAI_MODEL=gpt-3.5-turbo

# Option 4: Reduce chunk count
TOP_K_CHUNKS=3
```

### Issue 6: Out of Memory Error

**Error**: `MemoryError` during embedding

**Solutions**:
```env
# Reduce batch processing
CHUNK_SIZE=500

# Reduce top-k values
TOP_K_PARENTS=2
TOP_K_CHUNKS=3

# System: Upgrade RAM or use Pinecone for distributed storage
```

### Issue 7: LangSmith Tracing Not Working

**Error**: Traces not appearing in LangSmith

**Solutions**:
- Verify `LANGSMITH_API_KEY` is correct
- Check `ENABLE_LANGSMITH=true`
- Ensure `LANGSMITH_PROJECT` exists
- Restart app after .env changes

### Getting Help

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
streamlit run streamlit_app.py

# Check configuration
python -c "from core.utils.config import Config; print(Config.get_summary())"

# Validate setup
python -c "from core.utils.config import Config; is_valid, warnings = Config.validate(); [print(w) for w in warnings]"
```

---

## 📊 Performance Metrics

Typical performance on modern hardware (macOS M1, 16GB RAM):

| Operation | Time | Notes |
|-----------|------|-------|
| **Embedding (per chunk)** | ~50-100ms | Depends on chunk size & model |
| **Similarity Search** | ~30-50ms | For 1000 chunks |
| **Parent Doc Retrieval** | ~20ms | Fast cosine similarity |
| **Chunk Assembly** | ~10-20ms | In-memory operation |
| **LLM Synthesis** | 2-10s | Depends on LLM and response length |
| **Cache Hit** | <10ms | If query cached |
| **PDF Ingestion** | 1-5s | Depends on PDF size |
| **Full Query** | 3-15s | RAG → Synthesis end-to-end |

### Optimization Tips

- ✅ Use smaller `CHUNK_SIZE` for faster processing
- ✅ Reduce `TOP_K_CHUNKS` for quicker synthesis
- ✅ Enable `ENABLE_CACHE` for repeated queries
- ✅ Use `gpt-3.5-turbo` instead of `gpt-4` for speed
- ✅ Set `LLM_TEMPERATURE=0.0` for faster inference
- ✅ Limit `LLM_MAX_TOKENS` to necessary length

---

## 🤝 Contributing

We welcome contributions! To improve the system:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature`
3. **Make** your improvements
4. **Test** thoroughly: `pytest tests/ -v`
5. **Commit**: `git commit -am "Add your feature"`
6. **Push**: `git push origin feature/your-feature`
7. **Submit** a Pull Request

### Areas for Contribution

- 🔧 Add new LLM providers
- 📊 Improve retrieval algorithms
- 🎨 Enhance UI/UX
- 📈 Performance optimizations
- 📚 Documentation improvements
- 🧪 Additional test coverage

---

## 📚 Additional Resources

- 📖 [Architecture Guide](./ARCHITECTURE.md) - Deep dive into system design
- 🔌 [Pinecone + LangSmith Setup](./PINECONE_LANGSMITH_GUIDE.md) - Production setup
- 📋 [Setup Guide](./SETUP.md) - Detailed installation steps
- 🚀 [Quick Reference](./QUICK_REFERENCE.md) - Common commands
- 💬 [Routing Guide](./ROUTING_TEST_GUIDE.md) - Testing routing logic

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

Built with:
- [LangChain](https://langchain.com) - LLM orchestration
- [Streamlit](https://streamlit.io) - Web UI
- [Pinecone](https://www.pinecone.io) - Vector database
- [LangSmith](https://smith.langchain.com) - Observability
- [Sentence Transformers](https://sbert.net) - Embeddings
- [OpenAI](https://openai.com) & [Google Gemini](https://makersuite.google.com) - LLMs

---

## 🚀 Quick Commands Reference

```bash
# Installation
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configuration
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Running
streamlit run streamlit_app.py

# Testing
pytest tests/ -v

# Debug mode
LOG_LEVEL=DEBUG streamlit run streamlit_app.py

# Check configuration
python -c "from core.utils.config import Config; print(Config.get_summary())"
```

---

**🌟 Built with ❤️ for intelligent document analysis and synthesis**

**Last Updated**: November 30, 2025  
**Version**: 1.0.0  
**Status**: Production Ready ✅
