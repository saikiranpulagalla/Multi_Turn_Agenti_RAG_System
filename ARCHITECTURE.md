# 🏗️ Architecture Documentation

This document describes the system architecture and design patterns used in the Agentic RAG System.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI                            │
│            (streamlit_app.py)                               │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│                   Router Agent                              │
│    (Intelligent Query Routing)                              │
│    ├── RAG Mode   (Document Search)                         │
│    ├── Web Mode   (Internet Search)                         │
│    └── Hybrid     (Both)                                    │
└────────┬───────────────┬──────────────────┬─────────────────┘
         │               │                  │
    ┌────▼─────┐  ┌────▼──────────┐  ┌─────▼────────┐
    │ RAG Agent │  │Web Search Ag. │  │Synthesis Ag. │
    │           │  │               │  │              │
    │• Ingest   │  │• Query Web    │  │• Merge Data  │
    │• Chunk    │  │• Parse Result │  │• LLM Synth   │
    │• Embed    │  │• Cache        │  │• Format Out  │
    │• Retrieve │  │• Rate Limit   │  │• Error Hdl   │
    └────┬─────┘  └────┬──────────┘  └─────┬────────┘
         │             │                   │
    ┌────▼─────────────▼───────────────────▼────┐
    │         LLM Client Abstraction             │
    │  ├── OpenAI (Primary)                      │
    │  └── Google Gemini (Fallback)              │
    └────┬───────────────────────────────────────┘
         │
    ┌────▼───────────────────────────────────┐
    │    Supporting Services                 │
    │  ├── Vector Store (In-Memory/FAISS)    │
    │  ├── Memory Service (Sessions)         │
    │  ├── Config Management                 │
    │  └── Logging & Tracing                 │
    └────────────────────────────────────────┘
```

## Core Components

### 1. Router Agent (`core/agents/router_agent.py`)

**Responsibility**: Intelligent routing of queries to appropriate backends

**Key Features:**
- Analyzes query text, length, and keywords
- Makes routing decisions based on configurable heuristics
- Supports three modes: RAG, Web, Hybrid
- Comprehensive error handling

**Decision Logic:**
```python
# Pseudo-code
if has_doc and query_is_short:
    route = "rag"
elif query_mentions_recent:
    route = "web"
elif query_asks_for_comparison:
    route = "hybrid"
else:
    route = "web" if not has_doc else "hybrid"
```

### 2. RAG Agent (`core/agents/rag_agent.py`)

**Responsibility**: Document ingestion, chunking, and retrieval

**Key Features:**
- PDF parsing with pdfplumber
- Intelligent chunking with overlapping windows
- Embedding generation using sentence-transformers
- Late-chunking retrieval strategy
- Parent-document indexing

**Workflow:**
```
PDF Input
    ↓
Extract Text (page-by-page)
    ↓
Create Chunks (with overlap)
    ↓
Generate Embeddings
    ↓
Store in Vector DB
    ↓
Index Parent Documents
    ↓
Ready for Retrieval
```

### 3. Web Search Agent (`core/agents/web_search_agent.py`)

**Responsibility**: Web search capability and caching

**Key Features:**
- SerpAPI integration for real web search
- Stub responses for demo/testing
- Request caching with TTL
- Error handling and retries
- Statistics tracking

**Flow:**
```
Query
    ↓
Check Cache
    ├─ Hit → Return Cached Results
    └─ Miss → Call SerpAPI
         ↓
    Parse Results
         ↓
    Cache Results
         ↓
    Return Results
```

### 4. Synthesis Agent (`core/agents/synthesis_agent.py`)

**Responsibility**: Merging and synthesizing results into coherent responses

**Key Features:**
- Prompt engineering for better outputs
- Multi-source synthesis (RAG + Web)
- Structured output format
- Error recovery with helpful messages

**Synthesis Process:**
```
RAG Chunks + Web Results
    ↓
Build Structured Prompt
    ↓
Call LLM with Config Parameters
    ↓
Parse Response
    ↓
Format Output with Metadata
    ↓
Return to User
```

### 5. Vector Store (`core/retriever/vectorstore_adapter.py`)

**Responsibility**: Vector storage and similarity search

**Implementation:**
- In-memory storage using NumPy
- Cosine similarity for ranking
- Support for metadata attachment
- Easy migration path to FAISS/Pinecone

**Operations:**
```python
add(id, embedding, metadata)      # Store vector
search(query_emb, top_k)          # Retrieve similar
get_stats()                        # Get store info
```

### 6. LLM Client (`core/llm_client.py`)

**Responsibility**: Unified LLM interface with fallback

**Key Features:**
- OpenAI support (primary)
- Google Gemini support (fallback)
- Retry logic
- Clear error messages
- Configurable parameters

**Error Handling:**
```
Try OpenAI
    └─ Fail → Try Gemini
         └─ Fail → Raise detailed error
```

## Advanced Retrieval: Late Chunking

### Why Late Chunking?

Traditional RAG uses small chunks for efficiency but loses context. Late chunking combines efficiency with context preservation.

### Implementation

```
Document
    ↓
Split into Large Chunks (2000 chars)
    ├─ Chunk 1: "Document overview..."
    ├─ Chunk 2: "Key findings..."
    └─ Chunk 3: "Conclusions..."
         ↓
For Each Large Chunk:
    ├─ Create embeddings
    ├─ Store as "parent document"
    └─ Store reference
         ↓
Query Comes In
    ├─ Generate query embedding
    ├─ Find similar parent chunks (top-3)
    ├─ Within each parent, refine to sub-chunks
    └─ Return top-6 most relevant sub-chunks
```

### Advantages

1. **Better Context**: Larger chunks retain document structure
2. **Accurate Retrieval**: Refining within parents improves accuracy
3. **Efficiency**: Fewer similarity computations than naive approach

## Data Flow

### Query Processing Flow

```
User Input (Query + Document Status)
    │
    ├─→ [Router] Decide routing strategy
    │       └─→ Analyze query keywords, length
    │       └─→ Check document availability
    │       └─→ Return: "rag" | "web" | "hybrid"
    │
    ├─→ [Retrieval] Based on route decision:
    │   │
    │   ├─→ RAG Path:
    │   │   ├─ Encode query to embeddings
    │   │   ├─ Find parent documents
    │   │   └─ Retrieve top-k chunks
    │   │
    │   └─→ Web Path:
    │       ├─ Check cache
    │       ├─ Call SerpAPI (if configured)
    │       └─ Return web results
    │
    ├─→ [Synthesis] Merge results
    │   ├─ Build prompt from chunks + web results
    │   ├─ Call LLM with temperature & max_tokens
    │   └─ Parse LLM response
    │
    └─→ [Output] Format and return results
        ├─ Summary text
        ├─ Metadata (sources, counts)
        └─ Debug info (timing, route taken)
```

## Configuration Architecture

### Centralized Config (`core/utils/config.py`)

```python
Config
├─ LLM Settings
│  ├─ OPENAI_API_KEY
│  ├─ GEMINI_API_KEY
│  └─ Model names
├─ Retrieval Settings
│  ├─ CHUNK_SIZE
│  ├─ TOP_K_CHUNKS
│  └─ Embedding model
├─ Performance Settings
│  ├─ ENABLE_CACHE
│  ├─ CACHE_TTL
│  └─ LOG_LEVEL
└─ Validation & Defaults
```

**Benefits:**
- Single source of truth
- Automatic validation on import
- Clear defaults
- Environment variable override

## Error Handling Strategy

### Layered Error Handling

```
Layer 1: Validation
    ├─ Input validation (empty strings, types)
    └─ Configuration validation (API keys, values)
        │
Layer 2: Try-Catch
    ├─ File operations (PDF reading)
    ├─ Network calls (API requests)
    └─ Model operations (embedding, LLM)
        │
Layer 3: Fallback
    ├─ Use Gemini if OpenAI fails
    ├─ Use cached results if API fails
    └─ Return meaningful error messages
        │
Layer 4: User Feedback
    ├─ Clear error messages
    ├─ Actionable solutions
    └─ Debug information
```

### Example Error Flow

```python
try:
    result = call_openai()
except APIError:
    logger.warning("OpenAI failed, trying Gemini")
    try:
        result = call_gemini()
    except APIError:
        logger.error("All LLMs failed")
        return helpful_error_message()
```

## Performance Considerations

### Caching Strategy

```
User Query
    ├─ Check Cache
    │  ├─ Hit (< 1s) → Return cached result
    │  └─ Miss → Proceed
    ├─ Check Vector DB
    │  ├─ Cached (1-2s) → Return
    │  └─ New → Generate & cache
    └─ Call LLM
       └─ Cache result (optional)
```

### Optimization Points

1. **Embedding Caching**: Store embeddings, don't regenerate
2. **Query Caching**: Cache synthesis results
3. **Batch Operations**: Process multiple chunks together
4. **Model Optimization**: Use efficient embedding model
5. **Async Operations**: (Future) Support concurrent requests

## Extensibility

### Adding New LLM Providers

```python
# core/llm_client.py
def call_llm(...):
    # Try OpenAI
    try:
        return call_openai(...)
    except:
        pass
    
    # Try Gemini
    try:
        return call_gemini(...)
    except:
        pass
    
    # Add new provider here
    try:
        return call_anthropic(...)
    except:
        pass
    
    raise Exception("All providers failed")
```

### Adding New Agents

```python
# Create new agent
class CustomAgent:
    def process(self, input_data):
        # Custom logic
        return output
        
# Register with router
router = RouterAgent(
    rag_agent=rag,
    web_agent=web,
    custom_agent=custom,  # Add new
    synth_agent=synth
)
```

### Using Different Vector DBs

```python
# Replace InMemoryVectorStore with:
from langchain.vectorstores import FAISS
# or
from pinecone import Index

# Update RAGAgent initialization
rag = RAGAgent(vectorstore=FAISSStore(), ...)
```

## Scalability Path

### Current (Development)
- In-memory storage
- Single process
- No distributed caching

### Near-term (Production)
- FAISS vector store
- Redis caching
- Containerized deployment

### Long-term (Enterprise)
- Pinecone/Weaviate vector DB
- Distributed architecture
- Multi-region deployment
- Advanced analytics

## Security Considerations

1. **API Key Management**
   - Never log API keys
   - Use .env files (not in version control)
   - Rotate keys regularly

2. **Data Privacy**
   - Process user documents locally
   - Don't send user data to external services unnecessarily
   - Clear cache appropriately

3. **Rate Limiting**
   - Implement request throttling
   - Cache to reduce API calls
   - Monitor usage

## Testing Strategy

```
Unit Tests
├─ Component tests (each agent)
├─ Config validation
└─ Error handling

Integration Tests
├─ End-to-end workflows
├─ Multi-agent coordination
└─ Error recovery

Performance Tests
├─ Throughput benchmarks
├─ Latency measurements
└─ Memory profiling
```

---

**Architecture Version**: 1.0  
**Last Updated**: 2024  
**Status**: Production-Ready
