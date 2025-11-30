# Pinecone + LangSmith Integration Guide

## Overview

This guide explains how to use the **Pinecone vector store** for persistent embeddings storage and **LangSmith tracing** for comprehensive agent observability.

---

## Quick Setup

### 1. Verify Environment Variables

Your `.env` file now has:

```dotenv
# Pinecone Configuration
USE_PINECONE=true
PINCONE_API_KEY=your-pinecone-api-key-here
PINCONE_HOST=https://your-project.svc.pinecone.io
PINCONE_INDEX_NAME=rag
EMBED_DIM=1024

# LangSmith Configuration
ENABLE_LANGSMITH=true
LANGSMITH_API_KEY=your-langsmith-api-key-here
LANGSMITH_PROJECT=agentic-rag
```

### 2. Install Dependencies

Run:
```powershell
pip install pinecone langsmith
```

Both packages are already installed in your environment.

### 3. Restart Streamlit

Stop the current Streamlit process and restart:

```powershell
streamlit run streamlit_app.py --server.headless true
```

The app will now:
- ✓ Use Pinecone for persistent vector storage
- ✓ Send all agent traces to LangSmith
- ✓ Fall back to in-memory store if Pinecone is unavailable

---

## LangSmith Tracing

### What Gets Traced?

The system traces:

1. **Router Decisions** – Which path (RAG/Web/Hybrid) is chosen
2. **Document Ingestion** – PDF upload and chunking
3. **Retrieval Operations** – Query embeddings and chunk retrieval
4. **LLM Calls** – Model, prompt length, response latency
5. **Synthesis Process** – Source counts and synthesis success

### Access the Dashboard

Open your LangSmith project:

```
https://smith.langchain.com/o/agentic-rag/projects/agentic-rag
```

### Log Output

When LangSmith is enabled, you'll see entries like:

```
INFO:agentic-rag:[router] decision: {'choice': 'rag', 'reason': 'heuristic', ...}
INFO:agentic-rag:[retrieval] query_len=42, results={'count': 6, 'parents_found': 2}
INFO:agentic-rag:[llm] model=gemini-2.0-flash, prompt_len=1250, response_len=450, latency_ms=3250.5
INFO:agentic-rag:[synthesis] synthesized -- {'query': 'what is...', 'rag_chunks': 6, 'web_results': 0}
```

---

## Pinecone Vector Store

### Architecture

**In-Memory Fallback + Pinecone Backend:**

```
Query
  ↓
[Vectorize with SentenceTransformer]
  ↓
[Try Pinecone Search] ← (primary)
  ↓
[If Pinecone fails → In-Memory Fallback]
  ↓
Results
```

### Key Features

- **Persistent Storage**: Embeddings persist across app restarts
- **Scalability**: Supports millions of embeddings
- **Automatic Fallback**: If Pinecone is unavailable, switches to in-memory
- **Batch Operations**: Efficient upsert and search

### Configuration

- **Dimension**: 1024 (matches your Pinecone index)
- **Metric**: Cosine similarity
- **Index Name**: `rag`
- **Serverless Mode**: ✓ (your setup)

### Manage Embeddings

Embeddings are stored with metadata:

```python
{
  "doc_id": "f31e48c0-25c6-47bf-a347-43b577bc3425",
  "offset": 0,
  "chunk_num": 0,
  "text": "First 500 chars of chunk..."
}
```

---

## Monitoring Agent Flow

### 1. Ingest a Document

When you upload a PDF:

- `RAGAgent` creates chunks using sliding-window strategy
- Each chunk is embedded (1024 dims) and indexed in Pinecone
- Parent documents are created for late-chunking retrieval
- **Trace**: "rag_ingest_complete" with chunk count and pages

### 2. Query Processing

When you submit a query:

- `RouterAgent` decides: RAG, Web, or Hybrid
- **Trace**: "router_decide" with reason (e.g., "rag_keywords")
- `RAGAgent.retrieve()` searches Pinecone
- **Trace**: "retrieval" with query length and chunks returned
- `SynthesisAgent` calls LLM
- **Trace**: "llm" with model, latency, and token counts

### 3. View in LangSmith

Visit your project dashboard to:
- ✓ Replay full conversation traces
- ✓ Debug agent decisions step-by-step
- ✓ Monitor LLM costs and latencies
- ✓ Identify retrieval gaps

Example trace structure:

```
Trace Root: query="what is climate change?"
  ├─ Router Decision: choice=rag, reason=heuristic
  ├─ RAG Retrieve: query_len=22, parents_found=2, chunks_returned=6
  ├─ LLM Call: model=gemini-2.0-flash, latency_ms=3150
  └─ Synthesis: success=true, sources={rag:6, web:0}
```

---

## Troubleshooting

### Pinecone Connection Fails

**Symptom**: `[vectorstore] Pinecone connection failed`

**Solution**:
1. Check `PINECONE_API_KEY` is correct (no extra spaces)
2. Verify `PINECONE_HOST` matches your Pinecone index host
3. Ensure your Pinecone project is active and not rate-limited

**Fallback**: System automatically switches to in-memory storage

### LangSmith Not Recording Traces

**Symptom**: No traces appear on LangSmith dashboard

**Solution**:
1. Verify `LANGSMITH_API_KEY` is set correctly
2. Check `ENABLE_LANGSMITH=true` in `.env`
3. Restart Streamlit after changing `.env`
4. Check logs: `INFO:agentic-rag:LangSmith tracer initialized: project=agentic-rag`

### Embedding Dimension Mismatch

**Symptom**: `Error: embedding dimension 384 does not match index dimension 1024`

**Solution**:
1. Your `.env` now has `EMBED_DIM=1024` (matches Pinecone)
2. If you change embedding models, update `EMBED_DIM` accordingly:
   - `all-MiniLM-L6-v2` → 384
   - `all-mpnet-base-v2` → 768
   - `all-roberta-large-v1` → 1024

---

## Advanced Configuration

### Switch Between Vector Stores

To use **in-memory only** (for testing):

```dotenv
USE_PINECONE=false
```

To use **Pinecone only** (disable fallback):

Edit `core/retriever/pinecone_vectorstore.py`:
```python
self.use_pinecone = True  # Force Pinecone, no fallback
```

### Custom LangSmith Project

To use a different LangSmith project:

```dotenv
LANGSMITH_PROJECT=my-custom-project
```

Then restart Streamlit and traces will appear in:
```
https://smith.langchain.com/o/agentic-rag/projects/my-custom-project
```

### Disable Tracing for Production

```dotenv
ENABLE_LANGSMITH=false
```

This reduces overhead but you lose observability.

---

## Performance Tips

1. **Batch Queries**: Submit multiple queries in succession to amortize connection costs
2. **Adjust TOP_K_CHUNKS**: Reduce from 6 to 3 for faster retrieval
3. **Monitor Latency**: Use LangSmith dashboard to identify bottlenecks
4. **Cache Warmup**: First query is slower; subsequent queries use cache

---

## Cleanup & Reset

### Clear All Embeddings from Pinecone

Add this to your app or run standalone:

```python
from core.retriever.pinecone_vectorstore import PineconeVectorStore
vs = PineconeVectorStore(
    api_key="your-key",
    host="your-host",
    index_name="rag",
    dimension=1024
)
vs.clear()
print("All embeddings cleared")
```

### Export Traces from LangSmith

Visit https://smith.langchain.com → Export runs as CSV/JSON

---

## Next Steps

1. **Upload a test document** – PDF will be chunked and stored in Pinecone
2. **Submit queries** – Watch traces appear in real-time on LangSmith
3. **Monitor performance** – Use LangSmith dashboard to optimize
4. **Scale up** – Pinecone supports millions of vectors; ready for production

---

## Support

For issues:
- **Pinecone**: https://docs.pinecone.io/
- **LangSmith**: https://docs.smith.langchain.com/
- **Project**: Check logs with `LOG_LEVEL=DEBUG`

