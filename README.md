# Multi-Turn Agentic RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system powered by intelligent agents and language models. This system orchestrates multiple specialized agents to provide intelligent document analysis, web search, and content synthesis.

## 🚀 Features

- **Multi-Agent Architecture**: Router, RAG, Web Search, and Synthesis agents working together
- **Smart Routing**: Intelligent decision-making about whether to use document search, web search, or both
- **Advanced Retrieval**: 
  - Late-chunking with overlapping windows
  - Parent-document retrieval
  - Cosine similarity ranking
  - **Pinecone Vector Store** for persistent, scalable embeddings (with in-memory fallback)
- **Multi-Model Support**: 
  - OpenAI (primary)
  - Google Gemini (fallback)
- **Configurable**: Centralized configuration with sensible defaults
- **Production-Ready**: 
  - Comprehensive error handling
  - Logging and tracing
  - Request caching
  - Type hints throughout
- **Observability**: 
  - **LangSmith Tracing** for full agent decision visibility
  - Structured logging with event tracing
- **Streamlit UI**: Interactive web interface for queries

## 🆕 What's New

- ✨ **Pinecone Integration**: Persistent vector storage with automatic fallback to in-memory
- ✨ **LangSmith Tracing**: Complete observability of agent decisions, retrievals, and LLM calls
- ✨ **Batch Operations**: Efficient embedding ingestion for PDFs
- ✨ **Enhanced Config**: Centralized configuration for all components

## 📋 Prerequisites

- Python 3.8+
- pip or conda
- OpenAI API key OR Google Gemini API key (or both for redundancy)
- (Optional) SerpAPI key for web search
- (Optional) Pinecone API key for persistent vector storage
- (Optional) LangSmith API key for agent tracing and observability

## 🔧 Installation

### 1. Clone or download the project

```bash
cd agentic-rag-system
```

### 2. Create a virtual environment (recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n rag-system python=3.10
conda activate rag-system
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file in the project root:

```env
# Required: At least one LLM API key
OPENAI_API_KEY=your-openai-api-key
GEMINI_API_KEY=your-gemini-api-key

# Optional: Web search
SERP_API_KEY=your-serpapi-api-key

# Optional: Model configuration
OPENAI_MODEL=gpt-4o-mini
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Optional: Retrieval tuning
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
TOP_K_PARENTS=3
TOP_K_CHUNKS=6

# Optional: Performance
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=1024
ENABLE_CACHE=true
CACHE_TTL_SECONDS=3600
```

## 🎯 Quick Start

### Using Streamlit UI (Recommended)

```bash
streamlit run streamlit_app.py
```

Then:
1. Upload a PDF document (optional but recommended)
2. Enter your query in the text area
3. Click "Run" to process

The system will:
- Route your query intelligently
- Retrieve relevant information
- Synthesize a comprehensive response

### Using Python directly

```python
from core.memory.memory_service import MemoryService
from core.retriever.vectorstore_adapter import InMemoryVectorStore
from core.agents.rag_agent import RAGAgent
from core.agents.web_search_agent import WebSearchAgent
from core.agents.synthesis_agent import SynthesisAgent
from core.agents.router_agent import RouterAgent

# Initialize components
memory = MemoryService()
vector_store = InMemoryVectorStore()
rag = RAGAgent(vector_store, memory)
web = WebSearchAgent()
synth = SynthesisAgent()
router = RouterAgent(rag, web, synth)

# Ingest a document
doc_id = rag.ingest_pdf("your_document.pdf")

# Process a query
payload = {
    "has_doc": True,
    "query_text": "Summarize the main points"
}
result = router.route(payload)
print(result["result"]["summary"])
```

## 📁 Project Structure

```
agentic-rag-system/
├── core/
│   ├── agents/                 # Intelligent agents
│   │   ├── router_agent.py     # Routing decisions
│   │   ├── rag_agent.py        # Document retrieval
│   │   ├── web_search_agent.py # Web search
│   │   ├── synthesis_agent.py  # Response synthesis
│   │   └── ...
│   ├── retriever/              # Retrieval strategies
│   │   ├── late_chunker.py     # Advanced chunking
│   │   └── vectorstore_adapter.py
│   ├── utils/                  # Utilities
│   │   ├── config.py           # Centralized configuration
│   │   └── logging_utils.py    # Logging and tracing
│   ├── memory/                 # Memory management
│   └── tools/                  # Document loading tools
├── llm/                        # LLM client abstractions
│   ├── gemini_client.py        # Google Gemini integration
│   └── prompts.py
├── ui/                         # UI components
│   └── app.py
├── streamlit_app.py            # Main Streamlit app
├── main.py                     # CLI entry point
├── requirements.txt
└── README.md
```

## 🤖 How It Works

### 1. Query Routing

The Router Agent analyzes your query and decides:
- **RAG Mode**: For document-specific questions
- **Web Mode**: For current/trending information
- **Hybrid Mode**: For comparative or comprehensive queries

### 2. Retrieval

- **For RAG**: Chunks documents using sliding windows with overlap
- **For Web**: Searches using SerpAPI (or stub if not configured)

### 3. Synthesis

- Merges results from chosen retrieval methods
- Uses LLM to create coherent, comprehensive responses
- Provides action items and recommendations

### 4. Caching

- Recent queries are cached for faster re-retrieval
- Configurable TTL (time-to-live)

## ⚙️ Configuration

### Core Settings (config.py)

```python
from core.utils.config import Config

# Access configuration
print(Config.CHUNK_SIZE)        # 1000
print(Config.TOP_K_CHUNKS)      # 6
print(Config.LLM_TEMPERATURE)   # 0.0

# Validate configuration
is_valid, warnings = Config.validate()

# Get summary
summary = Config.get_summary()
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| OPENAI_API_KEY | None | OpenAI API key |
| GEMINI_API_KEY | None | Google Gemini API key |
| SERP_API_KEY | None | SerpAPI key for web search |
| CHUNK_SIZE | 1000 | Document chunk size in characters |
| CHUNK_OVERLAP | 100 | Overlap between chunks |
| TOP_K_PARENTS | 3 | Number of parent docs to retrieve |
| TOP_K_CHUNKS | 6 | Number of chunks to synthesize |
| LLM_TEMPERATURE | 0.0 | LLM sampling temperature |
| LLM_MAX_TOKENS | 1024 | Max tokens in LLM response |
| ENABLE_CACHE | true | Enable response caching |
| CACHE_TTL_SECONDS | 3600 | Cache validity duration |

## 🧪 Testing & Validation

### API Key Verification

Use the Streamlit UI's "API Key Status" debug section to verify keys are working.

### Unit Tests

```bash
pytest tests/ -v
```

## 📊 Performance

- **Embedding**: ~100ms per chunk (depends on model)
- **Retrieval**: ~50ms for similarity search
- **Synthesis**: ~2-10s depending on LLM response time
- **Cache Hit**: <10ms

## 🐛 Troubleshooting

### No API Keys Configured
**Error**: "Neither OPENAI_API_KEY nor GEMINI_API_KEY is set"
**Solution**: Create `.env` file with at least one API key

### PDF Extraction Fails
**Error**: "Failed to read PDF"
**Solution**: Ensure PDF is readable and not password-protected

### Empty Document Content
**Error**: "PDF contains no extractable text"
**Solution**: Try a different PDF; some PDFs may have text as images

### Web Search Not Working
**Error**: "Web search returning stub responses"
**Solution**: Set SERP_API_KEY in .env file

### Slow Responses
**Issue**: Synthesis takes long time
**Solution**: 
- Increase LLM_TEMPERATURE for faster, less accurate responses
- Reduce LLM_MAX_TOKENS
- Enable ENABLE_CACHE for repeated queries

## 🎓 Advanced Usage

### Custom Chunking Strategy

```python
# Modify CHUNK_SIZE in .env or Config
os.environ["CHUNK_SIZE"] = "2000"  # Larger chunks
os.environ["CHUNK_OVERLAP"] = "200"  # More overlap
```

### Multiple Documents

```python
for pdf_file in ["doc1.pdf", "doc2.pdf", "doc3.pdf"]:
    doc_id = rag.ingest_pdf(pdf_file)
    print(f"Ingested: {doc_id}")
```

### Custom Routing Logic

```python
class CustomRouter(RouterAgent):
    def decide(self, payload):
        # Your custom logic
        if "specific_keyword" in payload["query_text"]:
            return "rag"
        return super().decide(payload)
```

### Enable Detailed Logging

```bash
export LOG_LEVEL=DEBUG
streamlit run streamlit_app.py
```

## 📝 Examples

### Example 1: Document Analysis

```
Query: "What are the main conclusions of this document?"
Route: RAG
Result: Summarizes key findings from your document
```

### Example 2: Current Events

```
Query: "What's the latest news on AI?"
Route: Web
Result: Latest information from web search
```

### Example 3: Comparative Analysis

```
Query: "Compare my document to current market trends"
Route: Hybrid
Result: Combines document insights with web information
```

## 🤝 Contributing

To improve the system:

1. Fork or clone the repository
2. Create a feature branch
3. Make your improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

[Add your license here]

## 📚 Advanced Configuration

### Pinecone Vector Store

For persistent, scalable embeddings storage:

```bash
# Set in .env
USE_PINECONE=true
PINECONE_API_KEY=your-key-here
PINECONE_HOST=your-host-url
PINECONE_INDEX_NAME=rag
EMBED_DIM=1024
```

See [Pinecone + LangSmith Guide](./PINECONE_LANGSMITH_GUIDE.md) for details.

### LangSmith Tracing

For full observability of agent decisions and LLM calls:

```bash
# Set in .env
ENABLE_LANGSMITH=true
LANGSMITH_API_KEY=your-key-here
LANGSMITH_PROJECT=agentic-rag
```

Then view traces at: https://smith.langchain.com/o/agentic-rag/projects/agentic-rag

See [Pinecone + LangSmith Guide](./PINECONE_LANGSMITH_GUIDE.md) for complete setup.

## 🆘 Support

For issues or questions:
1. Check the Troubleshooting section
2. Enable DEBUG logging for detailed information
3. Check API key validity
4. Review error messages in the Streamlit UI

## 🔄 Updates & Improvements

This system is continuously improved with:
- Better error handling and user feedback
- Enhanced retrieval algorithms
- Performance optimizations
- Additional LLM provider support
- Improved UI/UX
- **NEW**: Pinecone for scalable vector storage
- **NEW**: LangSmith for complete agent observability

---

**Made with ❤️ for intelligent document analysis and synthesis**
