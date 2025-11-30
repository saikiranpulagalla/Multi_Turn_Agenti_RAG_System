# 🚀 Agentic RAG System - Getting Started Guide

> **Get your intelligent document analysis system up and running in 5 minutes!**

---

## ⚡ 5-Minute Quick Start

### 1️⃣ Install Python Dependencies (2 min)

```bash
# Clone/navigate to project
cd agentic-rag-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Configure API Keys (2 min)

Create `.env` file in project root:

```bash
# At minimum, add ONE of these:
OPENAI_API_KEY=sk-your-key-here
# OR
GEMINI_API_KEY=your-key-here
```

Get free API keys:
- **OpenAI**: https://platform.openai.com/api-keys ($5 free credit)
- **Google Gemini**: https://makersuite.google.com/app/apikey (free tier)

### 3️⃣ Start the App (1 min)

```bash
streamlit run streamlit_app.py
```

App opens at: **http://localhost:8501**

---

## 🎯 First Steps

### Upload a Document
1. Click **"Upload PDF"** in the left sidebar
2. Select a PDF file
3. Wait for "✓ Successfully ingested" message

### Ask a Question
1. Type in the chat box: `"Summarize the document"`
2. Click **"Run Query"** or press Enter
3. Get your answer! 📝

### Try Different Questions
- "What are the key points?"
- "Explain this in simple terms"
- "What is [specific topic]?"

---

## 🔌 Optional: Enable Advanced Features

### Add Web Search (Optional)

Get free API key: https://serpapi.com

```env
SERP_API_KEY=your-key-here
```

Now queries can also search the web!

### Add Vector Database - Pinecone (Optional)

For persistent storage (survives app restart):

1. Create account: https://www.pinecone.io
2. Create an index (1024 dimensions)
3. Add to `.env`:

```env
USE_PINECONE=true
PINECONE_API_KEY=your-key
PINECONE_HOST=https://rag-xxxxx.svc.aped-4627-b74a.pinecone.io
```

### Add Observability - LangSmith (Optional)

See all agent decisions in real-time:

1. Sign up: https://smith.langchain.com
2. Add to `.env`:

```env
ENABLE_LANGSMITH=true
LANGSMITH_API_KEY=your-key
LANGSMITH_PROJECT=agentic-rag
```

3. View traces at: https://smith.langchain.com

---

## 🤔 Common Questions

### Q: What if I get "API key not found" error?
**A**: Make sure you created the `.env` file with your API key. Restart the app after editing.

### Q: Can I use both OpenAI and Gemini?
**A**: Yes! Add both keys. OpenAI is primary, Gemini is fallback if OpenAI fails.

### Q: How do I update the .env file?
**A**: Edit `.env` file and restart Streamlit. The app picks up changes immediately.

### Q: Can I upload multiple PDFs?
**A**: Yes! Upload them one by one. All get indexed and searchable together.

### Q: Where is my data stored?
**A**: 
- By default: In-memory (lost on app restart)
- With Pinecone: Persisted in Pinecone's cloud
- Local: In Python memory during session

### Q: How do I debug issues?
**A**: Set `LOG_LEVEL=DEBUG` in `.env` and check console output.

---

## 📊 System Architecture

```
Your Query
    ↓
Router Agent (Decides: Use docs or web search?)
    ├─→ RAG Path (Search uploaded documents)
    ├─→ Web Path (Search internet)
    └─→ Hybrid Path (Both)
    ↓
LLM (OpenAI or Gemini)
    ↓
Answer with Sources
```

---

## 🎓 Usage Examples

### Example 1: Document Q&A
```
Upload: research_paper.pdf
Query: "What are the main findings?"
Answer: [Extracts and summarizes from your PDF]
```

### Example 2: General Knowledge
```
Query: "What's new in AI?"
Answer: [Searches web for current info]
```

### Example 3: Hybrid
```
Upload: my_document.pdf
Query: "Compare my insights to current trends"
Answer: [Combines your doc + web search]
```

---

## ⚙️ Key Settings to Know

| Setting | What it Does | Default | Tip |
|---------|------------|---------|-----|
| `CHUNK_SIZE` | Document piece size | 1000 | Larger = more context, slower |
| `TOP_K_CHUNKS` | Results to return | 6 | More = slower but comprehensive |
| `RELEVANCE_THRESHOLD` | Minimum relevance score | 0.15 | Higher = stricter matching |
| `LLM_TEMPERATURE` | Response randomness | 0.0 | 0=exact, 1=creative |
| `ENABLE_CACHE` | Cache recent queries | true | Faster repeat queries |

---

## 🆘 Troubleshooting

### Problem: "Failed to read PDF"
- **Solution**: PDF might be corrupted or password-protected
- Try a different PDF file

### Problem: "No chunks above relevance threshold"
- **Solution**: Document doesn't match query well
- Try web search instead or rephrase query

### Problem: Slow responses (10+ seconds)
- **Solution**: 
  - Reduce `LLM_MAX_TOKENS` in `.env`
  - Use `gpt-3.5-turbo` instead of `gpt-4`
  - Reduce `TOP_K_CHUNKS` value

### Problem: Runs out of memory
- **Solution**:
  - Reduce `CHUNK_SIZE` (e.g., 500)
  - Use Pinecone for distributed storage
  - Restart app periodically

---

## 📝 Project Files Overview

```
agentic-rag-system/
├── streamlit_app.py          ← Main app (run this!)
├── requirements.txt          ← Dependencies
├── .env                      ← Your API keys (create this)
├── core/
│   ├── agents/               ← Smart agents (router, RAG, etc)
│   ├── retriever/            ← Retrieval logic
│   └── utils/                ← Config, logging
└── README_COMPREHENSIVE.md   ← Full documentation
```

---

## 🔄 Development Workflow

```bash
# 1. Activate environment
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows

# 2. Edit .env if needed
# (Add/change API keys)

# 3. Run app
streamlit run streamlit_app.py

# 4. Test in browser
# http://localhost:8501

# 5. Check logs in terminal for errors

# 6. Stop app (Ctrl+C)
```

---

## 🚀 Production Deployment

For deploying to production:

1. **Use Pinecone**: Enables persistent vector storage
2. **Set up LangSmith**: Monitor agent decisions
3. **Use environment variables**: Don't commit .env files
4. **Scale with Streamlit Cloud**: Free tier available at streamlit.io
5. **Add authentication**: Protect your app

See `PINECONE_LANGSMITH_GUIDE.md` for detailed setup.

---

## 💡 Pro Tips

✅ **Enable caching** for 10x faster repeated queries:
```env
ENABLE_CACHE=true
CACHE_TTL_SECONDS=3600
```

✅ **Use debug logging** to understand what's happening:
```env
LOG_LEVEL=DEBUG
```

✅ **Batch upload documents** for efficiency

✅ **Test with small chunks first**, then increase size

✅ **Monitor LangSmith** to see agent decision patterns

---

## 📚 Next Steps

1. ✅ **Explore**: Try different queries and documents
2. 📖 **Read**: Check `README_COMPREHENSIVE.md` for full docs
3. 🔌 **Integrate**: Add Pinecone/LangSmith for production
4. 🛠️ **Customize**: Adjust settings in `.env` for your use case
5. 🚀 **Deploy**: Deploy to Streamlit Cloud or your server

---

## 📞 Support

- 📖 Full docs: `README_COMPREHENSIVE.md`
- 🏗️ Architecture: `ARCHITECTURE.md`
- 🔌 Pinecone setup: `PINECONE_LANGSMITH_GUIDE.md`
- 💬 Routing test: `ROUTING_TEST_GUIDE.md`
- ⚡ Quick ref: `QUICK_REFERENCE.md`

---

## 🎉 You're Ready!

You now have an intelligent RAG system that can:
- ✅ Read and understand your documents
- ✅ Search the web for information
- ✅ Answer questions intelligently
- ✅ Combine both sources seamlessly

**Happy analyzing! 🚀**

---

**Last Updated**: November 30, 2025
**Quick Start Version**: 1.0
**Status**: Ready to Use ✅
