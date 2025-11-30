"""
Multi-turn Conversational RAG System UI with LangSmith tracing and Pinecone integration.
Supports document upload, intelligent routing (RAG/Web/Hybrid), and full conversation history.
"""
import streamlit as st
import time
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from core.agents.rag_agent import RAGAgent
from core.agents.router_agent import RouterAgent
from core.agents.synthesis_agent import SynthesisAgent
from core.agents.web_search_agent import WebSearchAgent
from core.memory.memory_service import MemoryService
from core.utils.config import Config
from core.utils.logging_utils import trace_event
import logging

logger = logging.getLogger("agentic-rag")

# ============================================================================
# STREAMLIT PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Multi-Turn Agentic RAG System")
st.markdown("*Powered by intelligent routing, LangSmith tracing, and Pinecone vector storage*")

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "memory_service" not in st.session_state:
    st.session_state.memory_service = MemoryService()
    st.session_state.session_id = st.session_state.memory_service.create_session()
    logger.info(f"Created new session: {st.session_state.session_id}")

if "rag_agent" not in st.session_state:
    st.session_state.rag_agent = RAGAgent()

if "synthesis_agent" not in st.session_state:
    st.session_state.synthesis_agent = SynthesisAgent()

if "web_agent" not in st.session_state:
    st.session_state.web_agent = WebSearchAgent()

if "router_agent" not in st.session_state:
    # Initialize router with all required agents
    st.session_state.router_agent = RouterAgent(
        rag_agent=st.session_state.rag_agent,
        web_agent=st.session_state.web_agent,
        synth_agent=st.session_state.synthesis_agent
    )

if "documents_ingested" not in st.session_state:
    st.session_state.documents_ingested = False

if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()  # Track which files have been processed

# ============================================================================
# SIDEBAR: DOCUMENT UPLOAD & SESSION MANAGEMENT
# ============================================================================

with st.sidebar:
    st.header("📚 Document Management")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type="pdf",
        key=f"pdf_uploader_{st.session_state.session_id}"
    )
    
    if uploaded_file is not None:
        # Only process if this file hasn't been processed yet
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        
        if file_key not in st.session_state.processed_files:
            with st.spinner("📥 Processing document..."):
                try:
                    # Save uploaded file temporarily (use proper temp directory for Windows/Linux)
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        temp_path = tmp.name
                    
                    # Ingest with RAGAgent (which uses Pinecone or fallback)
                    st.session_state.rag_agent.ingest_pdf(temp_path, doc_id=uploaded_file.name)
                    st.session_state.memory_service.register_ingested_doc(
                        st.session_state.session_id,
                        uploaded_file.name
                    )
                    
                    st.session_state.documents_ingested = True
                    st.session_state.ingested_docs.append(uploaded_file.name)
                    st.session_state.processed_files.add(file_key)  # Mark as processed
                    
                    trace_event("ui", "document_ingested", {
                        "doc_name": uploaded_file.name,
                        "session_id": st.session_state.session_id
                    })
                    
                    st.success(f"✅ Document ingested: {uploaded_file.name}")
                    logger.info(f"Successfully ingested: {uploaded_file.name}")
                    
                except Exception as e:
                    st.error(f"❌ Failed to ingest document: {str(e)}")
                    logger.error(f"Document ingestion failed: {str(e)}", exc_info=True)
                    trace_event("ui", "document_ingest_error", {"error": str(e)})
        else:
            st.info(f"✓ Already processed: {uploaded_file.name}")
    
    # Display ingested documents
    if st.session_state.ingested_docs:
        st.subheader("📖 Ingested Documents")
        for doc in st.session_state.ingested_docs:
            st.caption(f"✓ {doc}")
    else:
        st.caption("No documents ingested yet")
    
    # Session info
    st.divider()
    st.subheader("📊 Session Info")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Session ID", st.session_state.session_id[:8] + "...", delta=None)
    
    with col2:
        conv_history = st.session_state.memory_service.get_conversation_history(
            st.session_state.session_id
        )
        st.metric("Messages", len(conv_history), delta=None)
    
    # Configuration display
    st.divider()
    st.subheader("⚙️ Configuration")
    
    config_info = {
        "Embedding Model": Config.EMBED_MODEL_NAME.split("/")[-1],
        "Embedding Dim": str(Config.EMBED_DIM),
        "Vector Store": "Pinecone" if Config.USE_PINECONE else "In-Memory",
        "LLM": Config.OPENAI_MODEL if Config.OPENAI_API_KEY else Config.GEMINI_MODEL,
        "LangSmith": "✓" if Config.ENABLE_LANGSMITH else "✗",
    }
    
    for key, val in config_info.items():
        st.caption(f"**{key}:** {val}")
    
    # Clear session button
    st.divider()
    if st.button("🔄 Clear Session", key="clear_session_btn", use_container_width=True):
        # Clear memory service
        st.session_state.memory_service = MemoryService()
        st.session_state.session_id = st.session_state.memory_service.create_session()
        
        # Clear RAG agent vectorstore
        if hasattr(st.session_state.rag_agent, 'vs'):
            if hasattr(st.session_state.rag_agent.vs, 'clear'):
                st.session_state.rag_agent.vs.clear()
        
        # Reset document tracking
        st.session_state.documents_ingested = False
        st.session_state.ingested_docs = []
        
        trace_event("ui", "session_cleared", {"old_session_id": st.session_state.session_id})
        st.success("✅ Session cleared & documents removed")
        st.rerun()

# ============================================================================
# MAIN: CONVERSATION INTERFACE - IMPROVED CHAT UI
# ============================================================================

# Custom CSS for better chat styling
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        color: #000;
        font-size: 0.95rem;
        line-height: 1.4;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
        border-left: 4px solid #2196F3;
        color: #0d47a1;
    }
    .bot-message {
        background-color: #f5f5f5;
        margin-right: 2rem;
        border-left: 4px solid #4CAF50;
        color: #1b5e20;
    }
    .message-badge {
        font-size: 0.8rem;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        display: inline-block;
        width: fit-content;
        margin-bottom: 0.5rem;
        background-color: rgba(255, 255, 255, 0.5);
        color: #000;
        font-weight: bold;
    }
    .metadata-small {
        font-size: 0.75rem;
        color: #555;
        margin-top: 0.5rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

st.subheader("💬 Chat")

# Get conversation history from memory
conv_history = st.session_state.memory_service.get_conversation_history(
    st.session_state.session_id
)

# Display conversation history with improved styling
if conv_history:
    for turn in conv_history:
        # User message
        with st.container():
            user_msg_text = turn.get('user_message', '').strip()
            st.markdown(f"""
            <div class="chat-message user-message">
                <div style="color: #0d47a1; font-weight: bold;">👤 You</div>
                <div style="color: #0d47a1; margin-top: 0.3rem;">{user_msg_text}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Bot response
        with st.container():
            route = turn.get('route', 'unknown')
            route_emoji = {
                "rag": "🗂️",
                "web": "🌐",
                "hybrid": "🔀",
                "error": "❌"
            }.get(route, "❓")
            
            route_label = {
                "rag": "RAG",
                "web": "Web",
                "hybrid": "Hybrid",
                "error": "Error"
            }.get(route, "Unknown")
            
            bot_response = turn.get('bot_response', '').strip()
            metadata = turn.get('metadata', {})
            
            badge_html = f'<span class="message-badge">{route_emoji} {route_label}</span>'
            
            metadata_html = ""
            if metadata:
                sources = metadata.get('source_counts', {})
                latency = metadata.get('latency_ms', 0)
                metadata_html = f'<div class="metadata-small">📊 {sources.get("rag", 0)} doc chunks • {sources.get("web", 0)} web • ⏱️ {latency:.0f}ms</div>'
            
            st.markdown(f"""
            <div class="chat-message bot-message">
                <div style="color: #1b5e20; font-weight: bold;">🤖 Assistant {badge_html}</div>
                <div style="color: #1b5e20; margin-top: 0.5rem;">{bot_response}</div>
                {metadata_html}
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("💭 No messages yet. Start a conversation!")

# ============================================================================
# INPUT SECTION: IMPROVED QUERY INPUT
# ============================================================================

st.divider()

# Prepare routing context
conv_context = st.session_state.memory_service.get_context_summary(
    st.session_state.session_id
)

# Create a nice input section
col1, col2 = st.columns([0.85, 0.15], gap="small")

with col1:
    user_query = st.text_input(
        "Message...",
        placeholder="Ask anything about your documents or general questions",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("Send", use_container_width=True, type="primary", key="send_button")

# ============================================================================
# QUERY PROCESSING
# ============================================================================

if send_button and user_query and user_query.strip():
    start_time = time.time()
    
    with st.spinner("⏳ Processing..."):
        try:
            trace_event("ui", "query_submitted", {
                "session_id": st.session_state.session_id,
                "query_length": len(user_query),
                "conversation_turn": len(conv_history) + 1
            })
            
            # Build payload for router
            payload = {
                "query": user_query.strip(),
                "has_documents": st.session_state.documents_ingested,
                "use_web_search": True
            }
            
            # Route and synthesize with conversation context
            result = st.session_state.router_agent.route(
                payload,
                conversation_context=conv_context
            )
            
            # Extract response
            bot_response = result.get("response", {}).get("summary", "❌ No response generated")
            metadata = result.get("response", {}).get("meta", {})
            route_chosen = result.get("route", "unknown")
            
            # Record in memory
            st.session_state.memory_service.add_conversation_turn(
                st.session_state.session_id,
                user_message=user_query,
                bot_response=bot_response,
                route=route_chosen,
                sources=metadata.get("source_counts", {"rag": 0, "web": 0}),
                metadata=metadata
            )
            
            latency = (time.time() - start_time) * 1000
            
            trace_event("ui", "query_processed", {
                "route": route_chosen,
                "latency_ms": latency,
                "sources": metadata.get("source_counts", {})
            })
            
            logger.info(f"Query processed: route={route_chosen}, latency={latency:.0f}ms")
            
            # Rerun to display new message
            st.rerun()
            
        except Exception as e:
            error_msg = str(e)
            trace_event("ui", "query_error", {
                "error": error_msg,
                "session_id": st.session_state.session_id
            })
            st.error(f"❌ Error: {error_msg}")
            logger.error(f"Query processing failed: {error_msg}", exc_info=True)

elif send_button and not user_query:
    st.warning("⚠️ Please type a message first")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<div style='text-align: center; font-size: 0.85rem; color: #888;'>
<p>🚀 Agentic RAG System v2.0 | Multi-turn Conversations | LangSmith Tracing | Pinecone Vector Store</p>
<p>Upload documents, ask questions, and get intelligent routing-based responses.</p>
</div>
""", unsafe_allow_html=True)
