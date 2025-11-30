"""
RAG Agent: ingest documents, create chunks, index embeddings, retrieve with late-chunking & parent-doc retrieval.
This is a simplified orchestrator using sentence-transformers for embeddings.
Improved with better error handling, configurable chunking, metadata tracking, Pinecone support, and LangSmith tracing.
Smart intent detection: summarize/key points queries get all chunks, topic-specific queries use similarity search.
"""
from typing import List, Dict, Any, Optional
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from core.retriever.vectorstore_adapter import InMemoryVectorStore
from core.retriever.pinecone_vectorstore import PineconeVectorStore
from core.retriever.late_chunker import assemble_late_chunks, get_top_parent_docs
from core.utils.logging_utils import trace_event
from core.utils.config import Config
from core.utils.langsmith_tracer import get_tracer
import os
import uuid
import logging

logger = logging.getLogger("agentic-rag")

EMBED_MODEL_NAME = Config.EMBED_MODEL_NAME
CHUNK_SIZE = Config.CHUNK_SIZE
CHUNK_OVERLAP = Config.CHUNK_OVERLAP

try:
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    logger.info(f"Loaded embedding model: {EMBED_MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to load embedding model {EMBED_MODEL_NAME}: {e}")
    embedder = None

class RAGAgent:
    def __init__(self, vectorstore=None, memory_service=None):
        """
        Initialize RAG agent with optional vectorstore.
        If vectorstore is None, will initialize Pinecone or in-memory store based on config.
        """
        self.memory = memory_service
        
        # Initialize tracer
        self.tracer = get_tracer(
            enabled=Config.ENABLE_LANGSMITH,
            api_key=Config.LANGSMITH_API_KEY,
            project=Config.LANGSMITH_PROJECT
        )
        
        # Initialize vectorstore
        if vectorstore is not None:
            self.vs = vectorstore
        elif Config.USE_PINECONE and Config.PINECONE_API_KEY and Config.PINECONE_HOST:
            try:
                self.vs = PineconeVectorStore(
                    api_key=Config.PINECONE_API_KEY,
                    host=Config.PINECONE_HOST,
                    index_name=Config.PINECONE_INDEX_NAME,
                    dimension=Config.EMBED_DIM,
                    use_pinecone=True,
                )
                logger.info(f"RAGAgent initialized with Pinecone vectorstore (dim={Config.EMBED_DIM})")
            except Exception as e:
                logger.warning(f"Failed to initialize Pinecone, falling back to in-memory: {e}")
                self.vs = InMemoryVectorStore()
        else:
            self.vs = InMemoryVectorStore()
            logger.info("RAGAgent initialized with in-memory vectorstore")
        
        # chunk_index: doc_id -> list of chunks
        self.chunk_index: Dict[str, List[Dict[str, Any]]] = {}
        self.parent_vectors: List[tuple] = []  # list of (doc_id, parent_embedding)

    def ingest_pdf(self, pdf_path: str, doc_id: str = None):
        """
        Ingest a PDF document and create chunks with embeddings.
        Uses sliding window chunking for better context preservation.
        Returns doc_id on success, raises exception on failure.
        Traces ingestion with LangSmith.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not embedder:
            raise RuntimeError("Embedding model not initialized. Check EMBED_MODEL_NAME configuration.")
        
        doc_id = doc_id or str(uuid.uuid4())
        text = []
        
        # Trace ingestion start
        self.tracer.trace_agent_decision("rag_ingest", {"doc_id": doc_id, "path": pdf_path})
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        txt = page.extract_text() or ""
                        text.append(txt)
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num}: {e}")
                        continue
        except Exception as e:
            raise Exception(f"Failed to read PDF: {str(e)}")
        
        full_text = "\n".join(text)
        if not full_text.strip():
            raise ValueError("PDF contains no extractable text")
        
        # Create chunks with overlap for better context
        chunks = []
        embeddings_for_vs = []
        ids_for_vs = []
        metadata_for_vs = []
        
        try:
            chunk_positions = []
            for i in range(0, len(full_text), CHUNK_SIZE - CHUNK_OVERLAP):
                ctext = full_text[i:i+CHUNK_SIZE]
                if not ctext.strip():
                    continue
                
                chunk_positions.append(i)
                
                try:
                    emb = embedder.encode(ctext)
                    chunk_id = f"{doc_id}_c_{len(chunks)}"
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": ctext,
                        "embedding": emb,
                        "offset": i,
                        "length": len(ctext)
                    })
                    
                    # Prepare for batch upsert to vectorstore
                    embeddings_for_vs.append(emb)
                    ids_for_vs.append(chunk_id)
                    metadata_for_vs.append({
                        "doc_id": doc_id,
                        "offset": i,
                        "chunk_num": len(chunks) - 1,
                        "text": ctext[:500],  # Store first 500 chars for preview
                    })
                except Exception as e:
                    logger.error(f"Failed to encode chunk at offset {i}: {e}")
                    continue
            
            if not chunks:
                raise ValueError("No chunks created from PDF")
            
            # Batch add to vectorstore
            if embeddings_for_vs:
                try:
                    logger.info(f"Adding {len(embeddings_for_vs)} embeddings to vectorstore...")
                    # For Pinecone, add with doc_id tracking
                    if hasattr(self.vs, 'add'):
                        # PineconeVectorStore.add() with batch insert
                        self.vs.add(embeddings_for_vs, ids_for_vs, metadata_for_vs, doc_id=doc_id)
                        logger.info(f"✓ Successfully added {len(embeddings_for_vs)} chunks to Pinecone")
                    else:
                        # Fallback for InMemoryVectorStore
                        for emb, cid, meta in zip(embeddings_for_vs, ids_for_vs, metadata_for_vs):
                            self.vs.add(cid, emb, meta)
                        logger.info(f"✓ Successfully added {len(embeddings_for_vs)} chunks to in-memory store")
                except Exception as e:
                    logger.error(f"Failed to add chunks to vectorstore: {e}")
                    # Continue anyway
            
            # Parent doc embedding (from first ~2000 chars for efficiency)
            try:
                parent_text = full_text[:min(2000, len(full_text))]
                parent_emb = embedder.encode(parent_text)
                self.parent_vectors.append((doc_id, parent_emb))
            except Exception as e:
                logger.error(f"Failed to create parent embedding: {e}")
                # Continue anyway - chunks are still indexed
            
            self.chunk_index[doc_id] = chunks
            
            # Trace ingestion success
            self.tracer.trace_agent_decision("rag_ingest_complete", {
                "doc_id": doc_id,
                "chunks": len(chunks),
                "total_chars": len(full_text),
                "pages": len(text)
            })
            
            trace_event(doc_id, "ingested", {
                "chunks": len(chunks),
                "total_chars": len(full_text),
                "pages": len(text)
            })
            logger.info(f"Successfully ingested PDF {doc_id} with {len(chunks)} chunks")
            return doc_id
        except Exception as e:
            trace_event(doc_id, "ingest_error", {"error": str(e)})
            self.tracer.trace_agent_decision("rag_ingest_error", {"doc_id": doc_id, "error": str(e)})
            logger.error(f"Error ingesting PDF: {e}")
            raise

    def retrieve(self, query: str, top_k_parents: Optional[int] = None, top_k_chunks: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for the given query using late-chunking strategy.
        Smart detection of intent-based queries (summarize, key points, extract, etc.)
        vs. topic-specific queries (what is X, explain Y, etc.)
        
        For intent-based queries: Return all available chunks for LLM to process
        For topic-specific queries: Use similarity search to find relevant chunks
        
        Returns list of chunk dicts with 'chunk_id', 'text', and 'metadata' keys.
        Traces retrieval with LangSmith.
        """
        if not query or not query.strip():
            trace_event("retrieve", "empty_query", {"q": query})
            self.tracer.trace_agent_decision("rag_retrieve", {"status": "empty_query"})
            logger.warning("Empty query provided to retrieve")
            return []
        
        if not embedder:
            trace_event("retrieve", "embedder_error", {"q": query})
            self.tracer.trace_agent_decision("rag_retrieve", {"status": "embedder_error"})
            logger.error("Embedding model not available")
            return []
        
        if not self.parent_vectors:
            trace_event("retrieve", "no_documents", {"q": query})
            self.tracer.trace_agent_decision("rag_retrieve", {"status": "no_documents"})
            logger.info("No documents ingested yet")
            return []
        
        try:
            top_k_parents = top_k_parents or Config.TOP_K_PARENTS
            top_k_chunks = top_k_chunks or Config.TOP_K_CHUNKS
            
            # Intent-based query keywords that don't need similarity search
            INTENT_KEYWORDS = [
                "summarize", "summary", "overview",
                "key points", "important", "highlight",
                "extract", "main idea", "main points",
                "what.*contain", "what.*include",
                "explain", "describe", "tell me about",
                "list", "enumerate", "outline",
                "section", "chapter", "content",
                "overall", "general", "in total",
                "all", "entire", "whole"
            ]
            
            query_lower = query.lower().strip("?!.,:;")
            
            # Check if this is an intent-based query
            is_intent_query = any(kw in query_lower for kw in INTENT_KEYWORDS)
            
            if is_intent_query:
                # Intent-based: return all chunks for LLM to synthesize
                logger.info(f"Intent-based query detected: '{query[:60]}' - returning all chunks")
                trace_event("retrieve", "intent_query", {"query": query[:50], "type": "intent"})
                
                q_emb = embedder.encode(query)
                parent_ids = get_top_parent_docs(q_emb, self.parent_vectors, top_k=top_k_parents)
                
                if not parent_ids:
                    logger.warning(f"No parent documents found for query: {query[:100]}")
                    return []
                
                # Get all chunks (no similarity filtering)
                chunks = assemble_late_chunks(parent_ids, self.chunk_index, max_chunks=top_k_chunks)
                
                if chunks:
                    # Sort by order but don't filter by similarity
                    self.tracer.trace_retrieval(query, {
                        "count": len(chunks),
                        "parents_found": len(parent_ids),
                        "query_type": "intent",
                        "query_len": len(query)
                    })
                    trace_event("retrieve", "success", {
                        "query_len": len(query),
                        "parents_found": len(parent_ids),
                        "chunks_returned": len(chunks),
                        "query_type": "intent"
                    })
                    logger.info(f"Retrieved {len(chunks)} chunks for intent query: {query[:80]}")
                    return chunks
                else:
                    return []
            
            # Topic-specific query: use similarity search
            logger.info(f"Topic-specific query detected: '{query[:60]}' - using similarity search")
            trace_event("retrieve", "topic_query", {"query": query[:50], "type": "topic"})
            
            # Relevance threshold for similarity-based retrieval
            RELEVANCE_THRESHOLD = 0.15  # For all-MiniLM-L6-v2 model
            
            q_emb = embedder.encode(query)
            parent_ids = get_top_parent_docs(q_emb, self.parent_vectors, top_k=top_k_parents)
            
            if not parent_ids:
                trace_event("retrieve", "no_parents_found", {"q": query})
                self.tracer.trace_agent_decision("rag_retrieve", {"status": "no_parents", "query_len": len(query)})
                logger.warning(f"No parent documents found for query: {query[:100]}")
                return []
            
            chunks = assemble_late_chunks(parent_ids, self.chunk_index, max_chunks=top_k_chunks)
            
            # Rerank chunks by similarity to query and filter by relevance threshold
            if chunks:
                try:
                    scored_chunks = []
                    for chunk in chunks:
                        chunk_emb = np.array(chunk.get("embedding", []))
                        if len(chunk_emb) > 0:
                            from core.retriever.late_chunker import cosine_similarity
                            score = cosine_similarity(q_emb, chunk_emb)
                            scored_chunks.append((chunk, score))
                    
                    # Sort by score
                    scored_chunks.sort(key=lambda x: x[1], reverse=True)
                    
                    # Filter by relevance threshold - only keep relevant chunks
                    filtered_chunks = [c for c, score in scored_chunks if score >= RELEVANCE_THRESHOLD]
                    
                    if not filtered_chunks:
                        # No chunks above threshold - topic not in documents
                        logger.info(f"No chunks above relevance threshold (0.15) for topic query: {query[:80]}")
                        trace_event("retrieve", "below_threshold", {
                            "query_len": len(query),
                            "max_score": max([s for _, s in scored_chunks]) if scored_chunks else 0,
                            "threshold": RELEVANCE_THRESHOLD,
                            "query_type": "topic"
                        })
                        return []
                    
                    chunks = filtered_chunks[:top_k_chunks]
                except Exception as e:
                    logger.debug(f"Could not rerank chunks: {e}")
                    # Continue with all chunks if reranking fails
            
            # Trace retrieval
            self.tracer.trace_retrieval(query, {
                "count": len(chunks),
                "parents_found": len(parent_ids),
                "query_type": "topic",
                "query_len": len(query)
            })
            
            trace_event("retrieve", "success", {
                "query_len": len(query),
                "parents_found": len(parent_ids),
                "chunks_returned": len(chunks),
                "query_type": "topic"
            })
            logger.info(f"Retrieved {len(chunks)} relevant chunks for topic query: {query[:80]}")
            return chunks
        except Exception as e:
            trace_event("retrieve", "error", {"q": query[:100], "error": str(e)})
            self.tracer.trace_agent_decision("rag_retrieve", {"status": "error", "error": str(e)})
            logger.error(f"Error during retrieval: {e}", exc_info=True)
            return []
