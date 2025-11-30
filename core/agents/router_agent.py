"""
Router Agent: decides whether to route to RAG, Web Search, or Hybrid.
Improved with better decision logic, error handling, and LangSmith tracing.
"""
from typing import Dict, Any, Optional
from core.utils.logging_utils import trace_event
from core.utils.config import Config
from core.utils.langsmith_tracer import get_tracer
import logging

logger = logging.getLogger("agentic-rag")

# Keywords that indicate different search strategies
RAG_KEYWORDS = ["document", "file", "uploaded", "summarize", "analyze", "extract", "in the document", "in your documents", "from the document", "in this document", "chapter", "page", "section", "definition"]
WEB_KEYWORDS = ["latest", "current", "news", "today", "recent", "now", "today's", "2024", "2025", "latest news", "current news", "breaking news", "trending", "this week", "this month"]
HYBRID_KEYWORDS = ["compare", "versus", "vs", "difference", "both", "and", "different"]

# Queries that are not information requests (just greetings, acknowledgments, etc)
NON_INFO_QUERIES = ["hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "sure", "yes", "no", "yep", "nope", "cool", "nice", "good", "bad", "ok got it", "got it", "understood", "i see", "lol", "haha", "hmm"]


class RouterAgent:
    def __init__(self, rag_agent=None, web_agent=None, synth_agent=None):
        self.rag = rag_agent
        self.web = web_agent
        self.synth = synth_agent
        
        # Initialize tracer
        self.tracer = get_tracer(
            enabled=Config.ENABLE_LANGSMITH,
            api_key=Config.LANGSMITH_API_KEY,
            project=Config.LANGSMITH_PROJECT
        )

    def decide(self, payload: Dict[str, Any], conversation_context: str = "") -> str:
        """
        Decide routing strategy based on query characteristics and conversation context.
        Traces decision with LangSmith.
        
        Args:
            payload: Dict with 'query'/'query_text' and 'has_documents'/'has_doc'
            conversation_context: Previous conversation turns for context-aware routing
            
        Returns:
            'rag', 'web', or 'hybrid'
        """
        # Handle both old and new payload formats
        has_doc = payload.get("has_documents") or payload.get("has_doc", False)
        query_text = (payload.get("query") or payload.get("query_text", "")).lower().strip()
        query_type = payload.get("query_type", "").lower()
        
        # Empty query should not proceed
        if not query_text:
            choice = "web"
            trace_event("router", "empty_query", {"choice": choice})
            self.tracer.trace_agent_decision("router_decide", {"choice": choice, "reason": "empty_query"})
            return choice
        
        # Check if this is a non-information query (greeting, acknowledgment, etc)
        # These don't need routing to RAG or web
        if query_text in NON_INFO_QUERIES or len(query_text) <= 3:
            choice = "web"  # Will eventually return "No response needed" via synthesis
            trace_event("router", "non_info_query", {"query": query_text[:20], "choice": choice})
            self.tracer.trace_agent_decision("router_decide", {"choice": choice, "reason": "non_info_query"})
            return choice
        
        # Check for explicit query type hints
        if query_type in ("factual", "document", "rag"):
            choice = "rag" if has_doc else "web"
            trace_event("router", "type_hint", {"choice": choice, "type": query_type})
            self.tracer.trace_agent_decision("router_decide", {"choice": choice, "reason": f"type_hint_{query_type}"})
            return choice
        
        if query_type in ("news", "web", "current"):
            choice = "web"
            trace_event("router", "type_hint", {"choice": choice, "type": query_type})
            self.tracer.trace_agent_decision("router_decide", {"choice": choice, "reason": f"type_hint_{query_type}"})
            return choice
        
        if query_type in ("comparative", "comparison", "hybrid"):
            choice = "hybrid"
            trace_event("router", "type_hint", {"choice": choice, "type": query_type})
            self.tracer.trace_agent_decision("router_decide", {"choice": choice, "reason": f"type_hint_{query_type}"})
            return choice
        
        # Keyword-based routing - check in order of priority
        
        # 1. Check for RAG keywords FIRST if documents available
        # (e.g., "summarize", "analyze", "from the document")
        if has_doc and any(kw in query_text for kw in RAG_KEYWORDS):
            choice = "rag"
            trace_event("router", "rag_keywords", {"keywords": [kw for kw in RAG_KEYWORDS if kw in query_text]})
            self.tracer.trace_agent_decision("router_decide", {"choice": choice, "reason": "rag_keywords"})
            return choice
        
        # 2. Check for WEB keywords (news, current, latest, etc)
        if any(kw in query_text for kw in WEB_KEYWORDS):
            choice = "web"
            trace_event("router", "web_keywords", {"keywords": [kw for kw in WEB_KEYWORDS if kw in query_text]})
            self.tracer.trace_agent_decision("router_decide", {"choice": choice, "reason": "web_keywords"})
            return choice
        
        # 3. Check for hybrid keywords (compare, vs, difference, etc)
        if has_doc and any(kw in query_text for kw in HYBRID_KEYWORDS):
            choice = "hybrid"
            reason = "hybrid_keywords"
            trace_event("router", "hybrid_keywords", {"keywords": [kw for kw in HYBRID_KEYWORDS if kw in query_text]})
            self.tracer.trace_agent_decision("router_decide", {"choice": choice, "reason": reason})
            return choice
        
        # 4. Default logic based on document availability
        if has_doc:
            # Has documents - prefer RAG for general questions
            choice = "rag"
            reason = "default_with_documents"
        else:
            # No documents - use web for any question
            choice = "web"
            reason = "default_no_documents"
        
        logger.info(f"Routing decision:")
        logger.info(f"  Query: '{query_text[:60]}'")
        logger.info(f"  Has documents: {has_doc}")
        logger.info(f"  Route chosen: '{choice}'")
        logger.info(f"  Reason: {reason}")
        
        trace_event("router", "heuristic_decision", {
            "choice": choice,
            "has_doc": has_doc,
            "query_len": len(query_text)
        })
        self.tracer.trace_agent_decision("router_decide", {
            "choice": choice,
            "reason": "heuristic",
            "has_doc": has_doc,
            "query_len": len(query_text)
        })
        return choice

    def route(self, payload: Dict[str, Any], conversation_context: str = "") -> Dict[str, Any]:
        """
        Route the query to appropriate agent(s) and synthesize results.
        Supports multi-turn conversations with context awareness.
        Traces routing with LangSmith.
        
        Args:
            payload: Dict with query info (query/query_text and has_documents/has_doc)
            conversation_context: Previous conversation turns for context
            
        Returns:
            Dict with 'route' and 'response' keys
        """
        if not self.synth:
            logger.error("Synthesis agent not initialized")
            self.tracer.trace_agent_decision("router_route", {"status": "error", "reason": "synthesis_agent_missing"})
            return {
                "route": "error",
                "response": {
                    "summary": "❌ Error: Synthesis agent not initialized",
                    "meta": {"error": "synthesis_agent_missing"}
                }
            }
        
        # Handle both old and new payload formats
        query_text = (payload.get("query") or payload.get("query_text", "")).strip()
        if not query_text:
            logger.warning("Empty query provided to router")
            self.tracer.trace_agent_decision("router_route", {"status": "error", "reason": "empty_query"})
            return {
                "route": "error",
                "response": {
                    "summary": "❌ Error: Query text is empty",
                    "meta": {"error": "empty_query"}
                }
            }
        
        choice = self.decide(payload, conversation_context)
        
        try:
            logger.info(f"Routing to '{choice}' for query: {query_text[:80]}")
            self.tracer.trace_agent_decision("router_route_start", {"choice": choice, "query_len": len(query_text)})
            
            if choice == "rag":
                logger.debug("Executing RAG retrieval")
                if self.rag:
                    chunks = self.rag.retrieve(query_text)
                    
                    # If RAG returns no relevant chunks, fallback to web search
                    if not chunks:
                        logger.info("RAG returned no relevant chunks, falling back to web search")
                        trace_event("router", "rag_no_results_fallback", {"query": query_text[:50]})
                        if self.web:
                            web_results = self.web.search(query_text)
                            response = self.synth.synthesize(query_text, [], web_results, conversation_context=conversation_context)
                        else:
                            response = self.synth.synthesize(query_text, [], [], conversation_context=conversation_context)
                        self.tracer.trace_agent_decision("router_route_complete", {"choice": "web_fallback", "status": "success"})
                        return {"route": "web", "response": response}  # Return as web since that's what was used
                    
                    response = self.synth.synthesize(query_text, chunks, [], conversation_context=conversation_context)
                else:
                    logger.warning("RAG agent not available, falling back to web search")
                    if self.web:
                        web_results = self.web.search(query_text)
                        response = self.synth.synthesize(query_text, [], web_results, conversation_context=conversation_context)
                    else:
                        response = self.synth.synthesize(query_text, [], [], conversation_context=conversation_context)
                    self.tracer.trace_agent_decision("router_route_complete", {"choice": "web_fallback", "status": "success"})
                    return {"route": "web", "response": response}
                    
                self.tracer.trace_agent_decision("router_route_complete", {"choice": choice, "status": "success"})
                return {"route": "rag", "response": response}
            
            elif choice == "web":
                logger.debug("Executing web search")
                if self.web:
                    web_results = self.web.search(query_text)
                else:
                    logger.warning("Web agent not available, using stub response")
                    web_results = []
                response = self.synth.synthesize(query_text, [], web_results, conversation_context=conversation_context)
                self.tracer.trace_agent_decision("router_route_complete", {"choice": choice, "status": "success"})
                return {"route": "web", "response": response}
            
            elif choice == "hybrid":
                # Hybrid: use both sources
                logger.debug("Executing hybrid retrieval (RAG + web)")
                chunks = self.rag.retrieve(query_text) if self.rag else []
                web_results = self.web.search(query_text) if self.web else []
                response = self.synth.synthesize(query_text, chunks, web_results, conversation_context=conversation_context)
                self.tracer.trace_agent_decision("router_route_complete", {"choice": choice, "status": "success"})
                return {"route": "hybrid", "response": response}
            
            else:
                logger.warning(f"Unknown route choice: {choice}, defaulting to web")
                response = self.synth.synthesize(query_text, [], [], conversation_context=conversation_context)
                self.tracer.trace_agent_decision("router_route_complete", {"choice": "web", "status": "fallback"})
                return {"route": "web", "response": response}
            
        except Exception as e:
            logger.error(f"Error during routing: {e}", exc_info=True)
            trace_event("router", "error", {"error": str(e), "choice": choice})
            self.tracer.trace_agent_decision("router_route", {"status": "error", "reason": str(e)[:100]})
            return {
                "route": "error",
                "response": {
                    "summary": f"❌ Error during processing: {str(e)[:200]}",
                    "meta": {
                        "error": str(e),
                        "choice": choice,
                        "needs_api_key": "API_KEY" in str(e)
                    }
                }
            }
