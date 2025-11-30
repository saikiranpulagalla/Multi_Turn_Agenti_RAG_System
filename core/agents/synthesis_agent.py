"""
Synthesis agent: merges RAG chunks + web search snippets into a coherent summary and global analysis.
Uses LLM client to synthesize with improved prompting, error handling, and LangSmith tracing.
"""
from core.llm_client import call_llm
from core.utils.logging_utils import trace_event
from core.utils.config import Config
from core.utils.langsmith_tracer import get_tracer
from typing import List, Dict
import time
import logging

logger = logging.getLogger("agentic-rag")


def build_synthesis_prompt(query: str, rag_chunks: List[Dict], web_results: List[Dict], instructions: str = "", conversation_context: str = "") -> str:
    """
    Build a well-structured synthesis prompt from query, RAG chunks, and web results.
    Includes conversation context for multi-turn conversations.
    """
    parts = []
    
    # Conversation context section (if available)
    if conversation_context:
        parts.append("=" * 60)
        parts.append("CONVERSATION CONTEXT:")
        parts.append(conversation_context)
        parts.append("=" * 60)
    
    # Query section
    parts.append("=" * 60)
    parts.append("USER QUERY:")
    parts.append(query)
    parts.append("=" * 60)
    
    # Determine source type
    has_rag = bool(rag_chunks)
    has_web = bool(web_results)
    
    # RAG chunks section
    if rag_chunks:
        parts.append("\n## DOCUMENT CONTENT (from RAG):")
        for i, c in enumerate(rag_chunks, 1):
            chunk_text = c.get("text", "").strip()
            if chunk_text:
                # Limit chunk size in prompt to avoid token overflow
                max_chunk_display = 500
                display_text = chunk_text[:max_chunk_display]
                if len(chunk_text) > max_chunk_display:
                    display_text += f"\n... [truncated {len(chunk_text) - max_chunk_display} chars]"
                
                parts.append(f"\n### Chunk {i}:")
                parts.append(display_text)
    else:
        parts.append("\n## DOCUMENT CONTENT: None provided")
    
    # Web results section
    if web_results:
        parts.append("\n## WEB SEARCH RESULTS:")
        for i, w in enumerate(web_results, 1):
            title = w.get('title', '').strip()
            snippet = w.get('snippet', '').strip()
            link = w.get('link', '').strip()
            
            if title or snippet:
                parts.append(f"\n### Result {i}:")
                if title:
                    parts.append(f"**Title:** {title}")
                if snippet:
                    parts.append(f"**Content:** {snippet}")
                if link:
                    parts.append(f"**Link:** {link}")
    else:
        parts.append("\n## WEB SEARCH RESULTS: None provided")
    
    # Instructions section - adapt based on sources available
    parts.append("\n" + "=" * 60)
    parts.append("SYNTHESIS TASK:")
    parts.append("=" * 60)
    
    # Build task instructions based on what sources we have
    if has_rag and has_web:
        task_instructions = """
Please provide a comprehensive response with:

1. **Summary**: Concise 2-3 sentence summary addressing the query
2. **Key Findings**: Main points from both document and web sources
3. **Analysis**: Deeper insights and connections between sources
4. **Action Items**: 5 concrete next steps or recommendations

Format your response clearly with these sections."""
    elif has_web and not has_rag:
        task_instructions = """
Please provide a comprehensive response based on the web search results:

1. **Summary**: Direct answer to the user's query based on the search results
2. **Key Points**: Main information found in the search results
3. **Explanation**: Detailed explanation of the topic
4. **Additional Context**: Any relevant background or related information
5. **Further Reading**: Suggestions for where to learn more

Be specific and reference the search results provided."""
    else:
        task_instructions = """
Please provide a response to the user's query.

Unfortunately, no sources are available to draw from. Provide your best general knowledge response, 
but note the limitation."""
    
    parts.append(task_instructions)
    
    if instructions:
        parts.append(f"\n**Additional Instructions:** {instructions}")
    
    parts.append("=" * 60)
    
    return "\n".join(parts)


class SynthesisAgent:
    def __init__(self):
        # Initialize tracer
        self.tracer = get_tracer(
            enabled=Config.ENABLE_LANGSMITH,
            api_key=Config.LANGSMITH_API_KEY,
            project=Config.LANGSMITH_PROJECT
        )

    def synthesize(self, query: str, rag_chunks: List[Dict], web_results: List[Dict], instructions: str = "", conversation_context: str = "") -> Dict:
        """
        Synthesize a response from RAG chunks and web results.
        Returns dict with 'summary' and 'meta' keys.
        Traces LLM call with LangSmith.
        Supports multi-turn conversations with context.
        """
        if not query or not query.strip():
            self.tracer.trace_agent_decision("synthesis", {"status": "error", "reason": "empty_query"})
            return {
                "summary": "❌ Error: Query is empty",
                "meta": {
                    "error": "empty_query",
                    "source_counts": {"rag": len(rag_chunks), "web": len(web_results)}
                }
            }
        
        # Check if this is a non-information query (greeting, acknowledgment, etc)
        query_lower = query.lower().strip()
        NON_INFO_QUERIES = ["hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "sure", "yes", "no", "yep", "nope", "cool", "nice", "good", "bad", "ok got it", "got it", "understood", "i see", "lol", "haha", "hmm"]
        
        if query_lower in NON_INFO_QUERIES or len(query_lower) <= 3:
            self.tracer.trace_agent_decision("synthesis", {"status": "non_info_query", "query": query_lower})
            return {
                "summary": f"👋 {query.capitalize()}! How can I help you today?",
                "meta": {
                    "source_counts": {"rag": 0, "web": 0},
                    "type": "greeting",
                    "note": "non_information_query"
                }
            }
        
        # If both sources are empty, provide helpful message
        if not rag_chunks and not web_results:
            self.tracer.trace_agent_decision("synthesis", {"status": "no_sources"})
            return {
                "summary": "ℹ️ No relevant information found. Please upload a document or ensure web search is configured.",
                "meta": {
                    "source_counts": {"rag": 0, "web": 0},
                    "warning": "no_sources"
                }
            }
        
        try:
            prompt = build_synthesis_prompt(query, rag_chunks, web_results, instructions, conversation_context)
            
            # Use configured parameters
            start_time = time.time()
            resp = call_llm(
                prompt,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=Config.LLM_MAX_TOKENS
            )
            latency_ms = (time.time() - start_time) * 1000
            
            if not resp or not resp.strip():
                resp = "❌ No response generated from LLM."
            
            # Trace LLM call
            prompt_words = len(prompt.split())
            response_words = len(resp.split())
            self.tracer.trace_llm_call(
                model=Config.OPENAI_MODEL if Config.OPENAI_API_KEY else Config.GEMINI_MODEL,
                prompt_len=prompt_words,
                response_len=response_words,
                latency_ms=latency_ms
            )
            
            trace_event("synthesis", "synthesized", {
                "query": query[:100],
                "rag_chunks": len(rag_chunks),
                "web_results": len(web_results),
                "latency_ms": latency_ms
            })
            logger.info(f"Successfully synthesized response for query: {query[:80]}")
            
            return {
                "summary": resp,
                "meta": {
                    "source_counts": {"rag": len(rag_chunks), "web": len(web_results)},
                    "model": "llm",
                    "success": True,
                    "latency_ms": latency_ms
                }
            }
        except Exception as e:
            error_msg = str(e)
            trace_event("synthesis", "error", {
                "query": query[:100],
                "error": error_msg
            })
            self.tracer.trace_agent_decision("synthesis", {"status": "error", "error": error_msg[:100]})
            logger.error(f"Error during synthesis: {error_msg}", exc_info=True)
            
            # Provide helpful guidance for API key issues
            help_text = ""
            if "API_KEY" in error_msg or "key" in error_msg.lower():
                help_text = "\n\n**How to fix:**\n1. Set OPENAI_API_KEY or GEMINI_API_KEY in your .env file\n2. Restart the app"
            
            return {
                "summary": f"❌ LLM Error: {error_msg}{help_text}",
                "meta": {
                    "error": error_msg,
                    "source_counts": {"rag": len(rag_chunks), "web": len(web_results)},
                    "needs_api_key": "API_KEY" in error_msg,
                    "success": False
                }
            }
