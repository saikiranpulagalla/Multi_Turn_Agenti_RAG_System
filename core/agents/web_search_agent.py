"""
Web Search Agent - wrapper using SerpAPI or stub responses.
Includes caching and error handling for better reliability.
"""
import requests
from core.utils.logging_utils import trace_event
from core.utils.config import Config
import os
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("agentic-rag")

SERP_API_KEY = Config.SERP_API_KEY
SERP_API = "https://serpapi.com/search"


class WebSearchAgent:
    def __init__(self, cache_ttl_seconds: int = 3600):
        """
        Initialize web search agent with optional caching.
        
        Args:
            cache_ttl_seconds: Cache time-to-live in seconds
        """
        self.cache: Dict[str, tuple] = {}  # query -> (results, timestamp)
        self.cache_ttl = cache_ttl_seconds
        self.request_count = 0
        self.error_count = 0

    def _is_cache_valid(self, query: str) -> bool:
        """Check if cached result is still valid."""
        if query not in self.cache:
            return False
        
        results, timestamp = self.cache[query]
        elapsed = (datetime.now() - timestamp).total_seconds()
        return elapsed < self.cache_ttl

    def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Search the web for the given query.
        Uses SerpAPI first, then falls back to DuckDuckGo, then stub.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            
        Returns:
            List of dicts with 'title', 'snippet', and 'link' keys
        """
        if not query or not query.strip():
            trace_event("web_search", "empty_query", {"q": query})
            logger.warning("Empty query provided to web search")
            return []
        
        # Clean and enhance the query for better search results
        cleaned_query = self._enhance_query(query)
        
        # Check cache first (only for successful results, not stubs)
        if Config.ENABLE_CACHE and self._is_cache_valid(cleaned_query):
            logger.debug(f"Returning cached web search results for: {cleaned_query[:50]}")
            trace_event("web_search", "cache_hit", {"q": cleaned_query[:50]})
            results = self.cache[cleaned_query][0]
            # Don't return stub responses from cache
            if results and results[0].get("title") != "Search Result":
                return results
        
        # Try SerpAPI if configured
        if SERP_API_KEY:
            results = self._search_serpapi(cleaned_query, num_results)
            if results:
                # Cache successful results
                if Config.ENABLE_CACHE:
                    self.cache[cleaned_query] = (results, datetime.now())
                return results
        
        # Try DuckDuckGo as fallback
        logger.info(f"Falling back to DuckDuckGo search for: {cleaned_query[:50]}")
        results = self._search_duckduckgo(cleaned_query, num_results)
        if results:
            # Cache successful results
            if Config.ENABLE_CACHE:
                self.cache[cleaned_query] = (results, datetime.now())
            return results
        
        # Final fallback: stub response (don't cache this!)
        logger.debug(f"Using stub response for web search: {cleaned_query[:50]}")
        trace_event("web_search", "stub_response", {"q": cleaned_query[:50]})
        stub_results = [{
            "title": "Search Result",
            "snippet": f"Web search for '{cleaned_query}' returned no results. Try setting SERP_API_KEY for better results.",
            "link": ""
        }]
        # NOTE: We don't cache stub responses so next attempt can try real search
        
        return stub_results

    def _enhance_query(self, query: str) -> str:
        """
        Enhance query for better search results.
        Removes common question words, punctuation, and adds context.
        
        Args:
            query: Original query
            
        Returns:
            Enhanced query for web search
        """
        # Remove leading question words and punctuation
        enhanced = query.strip()
        
        # Remove "about", "what is", "how to", etc
        question_starters = ["about ", "what is ", "what are ", "how to ", "how do ", "tell me about ", "explain ", "describe "]
        for starter in question_starters:
            if enhanced.lower().startswith(starter):
                enhanced = enhanced[len(starter):].strip()
        
        # Remove trailing question marks
        enhanced = enhanced.rstrip("?!")
        
        # Remove extra spaces
        enhanced = " ".join(enhanced.split())
        
        logger.debug(f"Enhanced query from '{query}' to '{enhanced}'")
        trace_event("web_search", "query_enhanced", {"original": query[:50], "enhanced": enhanced[:50]})
        
        return enhanced if enhanced else query

    def _search_serpapi(self, query: str, num_results: int = 5) -> Optional[List[Dict[str, str]]]:
        """
        Perform actual web search using SerpAPI.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results or None if request failed
        """
        try:
            logger.debug(f"Calling SerpAPI for: {query[:50]}")
            params = {
                "q": query,
                "api_key": SERP_API_KEY,
                "num": num_results
            }
            
            resp = requests.get(SERP_API, params=params, timeout=10)
            resp.raise_for_status()
            
            self.request_count += 1
            data = resp.json()
            
            # Parse results from SerpAPI response
            results = []
            for r in data.get("organic_results", [])[:num_results]:
                result = {
                    "title": r.get("title", ""),
                    "snippet": r.get("snippet", ""),
                    "link": r.get("link", "")
                }
                if result["title"] or result["snippet"]:  # Only include non-empty results
                    results.append(result)
            
            logger.info(f"SerpAPI returned {len(results)} results for: {query[:50]}")
            trace_event("web_search", "serpapi_success", {
                "q": query[:50],
                "count": len(results),
                "request_num": self.request_count
            })
            return results
            
        except requests.exceptions.Timeout:
            logger.warning(f"SerpAPI timeout for query: {query[:50]}")
            trace_event("web_search", "timeout", {"q": query[:50]})
            self.error_count += 1
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"SerpAPI request error: {e}")
            trace_event("web_search", "api_error", {"q": query[:50], "error": str(e)[:100]})
            self.error_count += 1
            return None
        except Exception as e:
            logger.error(f"Unexpected error in SerpAPI search: {e}")
            trace_event("web_search", "error", {"q": query[:50], "error": str(e)[:100]})
            self.error_count += 1
            return None

    def _search_duckduckgo(self, query: str, num_results: int = 5) -> Optional[List[Dict[str, str]]]:
        """
        Perform web search using DuckDuckGo (no API key needed).
        Uses duckduckgo-search library if available.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results or None if unavailable
        """
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning("duckduckgo-search library not available")
            trace_event("web_search", "duckduckgo_not_installed", {"q": query[:50]})
            return None
        
        try:
            logger.debug(f"Calling DuckDuckGo for: {query[:50]}")
            results = []
            
            with DDGS() as ddgs:
                for result in ddgs.text(query, max_results=num_results):
                    results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("body", ""),
                        "link": result.get("href", "")
                    })
            
            if results:
                logger.info(f"DuckDuckGo returned {len(results)} results for: {query[:50]}")
                trace_event("web_search", "duckduckgo_success", {
                    "q": query[:50],
                    "count": len(results)
                })
                return results
            else:
                logger.debug(f"DuckDuckGo returned no results for: {query[:50]}")
                trace_event("web_search", "duckduckgo_no_results", {"q": query[:50]})
                return None
                
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            trace_event("web_search", "duckduckgo_error", {"q": query[:50], "error": str(e)[:100]})
            return None

    def get_stats(self) -> Dict[str, int]:
        """Get agent statistics."""
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "cache_size": len(self.cache)
        }
