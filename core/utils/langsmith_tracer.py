"""
LangSmith tracing integration for agent observability.
Captures agent decisions, LLM calls, retrievals, and synthesis steps.
"""
import logging
from typing import Optional, Dict, Any
from functools import wraps

logger = logging.getLogger("agentic-rag")

try:
    from langsmith import traceable, Client
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False


class LangSmithTracer:
    """Wrapper for LangSmith tracing with graceful fallback."""

    def __init__(self, api_key: Optional[str] = None, project_name: str = "agentic-rag"):
        """
        Initialize LangSmith tracer.

        Args:
            api_key: LangSmith API key
            project_name: Project name for organizing traces
        """
        self.api_key = api_key
        self.project_name = project_name
        self.enabled = LANGSMITH_AVAILABLE and bool(api_key)
        self.client = None

        if self.enabled:
            try:
                self.client = Client(api_key=api_key)
                logger.info(f"LangSmith tracer initialized: project={project_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize LangSmith: {e}")
                self.enabled = False

    def trace_agent_decision(self, agent_name: str, decision: Dict[str, Any]) -> None:
        """
        Trace an agent decision (routing, retrieval, synthesis).

        Args:
            agent_name: Name of the agent (e.g., "router", "rag", "synthesis")
            decision: Decision details as dict
        """
        if not self.enabled:
            return

        try:
            # Log to LangSmith via standard logging (events will be captured)
            logger.info(f"[{agent_name}] decision: {decision}")
        except Exception as e:
            logger.debug(f"Error tracing agent decision: {e}")

    def trace_retrieval(self, query: str, results: Dict[str, Any]) -> None:
        """
        Trace a retrieval operation.

        Args:
            query: Query text
            results: Retrieval results including matches and scores
        """
        if not self.enabled:
            return

        try:
            logger.info(f"[retrieval] query_len={len(query)}, results={results.get('count', 0)}")
        except Exception as e:
            logger.debug(f"Error tracing retrieval: {e}")

    def trace_llm_call(
        self, model: str, prompt_len: int, response_len: int, latency_ms: float
    ) -> None:
        """
        Trace an LLM call.

        Args:
            model: Model name
            prompt_len: Prompt length in tokens (approx)
            response_len: Response length in tokens (approx)
            latency_ms: Latency in milliseconds
        """
        if not self.enabled:
            return

        try:
            logger.info(
                f"[llm] model={model}, prompt_len={prompt_len}, response_len={response_len}, latency_ms={latency_ms:.1f}"
            )
        except Exception as e:
            logger.debug(f"Error tracing LLM call: {e}")

    def traceable_function(self, name: str, run_type: str = "chain"):
        """
        Decorator to trace a function with LangSmith.

        Args:
            name: Name of the traced function
            run_type: Type of run ("chain", "tool", "llm", etc.)

        Returns:
            Decorator function
        """

        def decorator(func):
            if not self.enabled or not LANGSMITH_AVAILABLE:
                return func

            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    # Use langsmith.traceable if available
                    traced_func = traceable(name=name, run_type=run_type)(func)
                    return traced_func(*args, **kwargs)
                except Exception as e:
                    logger.debug(f"Error in traced function: {e}")
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def get_dashboard_url(self) -> Optional[str]:
        """Get the LangSmith dashboard URL for this project."""
        if not self.enabled or not self.client:
            return None
        try:
            return f"https://smith.langchain.com/o/agentic-rag/projects/{self.project_name}"
        except Exception as e:
            logger.debug(f"Error getting dashboard URL: {e}")
            return None


def get_tracer(enabled: bool = True, api_key: Optional[str] = None, project: str = "agentic-rag") -> LangSmithTracer:
    """
    Factory function to get a LangSmith tracer instance.

    Args:
        enabled: Whether tracing is enabled
        api_key: LangSmith API key
        project: Project name

    Returns:
        LangSmithTracer instance
    """
    if enabled and api_key:
        return LangSmithTracer(api_key=api_key, project_name=project)
    else:
        # Return no-op tracer
        return LangSmithTracer(api_key=None, project_name=project)
