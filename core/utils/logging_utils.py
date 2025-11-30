"""
Simple logging & trace helpers. Extend with LangSmith instrumentation.
"""
import logging
import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("agentic-rag")

def trace_event(session_id: str, event_name: str, payload: dict):
    """
    Hook for LangSmith or other tracing. Extend to push to LangSmith.
    For now, logs structured.
    """
    logger.info("[%s] %s -- %s", session_id, event_name, payload)
