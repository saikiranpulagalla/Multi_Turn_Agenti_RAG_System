"""
Centralized configuration management.
Loads from .env and provides sensible defaults.
Validates all required API keys and settings.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Centralized configuration with validation and defaults."""
    
    # LLM Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-pro")
    
    # Embedding Configuration
    EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    EMBED_DIM: int = int(os.getenv("EMBED_DIM", "384"))  # Pinecone index dimension; can be overridden via env
    
    # Retrieval Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    TOP_K_PARENTS: int = int(os.getenv("TOP_K_PARENTS", "3"))
    TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "6"))
    
    # LLM Parameters
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    
    # Web Search Configuration
    SERP_API_KEY: Optional[str] = os.getenv("SERP_API_KEY")
    WEB_SEARCH_RESULTS: int = int(os.getenv("WEB_SEARCH_RESULTS", "5"))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_TRACING: bool = os.getenv("ENABLE_TRACING", "true").lower() in ("true", "1", "yes")
    
    # Cache Configuration
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "true").lower() in ("true", "1", "yes")
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    
    # Pinecone Vector Store Configuration
    USE_PINECONE: bool = os.getenv("USE_PINECONE", "true").lower() in ("true", "1", "yes")
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    PINECONE_HOST: Optional[str] = os.getenv("PINECONE_HOST")  # e.g., https://rag-xxx.svc.aped-4627-b74a.pinecone.io
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "rag")
    
    # LangSmith Tracing Configuration
    LANGSMITH_API_KEY: Optional[str] = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "agentic-rag")
    ENABLE_LANGSMITH: bool = os.getenv("ENABLE_LANGSMITH", "true").lower() in ("true", "1", "yes")
    
    @classmethod
    def validate(cls) -> tuple:
        """
        Validate configuration.
        Returns (is_valid, list_of_warnings)
        """
        warnings = []
        
        # Check LLM configuration
        if not cls.OPENAI_API_KEY and not cls.GEMINI_API_KEY:
            warnings.append("⚠️ WARNING: Neither OPENAI_API_KEY nor GEMINI_API_KEY is set. LLM features will not work.")
        
        # Check web search configuration
        if not cls.SERP_API_KEY:
            warnings.append("⚠️ WARNING: SERP_API_KEY not set. Web search will use stub responses.")
        
        # Check Pinecone configuration if enabled
        if cls.USE_PINECONE and not cls.PINECONE_API_KEY:
            warnings.append("⚠️ WARNING: PINECONE_API_KEY not set but USE_PINECONE=true. Vector store will fall back to in-memory.")
        
        if cls.USE_PINECONE and not cls.PINECONE_HOST:
            warnings.append("⚠️ WARNING: PINECONE_HOST not set but USE_PINECONE=true. Vector store will fall back to in-memory.")
        
        # Check LangSmith configuration if enabled
        if cls.ENABLE_LANGSMITH and not cls.LANGSMITH_API_KEY:
            warnings.append("⚠️ WARNING: LANGSMITH_API_KEY not set but ENABLE_LANGSMITH=true. Tracing will be disabled.")
        
        # Validate numeric configurations
        if cls.CHUNK_SIZE <= 0:
            warnings.append("⚠️ WARNING: CHUNK_SIZE must be positive, using default 1000")
            cls.CHUNK_SIZE = 1000
        
        if cls.TOP_K_CHUNKS <= 0:
            warnings.append("⚠️ WARNING: TOP_K_CHUNKS must be positive, using default 6")
            cls.TOP_K_CHUNKS = 6
        
        if cls.LLM_TEMPERATURE < 0 or cls.LLM_TEMPERATURE > 1:
            warnings.append("⚠️ WARNING: LLM_TEMPERATURE must be between 0 and 1, using default 0.0")
            cls.LLM_TEMPERATURE = 0.0
        
        return (len(warnings) == 0 or (cls.OPENAI_API_KEY or cls.GEMINI_API_KEY)), warnings
    
    @classmethod
    def get_summary(cls) -> dict:
        """Get configuration summary for logging/debugging."""
        return {
            "openai_configured": bool(cls.OPENAI_API_KEY),
            "gemini_configured": bool(cls.GEMINI_API_KEY),
            "web_search_configured": bool(cls.SERP_API_KEY),
            "pinecone_configured": bool(cls.PINECONE_API_KEY and cls.PINECONE_HOST),
            "langsmith_enabled": cls.ENABLE_LANGSMITH and bool(cls.LANGSMITH_API_KEY),
            "chunk_size": cls.CHUNK_SIZE,
            "embedding_model": cls.EMBED_MODEL_NAME,
            "embedding_dim": cls.EMBED_DIM,
            "cache_enabled": cls.ENABLE_CACHE,
            "tracing_enabled": cls.ENABLE_TRACING,
        }


# Validate on import
_is_valid, _warnings = Config.validate()
if _warnings:
    import logging
    logger = logging.getLogger("agentic-rag")
    for warning in _warnings:
        logger.warning(warning)
