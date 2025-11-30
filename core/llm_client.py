"""
LLM client abstraction.
Uses OpenAI as primary, falls back to Gemini if OpenAI key is not present.
Keep responses deterministic via temperature control.
Includes comprehensive error handling and retry logic.
"""
import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("agentic-rag")

# Try OpenAI first
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not installed")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_AVAILABLE and OPENAI_API_KEY else None

# Fallback to Gemini
try:
    from llm.gemini_client import call_gemini
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    call_gemini = None
    logger.debug("Gemini client not available")


def call_llm(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    stop: Optional[list] = None,
    retry_count: int = 1
) -> str:
    """
    Single synchronous LLM call with fallback support.
    
    Args:
        prompt: The input prompt
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate
        stop: Stop sequences
        retry_count: Number of retry attempts
        
    Returns:
        Response string
        
    Raises:
        Exception: If both OpenAI and Gemini fail
    """
    error_messages = []
    
    # Try OpenAI first
    if client:
        for attempt in range(retry_count):
            try:
                logger.debug(f"Calling OpenAI API (attempt {attempt + 1}/{retry_count})")
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                )
                result = resp.choices[0].message.content.strip()
                logger.info(f"OpenAI API call succeeded with {len(result)} chars response")
                return result
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"OpenAI API error (attempt {attempt + 1}): {error_msg}")
                error_messages.append(f"OpenAI: {error_msg}")
                if attempt < retry_count - 1:
                    continue
    
    # Fallback to Gemini
    gemini_key = os.getenv("GEMINI_API_KEY")
    if GEMINI_AVAILABLE and call_gemini and gemini_key:
        for attempt in range(retry_count):
            try:
                logger.debug(f"Calling Gemini API (attempt {attempt + 1}/{retry_count})")
                gemini_response = call_gemini(prompt, temperature=temperature, max_tokens=max_tokens)
                if gemini_response:
                    logger.info(f"Gemini API call succeeded with {len(gemini_response)} chars response")
                    return gemini_response
                else:
                    error_messages.append("Gemini API returned empty response")
                    if attempt < retry_count - 1:
                        continue
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Gemini API error (attempt {attempt + 1}): {error_msg}")
                error_messages.append(f"Gemini: {error_msg}")
                if attempt < retry_count - 1:
                    continue
    elif not GEMINI_AVAILABLE:
        error_messages.append("google-generativeai package not installed (run: pip install google-generativeai)")
    elif not gemini_key:
        error_messages.append("GEMINI_API_KEY not set")
    
    # Both failed - provide clear error message
    logger.error(f"All LLM backends failed: {error_messages}")
    
    error_summary = " | ".join(error_messages) if error_messages else "No LLM configured"
    raise Exception(
        f"LLM Error: {error_summary}\n\n"
        f"Please configure at least one LLM API key:\n"
        f"  - OPENAI_API_KEY for OpenAI\n"
        f"  - GEMINI_API_KEY for Google Gemini\n"
        f"Set these in your .env file and restart the app."
    )
