"""Gemini client for LLM calls.

Fallback option when OpenAI is not available.

This module prefers the configured ``GEMINI_MODEL`` (from env) and tries a
list of likely model names (including ``gemini-2.0-flash``). It also falls
back to listing available models when necessary.
"""
import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("agentic-rag")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

# Prefer a configured model if present; default to a recent flash model
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def list_available_models():
    """List all available Gemini models for debugging."""
    if not GEMINI_AVAILABLE or not genai or not GEMINI_API_KEY:
        return []
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        models = genai.list_models()
        available = []
        for model in models:
            if 'generateContent' in getattr(model, 'supported_generation_methods', []):
                available.append({
                    'name': model.name,
                    'display_name': getattr(model, 'display_name', 'N/A'),
                    'supported_methods': model.supported_generation_methods
                })
        return available
    except Exception as e:
        logger.debug(f"Error listing models: {e}")
        return []


def get_gemini_client(model_name: Optional[str] = None):
    """
    Initialize and return Gemini client if API key is available.
    Tries the configured model first, then a set of fallbacks, then lists
    available models as a last resort.
    """
    if not GEMINI_AVAILABLE or not genai:
        return None
    if not GEMINI_API_KEY:
        return None

    try:
        genai.configure(api_key=GEMINI_API_KEY)

        # Build an ordered list of models to try: user-specified first
        models_to_try = []
        if model_name:
            models_to_try.append(model_name)
        if GEMINI_MODEL and GEMINI_MODEL not in models_to_try:
            models_to_try.append(GEMINI_MODEL)

        # Common fallbacks to try
        fallbacks = [
            'gemini-2.0-flash',
            'gemini-1.5-pro-latest',
            'gemini-1.5-flash-latest',
            'models/gemini-2.0-flash',
            'models/gemini-1.5-pro',
            'models/gemini-1.5-flash',
        ]

        for f in fallbacks:
            if f not in models_to_try:
                models_to_try.append(f)

        last_error = None
        for model in models_to_try:
            try:
                logger.debug(f"Trying Gemini model: {model}")
                return genai.GenerativeModel(model)
            except Exception as e:
                last_error = str(e)
                logger.debug(f"Model {model} failed: {e}")
                continue

        # If all specific models fail, try to list available models
        try:
            available_models = genai.list_models()
            if available_models:
                for model in available_models:
                    if 'generateContent' in getattr(model, 'supported_generation_methods', []):
                        try:
                            return genai.GenerativeModel(model.name)
                        except Exception:
                            continue
        except Exception as list_error:
            logger.debug(f"Error listing available models: {list_error}")

        # Last resort: try without specifying model (uses default)
        try:
            return genai.GenerativeModel()
        except Exception as e:
            raise Exception(f"Failed to initialize any Gemini model. Last error: {last_error}")

    except Exception as e:
        raise Exception(f"Error configuring Gemini: {e}")


def call_gemini(prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
    """
    Call Gemini API with the given prompt.

    Args:
        prompt: The input prompt
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text response, or raises Exception on failure
    """
    if not GEMINI_AVAILABLE:
        raise Exception("google-generativeai package not installed. Run: pip install google-generativeai")

    if not GEMINI_API_KEY:
        raise Exception("GEMINI_API_KEY not set in environment variables")

    try:
        model = get_gemini_client()
    except Exception as e:
        raise Exception(f"Failed to initialize Gemini client: {str(e)}")

    if not model:
        raise Exception("Failed to initialize Gemini client. Check your API key and model names.")

    try:
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )

        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            if getattr(response.prompt_feedback, 'block_reason', None):
                raise Exception(f"Content blocked: {response.prompt_feedback.block_reason}")

        text = getattr(response, 'text', None)
        if not text:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                    if candidate.finish_reason != 1:  # 1 = STOP (normal)
                        raise Exception(f"Generation stopped: {candidate.finish_reason}")
            return ""

        return text.strip()
    except Exception as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg or "invalid" in error_msg.lower():
            raise Exception(f"Invalid Gemini API key. Please check your GEMINI_API_KEY in the .env file.")
        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            raise Exception(f"Gemini API quota exceeded: {error_msg}")
        elif "permission" in error_msg.lower() or "forbidden" in error_msg.lower():
            raise Exception(f"Gemini API permission denied: {error_msg}")
        else:
            raise Exception(f"Gemini API error: {error_msg}")

