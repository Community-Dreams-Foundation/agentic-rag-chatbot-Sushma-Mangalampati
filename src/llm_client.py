"""LLM client: OpenAI or Ollama (local)."""

import os

from openai import OpenAI

OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "llama3.2"


def _use_ollama() -> bool:
    """Check if we should use Ollama instead of OpenAI."""
    return os.environ.get("USE_OLLAMA", "").lower() in ("1", "true", "yes")


def get_client(api_key: str | None = None) -> OpenAI | None:
    """
    Return an OpenAI-compatible client.
    - If USE_OLLAMA=1: Ollama at localhost:11434 (no API key needed)
    - Else: OpenAI (requires api_key or OPENAI_API_KEY)
    """
    if _use_ollama():
        return OpenAI(
            base_url=os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL),
            api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
        )
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)


def get_model() -> str:
    """Return the model name for chat completions."""
    if _use_ollama():
        return os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    return "gpt-4o-mini"


def has_llm() -> bool:
    """Return True if we have a working LLM (Ollama or OpenAI with key)."""
    if _use_ollama():
        return True
    return bool(os.environ.get("OPENAI_API_KEY"))
