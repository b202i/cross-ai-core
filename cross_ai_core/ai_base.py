"""
ai_base.py — Abstract base class for all cross-ai-core provider handlers.

Every provider (Anthropic, xAI, OpenAI, Gemini, Perplexity) inherits from
BaseAIHandler and implements each classmethod.

Cache directory
---------------
Provider handlers cache responses in a directory resolved by
``_get_cache_dir()``.  The location defaults to ``~/.cross_api_cache/`` and
can be overridden by setting ``CROSS_API_CACHE_DIR`` in the environment (or
in ``~/.crossenv`` before importing the library):

    CROSS_API_CACHE_DIR=~/my-project/cache
"""

import os
from abc import ABC, abstractmethod


def _get_cache_dir() -> str:
    """
    Return the API response cache directory path (as a string).

    Resolution order:
      1. ``CROSS_API_CACHE_DIR`` environment variable
      2. ``~/.cross_api_cache/``  (default)
    """
    env_val = os.environ.get("CROSS_API_CACHE_DIR", "").strip()
    return os.path.expanduser(env_val if env_val else "~/.cross_api_cache")


class BaseAIHandler(ABC):  # Abstract Base Class

    @classmethod
    @abstractmethod
    def get_payload(cls, prompt: str):
        """Generate the payload for the given prompt."""
        pass

    @classmethod
    @abstractmethod
    def get_client(cls, *args, **kwargs):
        """Return the client for making API requests."""
        pass

    @classmethod
    @abstractmethod
    def get_cached_response(cls, *args, **kwargs):
        """Return the cached response function."""
        pass

    @classmethod
    @abstractmethod
    def get_model(cls) -> str:
        """Return the model identifier string (e.g. 'gpt-4o')."""
        pass

    @classmethod
    @abstractmethod
    def get_make(cls) -> str:
        """Return the provider make string (e.g. 'openai')."""
        pass

    @classmethod
    @abstractmethod
    def get_content(cls, response: dict) -> str:
        """Extract the text content from a raw API response dict."""
        pass

    @classmethod
    @abstractmethod
    def put_content(cls, report: str, response: dict) -> dict:
        """Inject updated text content back into a raw API response dict."""
        pass

    @classmethod
    @abstractmethod
    def get_data_content(cls, select_data: dict) -> str:
        """Extract text from a stored data record (gen_response wrapper)."""
        pass

    @classmethod
    @abstractmethod
    def get_title(cls, gen_content: dict) -> str:
        """Return the first line of the content as a title."""
        pass

    @classmethod
    @abstractmethod
    def get_usage(cls, response: dict) -> dict:
        """Return token usage dict: {input_tokens, output_tokens, total_tokens}."""
        pass


