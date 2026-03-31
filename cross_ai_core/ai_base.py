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

