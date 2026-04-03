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

import hashlib
import json
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

    # ── Concrete shared implementation ────────────────────────────────────────

    @classmethod
    def get_cached_response(
        cls, client, payload: dict, verbose: bool = False, use_cache: bool = True
    ) -> tuple[dict, bool]:
        """MD5-keyed file cache shared by all providers.

        Delegates the actual API call to ``cls._call_api(client, payload)``.
        Subclasses implement only ``_call_api``; they do not need to override
        this method unless they have unusual caching requirements.
        """
        if os.environ.get("CROSS_NO_CACHE"):
            if verbose:
                print("Cache disabled via CROSS_NO_CACHE; fetching fresh data.")
            return cls._call_api(client, payload), False

        if not use_cache:
            if verbose:
                print("Cache disabled; fetching fresh data.")
            return cls._call_api(client, payload), False

        # Deterministic MD5 hash of the full payload → unique key per request
        param_str = json.dumps(payload, sort_keys=True)
        md5_hash = hashlib.md5(param_str.encode("utf-8")).hexdigest()

        cache_dir = _get_cache_dir()
        cache_file = os.path.join(cache_dir, f"{md5_hash}.json")

        if os.path.exists(cache_file):
            if verbose:
                print(f"api_cache: hit  {cache_file}")
            with open(cache_file, "r") as f:
                return json.load(f), True

        if verbose:
            print("api_cache: miss — calling API")

        json_response = cls._call_api(client, payload)
        if not json_response:
            return {}, False

        os.makedirs(cache_dir, exist_ok=True)
        try:
            with open(cache_file, "w") as f:
                json.dump(json_response, f)
            if verbose:
                print(f"api_cache: saved {cache_file}")
        except Exception as e:
            print(f"api_cache: write error: {e}")

        return json_response, False

    # ── Abstract methods every provider must implement ────────────────────────

    @classmethod
    @abstractmethod
    def _call_api(cls, client, payload: dict) -> dict:
        """Make the actual API call and return the response as a plain dict.

        This is the only method providers need to implement to participate in
        caching.  ``get_cached_response`` calls this after a cache miss, or
        directly when ``use_cache=False``.
        """
        pass

    @classmethod
    @abstractmethod
    def get_payload(cls, prompt: str, system: str | None = None):
        """Generate the payload for the given prompt.

        Args:
            prompt: The user prompt text.
            system: Optional system instruction.  When *None* the provider's
                    default system prompt is used (the journalism persona).
        """
        pass

    @classmethod
    @abstractmethod
    def get_client(cls, *args, **kwargs):
        """Return the client for making API requests."""
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

