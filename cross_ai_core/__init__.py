"""
cross_ai_core — Multi-provider AI dispatcher with caching and error handling.

Public API
----------
::

    from cross_ai_core import process_prompt, get_content, get_default_ai

    result = process_prompt("gemini", prompt, verbose=False, use_cache=True)
    text   = get_content("gemini", result.response)

Supported providers: xai, anthropic, openai, perplexity, gemini

Adding a provider
-----------------
1. Create ``cross_ai_core/ai_<name>.py`` implementing ``BaseAIHandler``.
2. Register in ``cross_ai_core/ai_handler.py``: add to ``AI_HANDLER_REGISTRY``
   and ``AI_LIST``.

Cache
-----
Responses are cached by default in ``~/.cross_api_cache/``.
Override with the ``CROSS_API_CACHE_DIR`` environment variable.
Bypass per-call with ``use_cache=False``, or globally with ``CROSS_NO_CACHE=1``.

Keys
----
The library reads API keys from ``os.environ``.  The calling application is
responsible for loading ``.env`` or ``~/.crossenv`` before importing.
"""

from cross_ai_core.ai_handler import (   # noqa: F401
    AI_HANDLER_REGISTRY,
    AI_LIST,
    AIResponse,
    check_api_key,
    get_ai_list,
    get_ai_make,
    get_ai_model,
    get_content,
    get_data_content,
    get_data_title,
    get_default_ai,
    get_usage,
    process_prompt,
    put_content,
)
from cross_ai_core.ai_base import BaseAIHandler, _get_cache_dir  # noqa: F401
from cross_ai_core.ai_error_handler import handle_api_error       # noqa: F401

__version__ = "0.2.0"
__all__ = [
    # Core dispatch
    "process_prompt",
    "get_content",
    "put_content",
    "get_data_content",
    "get_data_title",
    "get_default_ai",
    "get_ai_model",
    "get_ai_make",
    "get_ai_list",
    "get_usage",
    "check_api_key",
    # Registry
    "AI_HANDLER_REGISTRY",
    "AI_LIST",
    "AIResponse",
    # Extension points
    "BaseAIHandler",
    "_get_cache_dir",
    # Error handling
    "handle_api_error",
]

