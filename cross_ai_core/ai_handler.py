import os

from .ai_anthropic import AnthropicHandler
from .ai_gemini import GeminiHandler
from .ai_openai import OpenAIHandler
from .ai_perplexity import PerplexityHandler
from .ai_xai import XAIHandler
from .ai_error_handler import handle_api_error


class AIResponse:
    """
    Wrapper for AI API responses with metadata.
    
    Provides backward compatibility by unpacking as a 4-tuple,
    while also exposing additional metadata like cache status.
    
    Example:
        # Old code (still works):
        payload, client, response, model = process_prompt(...)
        
        # New code (access metadata):
        result = process_prompt(...)
        if result.was_cached:
            print("Response was cached")
    """
    def __init__(self, payload, client, response, model, was_cached):
        self.payload = payload
        self.client = client
        self.response = response
        self.model = model
        self.was_cached = was_cached
    
    def __iter__(self):
        """Enable backward-compatible tuple unpacking."""
        return iter((self.payload, self.client, self.response, self.model))
    
    def __getitem__(self, index):
        """Enable backward-compatible indexing."""
        return (self.payload, self.client, self.response, self.model)[index]
    
    def __len__(self):
        """Return tuple length for compatibility."""
        return 4


AI_HANDLER_REGISTRY = {
    "xai": XAIHandler,
    "anthropic": AnthropicHandler,
    "openai": OpenAIHandler,
    "perplexity": PerplexityHandler,
    "gemini": GeminiHandler,
}

AI_LIST = ["xai", "anthropic", "openai", "perplexity", "gemini"]


def process_prompt(
    ai_key: str,
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    verbose: bool = False,
    use_cache: bool = True,
) -> "AIResponse":
    """
    Process a prompt with the specified AI.

    Args:
        ai_key:    Provider key, e.g. ``"gemini"`` or ``"anthropic"``.
        prompt:    User prompt text.
        system:    Optional system instruction; ``None`` uses the provider default.
        model:     Optional model override for this call.  Resolution order:
                   explicit ``model`` arg → ``<AI_KEY_UPPER>_MODEL`` env var →
                   handler default (e.g. ``"gemini-2.5-flash"``).
        verbose:   Print cache hit/miss messages.
        use_cache: Read from / write to the on-disk cache (overridden by
                   ``CROSS_NO_CACHE=1``).

    Returns:
        AIResponse: Wrapper object that unpacks as (payload, client, response, model)
                   for backward compatibility, but also provides .was_cached attribute.

    Raises:
        ValueError: For an unknown *ai_key*.
        Exception:  Re-raises any exception after graceful error handling.
    """
    handler_cls = AI_HANDLER_REGISTRY.get(ai_key)
    if not handler_cls:
        raise ValueError(f"Unsupported AI model: {ai_key}")

    try:
        # Build the payload using the centralized handler.
        payload = handler_cls.get_payload(prompt, system=system)

        # Resolve effective model: explicit arg → env var → handler default.
        # All five providers store the model under the "model" key in their payload.
        effective_model = (
            model
            or os.environ.get(f"{ai_key.upper()}_MODEL", "").strip()
            or None
        )
        if effective_model:
            payload["model"] = effective_model
        else:
            effective_model = handler_cls.get_model()

        client = handler_cls.get_client()
        cached_response, was_cached = handler_cls.get_cached_response(client, payload, verbose, use_cache)

        # Stamp the provider make so callers never need to pass it separately.
        # This makes the response self-describing for put_content_auto / get_content_auto.
        # We modify in-memory only — the cache file on disk is NOT changed.
        if isinstance(cached_response, dict):
            cached_response["_make"] = ai_key

        return AIResponse(payload, client, cached_response, effective_model, was_cached)

    except Exception as e:
        # Handle common API errors gracefully (quota, rate limit, etc.)
        # This will print user-friendly messages and exit on quota errors
        handle_api_error(e, ai_key, exit_on_quota=True, quiet=False)
        # If handle_api_error doesn't exit, re-raise the exception
        raise


def get_data_title(ai_key: str, data: dict):
    handler_cls = AI_HANDLER_REGISTRY.get(ai_key)
    if not handler_cls:
        raise ValueError(f"Unsupported AI model: {ai_key}")
    title = handler_cls.get_title(data)
    return title


def get_content(ai_key, response):
    handler_cls = AI_HANDLER_REGISTRY.get(ai_key)
    if not handler_cls:
        raise ValueError(f"Unsupported AI model: {ai_key}")
    return handler_cls.get_content(response)


def put_content(ai_key, report, response):
    handler_cls = AI_HANDLER_REGISTRY.get(ai_key)
    if not handler_cls:
        raise ValueError(f"Unsupported AI model: {ai_key}")
    return handler_cls.put_content(report, response)


def get_content_auto(response: dict) -> str:
    """Extract text from a self-describing response (requires ``_make`` to be embedded).

    ``process_prompt()`` stamps ``_make`` into every response it returns, so
    callers that hold a response from ``process_prompt()`` can use this helper
    without tracking which provider produced the response.

    Raises:
        ValueError: if ``_make`` is absent (e.g. a response loaded from an old
                    container that predates the stamp).  Use ``get_content(make,
                    response)`` in that case.
    """
    make = response.get("_make")
    if not make:
        raise ValueError(
            "Response is missing '_make'.  Use get_content(make, response) "
            "or ensure the response was produced by process_prompt()."
        )
    return get_content(make, response)


def put_content_auto(report: str, response: dict) -> dict:
    """Update text in a self-describing response (requires ``_make`` to be embedded).

    ``process_prompt()`` stamps ``_make`` into every response it returns, so
    callers that hold a response from ``process_prompt()`` can use this helper
    without tracking which provider produced the response.

    Raises:
        ValueError: if ``_make`` is absent (e.g. a response loaded from an old
                    container that predates the stamp).  Use ``put_content(make,
                    report, response)`` in that case.
    """
    make = response.get("_make")
    if not make:
        raise ValueError(
            "Response is missing '_make'.  Use put_content(make, report, response) "
            "or ensure the response was produced by process_prompt()."
        )
    return put_content(make, report, response)


def get_data_content(ai_key, select_data):
    handler_cls = AI_HANDLER_REGISTRY.get(ai_key)
    if not handler_cls:
        raise ValueError(f"Unsupported AI model: {ai_key}")
    content = handler_cls.get_data_content(select_data)
    return content


def get_ai_list() -> list[str]:
    """Return a copy of the ordered provider list.

    Returns a new list on every call so that callers who mutate the result
    (e.g. ``get_ai_list().remove("xai")``) do not corrupt the global ``AI_LIST``.
    """
    return list(AI_LIST)


def get_usage(ai_key: str, response: dict) -> dict:
    """
    Extract token usage from a raw API response dict.

    Delegates to the provider-specific handler so callers never need to
    know each provider's response schema.

    Returns:
        dict with keys:
            input_tokens  (int) — prompt / input tokens consumed
            output_tokens (int) — completion / output tokens generated
            total_tokens  (int) — sum of the above (computed if absent)
        All values default to 0 if the field is missing.
    """
    handler_cls = AI_HANDLER_REGISTRY.get(ai_key)
    if not handler_cls:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return handler_cls.get_usage(response)


def get_default_ai():
    """
    Return the user-configured default AI provider.

    Resolution order:
      1. ``DEFAULT_AI`` environment variable (set via st-admin or .env)
      2. First entry in AI_LIST

    Never hardcode a provider name — always call this function.
    """
    configured = os.environ.get("DEFAULT_AI", "").strip()
    if configured and configured in AI_HANDLER_REGISTRY:
        return configured
    return AI_LIST[0]


def get_ai_make(ai_key: str):
    handler_cls = AI_HANDLER_REGISTRY.get(ai_key)
    if not handler_cls:
        raise ValueError(f"Unsupported AI model: {ai_key}")
    return handler_cls.get_make()


def get_ai_model(ai_key: str) -> str:
    """Return the active model string for *ai_key*.

    Resolution order:
      1. ``<AI_KEY_UPPER>_MODEL`` environment variable  (e.g. ``XAI_MODEL=grok-3-latest``)
      2. Compiled-in handler default  (e.g. ``"grok-4-1-fast-reasoning"``)

    Set the env var in ``~/.crossenv`` or ``.env`` to switch models globally
    without touching code.  Use ``process_prompt(..., model="...")`` to override
    for a single call.

    Raises:
        ValueError: if *ai_key* is not a registered provider.
    """
    # Check env var override first
    env_model = os.environ.get(f"{ai_key.upper()}_MODEL", "").strip()
    if env_model:
        return env_model
    handler_cls = AI_HANDLER_REGISTRY.get(ai_key)
    if not handler_cls:
        raise ValueError(f"Unsupported AI model: {ai_key}")
    return handler_cls.get_model()


# ── API key validation ────────────────────────────────────────────────────────

# Map every registered make → the environment variable that holds its key.
_API_KEY_ENV_VARS: dict[str, str] = {
    "xai":        "XAI_API_KEY",
    "anthropic":  "ANTHROPIC_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "gemini":     "GEMINI_API_KEY",
}


def check_api_key(ai_make: str, paths_checked: list | None = None) -> bool:
    """
    Verify that the API key for *ai_make* is present in the environment.

    If the key is missing, prints a clear diagnostic showing which config
    files were searched (in load order) and the exact env-var name to add.

    Args:
        ai_make:       Provider make string, e.g. ``"xai"`` or ``"gemini"``.
        paths_checked: Ordered list of dotenv file paths that were attempted,
                       in load order (lowest → highest priority).  When omitted
                       the three canonical A1 paths are shown as defaults.

    Returns:
        ``True``  — key is present and non-empty.
        ``False`` — key is missing; diagnostic has already been printed.
    """
    env_var = _API_KEY_ENV_VARS.get(ai_make)
    if not env_var:
        return True  # unknown provider — let the SDK surface the real error

    if os.environ.get(env_var, "").strip():
        return True  # key present ✓

    if paths_checked is None:
        paths_checked = [
            os.path.expanduser("~/.crossenv"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
            os.path.join(os.getcwd(), ".env"),
        ]

    print(f"\n  ✗  API key not set for '{ai_make}'  (env var: {env_var})")
    print(f"     Config files searched (in load order):")
    for i, path in enumerate(paths_checked, 1):
        status = "✓ exists" if os.path.isfile(path) else "✗ not found"
        print(f"       {i}. {path}  [{status}]")
    print(f"     Fix: add  {env_var}=<your-key>  to one of the files above.\n")
    return False

