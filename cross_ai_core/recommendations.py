"""
cross_ai_core.recommendations — curated model recommendations per provider.

A small, hand-maintained dict per make.  Used by :func:`cross_ai_core.discovery.
get_available_models` to:

  * mark a model with ``is_recommended=True`` when it appears here, so the
    ``st-admin`` alias wizard can sort recommended first;
  * mark the **first** entry of each make's list as ``is_default=True`` so the
    wizard can show "★ default" next to it;
  * provide a graceful **fallback** when a provider's discovery API is
    unreachable (network down, bad key, SDK incompatibility) — the recommended
    set is the minimum the wizard should always be able to show.

Update this file whenever a new flagship model ships.  No code change is
required — bump the dict, cut a new ``cross-ai-core`` patch release.
"""
from __future__ import annotations


# Order matters: the FIRST entry per make is treated as that make's "default"
# recommendation in the wizard.  Keep it in sync (roughly) with each handler's
# compiled-in ``AI_MODEL`` constant.
RECOMMENDED_MODELS: dict[str, list[str]] = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
    ],
    "anthropic": [
        "claude-opus-4-5",
        "claude-sonnet-4-5",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-haiku-20241022",
    ],
    "xai": [
        "grok-4-1-fast-reasoning",
        "grok-4",
        "grok-3",
    ],
    "gemini": [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-pro-latest",
        "gemini-flash-latest",
    ],
    "perplexity": [
        "sonar-pro",
        "sonar",
        "sonar-reasoning",
    ],
}


def get_recommended(make: str) -> list[str]:
    """Return the curated recommendation list for *make* (empty list if unknown)."""
    return list(RECOMMENDED_MODELS.get(make, []))


def get_recommended_default(make: str) -> str | None:
    """Return the *first* recommended model id for *make*, or ``None``."""
    rec = RECOMMENDED_MODELS.get(make)
    return rec[0] if rec else None

