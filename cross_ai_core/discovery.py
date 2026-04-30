"""
cross_ai_core.discovery — provider model discovery (CAC-10h).

A single entry point — :func:`get_available_models(make, refresh=False)` —
returns a list of :class:`ModelInfo` describing the models that the caller's
API key can actually reach for *make*.

Caching
-------
Each make's last fetched list is cached as JSON in
``~/.cross_models_cache/<make>.json`` for **7 days**.  Override the directory
with ``CROSS_MODELS_CACHE_DIR``; bypass caching entirely by setting
``CROSS_NO_MODELS_CACHE=1`` in the environment.  The 7-day TTL is short
enough that newly released flagship models surface within a week without
manual intervention, and long enough that the wizard never blocks on a
network call during normal use.

Graceful degradation
--------------------
If the provider SDK call fails for any reason (network error, bad key,
unsupported method, throttling) the discovery layer falls back to the
curated :data:`cross_ai_core.recommendations.RECOMMENDED_MODELS` list for
that make.  Discovery is therefore **never** allowed to raise to the caller.

Recommendation metadata
-----------------------
Every returned :class:`ModelInfo` carries ``is_recommended`` (the model id
appears in :data:`RECOMMENDED_MODELS` for that make) and ``is_default`` (the
model id is the *first* entry in that list — the curated recommended
default).  These are advisory flags only; the alias wizard uses them for
ordering, never for filtering.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from .recommendations import RECOMMENDED_MODELS

_LOG = logging.getLogger(__name__)

#: Cache TTL for discovered model lists.  7 days, in seconds.
MODELS_CACHE_TTL_SECONDS: int = 7 * 24 * 60 * 60


@dataclass
class ModelInfo:
    """Lightweight metadata for one model offered by a provider."""

    id: str
    family: str = ""           # e.g. "gpt-4", "claude-3", "gemini-2.5"
    is_chat: bool = True
    created_at: int | None = None  # unix epoch seconds, if the SDK reports it
    is_default: bool = False
    is_recommended: bool = False

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "ModelInfo":
        # Tolerate extra keys from a future schema bump.
        return cls(
            id=data["id"],
            family=data.get("family", ""),
            is_chat=data.get("is_chat", True),
            created_at=data.get("created_at"),
            is_default=data.get("is_default", False),
            is_recommended=data.get("is_recommended", False),
        )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_dir() -> str:
    override = os.environ.get("CROSS_MODELS_CACHE_DIR", "").strip()
    path = os.path.expanduser(override) if override else os.path.expanduser("~/.cross_models_cache")
    os.makedirs(path, exist_ok=True)
    return path


def _cache_path(make: str) -> str:
    return os.path.join(_cache_dir(), f"{make}.json")


def _cache_disabled() -> bool:
    return os.environ.get("CROSS_NO_MODELS_CACHE", "").strip() not in ("", "0")


def _read_cache(make: str) -> list[ModelInfo] | None:
    """Return the cached model list for *make* if present and fresh, else ``None``."""
    if _cache_disabled():
        return None
    path = _cache_path(make)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            blob = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.debug("models cache read failed for %s: %s", make, exc)
        try:
            os.remove(path)
        except OSError:
            pass
        return None
    fetched_at = blob.get("fetched_at", 0)
    if (time.time() - fetched_at) > MODELS_CACHE_TTL_SECONDS:
        return None
    items = blob.get("models", [])
    try:
        return [ModelInfo.from_json(item) for item in items]
    except (KeyError, TypeError) as exc:
        _LOG.debug("models cache parse failed for %s: %s", make, exc)
        return None


def _write_cache(make: str, models: list[ModelInfo]) -> None:
    if _cache_disabled():
        return
    path = _cache_path(make)
    tmp = path + ".tmp"
    payload = {
        "make": make,
        "fetched_at": int(time.time()),
        "models": [m.to_json() for m in models],
    }
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except OSError as exc:
        _LOG.debug("models cache write failed for %s: %s", make, exc)
        try:
            os.remove(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Per-provider discovery
# ---------------------------------------------------------------------------

# Deny-list substrings for OpenAI: filters out non-chat / non-current models so
# the wizard isn't drowned in fine-tunes, audio, image, or embedding ids.
_OPENAI_DENY_SUBSTRINGS = (
    "embedding", "whisper", "tts", "dall-e", "dalle",
    "babbage", "davinci", "curie", "ada", "moderation",
    "audio", "image", "realtime", "search",
)


def _family_from_id(model_id: str) -> str:
    """Best-effort family extractor — strips date suffixes and minor versions."""
    parts = model_id.split("-")
    if len(parts) <= 2:
        return model_id
    # Drop trailing date-like or numeric suffix segments.
    while parts and (parts[-1].isdigit() or len(parts[-1]) >= 8 and parts[-1][:4].isdigit()):
        parts.pop()
    return "-".join(parts) if parts else model_id


def _list_openai_models(client) -> list[ModelInfo]:
    response = client.models.list()
    items = getattr(response, "data", None) or response  # SDK may return list-like
    out: list[ModelInfo] = []
    for item in items:
        mid = getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None)
        if not mid:
            continue
        low = mid.lower()
        if "gpt" not in low and "o1" not in low and "o3" not in low:
            continue
        if any(bad in low for bad in _OPENAI_DENY_SUBSTRINGS):
            continue
        if "ft:" in low or low.startswith("ft-"):
            continue  # fine-tunes
        created = getattr(item, "created", None)
        if created is None and isinstance(item, dict):
            created = item.get("created")
        out.append(ModelInfo(id=mid, family=_family_from_id(mid), created_at=created))
    return out


def _list_anthropic_models(client) -> list[ModelInfo]:
    # SDK ≥ 0.34 exposes .models.list()
    response = client.models.list()
    items = getattr(response, "data", None) or response
    out: list[ModelInfo] = []
    for item in items:
        mid = getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None)
        if not mid or not mid.startswith("claude-"):
            continue
        created = getattr(item, "created_at", None)
        if created is None and isinstance(item, dict):
            created = item.get("created_at")
        # created_at may be an ISO string — leave as-is for ordering best-effort
        out.append(ModelInfo(id=mid, family=_family_from_id(mid), created_at=_to_epoch(created)))
    return out


def _list_xai_models(client) -> list[ModelInfo]:
    response = client.models.list()
    items = getattr(response, "data", None) or response
    out: list[ModelInfo] = []
    for item in items:
        mid = getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None)
        if not mid or not mid.startswith("grok-"):
            continue
        created = getattr(item, "created", None)
        if created is None and isinstance(item, dict):
            created = item.get("created")
        out.append(ModelInfo(id=mid, family=_family_from_id(mid), created_at=created))
    return out


def _list_gemini_models(client) -> list[ModelInfo]:
    # google-genai client exposes .models.list() returning iterables of Model objects
    listing = client.models.list()
    out: list[ModelInfo] = []
    for item in listing:
        name = getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else None)
        if not name:
            continue
        # Filter to models that support text generation
        methods = (
            getattr(item, "supported_actions", None)
            or getattr(item, "supported_generation_methods", None)
            or (item.get("supported_actions") if isinstance(item, dict) else None)
            or []
        )
        if methods and "generateContent" not in methods:
            continue
        # Strip the "models/" prefix the API returns
        mid = name.split("/", 1)[1] if name.startswith("models/") else name
        out.append(ModelInfo(id=mid, family=_family_from_id(mid)))
    return out


def _list_perplexity_models(client) -> list[ModelInfo]:
    """Perplexity's ``/models`` endpoint is sparse — fall back to curated list.

    The SDK is OpenAI-compatible but the live endpoint sometimes returns 404 or
    an empty payload.  Try once, accept whatever comes back, and let the
    fallback layer fill in the curated recommendations if discovery yields
    nothing.
    """
    try:
        response = client.models.list()
    except Exception:  # noqa: BLE001 — handled by the outer fallback path
        return []
    items = getattr(response, "data", None) or response
    out: list[ModelInfo] = []
    for item in items:
        mid = getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None)
        if not mid:
            continue
        out.append(ModelInfo(id=mid, family=_family_from_id(mid)))
    return out


def _to_epoch(value) -> int | None:
    """Best-effort conversion of an SDK timestamp into unix epoch seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        # ISO-8601 best effort; fall back to None.
        from datetime import datetime
        try:
            return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
        except (ValueError, OSError):
            return None
    return None


# Wire each make to (handler-class-name → list-fn).  Lazy-imported in the
# dispatcher so that pulling in :func:`get_available_models` does not pay the
# cost of constructing a client.
_DISCOVERERS: dict[str, Callable[[object], list[ModelInfo]]] = {
    "openai":     _list_openai_models,
    "anthropic":  _list_anthropic_models,
    "xai":        _list_xai_models,
    "gemini":     _list_gemini_models,
    "perplexity": _list_perplexity_models,
}


def _fallback_from_recommendations(make: str) -> list[ModelInfo]:
    """Return curated recommendations as :class:`ModelInfo` items."""
    return [
        ModelInfo(id=mid, family=_family_from_id(mid), is_recommended=True)
        for mid in RECOMMENDED_MODELS.get(make, [])
    ]


def _annotate(make: str, models: list[ModelInfo]) -> list[ModelInfo]:
    """Stamp ``is_recommended`` / ``is_default`` from :data:`RECOMMENDED_MODELS`."""
    rec = RECOMMENDED_MODELS.get(make, [])
    rec_set = set(rec)
    default_id = rec[0] if rec else None
    for m in models:
        m.is_recommended = m.id in rec_set
        m.is_default = (m.id == default_id)
    return models


def _sort_models(make: str, models: list[ModelInfo]) -> list[ModelInfo]:
    """Recommended (in curated order) first; everything else newest-first."""
    rec = RECOMMENDED_MODELS.get(make, [])
    rec_index = {mid: i for i, mid in enumerate(rec)}
    def key(m: ModelInfo):
        if m.id in rec_index:
            return (0, rec_index[m.id], 0)
        # Negative created_at so newest comes first (None sorts last).
        ts = -(m.created_at or 0) if m.created_at else 0
        return (1, ts, m.id)
    return sorted(models, key=key)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_available_models(make: str, refresh: bool = False) -> list[ModelInfo]:
    """Return the available model list for *make*, sorted recommended-first.

    Parameters
    ----------
    make : str
        Provider name (``"openai"``, ``"anthropic"``, …).  Must be one of the
        keys of :data:`cross_ai_core.discovery._DISCOVERERS`.
    refresh : bool
        Force a fresh provider call, ignoring any cached entry.

    Returns
    -------
    list[ModelInfo]
        Always non-empty unless the make has no curated recommendations *and*
        the provider call returned nothing (an empty list is then returned).
        Never raises — network and SDK errors fall back to the curated list.
    """
    if make not in _DISCOVERERS:
        # Unknown make — return curated fallback (likely empty).
        return _annotate(make, _fallback_from_recommendations(make))

    # Try the cache first (unless refresh).
    if not refresh:
        cached = _read_cache(make)
        if cached is not None:
            return _sort_models(make, _annotate(make, cached))

    # Live provider call — never let it raise.
    try:
        from .ai_handler import _get_or_create_client, AI_HANDLER_REGISTRY
        handler_cls = AI_HANDLER_REGISTRY[make]
        client = _get_or_create_client(handler_cls, make)
        models = _DISCOVERERS[make](client)
    except Exception as exc:  # noqa: BLE001 — discovery is best-effort
        _LOG.info("model discovery failed for %s: %s — using curated fallback", make, exc)
        models = []

    if not models:
        models = _fallback_from_recommendations(make)
    else:
        # Splice in any recommended ids the live API hid (so wizard always
        # shows the curated set even on patchy SDKs).
        seen = {m.id for m in models}
        for mid in RECOMMENDED_MODELS.get(make, []):
            if mid not in seen:
                models.append(ModelInfo(id=mid, family=_family_from_id(mid)))

    models = _annotate(make, models)
    _write_cache(make, models)
    return _sort_models(make, models)

