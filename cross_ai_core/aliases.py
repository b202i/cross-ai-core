"""
cross_ai_core.aliases — alias layer for multi-model support (CAC-10).

An *alias* is a user-friendly name (e.g. ``"anthropic-opus"``) that resolves to
a (make, model) pair.  The alias layer lets callers reference more than one
model per provider without changing the on-disk container schema or the
``--ai`` CLI surface — every existing make string ("xai", "anthropic", …) is
auto-registered as an alias for itself with ``model=None`` (= handler default).

Storage
-------
Aliases live in ``~/.cross_ai_models.json``::

    {
      "anthropic-opus":   {"make": "anthropic", "model": "claude-opus-4-5"},
      "anthropic-sonnet": {"make": "anthropic", "model": "claude-sonnet-4-5"},
      "openai-mini":      {"make": "openai",    "model": "gpt-4o-mini"}
    }

Override the path with ``CROSS_AI_ALIASES_FILE``.  When the file is absent or
malformed, the registry falls back to "one alias per built-in make" — exactly
today's behaviour.

Resolution
----------
``resolve_alias(alias) -> AliasSpec`` returns ``(make, model)``.  ``model`` may
be ``None``, meaning "use the handler's compiled-in default".

Collisions
----------
A user-defined alias that *shadows* a built-in make name with a *different*
model is rejected at load time — silent override would break callers who think
``"anthropic"`` always means the handler default.
"""
from __future__ import annotations

import difflib
import json
import os
from collections import OrderedDict
from typing import NamedTuple


class AliasSpec(NamedTuple):
    make: str
    model: str | None


# Filled by _load_aliases() at import time; consumers should call get_aliases()
# rather than touching this directly.
_AI_ALIASES: "OrderedDict[str, AliasSpec]" = OrderedDict()
_ALIAS_LOAD_ERROR: str | None = None


def _aliases_file_path() -> str:
    """Path to the alias JSON file.  Override with ``CROSS_AI_ALIASES_FILE``."""
    override = os.environ.get("CROSS_AI_ALIASES_FILE", "").strip()
    if override:
        return os.path.expanduser(override)
    return os.path.expanduser("~/.cross_ai_models.json")


def _seed_built_ins(registry: "OrderedDict[str, AliasSpec]") -> None:
    """Auto-register one alias per built-in make → ``(make, None)``.

    Imported lazily to avoid a circular import (ai_handler imports this module).
    """
    from .ai_handler import AI_LIST  # local import: ai_handler imports us
    for make in AI_LIST:
        registry[make] = AliasSpec(make=make, model=None)


def _load_aliases() -> None:
    """Populate ``_AI_ALIASES`` from disk + built-in seeds.

    Order of operations:
      1. Seed every built-in make as a self-alias with ``model=None``.
      2. If the file exists and parses, merge user-defined aliases on top.
         User definitions are validated:
           - ``make`` must be a known provider.
           - An alias whose key matches a built-in make may only point at
             that same make with ``model=None`` (silent self-alias).  Any
             other definition with that name is a *collision* and is
             rejected — load fails loudly so the user fixes the file.
      3. On any error (missing file, malformed JSON, collision) — fall back
         to the seeded built-ins and store the error string for diagnostics.
    """
    global _ALIAS_LOAD_ERROR
    _ALIAS_LOAD_ERROR = None
    _AI_ALIASES.clear()
    _seed_built_ins(_AI_ALIASES)

    path = _aliases_file_path()
    if not os.path.isfile(path):
        return

    try:
        with open(path) as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _ALIAS_LOAD_ERROR = f"Could not read {path}: {exc}"
        return

    if not isinstance(raw, dict):
        _ALIAS_LOAD_ERROR = f"{path}: top-level value must be a JSON object."
        return

    from .ai_handler import AI_HANDLER_REGISTRY  # known makes

    # Two-pass: validate first, mutate second, so a collision aborts cleanly.
    validated: list[tuple[str, AliasSpec]] = []
    for alias, spec in raw.items():
        if not isinstance(alias, str) or not alias:
            _ALIAS_LOAD_ERROR = f"{path}: alias keys must be non-empty strings."
            return
        if not isinstance(spec, dict):
            _ALIAS_LOAD_ERROR = (
                f"{path}: value for {alias!r} must be an object with 'make' and 'model'."
            )
            return
        make = spec.get("make")
        model = spec.get("model")  # may legitimately be None
        if not isinstance(make, str) or make not in AI_HANDLER_REGISTRY:
            _ALIAS_LOAD_ERROR = (
                f"{path}: alias {alias!r} has unknown make {make!r}. "
                f"Known makes: {sorted(AI_HANDLER_REGISTRY)}"
            )
            return
        if model is not None and not isinstance(model, str):
            _ALIAS_LOAD_ERROR = (
                f"{path}: alias {alias!r} 'model' must be a string or null."
            )
            return
        # Collision check: user-defined alias name == built-in make name.
        # Allowed only when it resolves identically (same make, model=None).
        if alias in _AI_ALIASES and (
            _AI_ALIASES[alias].make != make or _AI_ALIASES[alias].model != model
        ):
            _ALIAS_LOAD_ERROR = (
                f"{path}: alias {alias!r} would shadow built-in make {alias!r} "
                f"with a different model ({model!r}). "
                f"Pick a different alias name (e.g. {make}-{(model or 'custom')!r})."
            )
            return
        validated.append((alias, AliasSpec(make=make, model=model)))

    for alias, spec in validated:
        _AI_ALIASES[alias] = spec


def reload_aliases() -> None:
    """Re-read ``~/.cross_ai_models.json`` from disk.

    Useful for tests and for ``st-admin`` after it edits the file.
    """
    _load_aliases()


def get_aliases() -> "OrderedDict[str, AliasSpec]":
    """Return the live alias registry (a reference — do not mutate)."""
    return _AI_ALIASES


def get_alias_load_error() -> str | None:
    """Return the last load error message, or ``None`` if loading succeeded."""
    return _ALIAS_LOAD_ERROR


def did_you_mean(bad: str, candidates) -> str | None:
    """Return the closest candidate to *bad*, or ``None`` if no good match."""
    matches = difflib.get_close_matches(bad, list(candidates), n=1, cutoff=0.6)
    return matches[0] if matches else None


def resolve_alias(alias: str) -> AliasSpec:
    """Return the (make, model) pair for *alias*.

    Resolution:
      1. Look up *alias* in the loaded registry.
      2. If absent, check whether *alias* is a registered make in
         ``AI_HANDLER_REGISTRY`` (covers test-registered mock providers and
         any provider added after import time) — if so, return a transient
         self-alias.
      3. Otherwise raise ``ValueError`` with a typo suggestion.
    """
    spec = _AI_ALIASES.get(alias)
    if spec is not None:
        return spec
    # Late-registered provider (e.g. a test mock) — accept as self-alias.
    from .ai_handler import AI_HANDLER_REGISTRY
    if alias in AI_HANDLER_REGISTRY:
        return AliasSpec(make=alias, model=None)
    suggestion = did_you_mean(alias, _AI_ALIASES.keys())
    hint = f" Did you mean {suggestion!r}?" if suggestion else ""
    # Keep the legacy "Unsupported AI" prefix so callers / tests that grep on
    # it continue to work.
    raise ValueError(
        f"Unsupported AI model: {alias!r}.{hint} "
        f"Known: {list(_AI_ALIASES.keys())}"
    )


def get_rate_limit_group(alias: str) -> tuple[str, int]:
    """Return ``(group_key, concurrency_cap)`` for *alias*.

    ``group_key`` is the resolved ``make`` — every alias sharing a make shares
    the same rate-limit group, so callers can key a semaphore on the group key
    to prevent multiple aliases from blowing through one provider's API quota.
    """
    from .ai_handler import get_rate_limit_concurrency
    make, _ = resolve_alias(alias)
    return make, get_rate_limit_concurrency(make)


# Load aliases at import time.  Done last so the helpers above are defined.
_load_aliases()

