"""
tests/test_aliases.py — CAC-10: alias registry + resolver tests.

Coverage:
    AliasSpec namedtuple shape
    _load_aliases — file absent (built-in seed), happy-path JSON, malformed JSON
    _load_aliases — collision rejection
    resolve_alias — exact, alias, did-you-mean, registered-make fallback
    did_you_mean — close match, no close match, exact match
    get_rate_limit_group — group key matches resolved make
    process_prompt — alias resolves to (make, model); _alias / _model stamped
    process_prompt — <ALIAS>_MODEL env var override; <MAKE>_MODEL fallback
    client cache — two aliases sharing a make share one client
    get_ai_list / get_default_ai — alias-aware
"""
import json
import os
from unittest.mock import MagicMock

import pytest

import cross_ai_core
from cross_ai_core.aliases import (
    AliasSpec,
    did_you_mean,
    get_alias_load_error,
    get_aliases,
    get_rate_limit_group,
    reload_aliases,
    resolve_alias,
)
from cross_ai_core.ai_handler import (
    AI_HANDLER_REGISTRY,
    AI_LIST,
    _client_cache,
    get_ai_list,
    get_ai_make,
    get_ai_model,
    get_default_ai,
    process_prompt,
    reset_client_cache,
)


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture
def alias_file(tmp_path, monkeypatch):
    """Point CROSS_AI_ALIASES_FILE at a fresh tmp file; reload aliases on entry/exit."""
    path = tmp_path / "cross_ai_models.json"
    monkeypatch.setenv("CROSS_AI_ALIASES_FILE", str(path))
    reload_aliases()
    yield path
    monkeypatch.delenv("CROSS_AI_ALIASES_FILE", raising=False)
    reload_aliases()


@pytest.fixture(autouse=True)
def _isolate_client_cache():
    """Every test gets a clean client cache."""
    reset_client_cache()
    yield
    reset_client_cache()


# ── AliasSpec ──────────────────────────────────────────────────────────────────

class TestAliasSpec:
    def test_spec_is_named_tuple(self):
        s = AliasSpec(make="anthropic", model="claude-opus-4-5")
        assert s.make == "anthropic"
        assert s.model == "claude-opus-4-5"
        assert s == ("anthropic", "claude-opus-4-5")

    def test_spec_model_can_be_none(self):
        s = AliasSpec(make="xai", model=None)
        assert s.model is None


# ── Loader ─────────────────────────────────────────────────────────────────────

class TestLoadAliases:
    def test_missing_file_seeds_built_ins(self, alias_file):
        # tmp file does not exist
        assert not alias_file.exists()
        reload_aliases()
        assert list(get_aliases().keys()) == AI_LIST
        for make in AI_LIST:
            assert get_aliases()[make] == AliasSpec(make=make, model=None)
        assert get_alias_load_error() is None

    def test_happy_path_user_aliases_loaded(self, alias_file):
        alias_file.write_text(json.dumps({
            "anthropic-opus":   {"make": "anthropic", "model": "claude-opus-4-5"},
            "anthropic-sonnet": {"make": "anthropic", "model": "claude-sonnet-4-5"},
        }))
        reload_aliases()
        keys = list(get_aliases().keys())
        # built-ins seeded first, user aliases appended in declaration order
        assert keys[:5] == AI_LIST
        assert "anthropic-opus" in keys
        assert "anthropic-sonnet" in keys
        assert get_aliases()["anthropic-opus"] == AliasSpec("anthropic", "claude-opus-4-5")

    def test_malformed_json_falls_back_to_built_ins(self, alias_file):
        alias_file.write_text("{not valid json")
        reload_aliases()
        # registry still has the built-ins
        assert list(get_aliases().keys()) == AI_LIST
        assert get_alias_load_error() is not None
        assert "Could not read" in get_alias_load_error()

    def test_unknown_make_rejected(self, alias_file):
        alias_file.write_text(json.dumps({
            "weird": {"make": "no_such_provider", "model": "x"},
        }))
        reload_aliases()
        # falls back to built-ins; load error is set
        assert "weird" not in get_aliases()
        assert get_alias_load_error() is not None
        assert "unknown make" in get_alias_load_error()

    def test_collision_with_built_in_make_rejected(self, alias_file):
        # Trying to redefine "anthropic" → claude-opus-4-5 must fail loud.
        alias_file.write_text(json.dumps({
            "anthropic": {"make": "anthropic", "model": "claude-opus-4-5"},
        }))
        reload_aliases()
        assert get_alias_load_error() is not None
        assert "shadow" in get_alias_load_error().lower()
        # Built-in self-alias preserved
        assert get_aliases()["anthropic"] == AliasSpec("anthropic", None)

    def test_silent_self_alias_allowed(self, alias_file):
        # Re-declaring "anthropic" with model=None is a no-op, not a collision.
        alias_file.write_text(json.dumps({
            "anthropic": {"make": "anthropic", "model": None},
        }))
        reload_aliases()
        assert get_alias_load_error() is None
        assert get_aliases()["anthropic"] == AliasSpec("anthropic", None)


# ── Resolver / did-you-mean ────────────────────────────────────────────────────

class TestResolveAlias:
    def test_built_in_make_resolves(self, alias_file):
        spec = resolve_alias("anthropic")
        assert spec == AliasSpec("anthropic", None)

    def test_user_alias_resolves(self, alias_file):
        alias_file.write_text(json.dumps({
            "opus": {"make": "anthropic", "model": "claude-opus-4-5"},
        }))
        reload_aliases()
        assert resolve_alias("opus") == AliasSpec("anthropic", "claude-opus-4-5")

    def test_unknown_alias_raises_with_suggestion(self, alias_file):
        with pytest.raises(ValueError, match="Did you mean 'anthropic'"):
            resolve_alias("antrhopic")  # typo

    def test_unknown_with_no_close_match_no_suggestion(self, alias_file):
        with pytest.raises(ValueError) as exc:
            resolve_alias("zzzzzz")
        assert "Did you mean" not in str(exc.value)

    def test_late_registered_make_resolves_as_self_alias(self, alias_file, monkeypatch):
        # Simulate a test mocking AI_HANDLER_REGISTRY at runtime.
        mock_handler = MagicMock()
        monkeypatch.setitem(AI_HANDLER_REGISTRY, "experimental", mock_handler)
        spec = resolve_alias("experimental")
        assert spec == AliasSpec("experimental", None)


class TestDidYouMean:
    def test_close_match(self):
        assert did_you_mean("anthorpic", ["anthropic", "openai", "xai"]) == "anthropic"

    def test_no_close_match(self):
        assert did_you_mean("zzzzzz", ["anthropic", "openai"]) is None

    def test_exact_match_returned(self):
        assert did_you_mean("openai", ["openai", "xai"]) == "openai"

    def test_empty_candidates(self):
        assert did_you_mean("anything", []) is None


# ── Rate-limit group ───────────────────────────────────────────────────────────

class TestRateLimitGroup:
    def test_group_key_is_make_for_built_in(self, alias_file):
        group, cap = get_rate_limit_group("anthropic")
        assert group == "anthropic"
        assert cap == 2  # CAC-5 default

    def test_two_aliases_same_make_share_group(self, alias_file):
        alias_file.write_text(json.dumps({
            "opus":   {"make": "anthropic", "model": "claude-opus-4-5"},
            "sonnet": {"make": "anthropic", "model": "claude-sonnet-4-5"},
        }))
        reload_aliases()
        g1, c1 = get_rate_limit_group("opus")
        g2, c2 = get_rate_limit_group("sonnet")
        assert g1 == g2 == "anthropic"
        assert c1 == c2 == 2


# ── process_prompt alias-awareness ─────────────────────────────────────────────

@pytest.fixture
def mock_handler(monkeypatch):
    """Register a mock provider as both 'mock_make' and a user alias for it."""
    handler = MagicMock()
    handler.get_payload.return_value = {"model": "mock-default"}
    handler.get_model.return_value = "mock-default"
    handler.get_cached_response.return_value = ({"text": "ok"}, False)
    monkeypatch.setitem(AI_HANDLER_REGISTRY, "mock_make", handler)
    yield handler
    AI_HANDLER_REGISTRY.pop("mock_make", None)


class TestProcessPromptAliasAware:
    def test_legacy_make_string_still_works(self, alias_file, mock_handler):
        result = process_prompt("mock_make", "hi", use_cache=False)
        assert result.response["_make"] == "mock_make"
        assert result.response["_alias"] == "mock_make"
        # No spec model, no override → handler default
        assert result.response["_model"] == "mock-default"

    def test_alias_resolves_to_make_and_model(self, alias_file, mock_handler):
        alias_file.write_text(json.dumps({
            "mock-fast": {"make": "mock_make", "model": "mock-fast-v1"},
        }))
        reload_aliases()
        result = process_prompt("mock-fast", "hi", use_cache=False)
        assert result.response["_make"] == "mock_make"
        assert result.response["_alias"] == "mock-fast"
        assert result.response["_model"] == "mock-fast-v1"
        assert result.model == "mock-fast-v1"

    def test_explicit_model_kwarg_wins(self, alias_file, mock_handler):
        alias_file.write_text(json.dumps({
            "mock-fast": {"make": "mock_make", "model": "mock-fast-v1"},
        }))
        reload_aliases()
        result = process_prompt("mock-fast", "hi", use_cache=False, model="overridden")
        assert result.response["_model"] == "overridden"

    def test_alias_env_var_overrides_spec(self, alias_file, mock_handler, monkeypatch):
        alias_file.write_text(json.dumps({
            "mock-fast": {"make": "mock_make", "model": "mock-fast-v1"},
        }))
        reload_aliases()
        monkeypatch.setenv("MOCK_FAST_MODEL", "from-alias-env")
        result = process_prompt("mock-fast", "hi", use_cache=False)
        assert result.response["_model"] == "from-alias-env"

    def test_make_env_var_legacy_fallback(self, alias_file, mock_handler, monkeypatch):
        # No alias defined for mock_make beyond self-seed; MOCK_MAKE_MODEL fires.
        monkeypatch.setenv("MOCK_MAKE_MODEL", "from-make-env")
        result = process_prompt("mock_make", "hi", use_cache=False)
        assert result.response["_model"] == "from-make-env"


# ── Client cache: aliases share clients ────────────────────────────────────────

class TestAliasesShareClient:
    def test_two_aliases_same_make_share_client(self, alias_file, monkeypatch):
        # Register a real-looking handler whose get_client() counts construction.
        construction_count = [0]
        sentinel_client = object()
        handler = MagicMock()
        handler.get_payload.return_value = {"model": "x"}
        handler.get_model.return_value = "x"
        handler.get_cached_response.side_effect = lambda *a, **kw: (
            (kw["client_factory"](), True) if False else
            (kw["client_factory"]() and {"text": "ok"}, False)
        )
        def _make_client():
            construction_count[0] += 1
            return sentinel_client
        handler.get_client.side_effect = _make_client
        monkeypatch.setitem(AI_HANDLER_REGISTRY, "mock_share", handler)

        alias_file.write_text(json.dumps({
            "share-a": {"make": "mock_share", "model": "a"},
            "share-b": {"make": "mock_share", "model": "b"},
        }))
        reload_aliases()
        process_prompt("share-a", "hi", use_cache=False)
        process_prompt("share-b", "hi", use_cache=False)
        # Both calls should share one client (cache keyed on make=mock_share)
        assert construction_count[0] == 1
        assert _client_cache.get("mock_share") is sentinel_client

        AI_HANDLER_REGISTRY.pop("mock_share", None)


# ── get_ai_list / get_default_ai alias-awareness ───────────────────────────────

class TestListAndDefault:
    def test_get_ai_list_returns_aliases(self, alias_file):
        alias_file.write_text(json.dumps({
            "opus": {"make": "anthropic", "model": "claude-opus-4-5"},
        }))
        reload_aliases()
        lst = get_ai_list()
        assert "opus" in lst
        # built-ins still present
        for m in AI_LIST:
            assert m in lst

    def test_get_default_ai_accepts_alias(self, alias_file, monkeypatch):
        alias_file.write_text(json.dumps({
            "opus": {"make": "anthropic", "model": "claude-opus-4-5"},
        }))
        reload_aliases()
        monkeypatch.setenv("DEFAULT_AI", "opus")
        assert get_default_ai() == "opus"

    def test_get_default_ai_unknown_falls_back(self, alias_file, monkeypatch):
        monkeypatch.setenv("DEFAULT_AI", "no-such-alias")
        # Falls back to first registry entry (xai)
        assert get_default_ai() == "xai"

    def test_get_ai_make_returns_resolved_make(self, alias_file):
        alias_file.write_text(json.dumps({
            "opus": {"make": "anthropic", "model": "claude-opus-4-5"},
        }))
        reload_aliases()
        # get_ai_make delegates to handler.get_make(), which returns the
        # provider's canonical make string.
        assert get_ai_make("opus") == get_ai_make("anthropic")

    def test_get_ai_model_alias_resolution(self, alias_file):
        alias_file.write_text(json.dumps({
            "opus": {"make": "anthropic", "model": "claude-opus-4-5"},
        }))
        reload_aliases()
        assert get_ai_model("opus") == "claude-opus-4-5"


# ── Public API exposure ────────────────────────────────────────────────────────

class TestPublicExports:
    def test_aliases_module_exports_in_package(self):
        for name in (
            "AliasSpec", "resolve_alias", "get_aliases",
            "get_alias_load_error", "did_you_mean", "get_rate_limit_group",
            "reload_aliases", "get_ai_make_list",
        ):
            assert hasattr(cross_ai_core, name), f"{name} not exported"

