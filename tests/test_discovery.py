"""
tests/test_discovery.py — CAC-10h: model discovery + cache + fallback.

Coverage:
    ModelInfo round-trip (to_json / from_json)
    Per-provider list filtering — openai (deny-list), anthropic, xai, gemini, perplexity
    7-day cache TTL — fresh hit, stale miss, refresh=True bypass
    Cache disable env var — CROSS_NO_MODELS_CACHE
    Cache file corruption recovery
    Graceful degradation — provider raises → curated fallback returned
    Annotation — is_recommended / is_default flags from RECOMMENDED_MODELS
    Sort order — recommended first (in curated order), then newest-first
    Splice-in — recommended ids missing from API are appended
    Public exports — get_available_models, ModelInfo, RECOMMENDED_MODELS
"""
from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import cross_ai_core
from cross_ai_core import discovery
from cross_ai_core.discovery import (
    MODELS_CACHE_TTL_SECONDS,
    ModelInfo,
    _annotate,
    _list_anthropic_models,
    _list_gemini_models,
    _list_openai_models,
    _list_perplexity_models,
    _list_xai_models,
    _sort_models,
    get_available_models,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Redirect ``~/.cross_models_cache/`` to a per-test tmp dir."""
    monkeypatch.setenv("CROSS_MODELS_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("CROSS_NO_MODELS_CACHE", raising=False)
    yield


def _model_obj(model_id: str, created: int | None = None):
    """Build a SDK-like model object that exposes ``.id`` and ``.created``."""
    return SimpleNamespace(id=model_id, created=created)


def _gemini_model_obj(name: str, methods=("generateContent",)):
    return SimpleNamespace(name=name, supported_actions=list(methods))


# ---------------------------------------------------------------------------
# ModelInfo
# ---------------------------------------------------------------------------

class TestModelInfo:
    def test_round_trip(self):
        m = ModelInfo(id="gpt-4o", family="gpt-4", is_chat=True,
                      created_at=1700000000, is_default=True, is_recommended=True)
        round_tripped = ModelInfo.from_json(m.to_json())
        assert round_tripped == m

    def test_from_json_tolerates_missing_optional_fields(self):
        m = ModelInfo.from_json({"id": "claude-opus-4-5"})
        assert m.id == "claude-opus-4-5"
        assert m.is_chat is True
        assert m.is_default is False
        assert m.is_recommended is False


# ---------------------------------------------------------------------------
# Per-provider filters
# ---------------------------------------------------------------------------

class TestOpenAIDiscovery:
    def test_filters_deny_list_and_keeps_gpt(self):
        client = MagicMock()
        client.models.list.return_value = SimpleNamespace(data=[
            _model_obj("gpt-4o", 1700000000),
            _model_obj("gpt-4o-mini", 1700000001),
            _model_obj("text-embedding-3-large"),     # denied
            _model_obj("whisper-1"),                   # denied
            _model_obj("dall-e-3"),                    # denied
            _model_obj("ft:gpt-4o:my-org::abc123"),    # fine-tune
            _model_obj("o1-preview", 1690000000),
        ])
        out = _list_openai_models(client)
        ids = [m.id for m in out]
        assert "gpt-4o" in ids
        assert "gpt-4o-mini" in ids
        assert "o1-preview" in ids
        assert "text-embedding-3-large" not in ids
        assert "whisper-1" not in ids
        assert "dall-e-3" not in ids
        assert all("ft:" not in mid for mid in ids)


class TestAnthropicDiscovery:
    def test_keeps_claude_only(self):
        client = MagicMock()
        client.models.list.return_value = SimpleNamespace(data=[
            SimpleNamespace(id="claude-opus-4-5", created_at=1730000000),
            SimpleNamespace(id="claude-sonnet-4-5", created_at=1729000000),
            SimpleNamespace(id="not-a-claude", created_at=None),
        ])
        out = _list_anthropic_models(client)
        ids = [m.id for m in out]
        assert ids == ["claude-opus-4-5", "claude-sonnet-4-5"]


class TestXAIDiscovery:
    def test_keeps_grok_only(self):
        client = MagicMock()
        client.models.list.return_value = SimpleNamespace(data=[
            _model_obj("grok-4-1-fast-reasoning", 1730000000),
            _model_obj("grok-3", 1720000000),
            _model_obj("not-grok"),
        ])
        out = _list_xai_models(client)
        ids = [m.id for m in out]
        assert ids == ["grok-4-1-fast-reasoning", "grok-3"]


class TestGeminiDiscovery:
    def test_keeps_generate_content_models_and_strips_prefix(self):
        client = MagicMock()
        client.models.list.return_value = [
            _gemini_model_obj("models/gemini-2.5-flash"),
            _gemini_model_obj("models/gemini-2.5-pro"),
            _gemini_model_obj("models/text-embedding-004", methods=("embedContent",)),
            _gemini_model_obj("models/aqa", methods=("generateAnswer",)),
        ]
        out = _list_gemini_models(client)
        ids = [m.id for m in out]
        assert "gemini-2.5-flash" in ids
        assert "gemini-2.5-pro" in ids
        assert "text-embedding-004" not in ids
        assert "aqa" not in ids


class TestPerplexityDiscovery:
    def test_returns_empty_on_endpoint_error(self):
        client = MagicMock()
        client.models.list.side_effect = RuntimeError("404 not found")
        out = _list_perplexity_models(client)
        assert out == []

    def test_returns_models_when_present(self):
        client = MagicMock()
        client.models.list.return_value = SimpleNamespace(data=[
            _model_obj("sonar-pro"),
            _model_obj("sonar"),
        ])
        out = _list_perplexity_models(client)
        assert [m.id for m in out] == ["sonar-pro", "sonar"]


# ---------------------------------------------------------------------------
# Cache TTL
# ---------------------------------------------------------------------------

class TestCacheTTL:
    def test_fresh_cache_is_returned(self, monkeypatch):
        monkeypatch.setattr(discovery, "_DISCOVERERS", {"openai": MagicMock(side_effect=AssertionError("must not call"))})
        # First populate the cache directly.
        cached = [ModelInfo(id="gpt-4o")]
        discovery._write_cache("openai", cached)
        result = discovery._read_cache("openai")
        assert result is not None and [m.id for m in result] == ["gpt-4o"]

    def test_stale_cache_is_ignored(self, tmp_path):
        path = discovery._cache_path("openai")
        with open(path, "w") as f:
            json.dump({
                "make": "openai",
                "fetched_at": int(time.time()) - MODELS_CACHE_TTL_SECONDS - 60,
                "models": [{"id": "gpt-4o"}],
            }, f)
        assert discovery._read_cache("openai") is None

    def test_refresh_bypasses_cache(self):
        discovery._write_cache("openai", [ModelInfo(id="cached-only")])
        live_models = [ModelInfo(id="gpt-4o")]
        with patch.object(discovery, "_get_or_create_client", create=True), \
             patch.dict(discovery._DISCOVERERS, {"openai": MagicMock(return_value=live_models)}, clear=False), \
             patch("cross_ai_core.discovery.RECOMMENDED_MODELS", {"openai": ["gpt-4o"]}):
            # Need the lazy import path to succeed
            from cross_ai_core.ai_handler import AI_HANDLER_REGISTRY  # noqa: F401
            with patch("cross_ai_core.ai_handler._get_or_create_client", return_value=MagicMock()):
                result = get_available_models("openai", refresh=True)
        ids = [m.id for m in result]
        assert "gpt-4o" in ids
        assert "cached-only" not in ids


class TestCacheDisable:
    def test_no_cache_env_skips_read_and_write(self, monkeypatch):
        monkeypatch.setenv("CROSS_NO_MODELS_CACHE", "1")
        # write should be a no-op
        discovery._write_cache("openai", [ModelInfo(id="gpt-4o")])
        assert not os.path.isfile(discovery._cache_path("openai"))
        # read should always return None
        assert discovery._read_cache("openai") is None


class TestCacheCorruption:
    def test_corrupt_cache_file_is_deleted(self):
        path = discovery._cache_path("openai")
        with open(path, "w") as f:
            f.write("{not valid json")
        assert discovery._read_cache("openai") is None
        assert not os.path.isfile(path)


# ---------------------------------------------------------------------------
# Annotation + sort order
# ---------------------------------------------------------------------------

class TestAnnotation:
    def test_is_recommended_and_is_default_stamped(self):
        models = [ModelInfo(id="gpt-4o"), ModelInfo(id="gpt-4o-mini"), ModelInfo(id="gpt-3.5-turbo")]
        out = _annotate("openai", models)
        ids = {m.id: m for m in out}
        assert ids["gpt-4o"].is_recommended and ids["gpt-4o"].is_default
        assert ids["gpt-4o-mini"].is_recommended and not ids["gpt-4o-mini"].is_default
        assert not ids["gpt-3.5-turbo"].is_recommended


class TestSortOrder:
    def test_recommended_appear_first_in_curated_order(self):
        models = [
            ModelInfo(id="gpt-3.5-turbo", created_at=1600000000),
            ModelInfo(id="gpt-4o-mini"),
            ModelInfo(id="gpt-4o"),
            ModelInfo(id="some-other-gpt", created_at=1750000000),
        ]
        ordered = _sort_models("openai", models)
        ids = [m.id for m in ordered]
        # "gpt-4o" comes before "gpt-4o-mini" because of curated order.
        assert ids.index("gpt-4o") < ids.index("gpt-4o-mini")
        # Newer non-recommended sorts before older non-recommended.
        assert ids.index("some-other-gpt") < ids.index("gpt-3.5-turbo")


# ---------------------------------------------------------------------------
# End-to-end: get_available_models() with mocked discovery + handler client
# ---------------------------------------------------------------------------

class TestGetAvailableModels:
    def test_falls_back_to_recommendations_on_api_error(self):
        with patch.dict(discovery._DISCOVERERS,
                        {"openai": MagicMock(side_effect=RuntimeError("boom"))}, clear=False), \
             patch("cross_ai_core.ai_handler._get_or_create_client", return_value=MagicMock()):
            out = get_available_models("openai", refresh=True)
        ids = [m.id for m in out]
        # Curated openai list ships with gpt-4o first
        assert "gpt-4o" in ids
        assert any(m.is_default for m in out)

    def test_splices_recommended_missing_from_api(self):
        # Live API returns only one model; curated list has more.
        live = [ModelInfo(id="gpt-4o", created_at=1730000000)]
        with patch.dict(discovery._DISCOVERERS,
                        {"openai": MagicMock(return_value=live)}, clear=False), \
             patch("cross_ai_core.ai_handler._get_or_create_client", return_value=MagicMock()):
            out = get_available_models("openai", refresh=True)
        ids = [m.id for m in out]
        # Both API-reported and curated-but-missing ids should be present.
        assert "gpt-4o" in ids
        assert "gpt-4o-mini" in ids   # curated, missing from API → spliced

    def test_unknown_make_returns_empty(self):
        out = get_available_models("not-a-real-provider")
        assert out == []

    def test_caches_after_live_call(self):
        live = [ModelInfo(id="gpt-4o", created_at=1730000000)]
        with patch.dict(discovery._DISCOVERERS,
                        {"openai": MagicMock(return_value=live)}, clear=False), \
             patch("cross_ai_core.ai_handler._get_or_create_client", return_value=MagicMock()):
            get_available_models("openai", refresh=True)
        assert os.path.isfile(discovery._cache_path("openai"))


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

class TestPackageExports:
    def test_get_available_models_is_exported(self):
        assert "get_available_models" in cross_ai_core.__all__
        assert cross_ai_core.get_available_models is get_available_models

    def test_modelinfo_and_recommendations_exported(self):
        assert cross_ai_core.ModelInfo is ModelInfo
        assert isinstance(cross_ai_core.RECOMMENDED_MODELS, dict)
        assert "openai" in cross_ai_core.RECOMMENDED_MODELS

