"""
tests/test_ai_handler.py — Tests for cross_ai_core.ai_handler

Coverage:
    AI_LIST / AI_HANDLER_REGISTRY  — completeness
    get_default_ai                 — env resolution, fallback, unknown value
    get_ai_list                    — returns correct list
    check_api_key                  — present/missing/unknown provider
    AIResponse                     — tuple unpacking, indexing, was_cached
    process_prompt                 — dispatches to handler; returns AIResponse
    process_prompt error path      — ValueError for unknown provider
    get_content / put_content      — dispatch wrappers
    get_usage                      — returns zero dict for unknown provider

All tests mock provider SDK calls — no real API calls are made.
"""
import os
from unittest.mock import MagicMock, patch

import pytest

from cross_ai_core.ai_handler import (
    AI_HANDLER_REGISTRY,
    AI_LIST,
    AIResponse,
    _API_KEY_ENV_VARS,
    check_api_key,
    get_ai_list,
    get_content,
    get_default_ai,
    get_usage,
    process_prompt,
    put_content,
)


# ── Registry / list completeness ───────────────────────────────────────────────

class TestRegistryCompleteness:
    EXPECTED = {"xai", "anthropic", "openai", "perplexity", "gemini"}

    def test_ai_list_contains_all_providers(self):
        assert set(AI_LIST) == self.EXPECTED

    def test_registry_contains_all_providers(self):
        assert set(AI_HANDLER_REGISTRY.keys()) == self.EXPECTED

    def test_registry_values_are_classes(self):
        for name, cls in AI_HANDLER_REGISTRY.items():
            assert isinstance(cls, type), f"{name} handler should be a class"

    def test_api_key_env_vars_covers_all_providers(self):
        assert set(_API_KEY_ENV_VARS.keys()) == self.EXPECTED


# ── get_default_ai ─────────────────────────────────────────────────────────────

class TestGetDefaultAi:
    def test_returns_first_in_list_when_no_env(self, monkeypatch):
        monkeypatch.delenv("DEFAULT_AI", raising=False)
        assert get_default_ai() == AI_LIST[0]

    def test_returns_env_var_when_valid(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_AI", "gemini")
        assert get_default_ai() == "gemini"

    def test_falls_back_when_env_var_is_unknown_provider(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_AI", "bogus_provider")
        assert get_default_ai() == AI_LIST[0]

    def test_falls_back_when_env_var_is_empty(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_AI", "")
        assert get_default_ai() == AI_LIST[0]

    def test_case_sensitive(self, monkeypatch):
        # "Gemini" (capital) is not a valid key — should fall back
        monkeypatch.setenv("DEFAULT_AI", "Gemini")
        assert get_default_ai() == AI_LIST[0]


# ── get_ai_list ────────────────────────────────────────────────────────────────

class TestGetAiList:
    def test_returns_list(self):
        result = get_ai_list()
        assert isinstance(result, list)

    def test_matches_ai_list_constant(self):
        assert get_ai_list() == AI_LIST


# ── check_api_key ──────────────────────────────────────────────────────────────

class TestCheckApiKey:
    def test_returns_true_when_key_present(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-abc123")
        assert check_api_key("gemini") is True

    def test_returns_false_when_key_missing(self, monkeypatch, capsys):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = check_api_key("gemini", paths_checked=[])
        assert result is False

    def test_missing_key_prints_diagnostic(self, monkeypatch, capsys):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        check_api_key("openai", paths_checked=["/fake/.env"])
        out = capsys.readouterr().out
        assert "OPENAI_API_KEY" in out
        assert "openai" in out

    def test_unknown_provider_returns_true(self, monkeypatch):
        # Unknown providers pass through — let the SDK surface the real error
        result = check_api_key("unknown_provider")
        assert result is True

    def test_whitespace_only_key_treated_as_missing(self, monkeypatch, capsys):
        monkeypatch.setenv("XAI_API_KEY", "   ")
        result = check_api_key("xai", paths_checked=[])
        assert result is False


# ── AIResponse ─────────────────────────────────────────────────────────────────

class TestAIResponse:
    def _make_response(self, was_cached=False):
        return AIResponse(
            payload={"prompt": "test"},
            client=object(),
            response={"text": "hello"},
            model="test-model",
            was_cached=was_cached,
        )

    def test_was_cached_false(self):
        r = self._make_response(was_cached=False)
        assert r.was_cached is False

    def test_was_cached_true(self):
        r = self._make_response(was_cached=True)
        assert r.was_cached is True

    def test_tuple_unpacking(self):
        r = self._make_response()
        payload, client, response, model = r
        assert payload == {"prompt": "test"}
        assert response == {"text": "hello"}
        assert model == "test-model"

    def test_index_access(self):
        r = self._make_response()
        assert r[2] == {"text": "hello"}
        assert r[3] == "test-model"

    def test_len_is_four(self):
        assert len(self._make_response()) == 4

    def test_attributes_accessible(self):
        r = self._make_response()
        assert r.payload == {"prompt": "test"}
        assert r.model == "test-model"


# ── process_prompt ─────────────────────────────────────────────────────────────

class TestProcessPrompt:
    """process_prompt() is tested with a fully mocked handler class."""

    def _mock_handler(self, response_data=None, was_cached=False):
        """Return a mock handler class that simulates a provider."""
        mock_cls = MagicMock()
        mock_cls.get_payload.return_value = {"mock": "payload"}
        mock_cls.get_client.return_value = MagicMock()
        mock_cls.get_cached_response.return_value = (
            response_data or {"choices": [{"message": {"content": "mocked"}}]},
            was_cached,
        )
        mock_cls.get_model.return_value = "mock-model-1"
        return mock_cls

    def test_returns_ai_response(self):
        mock_cls = self._mock_handler()
        with patch.dict(AI_HANDLER_REGISTRY, {"mock_ai": mock_cls}):
            result = process_prompt("mock_ai", "hello", verbose=False, use_cache=True)
        assert isinstance(result, AIResponse)

    def test_response_was_cached_flag_false(self):
        mock_cls = self._mock_handler(was_cached=False)
        with patch.dict(AI_HANDLER_REGISTRY, {"mock_ai": mock_cls}):
            result = process_prompt("mock_ai", "hello", verbose=False, use_cache=True)
        assert result.was_cached is False

    def test_response_was_cached_flag_true(self):
        mock_cls = self._mock_handler(was_cached=True)
        with patch.dict(AI_HANDLER_REGISTRY, {"mock_ai": mock_cls}):
            result = process_prompt("mock_ai", "hello", verbose=False, use_cache=True)
        assert result.was_cached is True

    def test_passes_prompt_to_get_payload(self):
        mock_cls = self._mock_handler()
        with patch.dict(AI_HANDLER_REGISTRY, {"mock_ai": mock_cls}):
            process_prompt("mock_ai", "my prompt", verbose=False, use_cache=True)
        mock_cls.get_payload.assert_called_once_with("my prompt")

    def test_passes_use_cache_to_get_cached_response(self):
        mock_cls = self._mock_handler()
        with patch.dict(AI_HANDLER_REGISTRY, {"mock_ai": mock_cls}):
            process_prompt("mock_ai", "p", verbose=False, use_cache=False)
        _, _, kwargs_or_args = (
            mock_cls.get_cached_response.call_args[0],
            mock_cls.get_cached_response.call_args[0],
            mock_cls.get_cached_response.call_args,
        )
        # use_cache=False must have been passed (positional or keyword)
        call_args = mock_cls.get_cached_response.call_args
        all_args = list(call_args.args) + list(call_args.kwargs.values())
        assert False in all_args

    def test_raises_value_error_for_unknown_provider(self):
        with pytest.raises(ValueError, match="Unsupported AI model"):
            process_prompt("totally_unknown", "hello", verbose=False, use_cache=False)

    def test_backward_compat_tuple_unpack(self):
        mock_cls = self._mock_handler()
        with patch.dict(AI_HANDLER_REGISTRY, {"mock_ai": mock_cls}):
            result = process_prompt("mock_ai", "hi", verbose=False, use_cache=False)
        payload, client, response, model = result   # must not raise
        assert model == "mock-model-1"


# ── get_content / put_content ──────────────────────────────────────────────────

class TestContentWrappers:
    def test_get_content_dispatches_to_handler(self):
        mock_cls = MagicMock()
        mock_cls.get_content.return_value = "extracted text"
        with patch.dict(AI_HANDLER_REGISTRY, {"mock_ai": mock_cls}):
            result = get_content("mock_ai", {"raw": "response"})
        assert result == "extracted text"
        mock_cls.get_content.assert_called_once_with({"raw": "response"})

    def test_put_content_dispatches_to_handler(self):
        mock_cls = MagicMock()
        mock_cls.put_content.return_value = {"updated": True}
        with patch.dict(AI_HANDLER_REGISTRY, {"mock_ai": mock_cls}):
            result = put_content("mock_ai", "report text", {"raw": "response"})
        assert result == {"updated": True}

    def test_get_content_raises_for_unknown_provider(self):
        with pytest.raises(ValueError):
            get_content("no_such_provider", {})


# ── get_usage ──────────────────────────────────────────────────────────────────

class TestGetUsage:
    def test_unknown_provider_returns_zeros(self):
        result = get_usage("no_such_provider", {})
        assert result == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def test_dispatches_to_handler(self):
        mock_cls = MagicMock()
        mock_cls.get_usage.return_value = {
            "input_tokens": 10, "output_tokens": 20, "total_tokens": 30
        }
        with patch.dict(AI_HANDLER_REGISTRY, {"mock_ai": mock_cls}):
            result = get_usage("mock_ai", {"usage": {}})
        assert result["total_tokens"] == 30

