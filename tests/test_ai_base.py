"""
tests/test_ai_base.py — Tests for cross_ai_core.ai_base

Coverage:
    _get_cache_dir   — env var resolution, tilde expansion, default fallback
    BaseAIHandler    — cannot be instantiated directly (ABC enforcement)
"""
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from cross_ai_core.ai_base import BaseAIHandler, _get_cache_dir


# ── _get_cache_dir ─────────────────────────────────────────────────────────────

class TestGetCacheDir:
    def test_default_is_cross_api_cache(self, monkeypatch):
        """No env var → ~/.cross_api_cache"""
        monkeypatch.delenv("CROSS_API_CACHE_DIR", raising=False)
        result = _get_cache_dir()
        assert result == os.path.expanduser("~/.cross_api_cache")

    def test_env_var_absolute_path(self, monkeypatch):
        """Absolute path is returned as-is."""
        monkeypatch.setenv("CROSS_API_CACHE_DIR", "/tmp/my_cache")
        assert _get_cache_dir() == "/tmp/my_cache"

    def test_env_var_tilde_expanded(self, monkeypatch):
        """Tilde in env var is expanded."""
        monkeypatch.setenv("CROSS_API_CACHE_DIR", "~/my_project/cache")
        result = _get_cache_dir()
        assert result == os.path.expanduser("~/my_project/cache")
        assert "~" not in result

    def test_env_var_empty_string_uses_default(self, monkeypatch):
        """Empty env var string falls back to default."""
        monkeypatch.setenv("CROSS_API_CACHE_DIR", "   ")
        result = _get_cache_dir()
        assert result == os.path.expanduser("~/.cross_api_cache")

    def test_returns_string(self, monkeypatch):
        """Return type is always str, not Path."""
        monkeypatch.delenv("CROSS_API_CACHE_DIR", raising=False)
        assert isinstance(_get_cache_dir(), str)


# ── BaseAIHandler (ABC) ────────────────────────────────────────────────────────

class TestBaseAIHandler:
    def test_cannot_instantiate_directly(self):
        """BaseAIHandler is abstract — instantiating it must raise TypeError."""
        with pytest.raises(TypeError):
            BaseAIHandler()

    def test_concrete_subclass_must_implement_all_methods(self):
        """A subclass missing any abstract method also cannot be instantiated."""
        class Incomplete(BaseAIHandler):
            pass   # implements nothing

        with pytest.raises(TypeError):
            Incomplete()

    def test_fully_implemented_subclass_is_instantiable(self):
        """A subclass that implements all abstract methods can be instantiated."""
        class Complete(BaseAIHandler):
            @classmethod
            def get_payload(cls, prompt):
                return {}

            @classmethod
            def get_client(cls):
                return None

            @classmethod
            def get_cached_response(cls, client, payload, verbose, use_cache):
                return {}, False

        # Should not raise
        instance = Complete()
        assert instance is not None

