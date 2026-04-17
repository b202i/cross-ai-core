"""
tests/test_ai_base.py — Tests for cross_ai_core.ai_base

Coverage:
    _get_cache_dir       — env var resolution, tilde expansion, default fallback
    BaseAIHandler        — cannot be instantiated directly (ABC enforcement)
    get_cached_response  — CROSS_NO_CACHE env var bypasses cache
"""
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

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
            def get_payload(cls, prompt, system=None):
                return {}

            @classmethod
            def get_client(cls):
                return None

            @classmethod
            def _call_api(cls, client, payload):
                return {}

            @classmethod
            def get_model(cls):
                return "test-model"

            @classmethod
            def get_make(cls):
                return "test"

            @classmethod
            def get_content(cls, response):
                return ""

            @classmethod
            def put_content(cls, report, response):
                return response

            @classmethod
            def get_data_content(cls, select_data):
                return ""

            @classmethod
            def get_title(cls, gen_content):
                return ""

            @classmethod
            def get_usage(cls, response):
                return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        # Should not raise
        instance = Complete()
        assert instance is not None


# ── get_cached_response / CROSS_NO_CACHE ──────────────────────────────────────

# A minimal concrete handler used by cache tests
class _FakeHandler(BaseAIHandler):
    @classmethod
    def get_payload(cls, prompt, system=None):
        return {"prompt": prompt}

    @classmethod
    def get_client(cls):
        return None

    @classmethod
    def _call_api(cls, client, payload):
        return {"result": "fresh"}

    @classmethod
    def get_model(cls):
        return "fake-model"

    @classmethod
    def get_make(cls):
        return "fake"

    @classmethod
    def get_content(cls, response):
        return response.get("result", "")

    @classmethod
    def put_content(cls, report, response):
        return response

    @classmethod
    def get_data_content(cls, select_data):
        return ""

    @classmethod
    def get_title(cls, gen_content):
        return ""

    @classmethod
    def get_usage(cls, response):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


class TestCacheAtomicWrite:
    """CAC-1: atomic write, flock, and corrupt-file recovery."""

    def test_cache_miss_writes_file(self, monkeypatch, tmp_path):
        """A cache miss calls _call_api and persists the result."""
        monkeypatch.delenv("CROSS_NO_CACHE", raising=False)
        monkeypatch.setenv("CROSS_API_CACHE_DIR", str(tmp_path))

        response, was_cached = _FakeHandler.get_cached_response(None, {"prompt": "hello"})

        assert response == {"result": "fresh"}
        assert was_cached is False
        # Exactly one .json file created; no stray .tmp files
        json_files = list(tmp_path.glob("*.json"))
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(json_files) == 1
        assert len(tmp_files) == 0

    def test_cache_hit_returns_cached_value(self, monkeypatch, tmp_path):
        """A warm cache returns the stored value without calling the API."""
        monkeypatch.delenv("CROSS_NO_CACHE", raising=False)
        monkeypatch.setenv("CROSS_API_CACHE_DIR", str(tmp_path))

        import hashlib
        payload = {"prompt": "hello"}
        key = hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        (tmp_path / f"{key}.json").write_text(json.dumps({"result": "cached"}))

        response, was_cached = _FakeHandler.get_cached_response(None, payload)

        assert response == {"result": "cached"}
        assert was_cached is True

    def test_corrupt_cache_file_falls_back_to_api(self, monkeypatch, tmp_path):
        """A truncated / corrupt cache file is deleted and a fresh API call is made."""
        monkeypatch.delenv("CROSS_NO_CACHE", raising=False)
        monkeypatch.setenv("CROSS_API_CACHE_DIR", str(tmp_path))

        import hashlib
        payload = {"prompt": "corrupt-test"}
        key = hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        corrupt_file = tmp_path / f"{key}.json"
        corrupt_file.write_text("{bad json{{{{")  # truncated / invalid

        # Must not raise; must return a fresh API response
        response, was_cached = _FakeHandler.get_cached_response(None, payload)

        assert response == {"result": "fresh"}
        assert was_cached is False
        # Corrupt file must have been replaced with a valid one
        assert corrupt_file.exists()
        reloaded = json.loads(corrupt_file.read_text())
        assert reloaded == {"result": "fresh"}

    def test_corrupt_cache_verbose_prints_message(self, monkeypatch, tmp_path, capsys):
        """verbose=True prints a message when a corrupt cache file is recovered."""
        monkeypatch.delenv("CROSS_NO_CACHE", raising=False)
        monkeypatch.setenv("CROSS_API_CACHE_DIR", str(tmp_path))

        import hashlib
        payload = {"prompt": "verbose-corrupt"}
        key = hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        (tmp_path / f"{key}.json").write_text("not valid json")

        _FakeHandler.get_cached_response(None, payload, verbose=True)

        out = capsys.readouterr().out
        assert "corrupt" in out.lower() or "refetching" in out.lower()

    def test_no_stray_tmp_file_after_successful_write(self, monkeypatch, tmp_path):
        """After a successful write, no .tmp file is left on disk."""
        monkeypatch.delenv("CROSS_NO_CACHE", raising=False)
        monkeypatch.setenv("CROSS_API_CACHE_DIR", str(tmp_path))

        _FakeHandler.get_cached_response(None, {"prompt": "tmp-check"})

        assert list(tmp_path.glob("*.tmp")) == []

    def test_written_cache_file_contains_valid_json(self, monkeypatch, tmp_path):
        """The written cache file is parseable JSON."""
        monkeypatch.delenv("CROSS_NO_CACHE", raising=False)
        monkeypatch.setenv("CROSS_API_CACHE_DIR", str(tmp_path))

        _FakeHandler.get_cached_response(None, {"prompt": "valid-json-check"})

        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert data == {"result": "fresh"}

    def test_make_key_absent_from_written_cache_file(self, monkeypatch, tmp_path):
        """_make must never be written to disk — it is a runtime-only annotation.

        This is the CAC-2 requirement: process_prompt() stamps _make in-memory
        after get_cached_response() returns, so _make should never appear in the
        file written by get_cached_response().
        """
        monkeypatch.delenv("CROSS_NO_CACHE", raising=False)
        monkeypatch.setenv("CROSS_API_CACHE_DIR", str(tmp_path))

        _FakeHandler.get_cached_response(None, {"prompt": "make-key-check"})

        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1
        raw = json.loads(json_files[0].read_text())
        assert "_make" not in raw, (
            "_make must not be persisted to disk; it is a runtime annotation only"
        )


class TestCrossNoCache:
    """CROSS_NO_CACHE=1 must bypass the on-disk cache."""

    def test_cross_no_cache_bypasses_cache(self, monkeypatch, tmp_path):
        """When CROSS_NO_CACHE is set, _call_api is called even if a cache file exists."""
        monkeypatch.setenv("CROSS_NO_CACHE", "1")
        monkeypatch.setenv("CROSS_API_CACHE_DIR", str(tmp_path))

        # Pre-populate a cache file so we can confirm it is *not* read
        import hashlib, json
        payload = {"prompt": "test"}
        key = hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        cache_file = tmp_path / f"{key}.json"
        cache_file.write_text(json.dumps({"result": "cached"}))

        response, was_cached = _FakeHandler.get_cached_response(None, payload)

        assert response == {"result": "fresh"}
        assert was_cached is False

    def test_cross_no_cache_empty_string_uses_cache(self, monkeypatch, tmp_path):
        """Empty CROSS_NO_CACHE string does NOT bypass the cache."""
        monkeypatch.setenv("CROSS_NO_CACHE", "")
        monkeypatch.setenv("CROSS_API_CACHE_DIR", str(tmp_path))

        import hashlib, json
        payload = {"prompt": "test"}
        key = hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        cache_file = tmp_path / f"{key}.json"
        cache_file.write_text(json.dumps({"result": "cached"}))

        response, was_cached = _FakeHandler.get_cached_response(None, payload)

        assert response == {"result": "cached"}
        assert was_cached is True

    def test_cross_no_cache_unset_uses_cache(self, monkeypatch, tmp_path):
        """When CROSS_NO_CACHE is not set, a warm cache hit is returned."""
        monkeypatch.delenv("CROSS_NO_CACHE", raising=False)
        monkeypatch.setenv("CROSS_API_CACHE_DIR", str(tmp_path))

        import hashlib, json
        payload = {"prompt": "test"}
        key = hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        cache_file = tmp_path / f"{key}.json"
        cache_file.write_text(json.dumps({"result": "cached"}))

        response, was_cached = _FakeHandler.get_cached_response(None, payload)

        assert response == {"result": "cached"}
        assert was_cached is True

    def test_cross_no_cache_takes_priority_over_use_cache_true(self, monkeypatch, tmp_path):
        """CROSS_NO_CACHE overrides use_cache=True."""
        monkeypatch.setenv("CROSS_NO_CACHE", "1")
        monkeypatch.setenv("CROSS_API_CACHE_DIR", str(tmp_path))

        import hashlib, json
        payload = {"prompt": "test"}
        key = hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        cache_file = tmp_path / f"{key}.json"
        cache_file.write_text(json.dumps({"result": "cached"}))

        response, was_cached = _FakeHandler.get_cached_response(None, payload, use_cache=True)

        assert response == {"result": "fresh"}
        assert was_cached is False

    def test_cross_no_cache_verbose_prints_message(self, monkeypatch, tmp_path, capsys):
        """CROSS_NO_CACHE with verbose=True prints an informational message."""
        monkeypatch.setenv("CROSS_NO_CACHE", "1")
        monkeypatch.setenv("CROSS_API_CACHE_DIR", str(tmp_path))

        _FakeHandler.get_cached_response(None, {"prompt": "test"}, verbose=True)

        captured = capsys.readouterr()
        assert "CROSS_NO_CACHE" in captured.out

