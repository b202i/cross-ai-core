"""
tests/test_ai_error_handler.py — Tests for cross_ai_core.ai_error_handler

Coverage:
    is_quota_error        — quota/billing keyword detection
    is_rate_limit_error   — 429 transient vs. quota distinction
    is_transient_error    — 5xx / overload keyword detection
    get_error_type        — classification dispatch
    handle_api_error      — output + exit behaviour
    get_ai_dashboard_url  — known and unknown providers
"""
import sys
from unittest.mock import patch

import pytest

from cross_ai_core.ai_error_handler import (
    get_ai_dashboard_url,
    get_error_type,
    handle_api_error,
    is_quota_error,
    is_rate_limit_error,
    is_transient_error,
)


# ── is_quota_error ─────────────────────────────────────────────────────────────

class TestIsQuotaError:
    @pytest.mark.parametrize("msg", [
        "exceeded your current quota",
        "insufficient_quota on your account",
        "billing limit reached",
        "please upgrade your plan",
        "credits exhausted",
        "spending limit exceeded",
    ])
    def test_returns_true_for_quota_keywords(self, msg):
        assert is_quota_error(Exception(msg)) is True

    @pytest.mark.parametrize("msg", [
        "429 too many requests",
        "503 service unavailable",
        "rate limit",
        "connection error",
    ])
    def test_returns_false_for_non_quota(self, msg):
        assert is_quota_error(Exception(msg)) is False


# ── is_rate_limit_error ────────────────────────────────────────────────────────

class TestIsRateLimitError:
    def test_returns_true_for_429_without_quota(self):
        assert is_rate_limit_error(Exception("429 too many requests")) is True

    def test_returns_true_for_rate_keyword(self):
        assert is_rate_limit_error(Exception("rate limit exceeded")) is True

    def test_returns_false_when_quota_keyword_present(self):
        # 429 with quota language is a billing error, not a transient rate limit
        assert is_rate_limit_error(Exception("429 exceeded your current quota")) is False

    def test_returns_false_for_503(self):
        assert is_rate_limit_error(Exception("503 service unavailable")) is False


# ── is_transient_error ─────────────────────────────────────────────────────────

class TestIsTransientError:
    @pytest.mark.parametrize("msg", [
        "503 service unavailable",
        "500 internal server error",
        "service overloaded",
        "try again later",
        "temporarily unavailable",
        "high demand",
    ])
    def test_returns_true_for_transient_keywords(self, msg):
        assert is_transient_error(Exception(msg)) is True

    def test_returns_false_for_unrelated_error(self):
        assert is_transient_error(Exception("invalid API key")) is False


# ── get_error_type ─────────────────────────────────────────────────────────────

class TestGetErrorType:
    def test_quota(self):
        assert get_error_type(Exception("exceeded your current quota")) == "quota"

    def test_rate_limit(self):
        assert get_error_type(Exception("429 rate limit")) == "rate_limit"

    def test_transient(self):
        assert get_error_type(Exception("503 unavailable")) == "transient"

    def test_other(self):
        assert get_error_type(Exception("some unexpected error")) == "other"


# ── handle_api_error ───────────────────────────────────────────────────────────

class TestHandleApiError:
    def test_quota_error_exits_by_default(self):
        with pytest.raises(SystemExit) as exc_info:
            handle_api_error(
                Exception("exceeded your current quota"),
                ai_name="openai",
                exit_on_quota=True,
                quiet=True,
            )
        assert exc_info.value.code == 1

    def test_quota_error_no_exit_when_disabled(self):
        # When exit_on_quota=False, should raise QuotaExceededError instead of SystemExit
        from cross_ai_core.ai_error_handler import QuotaExceededError
        with pytest.raises(QuotaExceededError) as exc_info:
            handle_api_error(
                Exception("exceeded your current quota"),
                ai_name="openai",
                exit_on_quota=False,
                quiet=True,
            )
        assert exc_info.value.ai_name == "openai"

    def test_rate_limit_returns_type_and_does_not_exit(self):
        # Rate limit errors now raise RateLimitError (not SystemExit)
        from cross_ai_core.ai_error_handler import RateLimitError
        with pytest.raises(RateLimitError) as exc_info:
            handle_api_error(
                Exception("429 rate limit"),
                ai_name="gemini",
                exit_on_quota=True,
                quiet=True,
            )
        assert exc_info.value.ai_name == "gemini"

    def test_transient_returns_type_and_does_not_exit(self):
        # Transient errors now raise TransientError (not SystemExit)
        from cross_ai_core.ai_error_handler import TransientError
        with pytest.raises(TransientError) as exc_info:
            handle_api_error(
                Exception("503 unavailable"),
                ai_name="xai",
                exit_on_quota=True,
                quiet=True,
            )
        assert exc_info.value.ai_name == "xai"

    def test_other_error_returns_type(self):
        result = handle_api_error(
            Exception("unexpected failure"),
            ai_name="anthropic",
            exit_on_quota=True,
            quiet=True,
        )
        assert result == "other"

    def test_quiet_suppresses_stderr(self, capsys):
        handle_api_error(
            Exception("unexpected failure"),
            ai_name="anthropic",
            exit_on_quota=False,
            quiet=True,
        )
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_not_quiet_prints_to_stderr(self, capsys):
        handle_api_error(
            Exception("unexpected failure"),
            ai_name="anthropic",
            exit_on_quota=False,
            quiet=False,
        )
        captured = capsys.readouterr()
        assert "anthropic" in captured.err


# ── get_ai_dashboard_url ───────────────────────────────────────────────────────

class TestGetAiDashboardUrl:
    @pytest.mark.parametrize("provider,expected_fragment", [
        ("openai",     "platform.openai.com"),
        ("anthropic",  "console.anthropic.com"),
        ("gemini",     "console.cloud.google.com"),
        ("xai",        "console.x.ai"),
        ("perplexity", "perplexity.ai"),
    ])
    def test_known_providers_return_url(self, provider, expected_fragment):
        url = get_ai_dashboard_url(provider)
        assert expected_fragment in url

    def test_unknown_provider_returns_fallback_string(self):
        url = get_ai_dashboard_url("unknown_provider")
        assert "unknown_provider" in url



# ── CAC-4A: "timeout" keyword routes to transient ─────────────────────────────

class TestTimeoutIsTransient:
    """CAC-4A — httpx/SDK timeout errors must be classified as transient."""

    @pytest.mark.parametrize("msg", [
        "httpx.ReadTimeout",
        "APITimeoutError: request timed out",
        "Connection timeout after 30s",
        "timeout waiting for response",
    ])
    def test_timeout_is_transient(self, msg):
        assert is_transient_error(Exception(msg)) is True

    def test_timeout_routes_to_transient_type(self):
        assert get_error_type(Exception("ReadTimeout")) == "transient"

    def test_timeout_not_misclassified_as_quota(self):
        assert is_quota_error(Exception("timeout")) is False


# ── CAC-4B: retry_budget kwarg ────────────────────────────────────────────────

class TestRetryBudget:
    """CAC-4B — retry_budget caps total retry time."""

    def test_retry_budget_zero_fails_immediately(self):
        """retry_budget=0 means no sleep, fail on first transient error."""
        from cross_ai_core.ai_error_handler import retry_with_backoff, RateLimitError
        calls = 0

        def failing():
            nonlocal calls
            calls += 1
            raise RateLimitError("rate limit", ai_name="xai")

        with pytest.raises(Exception):
            retry_with_backoff(failing, "xai", retry_budget=0, quiet=True)
        # Should have been called only once (no retries after budget=0)
        assert calls == 1

    def test_retry_budget_exhaustion_raises_last_error(self, monkeypatch):
        """When budget is exhausted after one capped sleep, no further retries happen."""
        from cross_ai_core.ai_error_handler import retry_with_backoff, TransientError
        import time

        sleep_calls = []
        monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

        def failing():
            raise TransientError("503 overloaded", ai_name="openai")

        with pytest.raises(Exception):
            retry_with_backoff(
                failing, "openai",
                max_retries=5,
                wait_seconds=30,
                retry_budget=10,  # budget < wait_seconds, caps first sleep to 10
                quiet=True,
            )
        # Sleep is called once with the capped value (min(30, 10)=10),
        # then budget_remaining=0 so subsequent retries break immediately.
        assert len(sleep_calls) == 1
        assert sleep_calls[0] == 10.0

    def test_retry_budget_none_uses_normal_backoff(self, monkeypatch):
        """retry_budget=None (default) still does normal exponential backoff."""
        from cross_ai_core.ai_error_handler import retry_with_backoff, TransientError
        import time

        sleep_calls = []
        monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

        attempt_count = 0

        def failing():
            nonlocal attempt_count
            attempt_count += 1
            raise TransientError("503 overloaded", ai_name="gemini")

        with pytest.raises(Exception):
            retry_with_backoff(
                failing, "gemini",
                max_retries=3,
                wait_seconds=1,
                retry_budget=None,
                quiet=True,
            )
        # Should have slept twice (between attempt 1→2 and 2→3)
        assert len(sleep_calls) == 2
        assert sleep_calls[0] == 1   # first sleep = wait_seconds
        assert sleep_calls[1] == 2   # second sleep = wait_seconds * 2
