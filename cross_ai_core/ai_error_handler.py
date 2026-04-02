#!/usr/bin/env python3
"""
Centralized error handling for AI API errors across all cross tools.

Provides graceful handling of common API errors including:
- Quota/billing errors (401, 429 with quota messages)
- Rate limiting (429 transient)
- Service unavailable (503, 500)
- Network errors

Usage:
    from ai_error_handler import handle_api_error, is_quota_error, is_rate_limit_error
    
    try:
        response = api_call()
    except Exception as e:
        handle_api_error(e, ai_name="perplexity", script_name="st-merge")
"""

import sys
import time


# ── Custom exception hierarchy ────────────────────────────────────────────────

class CrossAIError(Exception):
    """Base class for all cross-ai-core API errors."""
    def __init__(self, message: str, ai_name: str = ""):
        super().__init__(message)
        self.ai_name = ai_name


class QuotaExceededError(CrossAIError):
    """Raised when an API quota or billing limit is exceeded (permanent)."""


class RateLimitError(CrossAIError):
    """Raised when a transient rate-limit (429) is hit — retry is appropriate."""


class TransientError(CrossAIError):
    """Raised on transient service errors (503, 500, overload) — retry is appropriate."""


# Keywords that indicate permanent billing/quota issues (no retry)
QUOTA_ERROR_KEYWORDS = (
    "quota", "insufficient_quota", "credits", "spending limit", 
    "billing", "exhausted", "payment", "upgrade", "plan", 
    "subscribe", "exceeded your current quota",
)

# Keywords that indicate transient rate limiting or overload (retry-able)
TRANSIENT_ERROR_KEYWORDS = (
    "503", "500", "502", "504", "UNAVAILABLE", "overloaded", 
    "temporarily", "high demand", "try again", "rate limit",
    "too many requests",
)


def is_quota_error(error: Exception) -> bool:
    """
    Check if an error is due to quota/billing issues.
    
    Args:
        error: The exception to check
        
    Returns:
        True if this is a quota/billing error
    """
    err_str = str(error).lower()
    return any(keyword in err_str for keyword in QUOTA_ERROR_KEYWORDS)


def is_rate_limit_error(error: Exception) -> bool:
    """
    Check if an error is a transient rate limit (429 without quota keywords).
    
    Args:
        error: The exception to check
        
    Returns:
        True if this is a transient rate limit error
    """
    err_str = str(error).lower()
    has_rate_keywords = (
        "429" in err_str or 
        "rate" in err_str or
        "too many requests" in err_str
    )
    return has_rate_keywords and not is_quota_error(error)


def is_transient_error(error: Exception) -> bool:
    """
    Check if an error is transient (503, 500, overload, etc).
    
    Args:
        error: The exception to check
        
    Returns:
        True if this is a transient error worth retrying
    """
    err_str = str(error).lower()
    return any(keyword in err_str for keyword in TRANSIENT_ERROR_KEYWORDS)


def get_error_type(error: Exception) -> str:
    """
    Classify the error type.
    
    Returns:
        "quota", "rate_limit", "transient", or "other"
    """
    if is_quota_error(error):
        return "quota"
    elif is_rate_limit_error(error):
        return "rate_limit"
    elif is_transient_error(error):
        return "transient"
    else:
        return "other"


def get_ai_dashboard_url(ai_name: str) -> str:
    """Get the dashboard URL for an AI provider."""
    dashboards = {
        "perplexity": "https://www.perplexity.ai/settings/api",
        "openai": "https://platform.openai.com/settings/organization/billing",
        "anthropic": "https://console.anthropic.com/settings/billing",
        "xai": "https://console.x.ai/billing",
        "gemini": "https://console.cloud.google.com/billing",
    }
    return dashboards.get(ai_name, f"your {ai_name} dashboard")


def format_quota_error_message(ai_name: str, script_name: str = None) -> str:
    """
    Format a user-friendly message for quota errors.
    
    Args:
        ai_name: The AI provider name (e.g., "perplexity")
        script_name: Optional script name (e.g., "st-merge")
        
    Returns:
        Formatted error message
    """
    dashboard_url = get_ai_dashboard_url(ai_name)
    
    msg = f"\n{'='*70}\n"
    msg += f"  API Quota Exceeded: {ai_name}\n"
    msg += f"{'='*70}\n\n"
    msg += f"  You have exceeded your API quota or billing limit for {ai_name}.\n\n"
    msg += f"  To continue:\n"
    msg += f"    1. Check your billing and add credits at:\n"
    msg += f"       {dashboard_url}\n\n"
    msg += f"    2. Or use a different AI provider with:\n"
    if script_name:
        msg += f"       {script_name} --ai <provider> ...\n\n"
    else:
        msg += f"       --ai <provider> ...\n\n"
    msg += f"  Available providers: xai, anthropic, openai, perplexity, gemini\n"
    msg += f"{'='*70}\n"
    
    return msg


def format_rate_limit_message(ai_name: str, wait_seconds: int = 15) -> str:
    """Format message for rate limit errors."""
    msg = f"\n  Rate limit reached for {ai_name}.\n"
    msg += f"  Waiting {wait_seconds}s before retry...\n"
    return msg


def format_transient_error_message(ai_name: str, error: Exception) -> str:
    """Format message for transient errors."""
    msg = f"\n  {ai_name} service temporarily unavailable.\n"
    msg += f"  Error: {str(error)[:100]}\n"
    return msg


def handle_api_error(
    error: Exception,
    ai_name: str,
    script_name: str = None,
    exit_on_quota: bool = True,
    quiet: bool = False
) -> str:
    """
    Handle an API error gracefully with user-friendly messages.
    
    Args:
        error: The exception that occurred
        ai_name: Name of the AI provider (e.g., "perplexity")
        script_name: Name of the calling script (e.g., "st-merge")
        exit_on_quota: Whether to exit on quota errors (default: True)
        quiet: Whether to suppress output (default: False)
        
    Returns:
        Error type string: "quota", "rate_limit", "transient", or "other"
        
    Side effects:
        - Prints error messages (unless quiet=True)
        - May call sys.exit(1) for quota errors (if exit_on_quota=True)
    """
    error_type = get_error_type(error)
    
    if error_type == "quota":
        if not quiet:
            print(format_quota_error_message(ai_name, script_name), file=sys.stderr)
        if exit_on_quota:
            sys.exit(1)
        raise QuotaExceededError(str(error), ai_name=ai_name)

    elif error_type == "rate_limit":
        if not quiet:
            print(format_rate_limit_message(ai_name), file=sys.stderr)
        raise RateLimitError(str(error), ai_name=ai_name)

    elif error_type == "transient":
        if not quiet:
            print(format_transient_error_message(ai_name, error), file=sys.stderr)
        raise TransientError(str(error), ai_name=ai_name)

    else:
        # Unknown error - show raw error
        if not quiet:
            print(f"\n  {ai_name} API error: {str(error)[:200]}\n", file=sys.stderr)
    
    return error_type


def retry_with_backoff(
    func,
    ai_name: str,
    max_retries: int = 3,
    wait_seconds: int = 15,
    quiet: bool = False,
    script_name: str = None
):
    """
    Retry a function with exponential backoff on transient errors.
    
    Args:
        func: Function to call (no arguments - use lambda to bind)
        ai_name: Name of AI provider
        max_retries: Maximum retry attempts (default: 3)
        wait_seconds: Base wait time between retries (default: 15)
        quiet: Suppress output
        script_name: Name of calling script
        
    Returns:
        Function result on success
        
    Raises:
        QuotaExceededError: On permanent billing/quota failure (also sys.exit(1))
        RateLimitError / TransientError: After all retries exhausted
        Exception: Re-raises the last exception after all retries exhausted
    """
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except QuotaExceededError:
            # Quota errors are permanent — exit immediately
            sys.exit(1)
        except (RateLimitError, TransientError) as e:
            # Typed transient errors from process_prompt / handle_api_error
            last_error = e
            if attempt < max_retries:
                if not quiet:
                    print(f"  Retry {attempt}/{max_retries} in {wait_seconds}s...")
                time.sleep(wait_seconds)
                wait_seconds *= 2  # Exponential backoff
            else:
                break
        except Exception as e:
            # Raw/unknown exceptions from callers not using process_prompt
            last_error = e
            error_type = get_error_type(e)
            if error_type == "quota":
                if not quiet:
                    print(format_quota_error_message(ai_name, script_name), file=sys.stderr)
                sys.exit(1)
            elif error_type in ("rate_limit", "transient") and attempt < max_retries:
                if not quiet:
                    print(f"  {ai_name} transient error — retry {attempt}/{max_retries} in {wait_seconds}s...")
                time.sleep(wait_seconds)
                wait_seconds *= 2  # Exponential backoff
            else:
                if not quiet:
                    print(f"\n  {ai_name} API error: {str(e)[:200]}\n", file=sys.stderr)
                break
    
    # All retries exhausted
    if not quiet:
        print(f"\n  Error: {ai_name} failed after {max_retries} attempts.", file=sys.stderr)
        print(f"  Last error: {last_error}\n", file=sys.stderr)
    
    raise last_error

