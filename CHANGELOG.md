# Changelog

All notable changes to `cross-ai-core` are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.4.2] — 2026-04-08

### Added
- `get_content_auto(response)` — extracts text from a self-describing response using the `_make` key stamped by `process_prompt()`; raises `ValueError` if `_make` is absent
- `put_content_auto(report, response)` — updates text in a self-describing response the same way; raises `ValueError` if `_make` is absent
- `process_prompt()` now stamps `"_make": ai_key` into every `dict` response it returns (in-memory only — the on-disk cache is unchanged), making responses self-describing for the `_auto` helpers
- `retry_with_backoff()` exported from the public API (was implemented in 0.4.0 but inadvertently omitted from `__init__.py`)

---

## [0.4.1] — 2026-04-03

### Added
- `CROSS_NO_CACHE=1` environment variable support — set in `~/.crossenv` or `.env` to bypass the on-disk API response cache globally, without requiring `--no-cache` on every command. Takes priority over `use_cache=True` passed to `process_prompt()` / `get_cached_response()`.

---

## [0.4.0] — 2026-04-02

### Added
- `system=` keyword parameter on `process_prompt()` — override the provider's default system prompt per call; `None` falls back to the provider's built-in default
- `QuotaExceededError`, `RateLimitError`, `TransientError`, `CrossAIError` added to the public API (`__all__`)
- `retry_with_backoff()` added to the public API
- `get_usage()` — normalised token-count extraction across all providers
- `check_api_key()` — diagnostic helper that prints which `.env` files were searched and the exact env var to add
- `AIResponse.was_cached` attribute — know whether a response was served from disk cache
- `py.typed` PEP 561 marker — package is now recognised as typed by mypy / pyright
- `[tool.pytest.ini_options]` in `pyproject.toml` — test config consolidated
- `Documentation` URL in `pyproject.toml` → `docs/` folder
- `docs/api-reference.md` — full public API reference with parallel-call examples
- `docs/providers.md` — per-provider guide (model, API key, strengths, free tier)
- `COMMERCIAL_LICENSE.md`

### Fixed
- `BaseAIHandler` ABC now enforces all 9 required abstract methods (previously only 3)
- `_get_cache_dir` exported publicly as `get_cache_dir` alias (private name convention conflict resolved)
- `use_cache` default was `False` in OpenAI and Perplexity providers — now consistently `True` across all five
- `get_ai_make()` and `get_ai_model()` now raise `ValueError` on unknown key instead of `AttributeError`
- `__version__` now read from `importlib.metadata` — single source of truth in `pyproject.toml`
- `get_data_title` type hint corrected from `json` (module) to `dict`
- `load_dotenv("~/.crossenv")` in README corrected to `os.path.expanduser("~/.crossenv")` — tilde was not expanded by python-dotenv

### Changed
- `verbose` and `use_cache` are now keyword-only arguments on `process_prompt()` (enforced by `*,`)

---

## [0.3.0] — 2026-03 *(initial public release)*

### Added
- Five providers: `xai`, `anthropic`, `openai`, `perplexity`, `gemini`
- MD5-keyed response caching via `~/.cross_api_cache/`
- `AIResponse` wrapper with backward-compatible 4-tuple unpacking
- `handle_api_error()` — classifies quota / rate-limit / transient errors
- Optional-extras install model (`[anthropic]`, `[gemini]`, `[openai]`, `[xai]`, `[all]`)

