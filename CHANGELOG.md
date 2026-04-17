# Changelog

All notable changes to `cross-ai-core` are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.6.0] — 2026-04-17

This release rolls up the CAC-1 → CAC-9 hardening series. Headline changes:
the on-disk cache is now atomic + lock-protected, every SDK client is
constructed lazily and cached per-process, `process_prompt()` accepts a
`retry_budget`, and a new `get_rate_limit_concurrency()` helper exposes
recommended per-provider semaphore sizes (consumed by `cross-st`'s PAR-1
`st-cross --parallel` mode).

### Added
- **`get_rate_limit_concurrency(make) -> int`** — recommended max concurrent
  in-flight calls per provider. Defaults: `xai=3, anthropic=2, openai=3,
  perplexity=2, gemini=5`. Raises `KeyError` on unknown provider. Exported
  from the package root. (CAC-5)
- **`retry_budget` kwarg** on `process_prompt()` and the underlying
  `retry_with_backoff()` — caps total time spent retrying transient errors.
  Each backoff sleep is shortened to `min(wait, remaining)`; loop exits as
  soon as the budget hits zero. `retry_budget=0` disables retries entirely;
  `None` (default) preserves pre-0.6 unlimited-retry behaviour. (CAC-4)
- **`"timeout"` keyword** added to `TRANSIENT_ERROR_KEYWORDS` — `httpx.ReadTimeout`,
  `APITimeoutError`, and similar are now classified as transient and retried
  rather than surfacing immediately. (CAC-4)
- **Lazy + cached SDK clients per provider** — `process_prompt()` constructs
  each provider client at most once per process, behind a `threading.Lock`
  with double-checked locking. Cache hits no longer construct a client at
  all (factory lambda only invoked on miss / `use_cache=False` /
  `CROSS_NO_CACHE`). New `process_prompt(..., client=...)` kwarg lets callers
  inject explicit clients (e.g. test mocks). (CAC-8)
- **`reset_client_cache(make=None)`** — public helper to drop one or all
  cached clients. Required for test isolation, key rotation, and post-fork
  cleanup. Exported from `__init__.py`.
- **`CROSS_NO_CLIENT_CACHE=1`** env var — fully disables the SDK client
  cache (mirrors `CROSS_NO_CACHE` for response caching).
- **`AIResponse.__repr__`** — concise debug string showing model, cached/live
  flag, and a truncated content preview. (CAC-6)
- **96 new tests** — `TestCacheAtomicWrite`, `TestGetAiList`,
  `TestTimeoutIsTransient`, `TestRetryBudget`, `TestGetRateLimitConcurrency`,
  `TestAIResponseRepr`, `TestClientCache` (8 tests including a
  `threading.Barrier` race that proves only one client is constructed under
  concurrent first-use). 144 total, all pass.

### Changed
- **Cache writes are now atomic + lock-protected** — `BaseAIHandler` writes
  the cache file via temp-file + `os.rename()` (atomic on POSIX) under
  `fcntl.LOCK_EX`. Reads acquire `fcntl.LOCK_SH`. Corrupt cache files
  (`json.JSONDecodeError`/`OSError`) are now caught, deleted, and the call
  falls through to the live API instead of crashing the subprocess. Windows
  build falls back gracefully if `fcntl` is unavailable (no-op). (CAC-1)
- **`get_ai_list()` returns a copy** — `return list(AI_LIST)` — so mutating
  the return value no longer corrupts the global. Order is asserted as
  `["xai", "anthropic", "openai", "perplexity", "gemini"]`. (CAC-3)
- **`DEFAULT_SYSTEM`, `MAX_TOKENS`, `get_title()` lifted to
  `BaseAIHandler`** — duplicates removed from all 5 provider files
  (−79 lines). `get_title()` is now a concrete classmethod (was abstract).
  Module-level payload helpers reference the base-class constants. (CAC-7)
- **Type annotations tightened** — `BaseAIHandler.get_payload()` gains a
  `-> dict` return type; `str  None` annotation corrected to `"str | None"`.
  (CAC-6)

### Fixed
- **`NameError: name 'DEFAULT_SYSTEM' is not defined` in gemini provider** — CAC-7
  lifted `DEFAULT_SYSTEM` to `BaseAIHandler` and removed the local constant from all
  five provider files, but one call site in `GeminiHandler._call_api()` (the
  `payload.get("system_instruction", DEFAULT_SYSTEM)` fallback argument) was missed.
  Every gemini API call raised a `NameError` and was reported as a failure in
  `st-cross` Step 1. Fixed by replacing the bare name with
  `BaseAIHandler.DEFAULT_SYSTEM`. (`d788e69`)


- Stray commented `base_url="https://api.x.ai"` artifact in
  `ai_anthropic.py` `get_anthropic_client()`. (CAC-9)

### Notes for consumers
- **`cross-st` requirement bumps to `cross-ai-core>=0.6.0`** to use the new
  `get_rate_limit_concurrency()` helper from PAR-1.
- **Cache layer is fork-aware but not fork-safe across providers** — call
  `reset_client_cache()` after `os.fork()` to be safe. PAR-1 dodges this by
  using subprocesses (each child gets its own clean cache) — intentional.

---



### Added
- `model=` keyword parameter on `process_prompt()` — per-call model override.
  Resolution order: explicit `model` arg → `<AI_KEY_UPPER>_MODEL` env var → handler
  default (e.g. `gemini-2.5-flash`).  Setting `GEMINI_MODEL=gemini-2.5-pro` in
  `~/.crossenv` globally switches the model without touching code.
- `get_ai_model(make)` now checks the `<MAKE_UPPER>_MODEL` env var before returning
  the compiled-in handler default — same resolution order as `process_prompt()`.
- 11 new tests covering model override paths in `TestProcessPromptModel` and
  `TestGetAiModel` (105 total, all passing).

### Changed
- **Provider SDK minimums bumped** (openai is a major version break):
  - `openai>=2.0.0` (was `>=1.70.0`) — tested on openai 2.31.0
  - `anthropic>=0.86.0` (was `>=0.84.0`) — tested on anthropic 0.92.0
  - `google-genai>=1.69.0` (was `>=1.65.0`) — tested on google-genai 1.71.0
- `process_prompt()` docstring expanded to document all keyword parameters
- `get_ai_model()` docstring updated to describe env-var resolution order

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

