# cross-ai-core — Developer Experience & PyPI Compliance Review

*Reviewed against v0.3.0 · April 2026*

---

## TL;DR

The package is **well-conceived and mostly solid**. Install is frictionless, the optional-extras model is clean, and the test suite is genuinely useful. Several concrete issues were fixed in this pass; a handful of larger architectural improvements are flagged below as the project matures.

---

## 1. Install & First-Use Experience

### What works well
- **Optional-extras install is excellent.** `pip install "cross-ai-core[anthropic]"` installs exactly one provider SDK. Developers immediately understand the cost model.
- **`process_prompt` / `get_content` surface area is minimal.** Two calls to get text out is about as lean as it can be.
- **`check_api_key()` diagnostic output is genuinely helpful** — it shows which `.env` files were searched and the exact env var name. Most libraries just throw `AuthenticationError` and let the developer figure it out.
- **`AIResponse` backward-compat tuple unpacking** is a thoughtful migration bridge.

### Pain points (from integrating into ShangScore)
- **The system prompt is hardcoded** — `"You are a seasoned investigative reporter..."` is baked into every provider's payload builder. A health-tracking app is not a newsroom. `process_prompt()` needs a `system=` parameter (see §6 below).
- **`verbose` and `use_cache` have no defaults in `process_prompt()`** — every call site must spell them out explicitly. They should default to `verbose=False, use_cache=True`.
- **`load_dotenv("~/.crossenv")` in the README quickstart is wrong.** Python's `python-dotenv` does not expand `~`. It should be `load_dotenv(os.path.expanduser("~/.crossenv"))`. A new developer will copy-paste this, get a silent no-op, and wonder why their API key isn't loading.

---

## 2. Documentation

### README
| Issue | Severity |
|---|---|
| `load_dotenv("~/.crossenv")` tilde not expanded | 🔴 Bug |
| Dependency table uses no Markdown pipes — renders as a wall of text on PyPI | 🟡 Minor |
| No badges (PyPI version, Python versions, license) | 🟡 Minor |
| `check_api_key()`, `retry_with_backoff()`, and `AIResponse` attributes undocumented | 🟡 Minor |
| No CHANGELOG link | 🟢 Nice to have |
| "Adding a provider" section lists 5 methods but the contract now has 9 | 🟡 Minor |

### AGENTS.md
Currently empty — this is the file AI coding agents read first to understand the repo. Should describe the provider plugin pattern, release workflow, and point to `RELEASE.md`.

---

## 3. Python & PyPI Conventions

### Fixed in this pass ✅
| Item | File | What was wrong |
|---|---|---|
| Syntax error in type hint | `ai_handler.py` | `list  None` (missing `\|`) — `check_api_key` signature was invalid Python |
| Stale commented-out code | `ai_handler.py` | `# AI_LIST = ["xai", ...]` leftover debug line |
| `get_ai_make` / `get_ai_model` missing safety guard | `ai_handler.py` | Called `handler_cls.get_make()` without checking if handler was found — `AttributeError` on unknown key |
| `_get_cache_dir` exported publicly with private name | `__init__.py` | Private convention (`_` prefix) conflicts with being in `__all__`; exposed as `get_cache_dir` alias |
| `__version__` missing from `__all__` | `__init__.py` | Standard Python packaging convention |
| `BaseAIHandler` only enforced 3 of 9 required methods | `ai_base.py` | ABC protection was nearly useless — new providers could skip 6 methods and pass the `isinstance` check |
| Misplaced docstring in `get_gemini_config` | `ai_gemini.py` | `"""docstring"""` was placed *after* an `import` statement inside the function — ignored by `help()` |
| `use_cache` default `False` in OpenAI + Perplexity | `ai_openai.py`, `ai_perplexity.py` | Inconsistent with xAI and Anthropic (default `True`) — caching silently disabled for two providers |
| Missing `py.typed` marker | — | Without this, type checkers (mypy, pyright) treat the package as untyped |
| `py.typed` not declared in `pyproject.toml` | `pyproject.toml` | Marker file not included in wheel |
| `[tool.pytest.ini_options]` missing from `pyproject.toml` | `pyproject.toml` | Config was only in `pytest.ini`; modern convention is to consolidate in `pyproject.toml` |
| Incomplete `Complete` test subclass | `tests/test_ai_base.py` | Only implemented 3 abstract methods; after ABC fix, test correctly failed and was updated |

### Still outstanding (not fixed — require design decisions)

#### 🔴 `sys.exit()` inside a library
`handle_api_error(exit_on_quota=True)` calls `sys.exit(1)` and this is the **default** in `process_prompt`. This is a CLI tool pattern, not a library pattern. Any application that wraps `process_prompt` in a try/except or a background thread will be killed silently. The correct fix is to raise a custom exception:

```python
# Instead of:
if exit_on_quota:
    sys.exit(1)

# Raise a typed exception the caller can catch:
raise QuotaExceededError(f"Quota exceeded for {ai_name}")
```

Add `QuotaExceededError`, `RateLimitError`, `TransientError` to the public API. Keep `exit_on_quota=True` only in CLI entry points, not in the library core.

#### 🔴 Hardcoded system prompt
All 5 providers hardcode `"You are a seasoned investigative reporter..."`. This makes the library unusable for any non-journalism application without monkey-patching module-level constants. Fix:

```python
# Add system= parameter to process_prompt
def process_prompt(
    ai_key: str,
    prompt: str,
    *,
    system: str | None = None,
    verbose: bool = False,
    use_cache: bool = True,
) -> AIResponse: ...
```

Then thread `system` through `get_payload(prompt, system=system)` in each handler. The hardcoded value becomes the default.

#### 🟡 Massive caching code duplication (DRY violation)
The `get_cached_response` logic (MD5 hash → check cache dir → call API → write cache) is copy-pasted across all 5 provider files (~60 lines each, ~300 lines total). This should live in `BaseAIHandler` as a concrete `cached_call` method, with each provider only implementing a thin `_call_api(client, payload)` method. One cache bug fix currently requires editing 5 files.

```python
# BaseAIHandler gets a concrete method:
@classmethod
def get_cached_response(cls, client, payload, verbose, use_cache):
    if not use_cache:
        return cls._call_api(client, payload), False
    # ... MD5 cache logic once ...
    return cls._call_api(client, payload), False

@classmethod
@abstractmethod
def _call_api(cls, client, payload) -> dict:
    """Make the actual API call, return raw response dict."""
```

#### 🟡 Module docstrings placed after code
In `ai_xai.py`, `ai_openai.py`, `ai_anthropic.py`, and `ai_gemini.py` the module-level docstring (the long model family reference) is placed *after* the imports and constants — Python only treats a string literal as a module docstring if it is the very first statement. These are currently just floating string literals that `help(module)` and IDEs ignore. Move them to the top, or convert them to comments (`#`).

#### 🟢 Version not DRY
`version = "0.3.0"` appears in both `pyproject.toml` and `__init__.py`. The two can drift. Modern pattern:

```python
# In __init__.py — single source of truth:
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("cross-ai-core")
except PackageNotFoundError:
    __version__ = "0.0.0"  # running from source without install
```

Then remove the hardcoded string from `__init__.py`.

#### 🟢 No `CHANGELOG.md`
PyPI users have no way to see what changed between versions. Even a minimal `## [0.3.0]` section per release is better than nothing.

#### 🟢 No `CONTRIBUTING.md`
The AGENTS.md + RELEASE.md cover internal workflow well, but there's no file for external contributors.

#### 🟢 `get_data_title` uses `json` as a type hint
```python
def get_data_title(ai_key: str, data: json):
```
`json` is a module, not a type. Should be `dict`.

---

## 4. Are We Following Proper PyPI Conventions?

| Convention | Status |
|---|---|
| `pyproject.toml` with `[build-system]` | ✅ |
| Semantic versioning | ✅ |
| `README.md` as long description | ✅ |
| OSI-approved license + classifier | ✅ |
| `requires-python` specified | ✅ |
| Optional extras for provider SDKs | ✅ |
| `[dev]` extras for test dependencies | ✅ |
| `project.urls` (Homepage, Repository, Bug Tracker) | ✅ |
| `py.typed` marker for typed packages | ✅ *(fixed)* |
| `package-data` includes `py.typed` | ✅ *(fixed)* |
| `__version__` accessible and in `__all__` | ✅ *(fixed)* |
| No `setup.py` / `setup.cfg` remnants | ✅ |
| `CHANGELOG.md` | ❌ |
| Tests in `[dev]` extras (not `[tests]`) | ✅ (minor style point — `[tests]` is more conventional but either works) |
| `Changelog` URL in `project.urls` | ❌ |

---

## 5. From the ShangScore Integration Perspective

Usage in ShangScore (`shangscore_app.py`) is minimal right now — the library is imported but not yet active in scoring. When Phase 2 (AI Coach) lands:

- **Add `system=` parameter first** — the coaching prompt will need a fitness/health persona, not a journalist.
- **Consider `use_cache=False` for real-time health advice** — cached AI responses are fine for static content but not for "how did I do today?" queries.
- **`retry_with_backoff`** should be the default wrapper for background fetch threads (the Garmin 429 pattern already exists in the biometric collector — same pattern applies here).
- **`check_api_key()`** should be called at startup in `shangscore_app.py` to surface missing keys before the first user request, not on demand.

---

## 6. Recommended `process_prompt` Signature (v0.4.0)

```python
def process_prompt(
    ai_key: str,
    prompt: str,
    *,                          # force keyword-only after here
    system: str | None = None,  # NEW: overrides provider default
    verbose: bool = False,      # was positional, now keyword-only
    use_cache: bool = True,     # was positional, now keyword-only
) -> AIResponse:
```

Making `verbose` and `use_cache` keyword-only (`*,`) is a minor breaking change but catches a surprising number of call-site mistakes where positional args get swapped.

