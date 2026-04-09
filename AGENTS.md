# AGENTS.md — cross-ai-core

## What This Package Does
`cross-ai-core` is a multi-provider AI dispatcher extracted from the `cross-ai` application.
It provides a single `process_prompt()` interface across Anthropic, xAI, OpenAI, Google Gemini,
and Perplexity, with MD5-keyed response caching and unified error handling.

## Repo layout
```
cross_ai_core/
  __init__.py          ← public API re-exports
  ai_base.py           ← BaseAIHandler ABC + _get_cache_dir()
  ai_handler.py        ← registry, process_prompt(), get_default_ai(), check_api_key()
  ai_error_handler.py  ← quota / rate-limit / transient error classification
  ai_anthropic.py      ← Anthropic / Claude provider
  ai_xai.py            ← xAI / Grok provider
  ai_openai.py         ← OpenAI provider
  ai_gemini.py         ← Google Gemini provider
  ai_perplexity.py     ← Perplexity provider
pyproject.toml
README.md
```

## Key conventions

**Never call `load_dotenv()` in this library.** The calling application is responsible
for loading keys into `os.environ` before importing. The library only reads env vars.

**Never hardcode AI provider names or model strings.** Use `get_default_ai()` and
`get_ai_model(make)`.

**Cache path** is resolved by `_get_cache_dir()` in `ai_base.py` — reads
`CROSS_API_CACHE_DIR` env var, defaults to `~/.cross_api_cache/`.

## Adding a provider
1. Create `cross_ai_core/ai_<name>.py` implementing `BaseAIHandler`
2. Import `_get_cache_dir` from `.ai_base` for the cache directory
3. Register in `ai_handler.py`: add to `AI_HANDLER_REGISTRY` and `AI_LIST`
4. Bump version in `pyproject.toml`

## Development setup

Each repo has its **own `.venv`** — do not share the venv between `cross-ai-core` and `cross-ai`.

```bash
cd ~/github/cross-ai-core
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # installs the package + pytest + pytest-mock
```

To use the local development version inside `cross-ai` at the same time:

```bash
# In a separate terminal with cross-ai's venv active:
cd ~/github/cross
pip install -e ../cross-ai-core/    # editable — changes are picked up instantly
```

## Running tests

```bash
cd ~/github/cross-ai-core
source .venv/bin/activate
python -m pytest tests/ -v
```

Tests cover:
- `test_ai_base.py` — `_get_cache_dir` env resolution, `BaseAIHandler` ABC enforcement
- `test_ai_error_handler.py` — quota/rate-limit/transient classification, `handle_api_error` exit behaviour
- `test_ai_handler.py` — registry completeness, `get_default_ai`, `check_api_key`, `AIResponse` backward compat, `process_prompt` with mocked handlers, `_make` stamp, `get_content_auto`, `put_content_auto`

**Never call real AI APIs in tests.** Use `unittest.mock.patch.dict(AI_HANDLER_REGISTRY, ...)` to inject mock handler classes.



## Publishing to PyPI

See **[RELEASE.md](RELEASE.md)** for the full step-by-step process, including
first-time PyPI account and token setup, TestPyPI trial uploads, the version
bump checklist, tagging, and the hotfix workflow.

Quick reference (assumes `~/.pypirc` is already configured):

```bash
# bump version in pyproject.toml, then:
rm -rf dist/ && python -m build && twine check dist/*
twine upload --repository testpypi dist/*   # trial
twine upload dist/*                         # real
git tag v0.x.y && git push --tags
```

## Version bump checklist
1. Update `version` in `pyproject.toml` — this is the **single source of truth**; `__init__.py` reads the version via `importlib.metadata` and does **not** contain a hardcoded version string
2. Add entry to `CHANGELOG.md`
3. `git tag v0.x.y && git push --tags`
4. `python -m build && twine upload dist/*`
5. In `cross-ai/pyproject.toml`, bump the `cross-ai-core>=` lower bound if needed

