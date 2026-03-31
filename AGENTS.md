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
```bash
cd ~/github/cross-ai-core
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
```

To use the development version in `cross-ai`:
```bash
cd ~/github/cross
pip install -e ../cross-ai-core/    # editable install of sibling repo
```

## Publishing to PyPI
```bash
# One-time setup (if not already done)
pip install build twine

# Build
cd ~/github/cross-ai-core
python -m build

# Check
twine check dist/*

# Upload (prompts for PyPI credentials or API token)
twine upload dist/*
```

After publishing, update `cross-ai/pyproject.toml` to pin a released version:
```toml
"cross-ai-core>=0.1.0",
```

## Version bump checklist
1. Update `version` in `pyproject.toml`
2. Update `__version__` in `cross_ai_core/__init__.py`
3. `git tag v0.x.0 && git push --tags`
4. `python -m build && twine upload dist/*`
5. In `cross-ai/pyproject.toml`, bump the `cross-ai-core>=` lower bound if needed

