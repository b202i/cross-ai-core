# cross-ai-core

[![PyPI version](https://img.shields.io/pypi/v/cross-ai-core.svg)](https://pypi.org/project/cross-ai-core/)
[![Python](https://img.shields.io/pypi/pyversions/cross-ai-core)](https://pypi.org/project/cross-ai-core/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Multi-provider AI dispatcher with MD5-keyed response caching and unified error handling.

Supports **Anthropic**, **xAI (Grok)**, **OpenAI**, **Google Gemini**, and **Perplexity** through a single consistent interface.

## Requirements

- **Python 3.10 or newer** (3.11 recommended for development)
- No upper version limit — tested on 3.10–3.13

## Install

Install only the provider(s) you need:

```bash
pip install "cross-ai-core[anthropic]"   # Claude
pip install "cross-ai-core[gemini]"      # Google Gemini
pip install "cross-ai-core[openai]"      # OpenAI (ChatGPT)
pip install "cross-ai-core[xai]"         # xAI Grok  (uses the OpenAI SDK)
pip install cross-ai-core                # Perplexity only (uses requests, no extra SDK)
```

Install all providers at once (used by [cross-st](https://github.com/b202i/cross-st), which runs all 5 simultaneously):

```bash
pip install "cross-ai-core[all]"
```

## Dependencies

`requests` is always installed — it is used for the Perplexity provider and general HTTP.  
The three provider SDKs are optional extras; pip installs only what you request.

| Extra | Package | Version | Providers covered |
|-------|---------|---------|-------------------|
| *(base)* | `requests` | ≥2.32.4 | Perplexity |
| `[anthropic]` | `anthropic` | ≥0.84.0 | Anthropic / Claude |
| `[gemini]` | `google-genai` | ≥1.65.0 | Google Gemini |
| `[openai]` | `openai` | ≥1.70.0 | OpenAI |
| `[xai]` | `openai` | ≥1.70.0 | xAI / Grok (OpenAI-compatible API) |
| `[all]` | all three above | — | All 5 providers |

## Quick start

```python
import os
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.crossenv"))  # your app loads keys; the library reads os.environ

from cross_ai_core import process_prompt, get_content, get_default_ai

provider = get_default_ai()         # reads DEFAULT_AI from env, falls back to "xai"
result   = process_prompt(
    provider,
    "Explain transformer attention in 3 sentences.",
    system="You are a concise technical writer.",   # omit to use each provider's default
    verbose=False,
    use_cache=True,
)
print(get_content(provider, result.response))
```

## Configuration (environment variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `DEFAULT_AI` | `xai` | Default provider when none is specified |
| `XAI_API_KEY` | — | xAI / Grok API key |
| `ANTHROPIC_API_KEY` | — | Anthropic / Claude API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `PERPLEXITY_API_KEY` | — | Perplexity API key |
| `CROSS_API_CACHE_DIR` | `~/.cross_api_cache/` | Response cache directory |
| `CROSS_NO_CACHE` | — | Set to `1` to disable caching globally |

The library only reads from `os.environ` — it never calls `load_dotenv()` itself.  
Load your `.env` or `~/.crossenv` before importing.  
You only need to set API keys for the providers you actually use.

## Caching

Responses are cached by MD5 hash of the request payload in `~/.cross_api_cache/`.  
The cache is safe to delete at any time.

```python
# Bypass cache for one call
result = process_prompt(provider, prompt, verbose=False, use_cache=False)

# Check if a response was served from cache
if result.was_cached:
    print("from cache")
```

## Development

```bash
cd ~/github/cross-ai-core
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"     # installs the package + pytest + pytest-mock
```

Run the test suite:

```bash
python -m pytest tests/ -v
```

Tests use mocks — no real API keys required.

> **Note:** Keep each repo's `.venv` separate; do not share it with dependent projects.

## Adding a provider

1. Create `cross_ai_core/ai_<name>.py` implementing `BaseAIHandler`
   (`get_payload`, `get_client`, `get_cached_response`, `get_model`, `get_make`,
   `get_content`, `put_content`, `get_data_content`, `get_title`, `get_usage`).
2. Register in `cross_ai_core/ai_handler.py`: add to `AI_HANDLER_REGISTRY` and `AI_LIST`.

## Documentation

- [API reference](docs/api-reference.md) — all public functions, `AIResponse`, parallel calls, error handling
- [Providers](docs/providers.md) — per-provider guide: models, API keys, strengths, free tiers
- [Changelog](CHANGELOG.md)

## Used by

| Project | PyPI | Description |
|---------|------|-------------|
| **cross-st** | [`cross-st`](https://pypi.org/project/cross-st/) | Multi-AI research reports with cross-product fact-checking. Installs this package automatically via `cross-ai-core[all]`. Full CLI toolkit — `pipx install cross-st`. |

> Building something with `cross-ai-core`? Open a PR or issue to get listed here.

## License

MIT — free for personal, academic, and open-source use.  
See [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) for organizational and commercial use.
