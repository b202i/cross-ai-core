# cross-ai-core

Multi-provider AI dispatcher with MD5-keyed response caching and unified error handling.

Supports **Anthropic**, **xAI**, **OpenAI**, **Google Gemini**, and **Perplexity** through a single consistent interface.

## Requirements

- **Python 3.10 or newer** (3.11 recommended for development)
- No upper version limit — tested on 3.10–3.13

## Install

```bash
pip install cross-ai-core
```

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `anthropic` | ≥0.84.0 | Anthropic / Claude API client |
| `google-genai` | ≥1.65.0 | Google Gemini API client |
| `openai` | ≥1.70.0 | OpenAI and xAI (Grok) API client |
| `requests` | ≥2.32.4 | HTTP for Perplexity API |

## Quick start

```python
from dotenv import load_dotenv
load_dotenv("~/.crossenv")          # your app loads keys; the library reads os.environ

from cross_ai_core import process_prompt, get_content, get_default_ai

provider = get_default_ai()         # reads DEFAULT_AI from env, falls back to "xai"
result   = process_prompt(provider, "Explain transformer attention in 3 sentences.",
                          verbose=False, use_cache=True)
print(get_content(provider, result.response))
```

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
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

## Adding a provider

1. Create `cross_ai_core/ai_<name>.py` implementing `BaseAIHandler`
   (`get_payload`, `get_client`, `get_cached_response`, `get_model`, `get_make`,
   `get_content`, `put_content`, `get_data_content`, `get_title`, `get_usage`).
2. Register in `cross_ai_core/ai_handler.py`: add to `AI_HANDLER_REGISTRY` and `AI_LIST`.

## License

MIT
