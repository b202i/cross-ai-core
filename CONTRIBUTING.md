# Contributing to cross-ai-core

Thank you for your interest in contributing. This document covers everything
you need to get from zero to a passing pull request.

---

## Development setup

Each repo has its own `.venv` — do not share it with `cross-ai` or other projects.

```bash
git clone https://github.com/b202i/cross-ai-core.git
cd cross-ai-core
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"     # package + pytest + pytest-mock
```

## Running tests

```bash
python -m pytest tests/ -v
```

All 80 tests must pass before opening a PR. No real API calls are made —
everything is mocked.

---

## Adding a provider

Adding a new AI provider is the most common contribution. Follow these four steps.

### 1. Create `cross_ai_core/ai_<name>.py`

Copy the structure from an existing provider (e.g. `ai_openai.py`).
Your handler class must subclass `BaseAIHandler` and implement every abstract
method. The only method that touches the network is `_call_api`:

```python
from .ai_base import BaseAIHandler

class MyProviderHandler(BaseAIHandler):

    @classmethod
    def _call_api(cls, client, payload: dict) -> dict:
        """Make the actual API call; return a plain JSON-serialisable dict."""
        response = client.some_method(**payload)
        return response.to_dict()   # or json.loads(response.to_json()), etc.

    @classmethod
    def get_payload(cls, prompt: str, system: str | None = None) -> dict:
        return {
            "model": AI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "system": system or DEFAULT_SYSTEM,
        }

    @classmethod
    def get_client(cls):
        import myprovider_sdk
        return myprovider_sdk.Client(api_key=os.getenv("MYPROVIDER_API_KEY"))

    # ... get_model, get_make, get_content, put_content,
    #     get_data_content, get_title, get_usage
```

`get_cached_response` is **already implemented** in `BaseAIHandler` — you get
MD5 caching for free just by implementing `_call_api`.

### 2. Register in `ai_handler.py`

```python
from .ai_myprovider import MyProviderHandler

AI_HANDLER_REGISTRY = {
    ...
    "myprovider": MyProviderHandler,
}

AI_LIST = [..., "myprovider"]
```

Also add the API key env var to `_API_KEY_ENV_VARS`:

```python
_API_KEY_ENV_VARS = {
    ...
    "myprovider": "MYPROVIDER_API_KEY",
}
```

### 3. Add the optional extra to `pyproject.toml`

```toml
[project.optional-dependencies]
myprovider = ["myprovider-sdk>=1.0"]
all = [..., "myprovider-sdk>=1.0"]
```

### 4. Update `docs/providers.md`

Add a section following the same format as the existing five providers:
key variable, default model, get-a-key link, strengths, notes.

---

## Coding conventions

- **Never call `load_dotenv()`** in library code. The caller loads keys.
- **Never hardcode provider names or model strings** in `ai_handler.py`.
  Use `get_default_ai()` and `get_ai_model(make)`.
- **Never call real AI APIs in tests.** Use `patch.dict(AI_HANDLER_REGISTRY, ...)`.
- Keep `_call_api` thin — one API call, one dict return. Error classification
  and retries are handled by the caller (`handle_api_error`, `retry_with_backoff`).

---

## Version bump checklist

Before opening a PR that changes behaviour:

1. Update `version` in `pyproject.toml`
2. Add a `## [x.y.z]` section to `CHANGELOG.md`
3. Update `docs/providers.md` if the PR adds or changes a provider

The maintainer will tag and publish to PyPI after merge.

---

## Pull request checklist

- [ ] `python -m pytest tests/ -v` — all tests pass
- [ ] New provider: all 4 steps above completed
- [ ] Bug fix: regression test added
- [ ] `CHANGELOG.md` updated
- [ ] No real API keys committed

---

## Questions?

Open a [GitHub issue](https://github.com/b202i/cross-ai-core/issues) or
email matt@makermattdesign.com.

