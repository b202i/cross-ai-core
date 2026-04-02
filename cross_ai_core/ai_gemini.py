import os
import json
import hashlib

from .ai_base import BaseAIHandler, _get_cache_dir

AI_MAKE = "gemini"
AI_MODEL = "gemini-2.5-flash"   # 02mar26 — confirmed available via list_gemini_models.py
MAX_TOKENS = 16000              # gemini-2.5-flash supports up to 65k output tokens

"""
Google Gemini Model Families (via google-genai SDK, v1beta)
Confirmed available models from list_gemini_models.py (02mar26):

# Gemini 2.5 Family (Current Default — all use short IDs on v1beta)
gemini-2.5-flash                 # Best price/performance; 1M context — CURRENT DEFAULT
gemini-2.5-pro                   # Most capable; deep reasoning, long context
gemini-2.5-flash-lite            # Fastest, most cost-efficient 2.5 variant
gemini-2.5-flash-image           # Flash with image generation support
gemini-2.5-flash-preview-tts     # Text-to-speech preview
gemini-2.5-pro-preview-tts       # Pro text-to-speech preview
gemini-2.5-flash-lite-preview-09-2025  # Dated preview snapshot

# Gemini 2.0 Family (GA)
gemini-2.0-flash                 # Balanced speed and capability; multimodal
gemini-2.0-flash-001             # Pinned snapshot of gemini-2.0-flash
gemini-2.0-flash-lite            # Cost-efficient GA variant
gemini-2.0-flash-lite-001        # Pinned snapshot of gemini-2.0-flash-lite
gemini-2.0-flash-exp-image-generation  # Experimental image generation

# Gemini 3 Family (Very early preview)
gemini-3-pro-preview             # Early preview of Gemini 3 Pro
gemini-3-flash-preview           # Early preview of Gemini 3 Flash
gemini-3.1-pro-preview           # Gemini 3.1 Pro preview
gemini-3.1-flash-image-preview   # 3.1 Flash with image generation

# Convenience Aliases
gemini-flash-latest              # Always latest Flash model
gemini-flash-lite-latest         # Always latest Flash Lite model
gemini-pro-latest                # Always latest Pro model

# Gemma (Open Weights Models)
gemma-3-1b-it / gemma-3-4b-it / gemma-3-12b-it / gemma-3-27b-it
gemma-3n-e4b-it / gemma-3n-e2b-it

# Gemini 1.5 (Legacy — NOT available on this account's v1beta endpoint)

# Differences Between Gemini Model Families

Gemini 2.5 Family (Current)

gemini-2.5-flash:
  - Best balance of speed, cost, and intelligence; recommended for report generation
  - Supports thinking mode (adaptive reasoning budget) for deeper analysis
  - 1M token context window; strong multimodal capabilities
  - Short model ID works on v1beta — no versioned suffix needed

gemini-2.5-pro:
  - Most intelligent model; best for complex reasoning and research
  - Long context window (1M tokens); ideal for wiki-style long-form reports
  - Higher cost; reserve for the most demanding tasks

gemini-2.5-flash-lite:
  - Fastest and most cost-efficient; best for high-volume tasks

Gemini 2.0 Family (GA Fallback)

gemini-2.0-flash:
  - Fully GA; reliable quota on all paid accounts
  - Strong multimodal and agentic capabilities
  - Use as fallback if 2.5 quota is exhausted

Gemini 3 (Very Early Preview)
  - Experimental; expect breaking changes, instability, and limited availability

# API Versions
v1alpha: Early access; unstable
v1beta:  Required for systemInstruction and GenerateContentConfig — use this
v1:      GA endpoint; does NOT support systemInstruction field
"""


class GeminiHandler(BaseAIHandler):

    @classmethod
    def get_payload(cls, prompt: str):
        return get_gemini_payload(prompt)

    @classmethod
    def get_client(cls):
        return get_gemini_client()

    @classmethod
    def get_cached_response(cls, client, payload, verbose, use_cache):
        return get_gemini_cached_response(client, payload, verbose, use_cache)

    @classmethod
    def get_model(cls):
        return AI_MODEL

    @classmethod
    def get_make(cls):
        return AI_MAKE

    @classmethod
    def get_content(cls, gen_content):
        return get_content(gen_content)

    @classmethod
    def put_content(cls, report, gen_content):  # Used by st-fix
        return put_content(report, gen_content)

    @classmethod
    def get_data_content(cls, select_data):
        return get_data_content(select_data)

    @classmethod
    def get_title(cls, gen_content):
        return get_title(gen_content)

    @classmethod
    def get_usage(cls, response: dict) -> dict:
        """Extract token counts from a Gemini response dict.
        Gemini flattens usage_metadata to the top level in get_json_response."""
        inp = response.get("prompt_token_count", 0)
        out = response.get("candidates_token_count", 0)
        tot = response.get("total_token_count") or (inp + out)
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": tot}


def get_json_response(response):
    json_response = {
        "model_version": response.model_version,
        "text": response.text,
        "candidates_token_count": response.usage_metadata.candidates_token_count,
        "prompt_token_count": response.usage_metadata.prompt_token_count,
        "total_token_count": response.usage_metadata.total_token_count,
    }
    return json_response


def get_gemini_cached_response(client, payload, verbose=False, use_cache=False):
    if not use_cache:
        if verbose:
            print("Cache disabled; fetching fresh data.")
        response = client.models.generate_content(
            model=AI_MODEL,
            contents=payload["contents"],
            config=get_gemini_config(),
        )
        json_response = get_json_response(response)
        return json_response, False  # Not cached
    else:
        # Only serialize JSON-safe fields for hashing
        param_str = json.dumps({"model": AI_MODEL, "contents": payload["contents"]}, sort_keys=True)
        md5_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()

        # Construct the cache file path
        cache_dir = _get_cache_dir()
        cache_file = os.path.join(cache_dir, f"{md5_hash}.json")

        # Check if the response is already in cache
        if os.path.exists(cache_file):
            if verbose:
                print(f"api_cache: Using cache_file: {cache_file}")
            with open(cache_file, 'r') as f:
                return json.load(f), True  # Cached
        else:
            if verbose:
                print("api_cache: cache miss, submitting API request")

            # If not in cache, fetch the response
            response = client.models.generate_content(
                model=AI_MODEL,
                contents=payload["contents"],
                config=get_gemini_config(),
            )
            json_response = get_json_response(response)

            # Save to cache
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                if verbose:
                    print(f"api_cache: api_cache/ dir created: {cache_dir}")

            try:
                with open(cache_file, 'w') as f:
                    json.dump(json_response, f)
                    if verbose:
                        print(f"api_cache: file created: {cache_file}")
            except Exception as e:
                print(f"api_cache: file write error: {str(e)}")

            return json_response, False  # Fresh API call, not cached


def get_gemini_config():
    """Return the GenerateContentConfig separately — not stored in the JSON payload."""
    from google.genai import types
    return types.GenerateContentConfig(
        system_instruction="You are a seasoned investigative reporter, "
                           "striving to be accurate, fair and balanced.",
        max_output_tokens=MAX_TOKENS,
        temperature=0.7,        # 0.0 = deterministic, 1.0 = creative; 0.7 balances both
        top_p=0.95,             # Nucleus sampling; limits token selection to top 95% probability mass
    )


def get_gemini_payload(prompt_from_file):
    # Gemini payload contains only JSON-serializable fields for storage and hashing.
    # GenerateContentConfig is built separately in get_gemini_config() and never stored.
    payload = {
        "model": AI_MODEL,
        "contents": prompt_from_file,
    }
    return payload


def get_gemini_client():
    from google import genai
    from google.genai import types
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
        http_options=types.HttpOptions(api_version='v1beta')  # v1beta required: v1 does not support systemInstruction
    )
    return client


def get_title(story_instance):
    content = get_data_content(story_instance)
    title = content.splitlines()[0]
    return title


def get_content(gen_response):
    text = gen_response["text"]
    return text


def put_content(report, gen_response):
    gen_response["text"] = report
    return gen_response


def get_data_content(select_data):
    content = select_data["gen_response"]["text"]
    return content
