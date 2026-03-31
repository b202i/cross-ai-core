import hashlib
import json
import os

from anthropic import Anthropic

from .ai_base import BaseAIHandler, _get_cache_dir

AI_MAKE = "xai"
AI_MODEL = "grok-4-1-fast-reasoning"
MAX_TOKENS = 16000

"""
xAI Grok Model Family (via Anthropic-compatible API at https://api.x.ai)

grok-4-1-fast-reasoning  # Fast, reasoning-optimized Grok 4 variant; best for interactive tasks

# Grok 4 Family (Current - 2025/2026)
grok-4-latest                    # Most capable Grok model; advanced reasoning, deep analysis
grok-4-0709                      # Dated snapshot of Grok 4 (09 Jul 2025)

# Grok 3 Family
grok-3-latest                    # Balanced capability and speed; strong for long-form content
grok-3-fast-latest               # Faster Grok 3 variant; optimized for throughput
grok-3-mini-latest               # Lightweight Grok 3; cost-efficient, high-volume tasks
grok-3-mini-fast-latest          # Fastest, most cost-efficient Grok 3 variant

# Grok 2 Family (Legacy)
grok-2-latest                    # Previous generation; general-purpose tasks
grok-2-vision-latest             # Grok 2 with vision/multimodal support

# Grok Beta (Legacy)
grok-beta                        # Original public Grok model; superseded by Grok 2+

# Differences Between Grok Model Families

Grok 4 Family (Current)

grok-4-latest:
  - Most intelligent Grok model; best for complex reasoning and research synthesis
  - Designed to compete with frontier models on coding, math, and analysis
  - Highest capability and cost; ideal for wiki-style report generation
  - Uses xAI's Anthropic-compatible API; drop-in with existing Anthropic SDK usage

Grok 3 Family

grok-3-latest:
  - Strong general-purpose model; best balance of quality and speed in Grok 3
  - Excellent for long-form content, summarization, and structured reports
grok-3-fast-latest:
  - Optimized for lower latency; good for interactive or real-time tasks
grok-3-mini-latest / grok-3-mini-fast-latest:
  - Cost-efficient variants for high-volume or simpler tasks

Grok 2 Family (Legacy)

grok-2-latest:       General-purpose; superseded by Grok 3+
grok-2-vision-latest: Adds image/vision understanding to Grok 2

# API Notes
xAI uses an Anthropic-compatible API endpoint (https://api.x.ai).
The Anthropic SDK is used directly with a custom base_url — no separate SDK required.
Authentication uses the XAI_API_KEY environment variable (xai_api_key in .env).
"""


class XAIHandler(BaseAIHandler):

    @classmethod
    def get_payload(cls, prompt: str):
        return get_xai_payload(prompt)

    @classmethod
    def get_client(cls):
        return get_xai_client()

    @classmethod
    def get_cached_response(cls, client, payload, verbose, use_cache):
        return get_xai_cached_response(client, payload, verbose, use_cache)

    @classmethod
    def get_model(cls):
        return AI_MODEL

    @classmethod
    def get_make(cls):
        return AI_MAKE

    @classmethod
    def get_content(cls,  gen_content):
        return get_content(gen_content)

    @classmethod
    def put_content(cls, report, gen_content):
        return put_content(report, gen_content)

    @classmethod
    def get_data_content(cls,  select_data):
        return get_data_content(select_data)

    @classmethod
    def get_title(cls, gen_content):
        return get_title(gen_content)

    @classmethod
    def get_usage(cls, response: dict) -> dict:
        """Extract token counts from an xAI (Anthropic-compatible) response dict."""
        usage = response.get("usage", {})
        inp = usage.get("input_tokens", 0)
        out = usage.get("output_tokens", 0)
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}


def get_xai_cached_response(client, payload, verbose=False, use_cache=True):
    if not use_cache:
        if verbose:
            print("Cache disabled; fetching fresh data.")
        response = client.messages.create(**payload)
        json_str = response.to_json()
        json_response = json.loads(json_str)
        return json_response, False  # Not cached
    else:
        # Convert param to a string for hashing
        param_str = json.dumps(payload, sort_keys=True)
        md5_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()

        # Construct the cache file path
        cache_dir = _get_cache_dir()
        cache_file = os.path.join(cache_dir, f"{md5_hash}.json")

        # Check if the response is already in cache
        if os.path.exists(cache_file):
            if verbose:
                print(f"api_cache: Using api_cache: {cache_file}")
            with open(cache_file, 'r') as f:
                return json.load(f), True  # Cached
        else:
            if verbose:
                print("api_cache: cache miss, submitting API request")

            # If not in cache, fetch the response
            response = client.messages.create(**payload)
            json_str = response.to_json()
            json_response = None

            try:
                json_response = json.loads(json_str)
            except ValueError as e:
                print(f"api_cache: failed json conversion: {e}")

            if json_response is None:
                return {}, False

            # Save to cache
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                if verbose:
                    print(f"api_cache: api_cache dir created: {cache_dir}")

            try:
                with open(cache_file, 'w') as f:
                    json.dump(json_response, f)
            except Exception as e:
                print(f"api_cache: An error occurred: {str(e)}")

            if verbose:
                print(f"api_cache: api_cache created: {cache_file}")

            return json_response, False  # Fresh API call, not cached


def get_xai_payload(prompt):
    gen_payload = {  # Store parameters into a dictionary for calling xAI and saving with the output
        "model": AI_MODEL,
        "max_tokens": MAX_TOKENS,
        "system": "You are a seasoned investigative reporter, "
                  "striving to be accurate, fair and balanced.",
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }
    return gen_payload


def get_xai_client():
    # xAI uses an Anthropic-compatible API; the Anthropic SDK is used with a custom base_url
    client = Anthropic(
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai",
    )
    return client


def get_title(story_instance):
    content = get_data_content(story_instance)
    title = content.splitlines()[0]
    return title


def get_content(gen_response):
    # Scan content blocks by key in case future Grok models return multiple blocks
    for block in gen_response["content"]:
        if block.get("type") == "text":
            return block["text"]
    return gen_response["content"][0]["text"]  # fallback


def put_content(report, gen_response):
    # Scan content blocks by key; update the text block
    for block in gen_response["content"]:
        if block.get("type") == "text":
            block["text"] = report
            return gen_response
    gen_response["content"][0]["text"] = report  # fallback
    return gen_response


def get_data_content(select_data):
    # Scan content blocks by key; robust against multi-block responses
    for block in select_data["gen_response"]["content"]:
        if "text" in block:
            return block["text"]
    raise ValueError("No element with 'text' key found in content array")
