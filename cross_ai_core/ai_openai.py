import os
import json
import hashlib
from .ai_base import BaseAIHandler, _get_cache_dir

AI_MAKE = "openai"
AI_MODEL = "gpt-4o"  # 02mar26
MAX_TOKENS = 16000

"""
OpenAI GPT Model Family (via OpenAI API at https://api.openai.com)

# GPT-4.5 Family
gpt-4.5-preview                  # Most capable GPT model; requires special API tier access

# GPT-4o Family (Current Default)
gpt-4o                           # Flagship multimodal model; text, image, audio; best speed/quality balance
gpt-4o-mini                      # Fast, cost-efficient; good for high-volume tasks
gpt-4o-audio-preview             # GPT-4o with native audio input/output support

# o-Series Reasoning Models
o3                               # Most powerful reasoning model; best for complex multi-step problems
o3-mini                          # Faster, cost-efficient reasoning; strong at math and coding
o4-mini                          # Latest compact reasoning model; fast and cost-efficient
o1                               # Original deep reasoning model; thorough but slower
o1-mini                          # Lightweight o1; cost-efficient reasoning

# Differences Between OpenAI Model Families

GPT-4.5 Family

gpt-4.5-preview:
  - Most capable GPT model; best for nuanced instruction following and long-form writing
  - Requires special API tier access — not available on standard accounts
  - Higher cost; upgrade path when API access is available

GPT-4o Family (Current Default)

gpt-4o:
  - Flagship multimodal model; handles text, images, and audio natively
  - Best balance of speed, capability, and cost for most tasks
  - Strong for summarization, Q&A, content generation, and data analysis
  - Recommended default for wiki-style report generation
gpt-4o-mini:
  - Lightweight, fast, and cost-efficient variant of GPT-4o
  - Best for high-throughput or latency-sensitive tasks

o-Series Reasoning Models

o3 / o4-mini:
  - Purpose-built for deep, multi-step reasoning tasks
  - Best for math, coding, logic puzzles, and scientific analysis
  - o3: maximum capability; o4-mini: fast and cost-efficient
  - Note: reasoning models do not support system messages or temperature

# API Notes
OpenAI uses the native OpenAI SDK (not an Anthropic-compatible wrapper).
The openai.OpenAI client is used directly.
Authentication uses the OPENAI_API_KEY environment variable (openai_api_key in .env).
Note: o-series reasoning models do not support system role or temperature parameters.
"""


class OpenAIHandler(BaseAIHandler):

    @classmethod
    def get_payload(cls, prompt: str):
        return get_openai_payload(prompt)

    @classmethod
    def get_client(cls):
        return get_openai_client()

    @classmethod
    def get_cached_response(cls, client, payload, verbose, use_cache):
        return get_openai_cached_response(client, payload, verbose, use_cache)

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
        """Extract token counts from an OpenAI-format response dict."""
        usage = response.get("usage", {})
        inp = usage.get("prompt_tokens", 0)
        out = usage.get("completion_tokens", 0)
        tot = usage.get("total_tokens") or (inp + out)
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": tot}


def get_openai_cached_response(client, payload, verbose=False, use_cache=True):

    if not use_cache:
        if verbose:
            print("Cache disabled; fetching fresh data.")
        response = client.chat.completions.create(**payload)
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
                print(f"api_cache: Using cache_file: {cache_file}")
            with open(cache_file, 'r') as f:
                return json.load(f), True  # Cached
        else:
            if verbose:
                print("api_cache: cache miss, submitting API request")

            # If not in cache, fetch the response
            response = client.chat.completions.create(**payload)
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
                    print(f"api_cache: api_cache/ dir created: {cache_dir}")

            try:
                with open(cache_file, 'w') as f:
                    json.dump(json_response, f)
                    if verbose:
                        print(f"api_cache: file created: {cache_file}")
            except Exception as e:
                print(f"api_cache: file write error: {str(e)}")

            return json_response, False  # Fresh API call, not cached


def get_openai_payload(prompt_from_file):
    payload = {
        "model": AI_MODEL,
        "messages": [
            {
                "role": "system",           # System instruction — consistent with other AI handlers
                "content": "You are a seasoned investigative reporter, "
                           "striving to be accurate, fair and balanced.",
            },
            {
                "role": "user",
                "content": prompt_from_file,
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,         # 0.0 = deterministic, 1.0 = creative; 0.7 balances both
        "top_p": 0.95,              # Nucleus sampling; limits token selection to top 95% probability mass
    }
    return payload


def get_openai_client():
    import openai
    # OpenAI native SDK — not a compatibility wrapper
    client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    return client


def get_title(story_instance):
    content = get_data_content(story_instance)
    title = content.splitlines()[0]
    return title


def get_content(gen_response):
    text = gen_response["choices"][0]["message"]["content"]
    return text


def put_content(report, gen_response):
    gen_response["choices"][0]["message"]["content"] = report
    return gen_response


def get_data_content(select_data):
    content = select_data["gen_response"]["choices"][0]["message"]["content"]
    return content
