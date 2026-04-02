import hashlib
import json
import os


from .ai_base import BaseAIHandler, _get_cache_dir

AI_MAKE = "anthropic"
AI_MODEL = "claude-opus-4-5"  # 02mar26
MAX_TOKENS = 16000             # min 16000 required for extended thinking

"""
or direct Anthropic API:

# Claude 3 Family (Legacy)
claude-3-haiku-20240307
claude-3-sonnet-20240229
claude-3-opus-20240229

# Claude 3.5 Family
claude-3-5-haiku-20241022
claude-3-5-sonnet-v2-20241022

# Claude 3.7 Family
claude-3-7-sonnet-20250219       # First hybrid reasoning model; extended thinking mode

# Claude 4 Family (Current - 2025/2026)
claude-sonnet-4-5                # Latest Sonnet; fast, highly capable, best for most tasks
claude-opus-4-5                  # Most intelligent Claude 4 model; best for complex reasoning

# Differences Between Claude Model Families

Claude 4 Family (Current)

claude-opus-4-5:
  - Most intelligent model in the Claude 4 family
  - Best for advanced reasoning, research synthesis, and complex multi-step tasks
  - Highest capability, higher cost; ideal for wiki-style report generation
  - Supports extended thinking for deep analytical work

claude-sonnet-4-5:
  - Balanced intelligence and speed in the Claude 4 family
  - Best for high-throughput tasks, data processing, and content generation
  - Faster and more cost-effective than Opus 4

Claude 3.7 Family

claude-3-7-sonnet-20250219:
  - First hybrid reasoning model on the market
  - Extended thinking capabilities for complex problem-solving and nuanced analysis
  - Strong for strategic analysis and advanced coding tasks

Claude 3.5 Family

claude-3-5-sonnet-v2-20241022:
  - Most intelligent in the 3.5 family; balances top-tier performance with speed
claude-3-5-haiku-20241022:
  - Fastest and most cost-effective in the 3.5 family

Claude 3 Family (Legacy)

Opus:   Strong on highly complex tasks like math and coding; R&D and advanced analysis
Sonnet: Balances intelligence and speed for high-throughput tasks
Haiku:  Near-instant responsiveness; live support, translations, content moderation
"""


class AnthropicHandler(BaseAIHandler):

    @classmethod
    def get_payload(cls, prompt: str):
        return get_anthropic_payload(prompt)

    @classmethod
    def get_client(cls):
        return get_anthropic_client()

    @classmethod
    def get_cached_response(cls, client, payload, verbose, use_cache):
        return get_anthropic_cached_response(client, payload, verbose, use_cache)

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
    def put_content(cls, report, gen_content):
        return put_content(report, gen_content)

    @classmethod
    def get_data_content(cls, select_data):
        return get_data_content(select_data)

    @classmethod
    def get_title(cls, gen_content):
        return get_title(gen_content)

    @classmethod
    def get_usage(cls, response: dict) -> dict:
        """Extract token counts from an Anthropic-format response dict.
        Note: extended thinking tokens are included in output_tokens."""
        usage = response.get("usage", {})
        inp = usage.get("input_tokens", 0)
        out = usage.get("output_tokens", 0)
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}


def get_anthropic_cached_response(client, payload, verbose=False, use_cache=True):
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
        json_response = {"message": "init"}
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
                return {}

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


def get_anthropic_payload(prompt):
    gen_payload = {  # Store parameters into a dictionary for calling X and saving with the output
        "model": AI_MODEL,
        "max_tokens": MAX_TOKENS,
        "thinking": {                  # Extended thinking: deep reasoning mode (Claude 4 / 3.7+)
            "type": "enabled",
            "budget_tokens": 10000,    # Tokens reserved for internal reasoning (must be < max_tokens)
        },
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


def get_anthropic_client():
    from anthropic import Anthropic
    client = Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        # base_url="https://api.x.ai",
    )
    return client


def get_title(story_instance):
    content = get_data_content(story_instance)
    title = content.splitlines()[0]
    return title


def get_content(gen_response):
    # Extended thinking returns multiple blocks; find the text block
    for block in gen_response["content"]:
        if block.get("type") == "text":
            return block["text"]
    return gen_response["content"][0]["text"]  # fallback


def put_content(report, gen_response):
    # Extended thinking returns multiple blocks; update the text block
    for block in gen_response["content"]:
        if block.get("type") == "text":
            block["text"] = report
            return gen_response
    gen_response["content"][0]["text"] = report  # fallback
    return gen_response


def get_data_content(select_data):
    # Extended thinking responses contain multiple blocks; find the text block
    for block in select_data["gen_response"]["content"]:
        if block.get("type") == "text":
            return block["text"]
    return select_data["gen_response"]["content"][0]["text"]  # fallback
