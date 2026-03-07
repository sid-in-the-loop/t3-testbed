"""
LLM API wrapper for WebWalkerQA PoC.

Thin wrapper around LiteLLM for direct (non-MCP) calls.
Used by both s1 and T³ methods.

Supports any LiteLLM-compatible model string:
  openai/gpt-4o-mini
  gemini/gemini-2.5-flash
  bedrock/...
  anthropic/claude-...
"""

import os
import asyncio
import logging
from typing import Optional

import litellm
from litellm.caching import Cache
litellm.drop_params = True  # Drop unsupported params silently

logger = logging.getLogger(__name__)

# LLM Caching disabled due to disk I/O issues on cluster filesystem

# Configure gpt-5-nano with provided pricing
litellm.model_cost["gpt-5-nano"] = {
    "max_tokens": 128000,
    "input_cost_per_token": 0.05 / 1e6,
    "output_cost_per_token": 0.005 / 1e6,
    "cache_read_input_token_cost": 0.40 / 1e6,
}
litellm.model_alias_map["gpt-5-nano"] = "openai/gpt-4o-mini"


async def call_llm(
    messages: list[dict],
    model: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    seed: Optional[int] = None,
) -> tuple[str, int, int]:
    """
    Call LLM and return (text_content, prompt_tokens, output_tokens).

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        model: LiteLLM model string.
        max_tokens: Maximum output tokens.
        temperature: Sampling temperature.
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple of (text, prompt_tokens, output_tokens).
    """
    if os.getenv("OPENAI_API_KEY"):
        litellm.api_key = os.getenv("OPENAI_API_KEY")

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed is not None:
        kwargs["seed"] = seed

    # Retry logic (same pattern as existing codebase)
    last_exc = None
    for attempt in range(5):
        try:
            response = await litellm.acompletion(**kwargs)
            break
        except Exception as e:
            last_exc = e
            if attempt < 4:
                wait = min(8.0 * (2 ** attempt), 64.0)
                await asyncio.sleep(wait)
    else:
        raise last_exc

    choice = response.choices[0]
    content = choice.message.content or ""

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

    return content, prompt_tokens, output_tokens


def call_llm_sync(
    messages: list[dict],
    model: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    seed: Optional[int] = None,
) -> tuple[str, int, int]:
    """Synchronous wrapper around call_llm."""
    return asyncio.run(call_llm(messages, model, max_tokens, temperature, seed))
