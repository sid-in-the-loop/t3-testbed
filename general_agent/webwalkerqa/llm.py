import os
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_MODEL_ALIASES = {
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt4o-mini": "openai/gpt-4o-mini",
    "gpt-4.1-mini": "openai/gpt-4.1-mini",
    "gpt4.1-mini": "openai/gpt-4.1-mini",
    "gpt-5-nano": "openai/gpt-5-nano",
    "gemini-2.5-flash": "gemini/gemini-2.5-flash",
    "gemini-3-flash-preview": "gemini/gemini-3-flash-preview",
    "qwen3-1.7b": "openai/Qwen/Qwen3-1.7B",
    "qwen3-4b": "openai/Qwen/Qwen3-4B",
    "qwen3-8b": "openai/Qwen/Qwen3-8B",
    "qwq-32b": "openai/Qwen/QwQ-32B-Preview",
    "gemma3-4b": "openai/google/gemma-3-4b-it",
    "gemma3-12b": "openai/google/gemma-3-12b-it",
}

_DEFAULT_API_BASE: Optional[str] = None


def set_api_base(base: Optional[str]):
    """Set the default api_base for all LLM calls (used for vLLM endpoints)."""
    global _DEFAULT_API_BASE
    _DEFAULT_API_BASE = base


def normalize_model(model: str) -> str:
    """Return LiteLLM model string; resolve alias if present (e.g. gpt4.1-mini -> openai/gpt-4.1-mini)."""
    s = (model or "").strip()
    if not s:
        return "openai/gpt-4o-mini"
    if "/" in s:
        return s
    return _MODEL_ALIASES.get(s.lower(), f"openai/{s}")


async def call_llm(
    messages: list[dict],
    model: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    seed: Optional[int] = None,
    api_base: Optional[str] = None,
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
    try:
        import litellm
        litellm.drop_params = True
    except ImportError:
        raise ImportError("litellm is required. Install with: pip install litellm")

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    effective_base = api_base or _DEFAULT_API_BASE
    if effective_base:
        kwargs["api_base"] = effective_base
    if seed is not None:
        seed_val = int(seed) % (2**32)
        kwargs["seed"] = seed_val

    if "gemini" in model.lower():
        kwargs["max_output_tokens"] = max_tokens

    if "qwen3" in model.lower():
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    last_exc = None
    context_truncated = False
    for attempt in range(5):
        try:
            response = await litellm.acompletion(**kwargs)
            break
        except Exception as e:
            last_exc = e
            err_name = type(e).__name__
            if "ContextWindowExceeded" in err_name:
                if not context_truncated:
                    msgs = kwargs.get("messages", [])
                    if msgs:
                        longest_idx = max(range(len(msgs)), key=lambda i: len(str(msgs[i].get("content", ""))))
                        content_str = str(msgs[longest_idx].get("content", ""))
                        if len(content_str) > 1000:
                            half = len(content_str) // 2
                            keep_head = content_str[: half // 2]
                            keep_tail = content_str[-(half // 2):]
                            msgs[longest_idx] = {**msgs[longest_idx], "content": keep_head + "\n[...truncated...]\n" + keep_tail}
                            context_truncated = True
                            continue
                raise
            if attempt < 4:
                is_conn = any(s in err_name for s in ("Connection", "InternalServer", "Timeout", "ServiceUnavailable"))
                base = 16.0 if is_conn else 8.0
                wait = min(base * (2 ** attempt), 120.0)
                await asyncio.sleep(wait)
    else:
        raise last_exc

    choice = response.choices[0]
    content = choice.message.content or ""

    if "gpt-oss-20b" in model.lower():
        if "<|channel|>" in content and "<|message|>" in content:
            after_channel = content.split("<|channel|>", 1)[1]
            channel_name, after_msg = after_channel.split("<|message|>", 1)
            content = after_msg if channel_name.strip() == "final" else ""

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
