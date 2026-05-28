"""
Web search wrapper for WebWalkerQA PoC.

Thin wrapper around the Serper API (already used by the deepresearch benchmark).
Falls back to a stub when no API key is available (useful for unit tests).

Uses async HTTP (httpx) so search does not block the asyncio event loop when
running many concurrent rollouts.

Usage:
    from webwalkerqa.search import web_search
    results = await web_search("who wrote Pride and Prejudice")
"""

import asyncio
import os
import re
import json
import time
import random
import sys
from pathlib import Path
from typing import Optional

# Reuse retrieval.py from the deepresearch benchmark if available
_RETRIEVAL_PATH = Path(__file__).parent.parent.parent / "benchmarks" / "deepresearch_llm_modeling"
if _RETRIEVAL_PATH.exists() and str(_RETRIEVAL_PATH) not in sys.path:
    sys.path.insert(0, str(_RETRIEVAL_PATH))

# Lazy singleton for async HTTP client (connection pooling across many concurrent searches)
_http_client: Optional["httpx.AsyncClient"] = None


def _get_http_client() -> "httpx.AsyncClient":
    """Return a shared AsyncClient; create on first use."""
    global _http_client
    if _http_client is None:
        import httpx
        _http_client = httpx.AsyncClient(timeout=20.0)
    return _http_client


async def web_search(query: str, max_chars: int = 3000) -> str:
    """
    Search and return a formatted result string (async, non-blocking).

    Backend is selected by the SEARCH_BACKEND env var:
      - "clueweb" → ClueWeb22 API (https://www.clueweb22.us/wiki18/search)
      - anything else / unset → Serper (default)
    """
    query = query.strip()
    if not query:
        return "No results: empty query."

    backend = os.getenv("SEARCH_BACKEND", "serper").lower()

    try:
        if backend == "clueweb":
            result = await _query_clueweb_async(query)
        else:
            api_key = os.getenv("SERPER_API_KEY")
            if not api_key:
                return f"[Search unavailable: SERPER_API_KEY not set] Query was: {query}"
            result = await _query_serper_async(query, api_key)
    except Exception as e:
        return f"[Search error: {e}] Query: {query}"

    if len(result) > max_chars:
        result = result[:max_chars] + "\n[... truncated ...]"
    return result


async def _query_clueweb_async(query: str, k: int = 3) -> str:
    """ClueWeb22 search API — no key required, returns wiki-style passages."""
    import httpx
    url = "https://www.clueweb22.us/wiki18/search"
    params = {"query": query, "k": k}
    client = _get_http_client()

    for attempt in range(5):
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            if attempt == 4:
                raise
            await asyncio.sleep(min(2.0, 0.3 * (attempt + 1)))

    results = data if isinstance(data, list) else data.get("results", [])
    if not results:
        return f"No results found for: '{query}'"

    snippets = []
    for i, item in enumerate(results[:k], 1):
        title = item.get("title", item.get("id", ""))
        text = item.get("contents", item.get("text", item.get("passage", "")))
        snippets.append(f"{i}. {title}\n   {text[:500]}")

    return f"Search results for '{query}':\n\n" + "\n\n".join(snippets)


async def _query_serper_async(query: str, api_key: str) -> str:
    """Async Serper API call, returns formatted string. Does not block the event loop."""
    import httpx

    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": 10}
    client = _get_http_client()

    last_error = None
    data = None
    for attempt in range(5):
        try:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            last_error = e
            if attempt == 4:
                raise
            await asyncio.sleep(min(2.0, 0.3 * (attempt + 1)))

    assert data is not None
    organic = data.get("organic", [])
    if not organic:
        return f"No results found for: '{query}'"

    snippets = []
    for i, page in enumerate(organic[:10], 1):
        title = page.get("title", "")
        link = page.get("link", "")
        snippet = page.get("snippet", "")
        date = page.get("date", "")
        parts = [f"{i}. [{title}]({link})"]
        if date:
            parts.append(f"   Date: {date}")
        if snippet:
            parts.append(f"   {snippet}")
        snippets.append("\n".join(parts))

    return f"Search results for '{query}':\n\n" + "\n\n".join(snippets)
