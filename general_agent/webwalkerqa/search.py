"""
Web search wrapper for WebWalkerQA PoC.

Thin wrapper around the Serper API (already used by the deepresearch benchmark).
Falls back to a stub when no API key is available (useful for unit tests).

Usage:
    from webwalkerqa.search import web_search
    results = web_search("who wrote Pride and Prejudice")
"""

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


def web_search(query: str, max_chars: int = 3000) -> str:
    """
    Search the web and return a formatted result string.

    Args:
        query: Search query.
        max_chars: Truncate result to this many characters.

    Returns:
        Formatted search result string (snippets from top results).
    """
    query = query.strip()
    if not query:
        return "No results: empty query."

    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return f"[Search unavailable: SERPER_API_KEY not set] Query was: {query}"

    try:
        result = _query_serper(query, api_key)
    except Exception as e:
        return f"[Search error: {e}] Query: {query}"

    if len(result) > max_chars:
        result = result[:max_chars] + "\n[... truncated ...]"
    return result


def _query_serper(query: str, api_key: str) -> str:
    """Direct Serper API call, returns formatted string."""
    import requests

    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": 10}

    for attempt in range(5):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(min(2.0, 0.3 * (attempt + 1)))

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
