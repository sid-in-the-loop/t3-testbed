"""
Web search wrapper for WebWalkerQA PoC.

Thin wrapper around the Serper API.
Now async-native to avoid blocking the event loop during high-concurrency runs.

Usage:
    from webwalkerqa.search import web_search
    results = await web_search("who wrote Pride and Prejudice")
"""

import os
import json
import asyncio
import sys
import httpx
import sqlite3
import hashlib
from pathlib import Path
from typing import Optional

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DB = CACHE_DIR / "search_cache.db"

class SearchCache:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS search_results "
                "(query_hash TEXT PRIMARY KEY, query TEXT, result TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"
            )

    def get(self, query: str) -> Optional[str]:
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT result FROM search_results WHERE query_hash = ?", (query_hash,))
                row = cursor.fetchone()
                return row[0] if row else None
        except sqlite3.OperationalError as e:
            # Handle potential locking or disk I/O issues in concurrent environments
            print(f"SearchCache.get Error: {e}")
            return None

    def set(self, query: str, result: str):
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO search_results (query_hash, query, result) VALUES (?, ?, ?)",
                    (query_hash, query, result),
                )
        except sqlite3.OperationalError as e:
            # If database is locked or hit disk I/O error, skip caching this result
            print(f"SearchCache.set Error: {e}")
            pass

_cache = SearchCache(CACHE_DB)

async def web_search(query: str, max_chars: int = 3000) -> str:
    """
    Search the web and return a formatted result string (async).
    Uses a local SQLite cache to avoid redundant API calls.

    Args:
        query: Search query.
        max_chars: Truncate result to this many characters.

    Returns:
        Formatted search result string (snippets from top results).
    """
    query = query.strip()
    if not query:
        return "No results: empty query."

    # Check cache first
    cached_result = _cache.get(query)
    if cached_result:
        if len(cached_result) > max_chars:
            return cached_result[:max_chars] + "\n[... truncated ...]"
        return cached_result

    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return f"[Search unavailable: SERPER_API_KEY not set] Query was: {query}"

    try:
        result = await _query_serper(query, api_key)
        # Cache the full result before truncation
        if not result.startswith("[Search error"):
            _cache.set(query, result)
    except Exception as e:
        # Check for specific Serper error messages in the response if possible
        err_msg = str(e)
        return f"[Search error: {err_msg}] Query: {query}"

    if len(result) > max_chars:
        result = result[:max_chars] + "\n[... truncated ...]"
    return result


async def _query_serper(query: str, api_key: str) -> str:
    """Direct Serper API call via httpx, returns formatted string."""
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": 10}

    async with httpx.AsyncClient() as client:
        for attempt in range(5):
            try:
                resp = await client.post(url, headers=headers, json=payload, timeout=20.0)
                if resp.status_code == 400:
                    data = resp.json()
                    if data.get("message") == "Not enough credits":
                        return f"[Search error: Serper API out of credits] Query: {query}"
                
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                if attempt == 4:
                    raise
                await asyncio.sleep(min(2.0, 0.3 * (attempt + 1)))

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
