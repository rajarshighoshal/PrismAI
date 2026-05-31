"""Web search for the GROUNDED tier. Provider-pluggable, free-first.

Order when SEARCH_PROVIDER=auto (first configured wins):
  1. SearXNG    (SEARXNG_URL set)    — self-hosted, free, open-source (preferred)
  2. Tavily     (TAVILY_API_KEY set) — hosted, cheap, reliable
  3. DuckDuckGo (no key)             — zero-config free default (ddgs lib)

Every provider returns a list of {"title", "url", "snippet"} and fails soft to
[] so a search hiccup degrades to an ungrounded answer rather than an error.
"""
import aiohttp

from . import config


def _provider() -> str:
    p = config.SEARCH_PROVIDER
    if p in ("searxng", "tavily", "duckduckgo"):
        return p
    if config.SEARXNG_URL:
        return "searxng"
    if config.TAVILY_API_KEY:
        return "tavily"
    return "duckduckgo"


async def _searxng(query, n, session):
    params = {"q": query, "format": "json", "safesearch": "0"}
    async with session.get(
        f"{config.SEARXNG_URL}/search",
        params=params,
        timeout=aiohttp.ClientTimeout(total=config.SEARCH_TIMEOUT),
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
    out = []
    for r in (data.get("results") or [])[:n]:
        out.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
            }
        )
    return out


async def _tavily(query, n, session):
    payload = {
        "api_key": config.TAVILY_API_KEY,
        "query": query[:400],  # Tavily rejects queries over 400 chars
        "max_results": n,
        "search_depth": "basic",
    }
    async with session.post(
        "https://api.tavily.com/search",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=config.SEARCH_TIMEOUT),
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
    out = []
    for r in (data.get("results") or [])[:n]:
        out.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
            }
        )
    return out


async def _duckduckgo(query, n, session):
    # ddgs is a sync lib; run it off the event loop. Lazy-imported so the rest
    # of the service works even if it's not installed.
    import asyncio

    def _go():
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS  # older package name
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=n))

    rows = await asyncio.get_event_loop().run_in_executor(None, _go)
    out = []
    for r in rows[:n]:
        out.append(
            {
                "title": r.get("title", ""),
                "url": r.get("href") or r.get("url", ""),
                "snippet": r.get("body") or r.get("snippet", ""),
            }
        )
    return out


async def search(query, *, max_results=None, session=None):
    """Return [{title, url, snippet}]. Fails soft to []."""
    if not (config.ENABLE_WEB_SEARCH and query.strip()):
        return []
    n = max_results or config.SEARCH_MAX_RESULTS
    own = session is None
    if own:
        session = aiohttp.ClientSession()
    try:
        provider = _provider()
        if provider == "searxng":
            return await _searxng(query, n, session)
        if provider == "tavily":
            return await _tavily(query, n, session)
        return await _duckduckgo(query, n, session)
    except Exception:
        return []
    finally:
        if own:
            await session.close()


def format_context(results) -> str:
    """Render search results into a compact, citeable context block."""
    if not results:
        return ""
    lines = []
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        snippet = " ".join((r.get("snippet") or "").split())
        lines.append(f"[{i}] {title}\n{url}\n{snippet}")
    return "\n\n".join(lines)
