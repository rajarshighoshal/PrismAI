"""Web search for the GROUNDED tier. Provider-pluggable, free-first.

Order when SEARCH_PROVIDER=auto (first configured wins):
  1. SearXNG    (SEARXNG_URL set)    — self-hosted, free, open-source (preferred)
  2. Tavily     (TAVILY_API_KEY set) — hosted, cheap, reliable
  3. DuckDuckGo (no key)             — zero-config free default (ddgs lib)

Every provider returns a list of {"title", "url", "snippet"} and fails soft to
[] so a search hiccup degrades to an ungrounded answer rather than an error.
"""
import asyncio

try:
    import aiohttp
except ImportError:  # lets offline tests run without installed service deps
    aiohttp = None

from . import config


def _require_aiohttp():
    if aiohttp is None:
        raise RuntimeError("aiohttp is required for live web search calls")


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
    # Parity with the proven router_fn path: advanced depth + the AI summary,
    # with a small retry on transient failures. Tavily rejects queries >400 chars.
    payload = {
        "api_key": config.TAVILY_API_KEY,
        "query": query[:400],
        "search_depth": "advanced",
        "include_answer": True,
        "max_results": n,
    }
    for attempt in range(3):
        try:
            async with session.post(
                "https://api.tavily.com/search",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=config.SEARCH_TIMEOUT),
            ) as resp:
                if resp.status in (429, 500, 502, 503, 504):
                    raise aiohttp.ClientError(f"tavily {resp.status}")
                resp.raise_for_status()
                data = await resp.json()
            break
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < 2:
                await asyncio.sleep(0.5 * (attempt + 1))
            else:
                raise
    out = []
    # Tavily's synthesized answer is high-signal grounding — surface it as [1]
    # so the model can cite it, mirroring router_fn's "Tavily AI Summary".
    answer = (data.get("answer") or "").strip()
    if answer:
        out.append({"title": "Tavily AI summary", "url": "", "snippet": answer})
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
        _require_aiohttp()
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
