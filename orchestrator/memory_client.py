"""Tool-server HTTP clients: chat memory, deliverable store, last-active.

Each call uses its own short-lived session so a background write survives the
caller's session closing. Everything fails soft — memory is an enhancement,
never a turn-blocker.
"""
import aiohttp

from . import config


async def _memory_recall(chat_id: str, query: str, session=None) -> list[tuple[str, str]]:
    """Call tool-server memory recall. Uses its own session for independence."""
    if not chat_id:
        return []
    try:
        async with aiohttp.ClientSession() as own_session:
            async with own_session.post(
                f"{config.TOOL_SERVER_URL}/memory/recall",
                json={"chat_id": chat_id, "query": query, "top_k": 6},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=config.MEMORY_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [(t["role"], t["content"]) for t in data.get("turns", [])]
    except Exception:
        pass
    return []


async def _memory_store(chat_id: str, role: str, content: str, session=None) -> bool:
    """Call tool-server memory store. Uses its own session to survive caller's session closure."""
    if not chat_id:
        return False
    try:
        async with aiohttp.ClientSession() as own_session:
            async with own_session.post(
                f"{config.TOOL_SERVER_URL}/memory/store",
                json={"chat_id": chat_id, "role": role, "content": content},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=config.MEMORY_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("stored", False)
    except Exception:
        pass
    return False


async def _deliverable_store(chat_id: str, content: str, filename: str = "", fmt: str = "") -> bool:
    """Persist a delivered document so a LATER turn can edit the real artifact instead
    of rebuilding it from scratch. Fire-and-forget; failure just means no edit memory."""
    if not (chat_id and (content or "").strip()):
        return False
    try:
        async with aiohttp.ClientSession() as own_session:
            async with own_session.post(
                f"{config.TOOL_SERVER_URL}/deliverable/store",
                json={"chat_id": chat_id, "content": content, "filename": filename, "fmt": fmt},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=config.MEMORY_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    return (await resp.json()).get("stored", False)
    except Exception:
        pass
    return False


async def _deliverable_get(chat_id: str):
    """Fetch the latest delivered document for this chat (what a follow-up edits), or None."""
    if not chat_id:
        return None
    try:
        async with aiohttp.ClientSession() as own_session:
            async with own_session.post(
                f"{config.TOOL_SERVER_URL}/deliverable/get",
                json={"chat_id": chat_id},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=config.MEMORY_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    return (await resp.json()).get("deliverable")
    except Exception:
        pass
    return None


async def _last_active(chat_id: str):
    """Unix time of the chat's previous stored turn (for resume-after-a-gap awareness), or None."""
    if not chat_id:
        return None
    try:
        async with aiohttp.ClientSession() as own_session:
            async with own_session.post(
                f"{config.TOOL_SERVER_URL}/memory/last_active",
                json={"chat_id": chat_id},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=config.MEMORY_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    return (await resp.json()).get("last_active")
    except Exception:
        pass
    return None


async def _plan_store(chat_id: str, plan: dict) -> bool:
    """Persist a pending outline awaiting approval (chunked writer). Fire-and-forget."""
    if not (chat_id and isinstance(plan, dict)):
        return False
    try:
        async with aiohttp.ClientSession() as own_session:
            async with own_session.post(
                f"{config.TOOL_SERVER_URL}/plan/store",
                json={"chat_id": chat_id, "plan": plan},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=config.MEMORY_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    return (await resp.json()).get("stored", False)
    except Exception:
        pass
    return False


async def _plan_get(chat_id: str):
    """Fetch the pending outline awaiting approval for this chat, or None."""
    if not chat_id:
        return None
    try:
        async with aiohttp.ClientSession() as own_session:
            async with own_session.post(
                f"{config.TOOL_SERVER_URL}/plan/get",
                json={"chat_id": chat_id},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=config.MEMORY_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    return (await resp.json()).get("plan")
    except Exception:
        pass
    return None


async def _plan_clear(chat_id: str) -> bool:
    """Drop the pending plan once built or abandoned."""
    if not chat_id:
        return False
    try:
        async with aiohttp.ClientSession() as own_session:
            async with own_session.post(
                f"{config.TOOL_SERVER_URL}/plan/clear",
                json={"chat_id": chat_id},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=config.MEMORY_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    return (await resp.json()).get("cleared", False)
    except Exception:
        pass
    return False
