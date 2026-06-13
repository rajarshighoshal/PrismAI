"""OpenAI-compatible HTTP surface for the orchestrator. Exposes /v1/models, /v1/chat/completions, /health."""
import asyncio
import json
import logging
import time
import uuid

import aiohttp
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import config, dedup, pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

app = FastAPI(title="OWUI Orchestrator", version="0.1.0")


def _check_auth(request: Request):
    if config.ORCH_API_KEY:
        if request.headers.get("authorization", "") != f"Bearer {config.ORCH_API_KEY}":
            raise HTTPException(status_code=401, detail="invalid api key")


def _user_id(request: Request) -> str:
    # Forwarded by OWUI when ENABLE_FORWARD_USER_INFO_HEADERS=true.
    return (
        request.headers.get("x-openwebui-user-id")
        or request.headers.get("x-openwebui-user-email")
        or ""
    )


def _request_headers(request: Request) -> dict:
    return {k.lower(): v for k, v in request.headers.items()}


def _chunk(cid, model, *, role=None, content=None, reasoning=None, finish=None) -> str:
    delta = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    body = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return "data: " + json.dumps(body) + "\n\n"


@app.get("/health")
async def health():
    return {"status": "ok", "service": "orchestrator", "version": app.version}


@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    data = [
        {"id": config.ADVERTISED_CHAT_ID, "object": "model", "created": now, "owned_by": "orchestrator"},
    ]
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _check_auth(request)
    body = await request.json()
    messages = body.get("messages") or []
    want_stream = bool(body.get("stream"))
    model = body.get("model") or config.ADVERTISED_CHAT_ID
    user_id = _user_id(request)
    request_headers = _request_headers(request)
    cid = "chatcmpl-" + uuid.uuid4().hex

    owui_headers = {k: v for k, v in request_headers.items() if "openwebui" in k}
    log.info(f"[request] user={user_id} model={model} messages={len(messages)} owui_headers={owui_headers}")

    # The chat id is part of the key: identical text in a DIFFERENT chat must run fresh —
    # its side effects (deliverable store, memory) belong to that chat, and a cached
    # answer would leave the new chat with no document to edit.
    dkey = (dedup.make_key(messages, model, user_id,
                           request_headers.get("x-openwebui-chat-id", ""))
            if (dedup.enabled() and messages) else None)

    if want_stream:
        async def event_stream():
            yield _chunk(cid, model, role="assistant")

            # De-dup: replay a just-computed answer, or attach to an identical
            # request that is still in flight, before doing any real work.
            lead_fut = None
            if dkey is not None:
                mode, payload = dedup.begin(dkey)
                if mode == "cached":
                    yield _chunk(cid, model, content=payload)
                    yield _chunk(cid, model, finish="stop")
                    yield "data: [DONE]\n\n"
                    return
                if mode == "follow":
                    try:
                        answer = await asyncio.wait_for(
                            asyncio.shield(payload), timeout=config.DEDUP_WAIT_TIMEOUT
                        )
                        yield _chunk(cid, model, content=answer)
                        yield _chunk(cid, model, finish="stop")
                        yield "data: [DONE]\n\n"
                        return
                    except Exception as exc:
                        # the twin failed/timed out -> run our own below. Log it:
                        # a silent swallow here hid dedup-wait timeouts that then
                        # silently re-ran the whole pipeline.
                        log.info(f"[dedup] follower fell back to its own run: {type(exc).__name__}: {exc}")
                else:
                    lead_fut = payload  # we are the original

            session = aiohttp.ClientSession()
            parts, done_ok, err = [], False, None
            try:
                async for kind, text in pipeline.run(
                    messages,
                    user_id=user_id,
                    session=session,
                    request_headers=request_headers,
                    user_model=model,
                ):
                    if kind == "content":
                        parts.append(text)
                        yield _chunk(cid, model, content=text)
                    elif kind == "reasoning":
                        yield _chunk(cid, model, reasoning=text)
                done_ok = True
                yield _chunk(cid, model, finish="stop")
                yield "data: [DONE]\n\n"
            except Exception as exc:  # surface, don't hang the client
                err = exc
                # Log the full error, but show the user a clean message — never the
                # raw exception (it embeds a URL the next turn's agent would try to
                # fetch, poisoning the chat).
                log.warning(f"[pipeline] failed: {type(exc).__name__}: {exc}")
                yield _chunk(cid, model, content="\n\n⚠️ Something went wrong completing this turn. Please try again.")
                yield _chunk(cid, model, finish="stop")
                yield "data: [DONE]\n\n"
            finally:
                answer = "".join(parts)
                if lead_fut is not None:
                    if done_ok and answer.strip():
                        dedup.resolve(dkey, lead_fut, answer=answer)
                    else:  # error / empty / client abort -> followers run their own
                        dedup.resolve(dkey, lead_fut, exc=(err or RuntimeError("aborted")))
                elif dkey is not None and done_ok and answer.strip():
                    dedup.store(dkey, answer)  # follow-fallback still caches
                await session.close()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming (OWUI uses this for title/tag generation): collect content.
    cached = dedup.get_cached(dkey) if dkey is not None else None
    if cached is not None:
        content = cached
    else:
        session = aiohttp.ClientSession()
        try:
            parts = []
            async for kind, text in pipeline.run(
                messages,
                user_id=user_id,
                session=session,
                request_headers=request_headers,
                user_model=model,
            ):
                if kind == "content":
                    parts.append(text)
            content = "".join(parts)
            if dkey is not None and content.strip():
                dedup.store(dkey, content)
        except Exception as exc:
            log.warning(f"[pipeline] failed: {type(exc).__name__}: {exc}")
            content = "Something went wrong completing this turn. Please try again."
        finally:
            await session.close()

    return JSONResponse(
        {
            "id": cid,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )
