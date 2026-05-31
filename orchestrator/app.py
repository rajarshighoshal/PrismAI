"""OpenAI-compatible HTTP surface for the orchestrator.

OWUI registers this as a model connection (base_url = http://owui-orchestrator:8002/v1).
Exposes GET /v1/models and POST /v1/chat/completions (streaming + non-streaming),
plus /health. All real work lives in pipeline.run; this file just speaks the
OpenAI wire format and forwards the logged-in user's id from OWUI's headers.
"""
import json
import time
import uuid

import aiohttp
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import config, pipeline

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
        {"id": config.ADVERTISED_VISION_ID, "object": "model", "created": now, "owned_by": "orchestrator"},
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
    cid = "chatcmpl-" + uuid.uuid4().hex

    if want_stream:
        async def event_stream():
            session = aiohttp.ClientSession()
            try:
                yield _chunk(cid, model, role="assistant")
                async for kind, text in pipeline.run(messages, user_id=user_id, session=session):
                    if kind == "content":
                        yield _chunk(cid, model, content=text)
                    elif kind == "reasoning":
                        yield _chunk(cid, model, reasoning=text)
                yield _chunk(cid, model, finish="stop")
                yield "data: [DONE]\n\n"
            except Exception as exc:  # surface, don't hang the client
                yield _chunk(cid, model, content=f"\n\n[orchestrator error: {exc}]")
                yield _chunk(cid, model, finish="stop")
                yield "data: [DONE]\n\n"
            finally:
                await session.close()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming (OWUI uses this for title/tag generation): collect content.
    session = aiohttp.ClientSession()
    try:
        parts = []
        async for kind, text in pipeline.run(messages, user_id=user_id, session=session):
            if kind == "content":
                parts.append(text)
        content = "".join(parts)
    except Exception as exc:
        content = f"[orchestrator error: {exc}]"
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
