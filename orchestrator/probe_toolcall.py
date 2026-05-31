"""Gate test: do the open models on Fireworks support real OpenAI tool-calling?
Decides whether a 'truly agentic' (model-drives-tools) architecture is possible.
Uses FIREWORKS_API_KEY already in env. stdlib only. Never prints the key.
"""
import json, os, time, urllib.request, urllib.error

KEY = os.environ.get("FIREWORKS_API_KEY", "")
BASE = "https://api.fireworks.ai/inference/v1/chat/completions"
MODELS = [
    "accounts/fireworks/models/deepseek-v4-pro",
    "accounts/fireworks/models/kimi-k2p6",
    "accounts/fireworks/models/glm-5p1",
    "accounts/fireworks/models/gpt-oss-120b",
]
TOOLS = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}]
MSGS = [
    {"role": "system", "content": "You can call web_search when you need current facts."},
    {"role": "user", "content": "What is the price of Bitcoin right now?"},
]


def call(model):
    payload = {"model": model, "messages": MSGS, "tools": TOOLS,
               "tool_choice": "auto", "max_tokens": 400, "temperature": 0.0}
    req = urllib.request.Request(BASE, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}: {e.read().decode()[:140]}", time.time() - t0
    except Exception as e:
        return f"ERR {type(e).__name__}: {e}", time.time() - t0
    dt = time.time() - t0
    ch = d.get("choices", [{}])[0]
    tcs = (ch.get("message") or {}).get("tool_calls")
    if tcs:
        fn = tcs[0].get("function", {})
        try:
            q = json.loads(fn.get("arguments") or "{}").get("query", "")
        except Exception:
            q = "(bad args)"
        return f"TOOL_CALL OK  fn={fn.get('name')} query={q!r}", dt
    return f"no tool_call (finish={ch.get('finish_reason')})", dt


if not KEY:
    print("FIREWORKS_API_KEY not in env — cannot probe")
else:
    for m in MODELS:
        res, dt = call(m)
        print(f"{m.split('/')[-1]:16s} [{dt:4.1f}s]  {res}")
