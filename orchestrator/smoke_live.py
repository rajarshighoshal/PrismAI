"""LIVE smoke test — runs the DEPLOYED orchestrator end-to-end with REAL models.

The mocked contract suite verifies wiring; this verifies reality: real OWUI-style
RAG-wrapped requests (the actual prepended-template shape this instance produces),
real classifier judgement, real verifier, real export. It asserts the failures that
mocks structurally cannot catch:

  T1  generate: a grounded cover letter docx is delivered; the deliverable is stored;
      the letter date is not stripped to "[Date]".
  T2  edit ("I've finished my MS — update the doc"): the EDIT path fires — the new
      version stays near-identical to v1 (no reconstruction) and the version bumps.
  T3  add a USER-stated fact ("geometric probes for AI safety"): the verifier must
      KEEP it — the user's word grounds the user's facts.

Run ON the server (costs a few real model calls — run on demand, not CI):
  ssh chatserver 'cat > /tmp/smoke_live.py' < orchestrator/smoke_live.py
  ssh chatserver 'ORCH_ENV=$(find /opt /root /home -maxdepth 4 -name orchestrator.env \
      -path "*orchestrator*" 2>/dev/null | head -1) python3 /tmp/smoke_live.py'
"""
import json
import os
import re
import sys
import time
import urllib.request

ORCH = "http://localhost:8002/v1/chat/completions"
TOOLS = "http://localhost:8001"
CHAT_ID = f"smoke-{int(time.time())}"

# Synthetic resume/posting — small (cost), but realistic enough to exercise grounding.
SOURCES = (
    '<source id="1" name="resume.tex">Jane Tester — ML Engineer.\n'
    "M.S. in Computer Science, Machine Learning specialization, Georgia Tech, expected May 2026.\n"
    "Acme Corp: built a fraud model improving recall by 31%.\n"
    "Publication: parallel graph algorithm with 1,204x speedup, accepted to SC26.</source>\n"
    '<source id="2" name="posting.txt">PhD position in machine learning for genomics at '
    "Example University. Requires ML, statistics, and programming background.</source>"
)

# The REAL wrapping this OWUI instance produces (template saved in its config DB has no
# {{QUERY}} placeholder, so the rendered template is PREPENDED and the user's text
# follows the final </context>). Structure verified against the live container.
def owui_wrap(user_text: str) -> str:
    return (
        "### Task:\nRespond to the user query using the provided context, incorporating "
        "inline citations in the format [id] **only when the <source> tag includes an "
        'explicit id attribute** (e.g., <source id="1">).\n\n### Guidelines:\n- If you '
        "don't know the answer, clearly state that.\n\n### Output:\nProvide a clear and "
        "direct response to the user's query.\n\n"
        f"<context>\n{SOURCES}\n</context>\n\n{user_text}"
    )


def _key() -> str:
    env_path = os.environ.get("ORCH_ENV", "")
    if env_path and os.path.exists(env_path):
        for line in open(env_path):
            if line.strip().startswith("ORCH_API_KEY="):
                return line.strip().split("=", 1)[1]
    return os.environ.get("ORCH_API_KEY", "")


def post(url: str, payload: dict, headers: dict, timeout: int = 420) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def turn(history, user_text):
    msgs = history + [{"role": "user", "content": owui_wrap(user_text)}]
    headers = {"x-openwebui-chat-id": CHAT_ID, "x-openwebui-user-id": "smoke"}
    key = _key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    t0 = time.time()
    data = post(ORCH, {"model": "PrismAI", "stream": False, "messages": msgs}, headers)
    answer = data["choices"][0]["message"]["content"]
    print(f"  · turn took {time.time()-t0:.0f}s, answer {len(answer)} chars")
    return msgs + [{"role": "assistant", "content": answer}], answer


def deliverable(retries=8):
    for _ in range(retries):
        d = post(f"{TOOLS}/deliverable/get", {"chat_id": CHAT_ID}, {}, timeout=20).get("deliverable")
        if d:
            return d
        time.sleep(1.5)
    return None


def words(t):
    return set(re.findall(r"[a-z0-9]{3,}", t.lower()))


fails = []
def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'}: {name}")
    if not cond:
        fails.append(name)


print(f"chat_id={CHAT_ID}")

print("T1: generate cover letter + docx …")
hist, a1 = turn([], "I want to apply for this PhD. Can you write the cover letter for me? "
                    "Please generate a docx file once done.")
check("T1: response contains a download link", "Download" in a1 or "download" in a1)
v1 = deliverable()
check("T1: deliverable stored", bool(v1))
if not v1:
    print("ABORT: nothing stored — later turns are meaningless"); sys.exit(1)
check("T1: letter date not stripped to placeholder", "[Date]" not in v1["content"])
check("T1: grounded credential present (31% or 1,204x)",
      "31%" in v1["content"] or "1,204x" in v1["content"])

print("T2: surgical edit (finished MS) …")
hist, a2 = turn(hist, "btw I have already finished my MS in May 2026 — can you update the doc?")
v2 = deliverable()
check("T2: new version stored", bool(v2) and v2["version"] > v1["version"])
if v2:
    w1, w2 = words(v1["content"]), words(v2["content"])
    overlap = len(w1 & w2) / max(1, len(w1))
    print(f"  · v1→v2 word containment: {overlap:.2f}")
    check("T2: EDITED not reconstructed (containment ≥ 0.7)", overlap >= 0.7)
    check("T2: the document actually changed", v2["content"] != v1["content"])

print("T3: user-stated fact must survive the verifier …")
hist, a3 = turn(hist, "also add that I currently work on geometric probes for AI safety "
                      "and connect it to this position")
v3 = deliverable()
check("T3: new version stored", bool(v3) and v2 and v3["version"] > v2["version"])
check("T3: user-stated fact KEPT (geometric probes)",
      bool(v3) and "geometric probes" in v3["content"].lower())

print("\n" + ("ALL LIVE SMOKE CHECKS PASSED" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
