"""Multi-turn document editing: classify the edit intent, patch-in-place when the
change is small, verify + re-deliver otherwise. Split out of the agent god-module.
"""
import json
import re
import logging

from . import config, fireworks
from .owui import _last_user_text, _unwrap_owui, _user_source
from prism_core.messages import _text_of
from .timectx import _now_line
from .prompts import SYSTEM_EDIT_INTENT, SYSTEM_EDIT_PATCH
from .prose import _voice_pass
from .verifier import _verified_or_blocked, _summarize_correction, _WORD_RE
from .delivery import _persist_turn, _repackage_deliverable, _same_doc

log = logging.getLogger(__name__)

async def _classify_edit_once(payload, *, session=None) -> dict:
    # The PRO model judges this gate: flash-at-low-reasoning misread even "can you
    # update the doc?" as 'new' (live smoke, repeatedly, regardless of prompt wording).
    # The gate only fires in chats that already delivered a document — pennies, and a
    # wrong verdict here silently drops the user's document.
    try:
        raw = await fireworks.complete(
            [{"role": "system", "content": SYSTEM_EDIT_INTENT},
             {"role": "user", "content": json.dumps(payload, ensure_ascii=True)}],
            config.GROUNDED_MODEL, max_tokens=600, temperature=0.0,
            session=session, label="gate:edit",
        )
        data = json.loads(re.search(r"\{.*\}", raw, flags=re.S).group(0))
        a = str(data.get("action", "new")).lower()
        if a in ("rename", "reformat", "edit", "voice"):
            return {"action": a, "filename": (data.get("filename") or "").strip(),
                    "format": (data.get("format") or "").strip().lower()}
    except Exception:
        pass
    return {"action": "new", "filename": "", "format": ""}

async def _classify_edit(last_user: str, prior: dict, *, messages=None, session=None) -> dict:
    """Classify a follow-up against the last delivered document: rename|reformat|edit|new.

    The two misroute directions are NOT symmetric: new-misread-as-edit is self-correcting,
    while edit-misread-as-new silently drops the document — so a 'new' verdict must win TWICE."""
    last_user = (last_user or "").strip()
    if not (last_user and prior and prior.get("content")):
        return {"action": "new"}
    recent = ""
    if messages:
        turns = [f"[{m.get('role')}]: {_unwrap_owui(_text_of(m.get('content')))[:200]}"
                 for m in messages[:-1] if m.get("role") in ("user", "assistant")]
        recent = "\n".join(turns[-4:])
    payload = {
        "recent_conversation": recent,
        "latest_user": last_user[:1500],
        "current_filename": prior.get("filename") or "document",
        "current_format": prior.get("fmt") or "docx",
    }
    result = await _classify_edit_once(payload, session=session)
    if result["action"] == "new":
        second = await _classify_edit_once(payload, session=session)
        if second["action"] != "new":
            result = second
    log.info(f"[edit-intent] action={result['action']} msg={last_user[:80]!r}")
    return result

def _edit_inject(prior: dict) -> str:
    """System directive for a surgical content edit of the prior document."""
    return (
        "REVISION TASK — the user is revising a document you already delivered in this "
        "chat. Here is that document, verbatim:\n\n"
        "--- CURRENT DOCUMENT ---\n" + (prior.get("content") or "").strip()
        + "\n--- END CURRENT DOCUMENT ---\n\n"
        "Make the change the user asks for — their INTENT, not a literal find-and-replace "
        "of their words. A 'fix this line' touches one line; a 'review and make X "
        "consistent' means reading the whole document and rewording every passage the "
        "intent genuinely covers, with judgment. Leave everything the request does not "
        "cover exactly as it is — never rewrite, re-order, or 'improve' beyond the ask. "
        "If the instruction is ambiguous or you are unsure what they want, output ONLY a "
        "short clarifying question (no document) — the user explicitly prefers being "
        "asked over being guessed at, and their answer comes straight back to you."
    )

def _is_doc_status_request(text: str) -> bool:
    t = (text or "").lower()
    return bool(re.search(r"\b(current|latest|this)\s+(doc|document|file)\s+(status|info|state)\b", t)
                or re.search(r"\b(show|what'?s|what is)\s+.*\b(doc|document|file)\s+status\b", t))

def _doc_status(prior: dict) -> str:
    content = prior.get("content") or ""
    return (
        "Current document:\n"
        f"- filename: {prior.get('filename') or 'document'}\n"
        f"- format: {(prior.get('fmt') or 'docx').upper()}\n"
        f"- version: {prior.get('version') or '?'}\n"
        f"- words: {len(_WORD_RE.findall(content)):,}\n"
        "\nYou can ask me to rename it, export it in another format, or revise a specific part."
    )

async def _try_patch_edit(baseline: str, instruction: str, *, session=None):
    """Apply targeted find→replace edits to the stored document. Returns patched text, or None when the change is too broad — caller falls back to a full re-emit."""
    if not baseline.strip():
        return None
    try:
        raw = await fireworks.complete(
            [{"role": "system", "content": SYSTEM_EDIT_PATCH + "\n\n--- DOCUMENT ---\n" + baseline},
             {"role": "user", "content": instruction}],
            config.GROUNDED_MODEL, max_tokens=config.DRAFT_MAX_TOKENS, temperature=0.0,
            session=session, label="edit:patch")
        data = json.loads(re.search(r"\{.*\}", raw, flags=re.S).group(0))
    except Exception:
        return None
    edits = data.get("edits")
    if data.get("broad") or not isinstance(edits, list) or not edits:
        return None
    text = baseline
    for e in edits:
        if not isinstance(e, dict):
            return None
        find = e.get("find") or ""
        repl = "" if e.get("replace") is None else str(e.get("replace"))
        if not find or text.count(find) != 1:   # must match exactly once, else fall back
            return None
        text = text.replace(find, repl, 1)
    return text if (text.strip() and text != baseline) else None

async def _dispatch_edit(messages, prior, chat_id, req_headers, session, show_work):
    """Handle multi-turn edit. Returns (handled, output_or_directive, baseline).
    If handled: output_or_directive is the response text, baseline is "".
    If not handled: output_or_directive is the edit directive, baseline is the prior doc."""
    if not (prior and prior.get("content")):
        return False, "", ""

    if _is_doc_status_request(_last_user_text(messages)):
        return True, _doc_status(prior), ""

    intent = await _classify_edit(_last_user_text(messages), prior, messages=messages, session=session)

    if intent["action"] in ("rename", "reformat"):
        fmt = intent.get("format") or prior.get("fmt") or "docx"
        filename = intent.get("filename") or prior.get("filename") or "document"
        link = await _repackage_deliverable(prior["content"], filename, fmt,
                                            chat_id=chat_id, headers=req_headers, session=session)
        verb = "Renamed" if intent["action"] == "rename" else f"Re-exported as {fmt.upper()}"
        return True, (f"📄 {verb} — download below.{link}" if link
                       else "I couldn't re-export that file — want me to try again?"), ""

    if intent["action"] == "voice":
        # On-demand voice route: the user is happy with the CONTENT and asked for a
        # different TONE ("make it more human", "warmer", "more formal"). Run the voice
        # pass on the stored document, verify (facts must not move), re-export.
        req = _last_user_text(messages).lower()
        register = ("formal" if any(c in req for c in ("formal", "professional", "polished", "corporate"))
                    else "warm")
        voiced = await _voice_pass(prior["content"], register, session=session)
        if not voiced or not voiced.strip() or voiced.strip() == prior["content"].strip():
            return True, "I couldn't give it a different voice this time — want me to try again?", ""
        src = ((_user_source(messages) + "\n\n" + prior["content"]).strip()
               if _user_source(messages).strip() else prior["content"])
        status, text = await _verified_or_blocked(messages, voiced, src, force=True, session=session)
        if status != "ok":
            return True, text, ""
        link = await _repackage_deliverable(text, prior.get("filename") or "document",
                                            prior.get("fmt") or "docx",
                                            chat_id=chat_id, headers=req_headers, session=session)
        output = (f"📄 Same content, {register} voice — download below.{link}" if link
                  else "I couldn't rebuild the file — want me to try again?")
        _persist_turn(chat_id, messages, text, session)
        return True, output, ""

    if intent["action"] != "edit":
        return False, "", ""

    baseline = (prior.get("content") or "").strip()

    # 1. Targeted in-place patches
    revised = await _try_patch_edit(baseline, _last_user_text(messages), session=session)
    patched = revised is not None

    # 2. Full rewrite if patches aren't clean
    if not patched:
        revised = ""
        try:
            revised = (await fireworks.complete(
                [{"role": "system", "content": _now_line() + "\n\n" + _edit_inject(prior)
                  + "\n\nOutput ONLY the complete revised document — no commentary."},
                 {"role": "user", "content": _last_user_text(messages)}],
                config.GROUNDED_MODEL, max_tokens=config.DRAFT_MAX_TOKENS,
                temperature=config.WRITER_TEMPERATURE, session=session, label="edit:write")).strip()
        except Exception as e:
            log.warning(f"[edit] directed revision failed, falling to normal flow: {e}")
        if revised and not _same_doc(revised, baseline) and "?" in revised and len(revised) < 1200:
            return True, revised, ""

    # 3. Verify + re-export
    if revised and (patched or _same_doc(revised, baseline)):
        src = ((_user_source(messages) + "\n\n" + baseline).strip()
               if _user_source(messages).strip() else baseline)
        status, text = await _verified_or_blocked(messages, revised, src, force=True, session=session)
        if status != "ok":
            return True, text, ""
        link = await _repackage_deliverable(text, prior.get("filename") or "document",
                                            prior.get("fmt") or "docx",
                                            chat_id=chat_id, headers=req_headers, session=session)
        summary = await _summarize_correction(baseline, text, session=session)
        output = ("📄 Updated — download below." + (("\n\n" + summary) if summary else "") + link) if link \
                 else "I couldn't rebuild the file — want me to try again?"
        _persist_turn(chat_id, messages, text, session)
        return True, output, ""
    # Fell through — inject prior doc as context for normal agent loop
    return False, _edit_inject(prior), baseline
