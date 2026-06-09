"""Tool server for OpenWebUI: file export + web fetch + citation lookup + chat memory.

OpenAPI-discoverable so OpenWebUI auto-registers each endpoint as a tool
the model can invoke. Stateless except for the memory DB.
"""
from __future__ import annotations

import asyncio
import base64
import csv
import ipaddress
import io
import logging
import os
import re
import socket
import tempfile
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urljoin, urlparse

import httpx
import pypandoc
import trafilatura
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

import memory  # same-directory module

logger = logging.getLogger("tool-server")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="OWUI Tool Server",
    description=(
        "File export (docx/pdf/md/csv), readable web extraction, and DOI→APA "
        "citation lookup. Used by OpenWebUI models via tool calls."
    ),
    version="0.3.0",
)

OPENWEBUI_BASE_URL = os.getenv("OPENWEBUI_BASE_URL", "http://open-webui:8080").rstrip("/")
OPENWEBUI_API_KEY = os.getenv("OPENWEBUI_API_KEY", "")
OPENWEBUI_ATTACH_EXPORTS = os.getenv("OPENWEBUI_ATTACH_EXPORTS", "true").lower() not in {
    "0",
    "false",
    "no",
}

# For /verify_grounding: the auditor LLM. Key from env; endpoint no-ops with a
# clear error if unset, so the server still starts without it.
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
VERIFY_MODEL = os.getenv("VERIFY_MODEL", "accounts/fireworks/models/deepseek-v4-flash")


# --- Request models -------------------------------------------------------

class ExportRequest(BaseModel):
    markdown: str = Field(
        ...,
        description=(
            "The markdown content to convert. May include headings, lists, "
            "in-text citations like (Author, 2024), code blocks, tables, links. "
            "Pass the complete final draft/content, not an outline or summary of it."
        ),
    )
    filename: Optional[str] = Field(
        None,
        description=(
            "Output filename WITHOUT extension. Defaults to 'document'. "
            "Non-alphanumeric chars (except - and _) are stripped for safety."
        ),
    )
    title: Optional[str] = Field(
        None,
        description=(
            "Document title for Word/PDF metadata. Shows in the Properties "
            "pane and on the first page if the template includes {{title}}."
        ),
    )


class CsvExportRequest(BaseModel):
    rows: list[dict] = Field(
        ...,
        description=(
            "List of row dicts. The keys of the first row become the column "
            "headers; all rows should share the same keys."
        ),
    )
    filename: Optional[str] = Field(
        None,
        description="Output filename WITHOUT extension. Defaults to 'data'.",
    )


class FetchUrlRequest(BaseModel):
    url: str = Field(
        ...,
        description="The full URL to fetch. Must include http:// or https://.",
    )
    max_chars: int = Field(
        8000,
        ge=500,
        le=50000,
        description=(
            "Maximum characters to return. Pages longer than this are truncated "
            "with a [...truncated] marker. Default 8000."
        ),
    )


class CitationRequest(BaseModel):
    doi: str = Field(
        ...,
        description=(
            "Digital Object Identifier (e.g. '10.1037/0033-2909.131.6.803'). "
            "Leading 'doi:' or 'https://doi.org/' is stripped automatically."
        ),
    )
    expected_title: Optional[str] = Field(
        None,
        description=(
            "The title (or distinctive words of it) you BELIEVE this DOI points to. "
            "Strongly recommended: the tool cross-checks the CrossRef record against "
            "this and returns verified=false if they don't match, so a valid-but-wrong "
            "DOI can't be cited by mistake. Leave null only if you have no expectation."
        ),
    )


class CitationSearchRequest(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Bibliographic search text — title, and optionally authors/year "
            "(e.g. 'Gathercole Alloway Working Memory and Learning 2008'). Use this "
            "instead of guessing DOIs when you only know the work, not its DOI."
        ),
    )
    rows: int = Field(
        3, ge=1, le=10,
        description="How many candidate matches to return (default 3).",
    )


# --- Helpers --------------------------------------------------------------

def _safe_filename(name: Optional[str], default: str) -> str:
    raw = (name or default).strip() or default
    # Drop a trailing extension the model tacked on (e.g. "...Letterdocx" or
    # "report.docx") so the export doesn't double it up ("...Letterdocx.docx").
    raw = re.sub(r"\.?(docx|pdf|md|markdown|csv)$", "", raw, flags=re.IGNORECASE)
    safe = re.sub(r"[^A-Za-z0-9_\- ]", "", raw).strip().replace(" ", "_")
    return safe or default


# Reasoning/tool-call markup that some models (notably DeepSeek V4) leak into
# tool arguments. If it reaches the exported file it corrupts the user-facing
# document, so every export path is sanitized at the choke point below.
_LEAK_PATTERNS = [
    # DeepSeek DSML tool-call markup, incl. the fullwidth-pipe variant seen live
    re.compile(r"<[｜|]?\s*DSML[｜|]?[^>]*>.*?(?=\n|$)", re.IGNORECASE | re.DOTALL),
    re.compile(r"</?\s*[｜|]?\s*(?:tool_calls?|invoke|parameter)\b[^>]*>", re.IGNORECASE),
    # Reasoning blocks
    re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<reasoning>.*?</reasoning>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<\|?(?:assistant|channel|analysis)\|?>", re.IGNORECASE),
]
# A leading planning preamble like "Let me analyze this carefully..." or
# "The user wants a ... paragraph. ... I'll condense." — strip the planning
# block when it appears at the very start of the content.
_PREAMBLE_CUE = re.compile(
    r"^(?:\s*(?:let me|i'?ll|i will|first,? let me|okay,? let me|now,? let me|"
    r"the user (?:wants|is asking|asked|needs)|we need to|here'?s my (?:plan|analysis)|"
    r"let'?s)\b.*?)(?=\n\s*\n)",
    re.IGNORECASE | re.DOTALL,
)
# A "Draft:" / "Here is the draft:" label that some models emit right before the
# real content. If it sits in the first part of the text, drop everything up to
# and including the label so only the deliverable remains.
_DRAFT_LABEL = re.compile(
    r"^.{0,600}?\b(?:here(?:'?s| is) the (?:draft|final|letter|response)|draft)\s*:\s*",
    re.IGNORECASE | re.DOTALL,
)


def _strip_reasoning_leak(markdown: str) -> str:
    """Remove model reasoning / tool-call markup that leaked into content."""
    if not markdown:
        return markdown
    cleaned = markdown
    for pat in _LEAK_PATTERNS:
        cleaned = pat.sub("", cleaned)
    # Drop a single leading planning preamble paragraph if present.
    m = _PREAMBLE_CUE.match(cleaned.lstrip("\n"))
    if m and len(m.group(0)) < 1200:
        cleaned = cleaned.lstrip("\n")[m.end():]
    # Drop a leading "Draft:"/"Here is the draft:" label if it sits up front.
    dm = _DRAFT_LABEL.match(cleaned.lstrip("\n"))
    if dm:
        cleaned = cleaned.lstrip("\n")[dm.end():]
    return cleaned.strip() + "\n"


def _convert_pandoc(markdown: str, fmt: str, extra_args: list[str]) -> bytes:
    markdown = _strip_reasoning_leak(markdown)
    # [N] markers: in a cited work (research paper with a References section) they're
    # real citations — keep them. Everywhere else (cover letters, emails, bios) they
    # are chat-only grounding/traceability markers — strip them from the document.
    has_refs = re.search(r"(?im)^#{0,6}\s*(references|bibliography|works cited|sources)\s*$", markdown)
    if not has_refs:
        markdown = re.sub(r"[ \t]*\[\d+\]", "", markdown)
    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
        out_path = Path(tmp.name)
    try:
        pypandoc.convert_text(
            markdown,
            to=fmt,
            # hard_line_breaks: a single newline becomes a line break (not joined),
            # so address blocks and signatures keep their layout instead of
            # collapsing onto one line.
            format="markdown+hard_line_breaks",
            outputfile=str(out_path),
            extra_args=extra_args,
        )
        return out_path.read_bytes()
    finally:
        out_path.unlink(missing_ok=True)


def _auth_header_from_request(request: Optional[Request]) -> dict[str, str]:
    # Prefer the configured service token. OWUI sends a model connection its own
    # connection key (often "none"), which is NOT a valid user token for the files
    # API — so the inbound header can't authenticate the upload. The service token
    # is the one that actually works; only fall back to inbound if it's unset.
    if OPENWEBUI_API_KEY:
        return {"Authorization": f"Bearer {OPENWEBUI_API_KEY}"}
    inbound_auth = request.headers.get("authorization") if request else ""
    if inbound_auth:
        return {"Authorization": inbound_auth}
    return {}


def _attach_file_to_openwebui(
    request: Optional[Request],
    data: bytes,
    media_type: str,
    filename: str,
) -> dict[str, Any]:
    if not OPENWEBUI_ATTACH_EXPORTS:
        return {"download_url": None, "attach_reason": "disabled"}

    headers = _auth_header_from_request(request)
    if not headers:
        return {"download_url": None, "attach_reason": "missing OpenWebUI auth"}

    try:
        with httpx.Client(timeout=20.0) as client:
            upload = client.post(
                f"{OPENWEBUI_BASE_URL}/api/v1/files/",
                headers={**headers, "Accept": "application/json"},
                files={"file": (filename, data, media_type)},
                data={"process": "false"},
            )
            upload.raise_for_status()
            uploaded = upload.json()
            file_id = uploaded.get("id") or uploaded.get("file", {}).get("id")
            if not file_id:
                return {"download_url": None, "attach_reason": "upload returned no file id"}

            # Relative URL — the user's browser session serves it (get_verified_user).
            download_url = f"/api/v1/files/{file_id}/content/{filename}"

            # Attach the file to the assistant message so it shows inline. Needs
            # the chat-id + message-id headers (forwarded by the orchestrator once
            # the forward_message_id OWUI patch is applied). The download link is
            # the fallback when they're absent.
            chat_id = request.headers.get("x-open-webui-chat-id") if request else ""
            message_id = request.headers.get("x-open-webui-message-id") if request else ""
            attached = False
            if chat_id and message_id:
                try:
                    file_item = {"type": "file", "id": file_id, "name": filename,
                                 "url": file_id, "size": len(data), "mime_type": media_type, "file": uploaded}
                    client.post(
                        f"{OPENWEBUI_BASE_URL}/api/v1/chats/{chat_id}/messages/{message_id}/event",
                        headers={**headers, "Accept": "application/json"},
                        json={"type": "files", "data": {"files": [file_item]}},
                    ).raise_for_status()
                    attached = True
                except Exception:
                    pass

            return {"openwebui_file_id": file_id, "download_url": download_url,
                    "attached_to_chat": attached}
    except Exception as e:
        logger.warning("OpenWebUI file upload failed for %s: %s", filename, e)
        return {"download_url": None, "attach_reason": f"upload failed: {type(e).__name__}"}


def _file_result(
    data: bytes,
    media_type: str,
    filename: str,
    request: Optional[Request] = None,
) -> list[Any]:
    b64 = base64.b64encode(data).decode("ascii")
    metadata = {
        "status": "success",
        "filename": filename,
        "mime_type": media_type,
        "bytes": len(data),
    }
    if request is not None:
        metadata.update(_attach_file_to_openwebui(request, data, media_type, filename))
    return [
        f"data:{media_type};base64,{b64}",
        metadata,
    ]


def _is_blocked_ip(ip: str) -> bool:
    parsed = ipaddress.ip_address(ip)
    return (
        parsed.is_private
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_multicast
        or parsed.is_reserved
        or parsed.is_unspecified
    )


def _validate_public_http_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise HTTPException(
            status_code=400,
            detail="url must start with http:// or https://",
        )
    host = parsed.hostname
    if not host:
        raise HTTPException(status_code=400, detail="url must include a hostname")
    host_l = host.lower().rstrip(".")
    if host_l in {"localhost", "host.docker.internal", "docker.for.mac.localhost"}:
        raise HTTPException(
            status_code=400,
            detail="private/internal URLs are not allowed",
        )
    try:
        if _is_blocked_ip(host_l):
            raise HTTPException(
                status_code=400,
                detail="private/internal URLs are not allowed",
            )
        return
    except ValueError:
        pass
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        infos = socket.getaddrinfo(host, port)
    except socket.gaierror as e:
        raise HTTPException(status_code=400, detail=f"could not resolve hostname: {e}")
    for info in infos:
        resolved_ip = info[4][0]
        if _is_blocked_ip(resolved_ip):
            raise HTTPException(
                status_code=400,
                detail="private/internal URLs are not allowed",
            )


def _download_public_url(url: str) -> str:
    current = url
    headers = {"User-Agent": "owui-tool-server/0.3.0"}
    with httpx.Client(timeout=15.0, follow_redirects=False, headers=headers) as client:
        for _ in range(5):
            _validate_public_http_url(current)
            response = client.get(current)
            if 300 <= response.status_code < 400 and response.headers.get("location"):
                current = urljoin(str(response.url), response.headers["location"])
                continue
            response.raise_for_status()
            return response.text
    raise HTTPException(status_code=508, detail="too many redirects")


# --- Endpoints ------------------------------------------------------------

@app.get("/health", summary="Liveness check", operation_id="health")
def health() -> dict:
    return {"status": "ok", "pandoc": pypandoc.get_pandoc_version()}


@app.post(
    "/export/docx",
    summary="Export markdown to a Microsoft Word .docx file",
    description=(
        "Convert APA-formatted (or any) markdown to a Word document. Returns "
        "an OpenWebUI-compatible file payload. Use when the user asks for a Word "
        "document, .docx export, or 'export to Word'. Generate the full final "
        "markdown first, then export that same content."
    ),
    response_description="OWUI file payload for a .docx file",
    operation_id="export_docx",
)
def export_docx(req: ExportRequest, request: Request) -> list[Any]:
    filename = _safe_filename(req.filename, "document")
    extra_args = ["--standalone"]
    if req.title:
        extra_args += ["--metadata", f"title={req.title}"]
    try:
        data = _convert_pandoc(req.markdown, "docx", extra_args)
    except Exception as e:
        logger.exception("docx export failed")
        raise HTTPException(status_code=500, detail=f"docx conversion failed: {e}")
    return _file_result(
        data,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        f"{filename}.docx",
        request,
    )


@app.post(
    "/export/pdf",
    summary="Export markdown to a PDF file",
    description=(
        "Convert markdown to a PDF document. Returns an OpenWebUI-compatible "
        "file payload. Use when the user asks for a PDF, a printable version, or "
        "a read-only share. Generate the full final markdown first, then export "
        "that same content."
    ),
    response_description="OWUI file payload for a PDF file",
    operation_id="export_pdf",
)
def export_pdf(req: ExportRequest, request: Request) -> list[Any]:
    filename = _safe_filename(req.filename, "document")
    # weasyprint = lightweight; xelatex would give publication-grade output
    # but bloats the image by ~3GB.
    extra_args = ["--standalone", "--pdf-engine=weasyprint"]
    if req.title:
        extra_args += ["--metadata", f"title={req.title}"]
    try:
        data = _convert_pandoc(req.markdown, "pdf", extra_args)
    except Exception as e:
        logger.exception("pdf export failed")
        raise HTTPException(status_code=500, detail=f"pdf conversion failed: {e}")
    return _file_result(data, "application/pdf", f"{filename}.pdf", request)


@app.post(
    "/export/markdown",
    summary="Save content as a .md file",
    description=(
        "Save markdown content as a downloadable .md file. Use when the user "
        "asks for a markdown export, a .md file, or wants to download the "
        "raw markdown of a draft. Export the complete content, not a summary."
    ),
    response_description="OWUI file payload for a .md file",
    operation_id="export_markdown",
)
def export_markdown(req: ExportRequest, request: Request) -> list[Any]:
    filename = _safe_filename(req.filename, "document")
    return _file_result(
        _strip_reasoning_leak(req.markdown).encode("utf-8"),
        "text/markdown",
        f"{filename}.md",
        request,
    )


@app.post(
    "/export/csv",
    summary="Export tabular data to a .csv file",
    description=(
        "Convert a list of dictionaries to a CSV file. The keys of the first "
        "row become the column headers. Use when the user asks for CSV, "
        "spreadsheet export, or wants tabular data they can open in Excel."
    ),
    response_description="OWUI file payload for a .csv file",
    operation_id="export_csv",
)
def export_csv(req: CsvExportRequest, request: Request) -> list[Any]:
    if not req.rows:
        raise HTTPException(status_code=400, detail="rows cannot be empty")
    filename = _safe_filename(req.filename, "data")
    fieldnames = list(req.rows[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(req.rows)
    return _file_result(
        buf.getvalue().encode("utf-8"),
        "text/csv",
        f"{filename}.csv",
        request,
    )


@app.post(
    "/fetch_url",
    summary="Fetch a URL and return its readable text content",
    description=(
        "Download a webpage and extract the main readable text (stripping nav, "
        "ads, boilerplate). Use when the user gives a specific URL to summarise "
        "or quote. Prefer this over web search for exact URL requests; use web "
        "search only when broader discovery is requested."
    ),
    operation_id="fetch_url",
)
def fetch_url(req: FetchUrlRequest) -> dict:
    try:
        downloaded = _download_public_url(req.url)
    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"fetch failed with HTTP {e.response.status_code}: {req.url}",
        )
    except Exception as e:
        logger.exception("fetch_url failed at download")
        raise HTTPException(status_code=502, detail=f"fetch failed: {e}")
    if not downloaded:
        raise HTTPException(status_code=502, detail=f"could not fetch {req.url}")
    text = trafilatura.extract(downloaded, include_links=True, include_tables=True) or ""
    truncated = False
    if len(text) > req.max_chars:
        text = text[: req.max_chars] + "\n\n[...truncated]"
        truncated = True
    return {
        "url": req.url,
        "chars": len(text),
        "truncated": truncated,
        "text": text,
    }


@app.post(
    "/lookup_doi_citation",
    summary="Look up a DOI on CrossRef and return an APA citation",
    description=(
        "Query CrossRef for a DOI and return the work as an APA-formatted "
        "citation string plus structured metadata (authors, year, title, "
        "journal). Use when the user gives a DOI and wants the formatted "
        "reference, or when verifying a citation they plan to use."
    ),
    operation_id="lookup_doi_citation",
)
def _title_similarity(a: str, b: str) -> float:
    """Token-overlap ratio (Jaccard) between two titles, case/punct-insensitive."""
    def toks(s: str) -> set:
        return set(re.findall(r"[a-z0-9]+", (s or "").lower()))
    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _crossref_to_apa(msg: dict, fallback_doi: str = "") -> dict:
    authors = msg.get("author", []) or []

    def _author_str(a: dict) -> str:
        family = a.get("family", "").strip()
        given = a.get("given", "").strip()
        initials = "".join(f"{p[0]}." for p in given.split() if p)
        return f"{family}, {initials}".strip(", ")

    author_parts = [_author_str(a) for a in authors[:20] if a.get("family")]
    if len(authors) > 20:
        author_parts.append("...")
    author_str = ", ".join(author_parts) if author_parts else "Unknown"

    issued = msg.get("issued") or msg.get("published") or {}
    date_parts = issued.get("date-parts") or [[None]]
    year = date_parts[0][0] if date_parts and date_parts[0] else "n.d."

    title = (msg.get("title") or [""])[0]
    container = (msg.get("container-title") or [""])[0]
    publisher = msg.get("publisher", "")
    volume = msg.get("volume", "")
    issue = msg.get("issue", "")
    pages = msg.get("page", "")
    doi_canonical = msg.get("DOI", fallback_doi)

    parts = [f"{author_str} ({year}). {title}."]
    if container:
        loc = f"*{container}*"
        if volume:
            loc += f", *{volume}*"
            if issue:
                loc += f"({issue})"
        if pages:
            loc += f", {pages}"
        parts.append(f"{loc}.")
    elif publisher:
        parts.append(f"{publisher}.")
    if doi_canonical:
        parts.append(f"https://doi.org/{doi_canonical}")
    apa = " ".join(parts)

    return {
        "doi": doi_canonical,
        "apa": apa,
        "title": title,
        "year": year,
        "container": container or publisher,
        "authors": [{"family": a.get("family"), "given": a.get("given")} for a in authors],
    }


def _fetch_crossref_work(doi: str) -> dict:
    try:
        r = httpx.get(
            f"https://api.crossref.org/works/{quote(doi, safe='')}",
            headers={"User-Agent": "owui-tool-server/0.3.0"},
            timeout=15.0,
        )
        r.raise_for_status()
        return r.json()["message"]
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=404,
            detail=f"CrossRef returned {e.response.status_code} for DOI {doi}",
        )
    except Exception as e:
        logger.exception("DOI lookup failed")
        raise HTTPException(status_code=502, detail=f"CrossRef lookup failed: {e}")


def lookup_doi_citation(req: CitationRequest) -> dict:
    doi = req.doi.strip()
    for prefix in ("doi:", "https://doi.org/", "http://doi.org/", "https://dx.doi.org/"):
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix):]
            break

    msg = _fetch_crossref_work(doi)
    result = _crossref_to_apa(msg, fallback_doi=doi)

    # Verification gate: if the caller told us what they expect, confirm the
    # CrossRef record actually matches. A valid DOI pointing at the WRONG work
    # (common with SAGE 10.4135 prefixes) must NOT be silently citable.
    if req.expected_title:
        score = _title_similarity(req.expected_title, result["title"])
        result["match_score"] = round(score, 2)
        result["verified"] = score >= 0.4
        if not result["verified"]:
            result["warning"] = (
                f"DOI resolves to '{result['title']}', which does NOT match the "
                f"expected title '{req.expected_title}'. Do not cite this DOI for "
                f"that work; search by title or cite from knowledge instead."
            )
    else:
        result["verified"] = None
        result["warning"] = (
            "No expected_title was provided, so this DOI was not cross-checked. "
            "Confirm the returned title matches the work you intend to cite."
        )
    return result


@app.post(
    "/search_citation",
    summary="Find a work's citation by title/author (no DOI needed)",
    description=(
        "Search CrossRef bibliographically by title and/or author and return the "
        "best APA matches with DOIs and a match_score. Use this INSTEAD of guessing "
        "DOIs when you know the work but not its DOI. If the top match_score is low, "
        "the work may not be in CrossRef — cite from knowledge rather than forcing it."
    ),
    operation_id="search_citation",
)
def search_citation(req: CitationSearchRequest) -> dict:
    try:
        r = httpx.get(
            "https://api.crossref.org/works",
            params={"query.bibliographic": req.query, "rows": req.rows},
            headers={"User-Agent": "owui-tool-server/0.3.0"},
            timeout=15.0,
        )
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", []) or []
    except Exception as e:
        logger.exception("citation search failed")
        raise HTTPException(status_code=502, detail=f"CrossRef search failed: {e}")

    results = []
    for msg in items:
        formatted = _crossref_to_apa(msg)
        formatted["match_score"] = round(_title_similarity(req.query, formatted["title"]), 2)
        results.append(formatted)
    results.sort(key=lambda x: x["match_score"], reverse=True)
    return {
        "query": req.query,
        "count": len(results),
        "results": results,
        "note": (
            "match_score is title-token overlap with your query, not authority. "
            "A low top score (< 0.3) likely means the work is not well indexed in "
            "CrossRef; prefer citing from knowledge over forcing a weak match."
        ),
    }


class VerifyGroundingRequest(BaseModel):
    source: str = Field(
        ...,
        description=(
            "The ground-truth material the draft must stay faithful to — the "
            "user's resume, the cited source's scope, the notes, or the posting. "
            "Every claim in the draft will be checked against ONLY this."
        ),
    )
    draft: str = Field(
        ...,
        description="The draft text to audit for unsupported / fabricated claims.",
    )


@app.post(
    "/verify_grounding",
    summary="Audit a draft for claims not supported by a source (anti-fabrication)",
    description=(
        "Strict grounding auditor. Returns the list of MATERIAL claims in the draft "
        "that are NOT supported by the source — invented skills/tools/metrics/scope, "
        "claims attributed to a source whose type cannot support them, hardened "
        "hedges, or format-constraint violations. Call this on your own draft BEFORE "
        "finalizing any source-bound document (cover letters, summaries of a cited "
        "work, resume bullets) so you can remove fabrications. Returns grounded=true "
        "with an empty list when every claim is supported."
    ),
    operation_id="verify_grounding",
)
def verify_grounding(req: VerifyGroundingRequest) -> dict:
    if not FIREWORKS_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="verify_grounding unavailable: FIREWORKS_API_KEY not set on the tool server.",
        )
    audit_sys = (
        "You are a grounding auditor. Flag ONLY claims in the DRAFT that introduce NEW "
        "factual information absent from the SOURCE — invented skills, tools, metrics, "
        "numbers, scope, credentials, or experience the source never states; claims "
        "attributed to a source whose type cannot support them (e.g. neuroscience pinned "
        "to a teachers' guide); or explicit format-constraint violations.\n"
        "A claim is SUPPORTED if the source states the same fact in ANY wording. Treat "
        "paraphrases, synonyms, and reasonable restatements as supported — e.g. source "
        "'knows Python' -> draft 'proficient in Python' is SUPPORTED (do NOT flag it); "
        "'built X' -> 'developed/engineered X' is SUPPORTED. Do NOT flag stylistic "
        "wording, tone, formatting, or true general knowledge presented as the author's "
        "own. When unsure whether something is a real fabrication vs a rephrase, do NOT "
        "flag it.\n"
        "Output a numbered list of only the genuinely-unsupported phrases, or the single "
        "word NONE if every claim is supported. No preamble, no reasoning."
    )
    payload = {
        "model": VERIFY_MODEL,
        "max_tokens": 1500,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": audit_sys},
            {"role": "user", "content": f"SOURCE:\n{req.source}\n\nDRAFT:\n{req.draft}\n\nUnsupported claims:"},
        ],
    }
    if "flash" in VERIFY_MODEL:  # deepseek-v4-flash: auditor, not a thinker
        payload["reasoning_effort"] = "none"
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(
                "https://api.fireworks.ai/inference/v1/chat/completions",
                headers={"Authorization": f"Bearer {FIREWORKS_API_KEY}",
                         "Content-Type": "application/json"},
                json=payload,
            )
            r.raise_for_status()
            content = (r.json()["choices"][0]["message"].get("content") or "").strip()
    except Exception as e:
        logger.exception("verify_grounding failed")
        raise HTTPException(status_code=502, detail=f"auditor call failed: {e}")

    grounded = content.upper().strip().startswith("NONE") or not content
    return {
        "grounded": grounded,
        "unsupported_claims": "" if grounded else content,
        "guidance": (
            "All claims are supported by the source."
            if grounded
            else "Remove or honestly hedge each listed claim, then re-finalize. "
            "For an out-of-scope citation, keep the true statement but drop the wrong "
            "citation rather than deleting the sentence."
        ),
    }


# ── Chat memory endpoints ──────────────────────────────────────────────

class MemoryStoreRequest(BaseModel):
    chat_id: str = Field(..., description="OWUI chat ID")
    role: str = Field(..., description="user or assistant")
    content: str = Field(..., description="Turn text content")


class MemoryRecallRequest(BaseModel):
    chat_id: str = Field(..., description="OWUI chat ID")
    query: str = Field(..., description="Recall query (typically the user's message)")
    exclude_hashes: list[str] = Field(default=[], description="Content hashes to exclude")
    top_k: int = Field(default=6, description="Max turns to return")


class MemorySweepRequest(BaseModel):
    chat_id: str = Field(..., description="OWUI chat ID to sweep")


class DeliverableStoreRequest(BaseModel):
    chat_id: str = Field(..., description="OWUI chat ID")
    content: str = Field(..., description="The verified deliverable text")
    filename: str = Field(default="", description="Delivered file name")
    fmt: str = Field(default="", description="Format: docx | pdf | md | …")


class DeliverableGetRequest(BaseModel):
    chat_id: str = Field(..., description="OWUI chat ID")


class LastActiveRequest(BaseModel):
    chat_id: str = Field(..., description="OWUI chat ID")


@app.on_event("startup")
async def startup_init_memory():
    """Pre-init the memory DB on startup so first request is fast."""
    await memory.get_conn()


@app.post(
    "/memory/store",
    summary="Store a chat turn in semantic memory",
    operation_id="memory_store",
)
async def memory_store(req: MemoryStoreRequest) -> dict:
    """Store a user or assistant turn for later semantic recall."""
    ok = await memory.store(req.chat_id, req.role, req.content)
    if ok:
        asyncio.create_task(memory.maybe_compress_chat(req.chat_id))
    return {"stored": ok, "chat_id": req.chat_id}


@app.post(
    "/memory/recall",
    summary="Recall relevant prior turns from chat memory",
    operation_id="memory_recall",
)
async def memory_recall(req: MemoryRecallRequest) -> dict:
    """Semantically search past turns in this chat and return the most relevant ones."""
    turns = await memory.recall(
        req.chat_id, req.query, req.exclude_hashes, req.top_k
    )
    return {
        "chat_id": req.chat_id,
        "count": len(turns),
        "turns": [{"role": r, "content": c} for r, c in turns],
    }


@app.post(
    "/memory/sweep",
    summary="Run TTL and referential sweep on memory",
    operation_id="memory_sweep",
)
async def memory_sweep(req: MemorySweepRequest) -> dict:
    """Clean up old or orphaned memory rows for a chat."""
    # Basic per-chat sweep: delete turns older than 90 days
    compacted = await memory.maybe_compress_chat(req.chat_id)
    removed = await memory.sweep_chat(req.chat_id)
    return {"chat_id": req.chat_id, "removed_rows": removed, "compacted_rows": compacted}


@app.post(
    "/deliverable/store",
    summary="Persist the document a turn delivered, for later editing",
    operation_id="deliverable_store",
)
async def deliverable_store(req: DeliverableStoreRequest) -> dict:
    """Store the verified deliverable so a follow-up turn can edit the real artifact."""
    version = await memory.store_deliverable(req.chat_id, req.content, req.filename, req.fmt)
    return {"stored": version > 0, "version": version, "chat_id": req.chat_id}


@app.post(
    "/deliverable/get",
    summary="Fetch the latest deliverable for a chat (the document to edit)",
    operation_id="deliverable_get",
)
async def deliverable_get(req: DeliverableGetRequest) -> dict:
    """Return the latest stored deliverable for this chat, or null."""
    d = await memory.get_deliverable(req.chat_id)
    return {"chat_id": req.chat_id, "deliverable": d}


@app.post(
    "/memory/last_active",
    summary="When this chat last had a stored turn (for resume-after-gap awareness)",
    operation_id="memory_last_active",
)
async def memory_last_active(req: LastActiveRequest) -> dict:
    """Return the unix time of the chat's most recent stored turn, or null."""
    return {"chat_id": req.chat_id, "last_active": await memory.last_active(req.chat_id)}
