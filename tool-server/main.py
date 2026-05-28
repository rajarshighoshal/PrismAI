"""Tool server for OpenWebUI: markdown -> docx/pdf via pandoc.

OpenAPI-discoverable so OpenWebUI can auto-register the endpoints as
tools the model can invoke. Stateless: each request is a self-contained
conversion. Listens on the internal docker network only — no auth.
"""
from __future__ import annotations

import io
import logging
import re
import tempfile
from pathlib import Path
from typing import Optional

import pypandoc
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("tool-server")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="OWUI Export Tool Server",
    description=(
        "Convert markdown to .docx / .pdf for OpenWebUI chats. Used by the "
        "model via tool calls when the user asks to export a draft."
    ),
    version="0.1.0",
)


class ExportRequest(BaseModel):
    markdown: str = Field(
        ...,
        description=(
            "The markdown content to convert. May include headings, lists, "
            "in-text citations like (Author, 2024), code blocks, tables, and links."
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
            "Document title shown in Word/PDF metadata (Properties pane) and "
            "rendered on the first page if the template includes {{title}}."
        ),
    )


def _safe_filename(name: Optional[str], default: str) -> str:
    raw = (name or default).strip() or default
    safe = re.sub(r"[^A-Za-z0-9_\- ]", "", raw).strip().replace(" ", "_")
    return safe or default


def _convert(markdown: str, fmt: str, extra_args: list[str]) -> bytes:
    """Run pandoc and return the binary output. Always cleans up the temp file."""
    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
        out_path = Path(tmp.name)
    try:
        pypandoc.convert_text(
            markdown,
            to=fmt,
            format="markdown",
            outputfile=str(out_path),
            extra_args=extra_args,
        )
        return out_path.read_bytes()
    finally:
        out_path.unlink(missing_ok=True)


@app.get("/health", summary="Liveness check")
def health() -> dict:
    return {"status": "ok", "pandoc": pypandoc.get_pandoc_version()}


@app.post(
    "/export/docx",
    summary="Export markdown to a Microsoft Word .docx file",
    description=(
        "Convert APA-formatted (or any) markdown to a Word document. Returns the "
        "binary .docx as a download. Use this when the user asks for a Word "
        "document, .docx export, or 'export to Word'."
    ),
    response_description="Binary .docx file",
)
def export_docx(req: ExportRequest) -> StreamingResponse:
    filename = _safe_filename(req.filename, "document")
    extra_args = ["--standalone"]
    if req.title:
        extra_args += ["--metadata", f"title={req.title}"]
    try:
        data = _convert(req.markdown, "docx", extra_args)
    except Exception as e:
        logger.exception("docx export failed")
        raise HTTPException(status_code=500, detail=f"docx conversion failed: {e}")
    return StreamingResponse(
        io.BytesIO(data),
        media_type=(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        headers={"Content-Disposition": f'attachment; filename="{filename}.docx"'},
    )


@app.post(
    "/export/pdf",
    summary="Export markdown to a PDF file",
    description=(
        "Convert markdown to a PDF document. Returns the binary PDF as a "
        "download. Use when the user asks for a PDF, a printable version, or a "
        "read-only share."
    ),
    response_description="Binary PDF file",
)
def export_pdf(req: ExportRequest) -> StreamingResponse:
    filename = _safe_filename(req.filename, "document")
    # weasyprint is the lightest reasonable PDF engine; xelatex would give
    # publication-grade typography but bloats the image by ~3GB.
    extra_args = ["--standalone", "--pdf-engine=weasyprint"]
    if req.title:
        extra_args += ["--metadata", f"title={req.title}"]
    try:
        data = _convert(req.markdown, "pdf", extra_args)
    except Exception as e:
        logger.exception("pdf export failed")
        raise HTTPException(status_code=500, detail=f"pdf conversion failed: {e}")
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}.pdf"'},
    )
