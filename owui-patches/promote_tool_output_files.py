"""Promote OpenAPI tool data URI outputs to OpenWebUI message files."""
import sys
from pathlib import Path

PATH = Path("/app/backend/open_webui/utils/middleware.py")

IMPORT_ANCHOR = "import os\n"
IO_IMPORT = "import io\n"

MODEL_IMPORT_ANCHOR = "from open_webui.models.models import Models\n"
FILE_IMPORTS = (
    "from open_webui.models.files import Files, FileForm\n"
    "from open_webui.storage.provider import Storage\n"
)

HELPER_MARKER = "# OWUI_TOOL_OUTPUT_FILE_HELPER_PATCH"
HELPER_ANCHOR = "\n\nasync def process_tool_result(\n"
HELPER = f"""

{HELPER_MARKER}
async def owui_tool_output_file_from_data_uri(request, user, data_uri, filename=None, mime_type=None):
    match = re.match(r"^data:([^;,]+)?(?:;[^,]*)?;base64,(.*)$", data_uri, re.DOTALL)
    if not match or not user:
        return None

    detected_mime = mime_type or match.group(1) or "application/octet-stream"
    try:
        raw = base64.b64decode(match.group(2), validate=True)
    except Exception:
        return None

    extensions = {{
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/pdf": ".pdf",
        "text/markdown": ".md",
        "text/csv": ".csv",
    }}
    safe_name = os.path.basename(filename or f"tool-output{{extensions.get(detected_mime, '.bin')}}") or "tool-output.bin"
    file_id = str(uuid4())
    stored_name = f"{{file_id}}_{{safe_name}}"

    user_id = getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)
    if not user_id:
        return None
    user_email = getattr(user, "email", "") or (user.get("email", "") if isinstance(user, dict) else "")
    user_name = getattr(user, "name", "") or (user.get("name", "") if isinstance(user, dict) else "")

    contents, file_path = await asyncio.to_thread(
        Storage.upload_file,
        io.BytesIO(raw),
        stored_name,
        {{
            "OpenWebUI-User-Email": user_email,
            "OpenWebUI-User-Id": user_id,
            "OpenWebUI-User-Name": user_name,
            "OpenWebUI-File-Id": file_id,
        }},
    )
    file_item = await Files.insert_new_file(
        user_id,
        FileForm(
            id=file_id,
            filename=safe_name,
            path=file_path,
            data={{}},
            meta={{
                "name": safe_name,
                "content_type": detected_mime,
                "size": len(contents),
                "data": {{"source": "tool_output"}},
            }},
        ),
    )
    if not file_item:
        return None

    return {{
        "type": "file",
        "id": file_item.id,
        "name": safe_name,
        "url": file_item.id,
        "size": len(contents),
        "content_type": detected_mime,
        "status": "uploaded",
        "file": file_item.model_dump(),
    }}
"""

OPENAPI_BRANCH_OLD = """        else:  # OpenAPI
            for item in tool_result:
                if isinstance(item, str) and item.startswith('data:'):
                    tool_result_files.append(
                        {
                            'type': 'data',
                            'content': item,
                        }
                    )
                    tool_result.remove(item)
"""

OPENAPI_BRANCH_NEW = """        else:  # OpenAPI
            remaining_tool_result = []
            for idx, item in enumerate(tool_result):
                if isinstance(item, str) and item.startswith('data:'):
                    file_meta = next(
                        (
                            candidate
                            for candidate in tool_result[idx + 1 :]
                            if isinstance(candidate, dict)
                            and (candidate.get('filename') or candidate.get('mime_type'))
                        ),
                        {},
                    )
                    uploaded_file = await owui_tool_output_file_from_data_uri(
                        request,
                        user,
                        item,
                        filename=file_meta.get('filename'),
                        mime_type=file_meta.get('mime_type'),
                    )
                    tool_result_files.append(uploaded_file or {'type': 'data', 'content': item})
                else:
                    remaining_tool_result.append(item)
            tool_result = remaining_tool_result
"""

PROMOTE_MARKER = "# OWUI_PROMOTE_TOOL_OUTPUT_FILES_PATCH"
PROMOTE_ANCHOR = """                        output.append(
                            {
                                'type': 'function_call_output',
"""
PROMOTE_BLOCK = f"""                        {PROMOTE_MARKER}
                        if display_files and metadata.get('chat_id') and metadata.get('message_id'):
                            message_files = await Chats.add_message_files_by_id_and_message_id(
                                metadata['chat_id'],
                                metadata['message_id'],
                                display_files,
                            )
                            if event_emitter:
                                await event_emitter(
                                    {{
                                        'type': 'files',
                                        'data': {{'files': message_files or display_files}},
                                    }}
                                )

{PROMOTE_ANCHOR}"""

OLD_AUTO_ATTACH_MARKER = "        # OWUI_TOOL_IDS_AUTO_ATTACH_PATCH\n"
OLD_AUTO_ATTACH_END = "        # Client side tools"


def replace_once(src: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if old not in src:
        raise RuntimeError(f"{label} anchor not found")
    updated = src.replace(old, new, 1)
    return updated, updated != src


def remove_old_auto_attach_patch(src: str) -> str:
    if OLD_AUTO_ATTACH_MARKER not in src:
        return src
    start = src.index(OLD_AUTO_ATTACH_MARKER)
    end = src.index(OLD_AUTO_ATTACH_END, start)
    return src[:start] + src[end:]


def main() -> int:
    if not PATH.exists():
        print(f"ERROR: target file not found: {PATH}", file=sys.stderr)
        return 2

    src = PATH.read_text()
    original = src

    src = remove_old_auto_attach_patch(src)

    if IO_IMPORT not in src:
        src, _ = replace_once(src, IMPORT_ANCHOR, IMPORT_ANCHOR + IO_IMPORT, "import")

    if FILE_IMPORTS not in src:
        src, _ = replace_once(src, MODEL_IMPORT_ANCHOR, MODEL_IMPORT_ANCHOR + FILE_IMPORTS, "file imports")

    if HELPER_MARKER not in src:
        src, _ = replace_once(src, HELPER_ANCHOR, HELPER + HELPER_ANCHOR, "helper")

    if OPENAPI_BRANCH_NEW not in src:
        src, _ = replace_once(src, OPENAPI_BRANCH_OLD, OPENAPI_BRANCH_NEW, "OpenAPI data URI branch")

    if PROMOTE_MARKER not in src:
        src, _ = replace_once(src, PROMOTE_ANCHOR, PROMOTE_BLOCK, "message file promotion")

    if src == original:
        print("file already in desired state - no-op")
    else:
        PATH.write_text(src)
        print(f"wrote {PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
