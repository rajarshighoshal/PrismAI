"""Patch OWUI's middleware to auto-attach all enabled tool servers.

OWUI 0.8.12 only attaches tool servers to a chat when the frontend
sends `metadata.tool_ids` containing `server:openapi:<id>` refs.
That requires per-chat UI opt-in — friction we don't want.

This patch makes the backend fall back: when no `tool_ids` were
specified, auto-populate them with `server:openapi:<id>` for every
registered tool server in `app.state.TOOL_SERVERS`. The standard
server-side execution path (which DOES wire up callables) then
handles invocation correctly.

Idempotent. Also cleans up an earlier broken version of this patch
that targeted `direct_tool_servers` (which is a frontend-only path
that doesn't attach the execution callable).
"""
import re
import sys
from pathlib import Path

PATH = Path("/app/backend/open_webui/utils/middleware.py")

OLD_MARKER = "# OWUI_TOOL_SERVERS_AUTO_ATTACH_PATCH"
NEW_MARKER = "# OWUI_TOOL_IDS_AUTO_ATTACH_PATCH"

NEW_ANCHOR = "        tool_ids = metadata.get('tool_ids', None)"
NEW_PATCH = (
    NEW_ANCHOR
    + "\n"
    + "        " + NEW_MARKER + "\n"
    + "        if not tool_ids:\n"
    + "            tool_ids = [\n"
    + "                f\"server:openapi:{s['id']}\"\n"
    + "                for s in (getattr(request.app.state, 'TOOL_SERVERS', None) or [])\n"
    + "                if s.get('id')\n"
    + "            ] or None"
)


def remove_old_patch(src: str) -> str:
    """Strip the old broken direct_tool_servers patch block, if present."""
    # Old block was:
    #   # OWUI_TOOL_SERVERS_AUTO_ATTACH_PATCH
    #   if not direct_tool_servers:
    #       <4 comment lines>
    #       direct_tool_servers = getattr(request.app.state, 'TOOL_SERVERS', None) or None
    # 7 lines total starting at the marker line, with consistent 8-space indent.
    pattern = re.compile(
        r"^[ \t]*" + re.escape(OLD_MARKER) + r"\n"
        r"(?:[ \t]*if not direct_tool_servers:\n"
        r"(?:[ \t]+.*\n){5})",
        re.MULTILINE,
    )
    return pattern.sub("", src)


# Patch block ends at this landmark line (the next line in stock OWUI
# after the tool_ids read). Used to strip any existing patch body so
# we can always re-apply the current desired body — keeps the live
# file in sync with the repo even when only the body changes.
PATCH_END_LANDMARK = "        # Client side tools"


def strip_current_patch(src: str) -> str:
    """Remove any existing NEW_MARKER block. No-op if not present."""
    marker_line = "        " + NEW_MARKER + "\n"
    if marker_line not in src:
        return src
    start = src.index(marker_line)
    end = src.index(PATCH_END_LANDMARK, start)
    return src[:start] + src[end:]


def apply_new_patch(src: str) -> str:
    if NEW_ANCHOR not in src:
        raise RuntimeError(
            "anchor line not found - OWUI version may have shifted; review middleware.py"
        )
    return src.replace(NEW_ANCHOR, NEW_PATCH, 1)


def main() -> int:
    if not PATH.exists():
        print(f"ERROR: target file not found: {PATH}", file=sys.stderr)
        return 2
    src = PATH.read_text()
    original = src

    src = remove_old_patch(src)
    src = strip_current_patch(src)
    try:
        src = apply_new_patch(src)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if src == original:
        print("file already in desired state - no-op")
    else:
        PATH.write_text(src)
        print(f"wrote {PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
