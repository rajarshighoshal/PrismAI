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
    + "            # Auto-attach all enabled tool servers when the frontend\n"
    + "            # doesn't specify any. Uses the standard server-side path so\n"
    + "            # callables are wired correctly for execution.\n"
    + "            _auto_servers = getattr(request.app.state, 'TOOL_SERVERS', None) or []\n"
    + "            _auto = [f\"server:openapi:{s['id']}\" for s in _auto_servers if s.get('id')]\n"
    + "            if _auto:\n"
    + "                tool_ids = _auto"
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


def apply_new_patch(src: str) -> str:
    if NEW_MARKER in src:
        return src
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
    if src != original:
        print("removed old broken direct_tool_servers patch")

    if NEW_MARKER in src:
        print("new tool_ids patch already present - no-op")
    else:
        try:
            src = apply_new_patch(src)
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        print("applied new tool_ids auto-attach patch")

    if src != original:
        PATH.write_text(src)
        print(f"wrote {PATH}")
    else:
        print("file already in desired state")
    return 0


if __name__ == "__main__":
    sys.exit(main())
