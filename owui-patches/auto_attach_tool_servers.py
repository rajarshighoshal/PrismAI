"""Patch OWUI's middleware to auto-attach all enabled tool servers.

OWUI 0.8.12 only attaches tool servers to a chat when the frontend
sends `metadata.tool_servers`. That requires the user to toggle Tools
in the chat composer per-chat — friction we don't want.

This patch makes the backend fall back to `app.state.TOOL_SERVERS`
(populated from `config.tool_server.connections` at startup) whenever
the frontend doesn't specify any. Result: the model sees tools in
every chat and decides when to call.

Idempotent — re-running is a no-op. Run inside the OWUI container:

    docker cp auto_attach_tool_servers.py open-webui:/tmp/
    docker exec open-webui python3 /tmp/auto_attach_tool_servers.py
    docker restart open-webui

After any OWUI image upgrade, re-run this script.
"""
import sys
from pathlib import Path

PATH = Path("/app/backend/open_webui/utils/middleware.py")
MARKER = "# OWUI_TOOL_SERVERS_AUTO_ATTACH_PATCH"
TARGET = "        direct_tool_servers = metadata.get('tool_servers', None)"
PATCHED = (
    TARGET
    + "\n"
    + "        " + MARKER + "\n"
    + "        if not direct_tool_servers:\n"
    + "            # Auto-attach all enabled tool servers when the frontend\n"
    + "            # doesn't specify any. OWUI 0.8.12 stock behavior requires\n"
    + "            # per-chat opt-in; this local patch removes that friction\n"
    + "            # so the model always has tools and decides when to call.\n"
    + "            direct_tool_servers = getattr(request.app.state, 'TOOL_SERVERS', None) or None"
)


def main() -> int:
    if not PATH.exists():
        print(f"ERROR: target file not found: {PATH}", file=sys.stderr)
        return 2
    src = PATH.read_text()
    if MARKER in src:
        print("already patched - no-op")
        return 0
    if TARGET not in src:
        print(
            "ERROR: target line not found. OWUI version may have shifted; "
            "review middleware.py manually before re-running.",
            file=sys.stderr,
        )
        return 1
    new = src.replace(TARGET, PATCHED, 1)
    PATH.write_text(new)
    print(f"patched {PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
