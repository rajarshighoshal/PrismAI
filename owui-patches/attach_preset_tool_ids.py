"""Merge a model preset's configured tool ids into the request tool_ids.

OWUI v0.9.5 reads request `tool_ids` only from the frontend payload; it does
NOT merge a workspace-model preset's `meta.toolIds` server-side. So external
OpenAPI tool servers listed on a preset (e.g. `server:openapi:owui-tool-server`)
never reach the model unless the user toggles them on per-chat. Builtin tools
are unaffected because they load via a separate `get_builtin_tools(model)` path,
which is why notes/tasks/knowledge show up but `export_docx` does not.

This patch injects the selected preset's own `meta.toolIds` into `tool_ids`
right where server-side tool resolution begins. Scoped to the selected preset
(not all registered servers globally), so it never attaches tools to models
that didn't ask for them — avoiding the "routing leaks" that the old global
auto-attach patch caused. Idempotent.
"""
import sys
from pathlib import Path

PATH = Path("/app/backend/open_webui/utils/middleware.py")
MARKER = "# OWUI_PRESET_TOOLIDS_MERGE_PATCH"
ANCHOR = "        tool_ids = metadata.get('tool_ids', None)\n"

BLOCK = (
    ANCHOR
    + "        " + MARKER + "\n"
    + "        try:\n"
    + "            _preset_meta = (model.get('info', {}) or {}).get('meta', {}) or {}\n"
    + "            _preset_tool_ids = _preset_meta.get('toolIds') or _preset_meta.get('tool_ids') or []\n"
    + "            if _preset_tool_ids:\n"
    + "                tool_ids = list(dict.fromkeys((tool_ids or []) + list(_preset_tool_ids)))\n"
    + "        except Exception:\n"
    + "            pass\n"
)


def main() -> int:
    if not PATH.exists():
        print("ERROR: target not found: " + str(PATH), file=sys.stderr)
        return 2
    src = PATH.read_text()
    if MARKER in src:
        print("preset toolIds merge already present - no-op")
        return 0
    if ANCHOR not in src:
        print("ERROR: anchor not found - OWUI version shifted; review middleware.py", file=sys.stderr)
        return 1
    src = src.replace(ANCHOR, BLOCK, 1)
    PATH.write_text(src)
    print("wrote " + str(PATH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
