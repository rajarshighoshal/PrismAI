"""Forward the OWUI message-id to OpenAI model connections.

openai.py forwards X-OpenWebUI-Chat-Id to a connection but NOT the message-id, so
a connection (the PrismAI orchestrator) can't attach exported files to the exact
assistant message. This adds the message-id forwarding next to the chat-id one
(and imports the header name if needed). Idempotent.
"""
import sys
from pathlib import Path

PATH = Path("/app/backend/open_webui/routers/openai.py")
MARKER = "# OWUI_FORWARD_MESSAGE_ID_PATCH"

IMPORT_ANCHOR = "    FORWARD_SESSION_INFO_HEADER_CHAT_ID,\n"
IMPORT_ADD = "    FORWARD_SESSION_INFO_HEADER_MESSAGE_ID,\n"

FWD_ANCHOR = "            headers[FORWARD_SESSION_INFO_HEADER_CHAT_ID] = metadata.get('chat_id')\n"
FWD_ADD = (
    "        " + MARKER + "\n"
    "        if metadata and metadata.get('message_id'):\n"
    "            headers[FORWARD_SESSION_INFO_HEADER_MESSAGE_ID] = metadata.get('message_id')\n"
)


def main() -> int:
    if not PATH.exists():
        print("ERROR: target not found: " + str(PATH), file=sys.stderr)
        return 2
    src = PATH.read_text()
    if MARKER in src:
        print("message-id forward already present - no-op")
        return 0
    if FWD_ANCHOR not in src:
        print("ERROR: forward anchor not found - OWUI version shifted", file=sys.stderr)
        return 1
    if IMPORT_ADD.strip() not in src and IMPORT_ANCHOR in src:
        src = src.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT_ADD, 1)
    src = src.replace(FWD_ANCHOR, FWD_ANCHOR + FWD_ADD, 1)
    PATH.write_text(src)
    print("wrote " + str(PATH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
