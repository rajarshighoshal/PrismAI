#!/bin/bash
# Weekly per-user STYLE-memory consolidation (style/intent only, never facts).
# Runs orchestrator/consolidate_style.py inside the open-webui container (which
# owns a writable webui.db). Sources FIREWORKS_API_KEY from the orchestrator env.
# Installed on the server as a weekly cron (Sun 4am); see README.
set -euo pipefail
REPO=/opt/owui-hybrid-router
FWKEY=$(grep -E '^FIREWORKS_API_KEY=' "$REPO/orchestrator/orchestrator.env" | cut -d= -f2-)
docker cp "$REPO/orchestrator/consolidate_style.py" open-webui:/tmp/consolidate_style.py
docker exec -e FIREWORKS_API_KEY="$FWKEY" open-webui python3 /tmp/consolidate_style.py
