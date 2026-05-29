#!/bin/bash
# Poll origin/main and run update.sh if HEAD moved.
# Intended for cron. Example log: /var/log/owui-auto-deploy.log
set -euo pipefail

REPO_DIR="/opt/owui-hybrid-router"
LOG="/var/log/owui-auto-deploy.log"
LOCK="/run/owui-auto-deploy.lock"

exec >> "$LOG" 2>&1
exec 9> "$LOCK"
flock -n 9 || { echo "$(date -Iseconds) skipped: another run in progress"; exit 0; }

cd "$REPO_DIR"
git fetch --quiet origin main

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "$(date -Iseconds) up to date at $LOCAL"
    exit 0
fi

echo "=== $(date -Iseconds) New commit: $LOCAL -> $REMOTE ==="
git pull --ff-only origin main
./update.sh
echo "=== $(date -Iseconds) Deploy complete ==="
