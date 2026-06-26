#!/bin/bash
# Off-site backup of the chat server's irreplaceable state to Cloudflare R2.
#
# Backs up (all consistent, never a half-written file):
#   - webui.db       OWUI chats / users / settings / functions  (open-webui container)
#   - router_mem.db  per-chat semantic memory + usage ledger     (owui-tool-server volume)
#   - uploads/       user-attached files                         (host data dir)
#
# Layout in the bucket (prefix defaults to "prismai"):
#   <prefix>/owui/webui.db                 latest snapshot (overwritten each run)
#   <prefix>/owui/router_mem.db            latest snapshot (overwritten each run)
#   <prefix>/owui/uploads/                 mirrored
#   <prefix>/snapshots/<UTC-date>/*.db     dated DB copies for point-in-time restore
#
# Credentials come from r2-backup.env next to this script (gitignored). This script
# carries NO secrets, so it is safe to commit. rclone is configured purely from env
# vars here, so it does not depend on ~/.config/rclone being present.
#
# Cron (root):  15 3 * * * /opt/owui-hybrid-router/backup-r2.sh >> /var/log/r2-backup.log 2>&1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${R2_BACKUP_ENV:-${SCRIPT_DIR}/r2-backup.env}"
DATA_DIR="${OWUI_DATA_DIR:-${SCRIPT_DIR}/data/owui}"
OWUI_CONTAINER="${OWUI_CONTAINER:-open-webui}"
TOOL_CONTAINER="${TOOL_CONTAINER:-owui-tool-server}"
SNAPSHOT_RETENTION_DAYS="${SNAPSHOT_RETENTION_DAYS:-14}"

log() { echo "$(date -Iseconds) $*"; }
die() { echo "$(date -Iseconds) ERROR: $*" >&2; exit 1; }

[ -f "$ENV_FILE" ] || die "env file not found: $ENV_FILE"
# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a

: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID missing in $ENV_FILE}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY missing in $ENV_FILE}"
: "${R2_ENDPOINT:?R2_ENDPOINT missing in $ENV_FILE}"
: "${R2_BUCKET:?R2_BUCKET missing in $ENV_FILE}"
R2_PREFIX="${R2_PREFIX:-prismai}"

command -v rclone >/dev/null 2>&1 || die "rclone not installed"
command -v docker >/dev/null 2>&1 || die "docker not installed"

# Self-contained rclone remote (named r2bk) built entirely from env — no rclone config file.
export RCLONE_CONFIG_R2BK_TYPE=s3
export RCLONE_CONFIG_R2BK_PROVIDER=Cloudflare
export RCLONE_CONFIG_R2BK_REGION=auto
export RCLONE_CONFIG_R2BK_NO_CHECK_BUCKET=true
export RCLONE_CONFIG_R2BK_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export RCLONE_CONFIG_R2BK_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export RCLONE_CONFIG_R2BK_ENDPOINT="$R2_ENDPOINT"

DEST="r2bk:${R2_BUCKET}/${R2_PREFIX}"
DATE_UTC="$(date -u +%F)"
STAGE="$(mktemp -d /tmp/r2-backup.XXXXXX)"
trap 'rm -rf "$STAGE"' EXIT

log "backup start -> ${DEST}"

# 1. Consistent SQLite snapshots via the live containers (sqlite .backup, WAL-safe).
#    Snapshot inside the container, then docker cp out — works for both bind mounts and
#    named volumes, and never copies a torn page from an in-flight write.
snapshot_db() {
    local container="$1" db_in_container="$2" out_name="$3"
    docker exec -i "$container" python3 - "$db_in_container" <<'PY'
import sqlite3, sys
src = sqlite3.connect(sys.argv[1])
dst = sqlite3.connect("/tmp/_r2_snap.db")
with dst:
    src.backup(dst)
dst.close(); src.close()
PY
    docker cp "$container:/tmp/_r2_snap.db" "$STAGE/$out_name"
    docker exec "$container" rm -f /tmp/_r2_snap.db
    log "snapshot ok: $out_name ($(stat -c%s "$STAGE/$out_name" 2>/dev/null || echo '?') bytes)"
}

snapshot_db "$OWUI_CONTAINER" /app/backend/data/webui.db webui.db
snapshot_db "$TOOL_CONTAINER" /app/backend/data/router_mem.db router_mem.db

# 2. Upload the DB snapshots: a stable "latest" copy + a dated point-in-time copy.
rclone copy "$STAGE/webui.db"       "${DEST}/owui/" --s3-no-check-bucket
rclone copy "$STAGE/router_mem.db"  "${DEST}/owui/" --s3-no-check-bucket
rclone copy "$STAGE/webui.db"       "${DEST}/snapshots/${DATE_UTC}/" --s3-no-check-bucket
rclone copy "$STAGE/router_mem.db"  "${DEST}/snapshots/${DATE_UTC}/" --s3-no-check-bucket
log "db snapshots uploaded (latest + snapshots/${DATE_UTC})"

# 3. Mirror user uploads (files the user attached). Skipped with a warning if absent.
if [ -d "${DATA_DIR}/uploads" ]; then
    rclone sync "${DATA_DIR}/uploads" "${DEST}/owui/uploads" --s3-no-check-bucket
    log "uploads synced from ${DATA_DIR}/uploads"
else
    log "WARNING: uploads dir not found at ${DATA_DIR}/uploads — skipped"
fi

# 4. Prune dated DB snapshots older than the retention window.
if [ "${SNAPSHOT_RETENTION_DAYS}" -gt 0 ] 2>/dev/null; then
    rclone delete "${DEST}/snapshots" --min-age "${SNAPSHOT_RETENTION_DAYS}d" --s3-no-check-bucket || true
    rclone rmdirs "${DEST}/snapshots" --leave-root --s3-no-check-bucket || true
    log "pruned snapshots older than ${SNAPSHOT_RETENTION_DAYS}d"
fi

log "backup complete"
