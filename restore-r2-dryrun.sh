#!/bin/bash
# Download latest R2 backups to /tmp and verify SQLite integrity.
# Never writes to production paths.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${R2_BACKUP_ENV:-${SCRIPT_DIR}/r2-backup.env}"
R2_PREFIX="${R2_PREFIX:-prismai}"

log() { echo "$(date -Iseconds) $*"; }
die() { echo "$(date -Iseconds) ERROR: $*" >&2; exit 1; }

[ -f "$ENV_FILE" ] || die "env file not found: $ENV_FILE"
# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a

: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID missing in $ENV_FILE}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY missing in $ENV_FILE}"
: "${R2_ENDPOINT:?R2_ENDPOINT missing in $ENV_FILE}"
: "${R2_BUCKET:?R2_BUCKET missing in $ENV_FILE}"

command -v rclone >/dev/null 2>&1 || die "rclone not installed"
command -v python3 >/dev/null 2>&1 || die "python3 not installed"

export RCLONE_CONFIG_R2BK_TYPE=s3
export RCLONE_CONFIG_R2BK_PROVIDER=Cloudflare
export RCLONE_CONFIG_R2BK_REGION=auto
export RCLONE_CONFIG_R2BK_NO_CHECK_BUCKET=true
export RCLONE_CONFIG_R2BK_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export RCLONE_CONFIG_R2BK_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export RCLONE_CONFIG_R2BK_ENDPOINT="$R2_ENDPOINT"

SRC="r2bk:${R2_BUCKET}/${R2_PREFIX}/owui"
STAGE="$(mktemp -d /tmp/r2-restore-dryrun.XXXXXX)"
trap 'rm -rf "$STAGE"' EXIT

log "restore dry-run download -> $STAGE"
rclone copy "${SRC}/webui.db" "$STAGE" --s3-no-check-bucket
rclone copy "${SRC}/router_mem.db" "$STAGE" --s3-no-check-bucket

python3 - "$STAGE/webui.db" "$STAGE/router_mem.db" <<'PY'
import os, sqlite3, sys
for path in sys.argv[1:]:
    con = sqlite3.connect(path)
    try:
        result = con.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        con.close()
    print(f"{os.path.basename(path)} bytes={os.path.getsize(path)} integrity={result}")
    if result.lower() != "ok":
        raise SystemExit(1)
PY

log "restore dry-run OK"
