#!/bin/bash
# Apply all OWUI patches in this directory to the running open-webui
# container, then restart. Idempotent: each patch script is a no-op if
# already applied.
#
# Run after every `deploy.sh` (which recreates the container and wipes
# patches) or after an OWUI image upgrade.
set -euo pipefail

CONTAINER="open-webui"
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
CHANGED=0

for patch in "$PATCH_DIR"/*.py; do
    name=$(basename "$patch")
    echo "--- Applying $name ---"
    docker cp "$patch" "${CONTAINER}:/tmp/${name}"
    if docker exec "$CONTAINER" python3 "/tmp/${name}" | tee /tmp/_patch_out; then
        # Treat any of these as "file was modified, needs restart":
        # - "patched ..." (legacy single-line scripts)
        # - "wrote ..."   (multi-step scripts that rewrite the file)
        # - "applied ..."  (multi-step scripts that announce per-stage)
        # - "removed ..."  (cleanup phases)
        if grep -qE "^(patched|wrote|applied|removed) " /tmp/_patch_out; then
            CHANGED=$((CHANGED + 1))
        fi
    else
        echo "ERROR applying $name — aborting before restart"
        exit 1
    fi
done

if [ "$CHANGED" -gt 0 ]; then
    echo ""
    echo "--- ${CHANGED} patch(es) applied — restarting ${CONTAINER} ---"
    docker restart "$CONTAINER"
    echo "Waiting for healthy..."
    for i in $(seq 1 18); do
        s=$(docker inspect -f '{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo unknown)
        echo "  ${i}: ${s}"
        [ "$s" = "healthy" ] && break
        sleep 5
    done
else
    echo ""
    echo "No new patches to apply — no restart needed."
fi
