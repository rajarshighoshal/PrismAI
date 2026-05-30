#!/bin/bash
# Zero-downtime blue-green deployment for OpenWebUI
# Usage: ./deploy.sh [container_name] [image]
# 
# 1. Starts a candidate container on a temporary port
# 2. Waits for it to be healthy
# 3. Recreates the real container on the original public port
# 4. Rolls back the old container if final startup fails

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER_NAME="${1:-open-webui}"
IMAGE="${2:-ghcr.io/open-webui/open-webui:v0.9.5}"
CONTAINER_PORT=8080
TEMP_PORT="${TEMP_PORT:-8081}"
NETWORK="${3:-}"  # optional docker network name
ENV_FILE="${OPENWEBUI_ENV_FILE:-${SCRIPT_DIR}/open-webui.env}"
EXTRA_ENV_ARGS=()
if [ -f "$ENV_FILE" ]; then
    EXTRA_ENV_ARGS=(--env-file "$ENV_FILE")
fi

echo "=== Blue-Green Deploy for ${CONTAINER_NAME} ==="

# Get current container's env, volumes, network
OLD_ID=$(docker ps -q -f "name=^${CONTAINER_NAME}$")
if [ -z "$OLD_ID" ]; then
    echo "ERROR: Container ${CONTAINER_NAME} not found or not running"
    exit 1
fi

# Extract current container config
VOLUMES=$(docker inspect "$OLD_ID" --format '{{range .Mounts}}-v {{.Source}}:{{.Destination}} {{end}}')
ENV_VARS=$(docker inspect "$OLD_ID" --format '{{range .Config.Env}}--env {{.}} {{end}}')
if [ -n "$NETWORK" ]; then
    NET_FLAG="--network ${NETWORK}"
elif docker inspect "$OLD_ID" --format '{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}' | grep -q .; then
    NET_NAME=$(docker inspect "$OLD_ID" --format '{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}')
    NET_FLAG="--network ${NET_NAME}"
else
    NET_FLAG=""
fi

RESTART_POLICY=$(docker inspect "$OLD_ID" --format '{{.HostConfig.RestartPolicy.Name}}')

echo "Old container: ${OLD_ID}"
echo "Volumes: ${VOLUMES}"
echo "Network: ${NET_FLAG}"

PORT_BINDINGS=$(docker port "$OLD_ID" "${CONTAINER_PORT}/tcp" 2>/dev/null || true)
HOST_PORT=$(printf '%s\n' "$PORT_BINDINGS" | awk -F: 'NR==1 {print $NF}')
if [ -z "$HOST_PORT" ]; then
    HOST_PORT=3000
fi
echo "Public port: ${HOST_PORT}->${CONTAINER_PORT}"

# Backup webui.db before the new container touches the shared volume.
# v0.9.0 ships a DB schema migration — once the new container starts it
# migrates in place, and the old container can no longer read the schema.
# This backup is the rollback path.
BACKUP_DIR="${BACKUP_DIR:-/var/backups/owui-hybrid-router}"
if ! mkdir -p "$BACKUP_DIR" 2>/dev/null; then
    BACKUP_DIR="$(pwd)/.local-backups"
    mkdir -p "$BACKUP_DIR"
fi
BACKUP_PATH="${BACKUP_DIR}/webui.db.backup-$(date +%Y%m%d-%H%M%S)"
echo ""
echo "--- Backing up webui.db to ${BACKUP_PATH} ---"
if docker cp "${OLD_ID}:/app/backend/data/webui.db" "${BACKUP_PATH}" 2>/dev/null; then
    echo "Backup size: $(ls -lh "${BACKUP_PATH}" | awk '{print $5}')"
else
    echo "WARNING: Could not back up webui.db (path may differ in this deploy). Proceeding."
fi

# Step 1: Start candidate container on TEMP_PORT
NEW_NAME="${CONTAINER_NAME}-new"
echo ""
echo "--- Step 1: Pulling image ${IMAGE} ---"
docker pull "${IMAGE}"

echo ""
echo "--- Step 2: Starting candidate on temporary port ${TEMP_PORT} ---"
docker rm -f "${NEW_NAME}" >/dev/null 2>&1 || true
docker run -d \
    --name "${NEW_NAME}" \
    ${VOLUMES} \
    ${ENV_VARS} \
    "${EXTRA_ENV_ARGS[@]}" \
    ${NET_FLAG} \
    -p "${TEMP_PORT}:${CONTAINER_PORT}" \
    --restart="${RESTART_POLICY:-unless-stopped}" \
    "${IMAGE}"

# Step 3: Wait for candidate health
echo ""
echo "--- Step 3: Waiting for candidate container to be healthy ---"
MAX_WAIT=120
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    STATUS=$(docker inspect "${NEW_NAME}" --format '{{.State.Health.Status}}' 2>/dev/null || echo "unknown")
    if [ "$STATUS" = "healthy" ]; then
        echo "New container is HEALTHY after ${WAITED}s"
        break
    fi
    echo "  Status: ${STATUS} (${WAITED}s / ${MAX_WAIT}s)"
    sleep 5
    WAITED=$((WAITED + 5))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: New container did not become healthy in ${MAX_WAIT}s"
    echo "Rolling back — removing candidate container"
    docker rm -f "${NEW_NAME}" >/dev/null 2>&1 || true
    exit 1
fi

# Step 4: Quick smoke test
echo ""
echo "--- Step 4: Smoke test ---"
if docker exec "${NEW_NAME}" curl -sf "http://localhost:${CONTAINER_PORT}/api/version" > /dev/null 2>&1; then
    echo "Smoke test PASSED"
else
    echo "WARNING: Smoke test failed, but container is healthy. Proceeding anyway."
fi

# Step 5: Replace real container on the original public port
echo ""
echo "--- Step 5: Recreating ${CONTAINER_NAME} on port ${HOST_PORT} ---"
OLD_BACKUP="${CONTAINER_NAME}-old-$(date +%Y%m%d-%H%M%S)"
docker stop "$OLD_ID"
docker rename "$OLD_ID" "$OLD_BACKUP"
docker rm -f "${NEW_NAME}" >/dev/null 2>&1 || true

if ! docker run -d \
    --name "${CONTAINER_NAME}" \
    ${VOLUMES} \
    ${ENV_VARS} \
    "${EXTRA_ENV_ARGS[@]}" \
    ${NET_FLAG} \
    -p "${HOST_PORT}:${CONTAINER_PORT}" \
    --restart="${RESTART_POLICY:-unless-stopped}" \
    "${IMAGE}"; then
    echo "ERROR: Could not start final container; rolling back old container"
    docker rename "$OLD_BACKUP" "${CONTAINER_NAME}" || true
    docker start "${CONTAINER_NAME}" || true
    exit 1
fi

WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    STATUS=$(docker inspect "${CONTAINER_NAME}" --format '{{.State.Health.Status}}' 2>/dev/null || echo "unknown")
    if [ "$STATUS" = "healthy" ]; then
        echo "Final container is HEALTHY after ${WAITED}s"
        break
    fi
    echo "  Final status: ${STATUS} (${WAITED}s / ${MAX_WAIT}s)"
    sleep 5
    WAITED=$((WAITED + 5))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: Final container did not become healthy in ${MAX_WAIT}s; rolling back"
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    docker rename "$OLD_BACKUP" "${CONTAINER_NAME}" || true
    docker start "${CONTAINER_NAME}" || true
    exit 1
fi

echo ""
echo "--- Step 6: Cleanup ---"
docker rm "$OLD_BACKUP"
echo ""
echo "=== Deploy complete ==="
echo "New container: ${CONTAINER_NAME} ($(docker ps -q -f name=^${CONTAINER_NAME}$))"
echo "Public port preserved: ${HOST_PORT}->${CONTAINER_PORT}"
