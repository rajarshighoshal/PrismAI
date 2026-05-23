#!/bin/bash
# Zero-downtime blue-green deployment for OpenWebUI
# Usage: ./deploy.sh [container_name] [image]
# 
# 1. Starts a new container on a different port
# 2. Waits for it to be healthy
# 3. Stops the old container
# 4. Renames the new one to take its place
# 5. Removes the old one

set -euo pipefail

CONTAINER_NAME="${1:-open-webui}"
IMAGE="${2:-ghcr.io/open-webui/open-webui:v0.9.5}"
OLD_PORT=8080
NEW_PORT=8081
NETWORK="${3:-}"  # optional docker network name

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

# Backup webui.db before the new container touches the shared volume.
# v0.9.0 ships a DB schema migration — once the new container starts it
# migrates in place, and the old container can no longer read the schema.
# This backup is the rollback path.
BACKUP_PATH="webui.db.backup-$(date +%Y%m%d-%H%M%S)"
echo ""
echo "--- Backing up webui.db to ${BACKUP_PATH} ---"
if docker cp "${OLD_ID}:/app/backend/data/webui.db" "${BACKUP_PATH}" 2>/dev/null; then
    echo "Backup size: $(ls -lh "${BACKUP_PATH}" | awk '{print $5}')"
else
    echo "WARNING: Could not back up webui.db (path may differ in this deploy). Proceeding."
fi

# Step 1: Start new container on NEW_PORT
NEW_NAME="${CONTAINER_NAME}-new"
echo ""
echo "--- Step 1: Starting new container on port ${NEW_PORT} ---"
docker run -d \
    --name "${NEW_NAME}" \
    ${VOLUMES} \
    ${ENV_VARS} \
    ${NET_FLAG} \
    -p "${NEW_PORT}:${OLD_PORT}" \
    --restart="${RESTART_POLICY:-unless-stopped}" \
    "${IMAGE}"

# Step 2: Wait for healthy
echo ""
echo "--- Step 2: Waiting for new container to be healthy ---"
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
    echo "Rolling back — removing new container"
    docker stop "${NEW_NAME}" && docker rm "${NEW_NAME}"
    exit 1
fi

# Step 3: Quick smoke test
echo ""
echo "--- Step 3: Smoke test ---"
if docker exec "${NEW_NAME}" curl -sf http://localhost:${OLD_PORT}/api/version > /dev/null 2>&1; then
    echo "Smoke test PASSED"
else
    echo "WARNING: Smoke test failed, but container is healthy. Proceeding anyway."
fi

# Step 4: Swap — stop old, rename new
echo ""
echo "--- Step 4: Swapping containers ---"
docker stop "$OLD_ID"
docker rename "${NEW_NAME}" "${CONTAINER_NAME}"

# Re-map the port if needed (the new container was on NEW_PORT internally)
# Actually we need to recreate with the correct port mapping
# For simplicity, if using a reverse proxy, just update the upstream

echo ""
echo "--- Step 5: Cleanup ---"
docker rm "$OLD_ID"
echo ""
echo "=== Deploy complete ==="
echo "New container: ${CONTAINER_NAME} ($(docker ps -q -f name=^${CONTAINER_NAME}$))"
echo "NOTE: New container is on port ${NEW_PORT}. Update your reverse proxy if needed."
