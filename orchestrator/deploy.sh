#!/bin/bash
# Build and run the OWUI orchestrator container.
# Joins the same docker network as open-webui so OWUI can reach it by name
# (http://owui-orchestrator:8002). Mirrors tool-server/deploy.sh.
set -euo pipefail

ORCH_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="owui-orchestrator:latest"
CONTAINER="owui-orchestrator"
BACKUP_IMAGE="owui-orchestrator:rollback"
HOST_PORT=8002
ENV_FILE="${ORCH_DIR}/orchestrator.env"
ENV_ARGS=()
if [ -f "$ENV_FILE" ]; then
    ENV_ARGS=(--env-file "$ENV_FILE")
fi

# Discover the network the existing open-webui container lives on.
NETWORK=$(docker inspect open-webui \
    --format '{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{"\n"}}{{end}}' \
    2>/dev/null | head -n 1)
if [ -z "$NETWORK" ]; then
    echo "ERROR: cannot find open-webui container or its network. Is OWUI running?"
    exit 1
fi
echo "Joining network: $NETWORK"

# Bind-mount OWUI's data dir read-only so the orchestrator can read per-user
# style profiles from webui.db. Optional: if not found, style memory just no-ops.
DATA_SRC=$(docker inspect open-webui \
    --format '{{range .Mounts}}{{if eq .Destination "/app/backend/data"}}{{.Source}}{{end}}{{end}}' \
    2>/dev/null || true)
MOUNT_ARGS=()
if [ -n "$DATA_SRC" ]; then
    echo "Mounting style db (read-only): $DATA_SRC -> /app/backend/data"
    MOUNT_ARGS=(-v "${DATA_SRC}:/app/backend/data:ro")
else
    echo "NOTE: open-webui data mount not found; style memory will no-op."
fi

if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "--- Saving rollback image ---"
    docker tag "$IMAGE" "$BACKUP_IMAGE"
fi

echo "--- Building image ---"
docker build -t "$IMAGE" "$ORCH_DIR"

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "--- Stopping + removing old container ---"
    docker stop "$CONTAINER" >/dev/null
    docker rm "$CONTAINER" >/dev/null
fi

echo "--- Running new container ---"
if ! docker run -d \
        --name "$CONTAINER" \
        --network "$NETWORK" \
        --restart unless-stopped \
        "${ENV_ARGS[@]}" \
        "${MOUNT_ARGS[@]}" \
        -p "127.0.0.1:${HOST_PORT}:8002" \
        "$IMAGE"; then
    echo "ERROR: failed to start new orchestrator container"
    if docker image inspect "$BACKUP_IMAGE" >/dev/null 2>&1; then
        docker run -d \
            --name "$CONTAINER" \
            --network "$NETWORK" \
            --restart unless-stopped \
            "${ENV_ARGS[@]}" \
            "${MOUNT_ARGS[@]}" \
            -p "127.0.0.1:${HOST_PORT}:8002" \
            "$BACKUP_IMAGE" >/dev/null
    fi
    exit 1
fi

echo "--- Waiting for /health (max 60s) ---"
for i in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:${HOST_PORT}/health" >/dev/null 2>&1; then
        echo "OK after ${i} attempt(s)"
        curl -s "http://127.0.0.1:${HOST_PORT}/health"
        echo
        echo
        echo "Reachable from OWUI as: http://${CONTAINER}:8002/v1"
        exit 0
    fi
    sleep 2
done

echo "FAILED to become healthy. Last logs:"
docker logs --tail 50 "$CONTAINER"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
if docker image inspect "$BACKUP_IMAGE" >/dev/null 2>&1; then
    echo "--- Rolling back to previous orchestrator image ---"
    docker run -d \
        --name "$CONTAINER" \
        --network "$NETWORK" \
        --restart unless-stopped \
        "${ENV_ARGS[@]}" \
        "${MOUNT_ARGS[@]}" \
        -p "127.0.0.1:${HOST_PORT}:8002" \
        "$BACKUP_IMAGE" >/dev/null
fi
exit 1
