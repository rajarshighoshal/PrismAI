#!/bin/bash
# Pull latest code and update the OpenWebUI function in the DB.
# No container restart needed — OWUI hot-loads functions on each request.
set -euo pipefail

REPO_DIR="/opt/owui-hybrid-router"
DB_PATH="/app/backend/data/webui.db"
CONTAINER="open-webui"
FUNCTION_ID="vector_router_interceptor"

echo "=== Updating OWUI Router Function ==="

# Step 1: Pull latest
echo "--- Pulling latest from git ---"
cd "$REPO_DIR"
git pull origin main

# Step 2: Copy the function file into the container
echo "--- Copying router_fn.py into container ---"
docker cp router_fn.py "${CONTAINER}:/tmp/router_fn.py"

# Step 3: Update the function in the OWUI database
echo "--- Updating function in database ---"
docker exec "$CONTAINER" python3 -c "
import sqlite3
with open('/tmp/router_fn.py', 'r') as f:
    content = f.read()
conn = sqlite3.connect('${DB_PATH}')
cur = conn.cursor()
cur.execute('UPDATE function SET content = ? WHERE id = ?', (content, '${FUNCTION_ID}'))
conn.commit()
print(f'Updated {cur.rowcount} row(s), {len(content)} chars')
conn.close()
"

if [ -x tool-server/deploy.sh ]; then
    echo "--- Rebuilding/restarting tool server ---"
    ./tool-server/deploy.sh
fi

# The orchestrator is a CONTAINER (not a hot-loaded OWUI function like router_fn.py), so a
# code change there needs a rebuild — otherwise auto-deploy silently ships only the router
# shim + tool-server while the orchestrator runs stale code. Built AFTER the tool-server
# since it calls it.
if [ -x orchestrator/deploy.sh ]; then
    echo "--- Rebuilding/restarting orchestrator ---"
    ./orchestrator/deploy.sh
fi

if [ -x owui-patches/apply.sh ]; then
    echo "--- Re-applying OWUI patches ---"
    ./owui-patches/apply.sh
fi

echo ""
echo "=== Update complete. Function hot-loaded; tool server, orchestrator, and patches synced. ==="
