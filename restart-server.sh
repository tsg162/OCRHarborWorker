#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PORT=$(grep '^PORT=' .env 2>/dev/null | cut -d= -f2 || echo 8080)

echo "Killing anything on port $PORT..."
fuser -k "$PORT/tcp" 2>/dev/null || true
sleep 0.5

echo "Starting worker on port $PORT..."
python3 -m ocrharbor_worker.main
