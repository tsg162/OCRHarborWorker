#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PORT=$(grep '^PORT=' .env 2>/dev/null | cut -d= -f2 || echo 5001)

echo "Killing anything on port $PORT..."
fuser -k "$PORT/tcp" 2>/dev/null || true
sleep 0.5

# --- Restart tunnel if token is configured ---
TUNNEL_TOKEN=$(grep '^OCRHARBOR_TUNNEL_TOKEN=' .env 2>/dev/null | cut -d= -f2 || true)
if [ -n "$TUNNEL_TOKEN" ]; then
    echo "Restarting Cloudflare tunnel..."
    pkill -f "cloudflared.*tunnel.*run" 2>/dev/null || true
    sleep 0.5
    nohup cloudflared tunnel run --token "$TUNNEL_TOKEN" > tunnel.log 2>&1 &
    TUNNEL_PID=$!
    echo "$TUNNEL_PID" > tunnel.pid
    echo "Tunnel PID: $TUNNEL_PID"
fi

LOG="$(pwd)/worker.log"

echo "Starting worker on port $PORT (logging to $LOG)..."
nohup python3 -m ocrharbor_worker.main > "$LOG" 2>&1 &
WORKER_PID=$!
echo "Worker PID: $WORKER_PID"

# Wait until the server reports ready (or bail after 120s)
echo "Waiting for server to be ready..."
for i in $(seq 1 120); do
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then
        echo "Worker process died. Last log lines:"
        tail -20 "$LOG"
        exit 1
    fi
    if grep -q "ready to accept jobs\|Uvicorn running on\|Application startup complete" "$LOG" 2>/dev/null; then
        echo "Server is up! (took ~${i}s)"
        tail -5 "$LOG"
        echo ""
        echo "Logs:   tail -f $LOG"
        echo "Stop:   kill $WORKER_PID"
        if [ -n "$TUNNEL_TOKEN" ]; then
            echo "Tunnel: tail -f $(pwd)/tunnel.log"
        fi
        exit 0
    fi
    sleep 1
done

echo "Timed out waiting for server. Last log lines:"
tail -20 "$LOG"
echo ""
echo "Worker may still be loading the model. Check: tail -f $LOG"
