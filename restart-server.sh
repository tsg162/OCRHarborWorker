#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

green() { echo -e "\033[32m$*\033[0m"; }
yellow() { echo -e "\033[33m$*\033[0m"; }
red() { echo -e "\033[31m$*\033[0m"; }

if [[ ! -f .env ]]; then
    red "Missing .env. Run ./install.sh first."
    exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

PORT="${PORT:-5001}"
PYTHON="${PYTHON_BIN:-python3}"
PIP="${PIP_BIN:-pip}"
TUNNEL_TOKEN="${OCRHARBOR_TUNNEL_TOKEN:-}"
LOG="$SCRIPT_DIR/worker.log"
TUNNEL_LOG="$SCRIPT_DIR/tunnel.log"
WORKER_PID_FILE="$SCRIPT_DIR/worker.pid"
TUNNEL_PID_FILE="$SCRIPT_DIR/tunnel.pid"
REPO_DIR="${OCRHARBOR_WORKER_REPO:-$SCRIPT_DIR}"
UPDATE=0

if [[ "${1:-}" == "--update" || "${OCRHARBOR_AUTO_UPDATE_ON_RESTART:-0}" == "1" ]]; then
    UPDATE=1
fi

echo "=== Restarting OCRHarbor Worker ==="
echo ""

if [[ "$UPDATE" == "1" ]]; then
    echo "[0/6] Updating worker repo..."
    if [[ ! -d "$REPO_DIR/.git" ]]; then
        red "Cannot update: git repo not found at $REPO_DIR"
        exit 1
    fi
    git -C "$REPO_DIR" pull --ff-only
    "$PIP" install -q "$REPO_DIR/ocrdoctotext_pkg"
    "$PIP" install -q -r "$REPO_DIR/requirements.txt"
    green "  Worker code updated"
    echo ""
fi

echo "[1/5] Stopping old worker..."
if [[ -f "$WORKER_PID_FILE" ]]; then
    OLD_PID="$(cat "$WORKER_PID_FILE")"
    if kill -0 "$OLD_PID" 2>/dev/null; then
        kill "$OLD_PID" 2>/dev/null || true
        for _ in $(seq 1 15); do
            kill -0 "$OLD_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$OLD_PID" 2>/dev/null; then
            yellow "  Worker did not stop gracefully; force killing"
            kill -9 "$OLD_PID" 2>/dev/null || true
            sleep 1
        fi
        echo "  Stopped PID $OLD_PID"
    fi
fi

if command -v fuser >/dev/null 2>&1 && fuser "$PORT/tcp" >/dev/null 2>&1; then
    fuser -k "$PORT/tcp" >/dev/null 2>&1 || true
    echo "  Freed port $PORT"
else
    pkill -f "python.*ocrharbor_worker.main" 2>/dev/null || true
fi
green "  Worker process stopped"

echo ""
echo "[2/5] Restarting Cloudflare tunnel..."
if [[ -z "$TUNNEL_TOKEN" ]]; then
    yellow "  No OCRHARBOR_TUNNEL_TOKEN in .env; skipping tunnel"
else
    if ! command -v cloudflared >/dev/null 2>&1; then
        red "  cloudflared is not installed. Run ./install.sh again."
        exit 1
    fi

    if [[ -f "$TUNNEL_PID_FILE" ]]; then
        OLD_TUNNEL_PID="$(cat "$TUNNEL_PID_FILE")"
        if kill -0 "$OLD_TUNNEL_PID" 2>/dev/null; then
            kill "$OLD_TUNNEL_PID" 2>/dev/null || true
            sleep 2
            kill -9 "$OLD_TUNNEL_PID" 2>/dev/null || true
        fi
    fi

    nohup cloudflared tunnel run --token "$TUNNEL_TOKEN" > "$TUNNEL_LOG" 2>&1 &
    TUNNEL_PID=$!
    echo "$TUNNEL_PID" > "$TUNNEL_PID_FILE"
    echo "  Tunnel PID: $TUNNEL_PID"

    TUNNEL_READY=false
    for i in $(seq 1 30); do
        if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
            red "  Tunnel process died. Last log lines:"
            tail -10 "$TUNNEL_LOG" 2>/dev/null || true
            exit 1
        fi
        if grep -q "Registered tunnel connection\|Connection.*registered\|INF " "$TUNNEL_LOG" 2>/dev/null; then
            TUNNEL_READY=true
            break
        fi
        sleep 1
    done

    if [[ "$TUNNEL_READY" == "true" ]]; then
        green "  Tunnel connected"
    else
        yellow "  Tunnel may still be connecting; continuing"
    fi
fi

echo ""
echo "[3/5] Starting worker on port $PORT..."
nohup "$PYTHON" -m ocrharbor_worker.main > "$LOG" 2>&1 &
WORKER_PID=$!
echo "$WORKER_PID" > "$WORKER_PID_FILE"
echo "  Worker PID: $WORKER_PID"

echo ""
echo "[4/5] Waiting for worker health..."
HEALTH_OK=false
for i in $(seq 1 180); do
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then
        red "  Worker process died after ${i}s. Last log lines:"
        tail -30 "$LOG" 2>/dev/null || true
        exit 1
    fi

    HTTP_CODE="$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/health" 2>/dev/null || echo "000")"
    if [[ "$HTTP_CODE" == "200" ]]; then
        HEALTH_OK=true
        break
    fi

    sleep 1
done

if [[ "$HEALTH_OK" != "true" ]]; then
    red "  Worker did not become healthy after 180s. Last log lines:"
    tail -30 "$LOG" 2>/dev/null || true
    exit 1
fi
green "  Local health check passed"

echo ""
echo "[5/5] Summary"
echo "  Worker PID:  $WORKER_PID"
echo "  Worker log:  tail -f $LOG"
if [[ -n "$TUNNEL_TOKEN" ]]; then
    echo "  Tunnel PID:  ${TUNNEL_PID:-unknown}"
    echo "  Tunnel log:  tail -f $TUNNEL_LOG"
fi
echo ""
green "Restart complete"
