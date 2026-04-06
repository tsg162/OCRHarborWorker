#!/usr/bin/env bash
#
# install.sh — Install and configure OCR GPU Worker on a remote instance.
#
# Usage:
#   bash install.sh [--port PORT]
#
# Default port: 5001

set -euo pipefail

PORT=5001

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash install.sh [--port PORT]"
            echo "Default port: 5001"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- Sanity check: warn if not under /workspace ---
if [ -d /workspace ] && [[ "$SCRIPT_DIR" != /workspace/* ]]; then
    echo "WARNING: You're installing from $SCRIPT_DIR"
    echo "         but /workspace exists. On VAST.ai the worker should live"
    echo "         under /workspace/ so it uses the large persistent volume"
    echo "         (the root disk is usually small and will fill up)."
    echo ""
    echo "         Expected: /workspace/OCRServer/worker-remote/"
    echo "         Got:      $SCRIPT_DIR"
    echo ""
    read -rp "Continue anyway? [y/N] " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo ""
        echo "To fix this:"
        echo "  cd /workspace"
        echo "  git clone <repo> OCRServer"
        echo "  cd OCRServer/worker-remote"
        echo "  bash install.sh"
        exit 1
    fi
fi

echo "=== OCR GPU Worker Installation ==="
echo "  Port: $PORT"
echo ""

# --- 1. Configure cache directories ---
echo "[1/5] Configuring cache directories..."
if [ -d /workspace ]; then
    CACHE_DIR="/workspace/.cache"
    echo "  Using /workspace (VAST.ai detected)"
else
    CACHE_DIR="$SCRIPT_DIR/.cache"
    echo "  Using local cache: $CACHE_DIR"
fi
export HF_HOME="$CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$PIP_CACHE_DIR"

# --- 2. Detect Python environment ---
echo "[2/5] Detecting Python environment..."
PIP="pip"
PYTHON="python3"

if [ -x /venv/main/bin/pip ]; then
    PIP="/venv/main/bin/pip"
    PYTHON="/venv/main/bin/python"
    echo "  Using VAST.ai venv: /venv/main/"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/pip" ]; then
    PIP="$VIRTUAL_ENV/bin/pip"
    PYTHON="$VIRTUAL_ENV/bin/python"
    echo "  Using active venv: $VIRTUAL_ENV"
else
    echo "  Using system Python"
    if ! command -v pip &>/dev/null; then
        apt-get update -qq && apt-get install -y -qq python3-pip
    fi
fi

# --- 3. Install dependencies ---
echo "[3/5] Installing dependencies..."
if ! $PYTHON -c "import ocrdoctotext" 2>/dev/null; then
    if [ -d "$SCRIPT_DIR/ocrdoctotext_pkg" ]; then
        $PIP install -q "$SCRIPT_DIR/ocrdoctotext_pkg/"
    elif [ -d "/workspace/OCRDocToText" ]; then
        $PIP install -q /workspace/OCRDocToText/
    else
        echo "  ERROR: ocrdoctotext package not found."
        echo "  Copy OCRDocToText to $SCRIPT_DIR/ocrdoctotext_pkg/ or /workspace/OCRDocToText/"
        exit 1
    fi
else
    echo "  ocrdoctotext already installed"
fi
$PIP install -q -r requirements.txt
echo "  Done"

# --- 4. Generate configuration ---
echo "[4/5] Generating configuration..."
SECRET=$($PYTHON -c "import secrets; print(secrets.token_urlsafe(32))")
cat > .env <<EOF
PORT=$PORT
WORKER_SECRET=$SECRET
HF_HOME=$HF_HOME
MAX_QUEUE_SIZE=500
JOB_TTL_SECONDS=3600
EOF
echo "  .env written"

# --- 5. Download model weights ---
echo "[5/5] Downloading model weights (may take a few minutes on first run)..."
$PYTHON -c "
import os
os.environ['HF_HOME'] = '$HF_HOME'
os.environ['TRANSFORMERS_CACHE'] = '$TRANSFORMERS_CACHE'
from ocrdoctotext import OCREngine
engine = OCREngine('lightonai/LightOnOCR-2-1B')
engine.load()
print('  Model loaded successfully')
"

# --- 6. Start tunnel if token is set ---
TUNNEL_TOKEN=$(grep '^OCRHARBOR_TUNNEL_TOKEN=' .env 2>/dev/null | cut -d= -f2 || true)
if [ -z "$TUNNEL_TOKEN" ]; then
    TUNNEL_TOKEN="${OCRHARBOR_TUNNEL_TOKEN:-}"
fi

if [ -n "$TUNNEL_TOKEN" ]; then
    echo "[6/6] Starting Cloudflare tunnel..."

    # Install cloudflared if missing
    if ! command -v cloudflared &>/dev/null; then
        echo "  Installing cloudflared..."
        curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
            -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared
    fi

    # Kill any existing tunnel
    pkill -f "cloudflared.*tunnel.*run" 2>/dev/null || true
    sleep 0.5

    nohup cloudflared tunnel run --token "$TUNNEL_TOKEN" > "$SCRIPT_DIR/tunnel.log" 2>&1 &
    TUNNEL_PID=$!
    echo "$TUNNEL_PID" > "$SCRIPT_DIR/tunnel.pid"

    # Wait briefly for tunnel to connect
    sleep 3
    if kill -0 "$TUNNEL_PID" 2>/dev/null; then
        echo "  Tunnel started (PID: $TUNNEL_PID)"
    else
        echo "  WARNING: Tunnel process died. Check tunnel.log"
    fi
else
    echo ""
    echo "  [SKIP] No OCRHARBOR_TUNNEL_TOKEN found — tunnel not started."
    echo "  To enable: add OCRHARBOR_TUNNEL_TOKEN=<token> to .env and re-run install.sh"
fi

echo ""
echo "========================================"
echo "  Installation complete!"
echo "========================================"
echo ""
echo "  WORKER_SECRET: $SECRET"
echo "  PORT:          $PORT"
if [ -n "$TUNNEL_TOKEN" ]; then
echo "  TUNNEL:        running (PID: ${TUNNEL_PID:-?})"
fi
echo ""
echo "Start the worker:"
echo "  cd $SCRIPT_DIR"
echo "  ./restart-server.sh"
echo ""
echo "Then on your control node, add this worker:"
echo "  ocrharbor servers add <name> <tunnel-url> --key $SECRET"
echo ""
echo "Logs:"
echo "  Worker: tail -f $SCRIPT_DIR/worker.log"
if [ -n "$TUNNEL_TOKEN" ]; then
echo "  Tunnel: tail -f $SCRIPT_DIR/tunnel.log"
fi
