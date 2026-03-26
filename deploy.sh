#!/usr/bin/env bash
#
# deploy.sh — Bootstrap the OCR GPU Worker on a fresh VAST.ai PyTorch instance.
#
# Usage:
#   1. SSH into the VAST.ai instance
#   2. Copy or clone the worker/ directory
#   3. Run: bash deploy.sh
#
# Prerequisites:
#   - VAST.ai instance using the "PyTorch (Vast)" template
#   - Port 8000 exposed in the instance config

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== OCR GPU Worker Deployment ==="
echo ""

# --- 1. Install system deps if needed ---
echo "[1/5] Checking system dependencies..."
if ! command -v pip &>/dev/null; then
    apt-get update && apt-get install -y python3-pip
fi

# --- 2. Install ocrdoctotext ---
echo "[2/5] Installing ocrdoctotext..."
if python3 -c "import ocrdoctotext" 2>/dev/null; then
    echo "  ocrdoctotext already installed"
else
    if [ -d "$SCRIPT_DIR/ocrdoctotext_pkg" ]; then
        pip install "$SCRIPT_DIR/ocrdoctotext_pkg/"
    elif [ -d "/workspace/OCRDocToText" ]; then
        pip install /workspace/OCRDocToText/
    else
        echo "  ERROR: ocrdoctotext not found."
        echo "  Copy the OCRDocToText project to $SCRIPT_DIR/ocrdoctotext_pkg/ or /workspace/OCRDocToText/"
        exit 1
    fi
fi

# --- 3. Install Python deps ---
echo "[3/5] Installing Python dependencies..."
pip install -r requirements.txt

# --- 4. Set up .env if missing ---
echo "[4/5] Checking .env configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  Created .env from .env.example — edit it now:"
    echo "    WORKER_SECRET  (required — shared secret for auth)"
    echo "    CALLBACK_URL   (control node webhook URL, optional)"
    echo "    CALLBACK_SECRET (key for webhook auth, optional)"
    echo ""
    read -rp "  Enter WORKER_SECRET (or press Enter to generate one): " secret
    if [ -z "$secret" ]; then
        secret=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
        echo "  Generated secret: $secret"
    fi
    sed -i "s|^WORKER_SECRET=.*|WORKER_SECRET=$secret|" .env

    read -rp "  Enter CALLBACK_URL (or press Enter to skip): " callback_url
    if [ -n "$callback_url" ]; then
        sed -i "s|^CALLBACK_URL=.*|CALLBACK_URL=$callback_url|" .env
    fi

    read -rp "  Enter CALLBACK_SECRET (or press Enter to skip): " callback_secret
    if [ -n "$callback_secret" ]; then
        sed -i "s|^CALLBACK_SECRET=.*|CALLBACK_SECRET=$callback_secret|" .env
    fi
fi

# --- 5. Pre-download model weights ---
echo "[5/5] Ensuring model weights are cached..."
python3 -c "
from ocrdoctotext import OCREngine
print('  Loading model (this may take a few minutes on first run)...')
engine = OCREngine('lightonai/LightOnOCR-2-1B')
engine.load()
print('  Model loaded successfully')
"

echo ""
echo "=== Deployment complete ==="
echo ""
echo "Start the worker with:"
echo "  cd $SCRIPT_DIR && python3 -m gpu_worker.main"
echo ""
echo "Or run in the background:"
echo "  cd $SCRIPT_DIR && nohup python3 -m gpu_worker.main > worker.log 2>&1 &"
echo ""
echo "Test health:"
echo "  curl http://localhost:8000/health"
