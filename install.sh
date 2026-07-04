#!/usr/bin/env bash
#
# install.sh - Install and configure OCRHarbor GPU Worker.
#
# Usage:
#   bash install.sh [--port PORT]
#
# This script is idempotent. It preserves an existing .env, generates missing
# values, installs dependencies, and caches model weights under /workspace when
# available. Use restart-server.sh to start or restart the worker and tunnel.

set -euo pipefail

PORT_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            PORT_ARG="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash install.sh [--port PORT]"
            echo "Default port: 5001"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

info() { echo "[INFO] $*"; }
ok() { echo "[ OK ] $*"; }
warn() { echo "[WARN] $*"; }
fatal() { echo "[ERROR] $*" >&2; exit 1; }

quote_env() {
    local value="${1:-}"
    value="${value//\'/\'\\\'\'}"
    printf "'%s'" "$value"
}

write_env_line() {
    local key="$1"
    local value="${2:-}"
    printf "%s=" "$key"
    quote_env "$value"
    printf "\n"
}

generate_secret() {
    "$PYTHON" - <<'PY'
import secrets
print("ocrw_tok_" + secrets.token_urlsafe(32))
PY
}

if [[ -f .env ]]; then
    info "Loading existing .env"
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

PORT="${PORT_ARG:-${PORT:-5001}}"
OCR_MODEL="${OCR_MODEL:-lightonai/LightOnOCR-2-1B}"
MAX_QUEUE_SIZE="${MAX_QUEUE_SIZE:-500}"
JOB_TTL_SECONDS="${JOB_TTL_SECONDS:-3600}"
CALLBACK_URL="${CALLBACK_URL:-}"
CALLBACK_SECRET="${CALLBACK_SECRET:-}"
OCRHARBOR_TUNNEL_TOKEN="${OCRHARBOR_TUNNEL_TOKEN:-}"

if [[ -d /workspace && "$SCRIPT_DIR" != /workspace/* && "${OCRHARBOR_ASSUME_YES:-0}" != "1" ]]; then
    warn "Installing from $SCRIPT_DIR, but /workspace exists."
    warn "On Vast.ai, clone OCRHarborWorker under /workspace to use the large persistent volume."
    read -rp "Continue anyway? [y/N] " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        exit 1
    fi
fi

echo "=== OCRHarbor Worker Installation ==="
echo "  Port: $PORT"
echo ""

info "Step 1/6: Checking disk space"
if [[ -d /workspace ]]; then
    workspace_avail=$(df -BG /workspace 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G')
    echo "  /workspace: ${workspace_avail:-unknown}G available"
    if [[ "${workspace_avail:-0}" =~ ^[0-9]+$ ]] && [[ "$workspace_avail" -lt 8 ]]; then
        warn "Less than 8GB free on /workspace. Model weights need several GB."
        if [[ "${OCRHARBOR_ASSUME_YES:-0}" != "1" ]]; then
            read -rp "Continue anyway? [y/N] " confirm
            if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
                exit 1
            fi
        fi
    fi
else
    warn "/workspace not found; using local cache paths."
fi

info "Step 2/6: Configuring cache directories"
if [[ -n "${OCRHARBOR_CACHE_DIR:-}" ]]; then
    CACHE_DIR="$OCRHARBOR_CACHE_DIR"
elif [[ -d /workspace ]]; then
    CACHE_DIR="/workspace/.cache"
else
    CACHE_DIR="$SCRIPT_DIR/.cache"
fi

HF_HOME="${HF_HOME:-$CACHE_DIR/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_DIR/pip}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$PIP_CACHE_DIR"
export HF_HOME TRANSFORMERS_CACHE PIP_CACHE_DIR OCR_MODEL
echo "  HF_HOME=$HF_HOME"
echo "  PIP_CACHE_DIR=$PIP_CACHE_DIR"

info "Step 3/6: Detecting Python environment"
PYTHON=""
PIP=""

if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN:-}" ]]; then
    PYTHON="$PYTHON_BIN"
    PIP="${PIP_BIN:-$(dirname "$PYTHON")/pip}"
    echo "  Using configured Python: $PYTHON"
elif [[ -x /venv/main/bin/python && -x /venv/main/bin/pip ]]; then
    PYTHON="/venv/main/bin/python"
    PIP="/venv/main/bin/pip"
    echo "  Using Vast.ai venv: /venv/main"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" && -x "$VIRTUAL_ENV/bin/pip" ]]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
    PIP="$VIRTUAL_ENV/bin/pip"
    echo "  Using active venv: $VIRTUAL_ENV"
else
    PYTHON="$(command -v python3 || true)"
    [[ -n "$PYTHON" ]] || fatal "python3 not found"
    PIP="$(command -v pip3 || command -v pip || true)"
    if [[ -z "$PIP" ]]; then
        info "pip not found; installing python3-pip"
        apt-get update -qq && apt-get install -y -qq python3-pip
        PIP="$(command -v pip3 || command -v pip || true)"
    fi
    [[ -n "$PIP" ]] || fatal "pip not found after installation"
    echo "  Using system Python: $PYTHON"
fi

ok "Python: $("$PYTHON" --version)"

info "Step 4/6: Installing Python dependencies"
if ! "$PYTHON" -c "import ocrdoctotext" 2>/dev/null; then
    if [[ -d "$SCRIPT_DIR/ocrdoctotext_pkg" ]]; then
        "$PIP" install -q "$SCRIPT_DIR/ocrdoctotext_pkg/"
    elif [[ -d "/workspace/OCRDocToText" ]]; then
        "$PIP" install -q /workspace/OCRDocToText/
    else
        fatal "ocrdoctotext package not found at $SCRIPT_DIR/ocrdoctotext_pkg or /workspace/OCRDocToText"
    fi
else
    echo "  ocrdoctotext already installed"
fi
"$PIP" install -q -r requirements.txt
ok "Dependencies installed"

info "Step 5/6: Writing configuration"
WORKER_SECRET="${WORKER_SECRET:-}"
if [[ -z "$WORKER_SECRET" || "$WORKER_SECRET" == "change-this-to-a-random-string" ]]; then
    WORKER_SECRET="$(generate_secret)"
    ok "Generated WORKER_SECRET"
else
    ok "Using existing WORKER_SECRET"
fi

{
    write_env_line "PORT" "$PORT"
    write_env_line "WORKER_SECRET" "$WORKER_SECRET"
    write_env_line "OCRHARBOR_TUNNEL_TOKEN" "$OCRHARBOR_TUNNEL_TOKEN"
    write_env_line "CALLBACK_URL" "$CALLBACK_URL"
    write_env_line "CALLBACK_SECRET" "$CALLBACK_SECRET"
    write_env_line "OCR_MODEL" "$OCR_MODEL"
    write_env_line "HF_HOME" "$HF_HOME"
    write_env_line "TRANSFORMERS_CACHE" "$TRANSFORMERS_CACHE"
    write_env_line "PIP_CACHE_DIR" "$PIP_CACHE_DIR"
    write_env_line "MAX_QUEUE_SIZE" "$MAX_QUEUE_SIZE"
    write_env_line "JOB_TTL_SECONDS" "$JOB_TTL_SECONDS"
    write_env_line "PYTHON_BIN" "$PYTHON"
    write_env_line "PIP_BIN" "$PIP"
} > .env
chmod 600 .env

{
    printf "export HF_HOME=%q\n" "$HF_HOME"
    printf "export TRANSFORMERS_CACHE=%q\n" "$TRANSFORMERS_CACHE"
    printf "export PIP_CACHE_DIR=%q\n" "$PIP_CACHE_DIR"
} > .cache_env
ok ".env written"

info "Step 6/6: Ensuring model weights are cached"
if [[ "${OCRHARBOR_SKIP_MODEL_DOWNLOAD:-0}" == "1" ]]; then
    warn "Skipping model download because OCRHARBOR_SKIP_MODEL_DOWNLOAD=1"
else
    "$PYTHON" - <<'PY'
import os
from ocrdoctotext import OCREngine

model = os.environ.get("OCR_MODEL", "lightonai/LightOnOCR-2-1B")
engine = OCREngine(model)
engine.load()
print("  Model loaded successfully")
PY
fi

if [[ -n "$OCRHARBOR_TUNNEL_TOKEN" ]]; then
    if ! command -v cloudflared >/dev/null 2>&1; then
        info "Installing cloudflared"
        arch="$(uname -m)"
        case "$arch" in
            x86_64|amd64) cf_arch="amd64" ;;
            aarch64|arm64) cf_arch="arm64" ;;
            *) fatal "Unsupported architecture for cloudflared: $arch" ;;
        esac
        curl -sL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${cf_arch}" \
            -o /usr/local/bin/cloudflared
        chmod +x /usr/local/bin/cloudflared
    fi
    ok "cloudflared available"
else
    warn "No OCRHARBOR_TUNNEL_TOKEN configured; tunnel will be skipped on restart"
fi

echo ""
echo "========================================"
echo "  Installation complete"
echo "========================================"
echo ""
echo "  WORKER_SECRET: $WORKER_SECRET"
echo "  PORT:          $PORT"
echo "  PYTHON:        $PYTHON"
echo ""
echo "Start or restart the worker:"
echo "  cd $SCRIPT_DIR"
echo "  ./restart-server.sh"
echo ""
echo "Logs:"
echo "  Worker: tail -f $SCRIPT_DIR/worker.log"
echo "  Tunnel: tail -f $SCRIPT_DIR/tunnel.log"
