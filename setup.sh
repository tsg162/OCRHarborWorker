#!/usr/bin/env bash
# setup.sh - One-command OCRHarbor worker bootstrap.
#
# Called by `ocrharbor deploy`:
#   setup.sh --deploy <tunnel-token> <worker-secret> <server-name> [--port PORT]
#
# Manual mode:
#   setup.sh <tunnel-token> [server-name] [--port PORT]

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  setup.sh --deploy <tunnel-token> <worker-secret> <server-name> [--port PORT]
  setup.sh <tunnel-token> [server-name] [--port PORT]

Examples:
  ocrharbor deploy gpu1
  bash <(curl -sL https://raw.githubusercontent.com/tsg162/OCRHarborWorker/main/setup.sh) --deploy eyJ... ocrw_tok_... gpu1
  ./setup.sh eyJ... gpu1
USAGE
}

DEPLOY=0
PORT="${OCRHARBOR_PORT:-5001}"
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --deploy)
            DEPLOY=1
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [[ "${#POSITIONAL[@]}" -lt 1 ]]; then
    usage
    exit 1
fi

TUNNEL_TOKEN="${POSITIONAL[0]}"
WORKER_SECRET=""
SERVER_NAME="$(hostname -s 2>/dev/null || echo worker)"

if [[ "$DEPLOY" == "1" ]]; then
    if [[ "${#POSITIONAL[@]}" -lt 2 ]]; then
        echo "Deploy mode requires <tunnel-token> and <worker-secret>." >&2
        usage
        exit 1
    fi
    WORKER_SECRET="${POSITIONAL[1]}"
    SERVER_NAME="${POSITIONAL[2]:-$SERVER_NAME}"
elif [[ "${POSITIONAL[1]:-}" == ocrw_tok_* ]]; then
    DEPLOY=1
    WORKER_SECRET="${POSITIONAL[1]}"
    SERVER_NAME="${POSITIONAL[2]:-$SERVER_NAME}"
else
    SERVER_NAME="${POSITIONAL[1]:-$SERVER_NAME}"
fi

REPO_URL="${OCRHARBOR_WORKER_REPO_URL:-https://github.com/tsg162/OCRHarborWorker.git}"
TUNNEL_SUFFIX="-ocr"
DOMAIN="${OCRHARBOR_DOMAIN:-gpuharbor.xyz}"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

header() { echo -e "\n${BOLD}${CYAN}> $*${NC}"; }

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

if ! command -v git >/dev/null 2>&1; then
    header "Installing git"
    apt-get update -qq && apt-get install -y -qq git
fi

REPO_DIR=""
if [[ -f "./install.sh" && -d "./ocrharbor_worker" ]]; then
    REPO_DIR="$(pwd)"
    header "Using repo in current directory"
elif [[ -f "/workspace/OCRHarborWorker/install.sh" ]]; then
    REPO_DIR="/workspace/OCRHarborWorker"
    header "Updating existing repo"
    git -C "$REPO_DIR" pull --ff-only
else
    if [[ -d "/workspace" ]]; then
        REPO_DIR="/workspace/OCRHarborWorker"
    else
        REPO_DIR="${HOME}/OCRHarborWorker"
    fi
    header "Cloning OCRHarborWorker to ${REPO_DIR}"
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

header "Writing .env"
{
    write_env_line "PORT" "$PORT"
    write_env_line "OCRHARBOR_TUNNEL_TOKEN" "$TUNNEL_TOKEN"
    if [[ -n "$WORKER_SECRET" ]]; then
        write_env_line "WORKER_SECRET" "$WORKER_SECRET"
    fi
    write_env_line "HF_HOME" "${HF_HOME:-/workspace/.cache/huggingface}"
    write_env_line "MAX_QUEUE_SIZE" "${MAX_QUEUE_SIZE:-500}"
    write_env_line "JOB_TTL_SECONDS" "${JOB_TTL_SECONDS:-3600}"
} > .env
chmod 600 .env
echo -e "${DIM}  .env written${NC}"

header "Installing worker"
chmod +x install.sh restart-server.sh
OCRHARBOR_ASSUME_YES=1 ./install.sh --port "$PORT"

header "Starting worker"
./restart-server.sh

if [[ "$DEPLOY" == "1" ]]; then
    echo ""
    echo -e "${BOLD}${GREEN}Done.${NC} Worker is running."
    echo -e "${DIM}The control node already has this worker registered.${NC}"
    echo -e "${DIM}Verify with: ocrharbor servers health ${SERVER_NAME}${NC}"
    echo ""
else
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    BASE_NAME="${SERVER_NAME%$TUNNEL_SUFFIX}"
    TUNNEL_URL="https://${BASE_NAME}${TUNNEL_SUFFIX}.${DOMAIN}"
    echo ""
    echo -e "${BOLD}${GREEN}Run this on your control node:${NC}"
    echo ""
    echo -e "  ${CYAN}ocrharbor servers add ${BASE_NAME} ${TUNNEL_URL} --key ${WORKER_SECRET}${NC}"
    echo ""
fi
