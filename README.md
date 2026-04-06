# OCRHarbor Worker

GPU OCR worker that runs on remote instances (Vast.ai, Runpod, bare metal) to process images with LightOnOCR-2-1B. Part of the [OCRHarbor](https://github.com/tsg162/OCRHarbor) ecosystem.

## Quick Start

### 1. Create a tunnel (one-time, from your laptop)

```bash
ocrharbor tunnels create gpu1
# Outputs: OCRHARBOR_TUNNEL_TOKEN=eyJ...
```

This creates a permanent URL `gpu1-ocr.gpuharbor.xyz` that survives instance restarts. See the [OCRHarbor README](https://github.com/tsg162/OCRHarbor#tunnel-management) for one-time Cloudflare setup.

### 2. Set up the worker (on the GPU instance)

```bash
cd /workspace
git clone https://github.com/tsg162/OCRHarborWorker.git
cd OCRHarborWorker

# Add tunnel token to .env
echo "OCRHARBOR_TUNNEL_TOKEN=eyJ...paste-token-here..." > .env

# Install
bash install.sh

# Start the worker (runs in background, logs to worker.log)
bash restart-server.sh
```

The install script will:
1. Install Python dependencies and the bundled OCR engine
2. Generate a `WORKER_SECRET` for API auth
3. Install `cloudflared` and connect the named tunnel (if token is set)

### 3. Register with Control Node (from your laptop)

```bash
ocrharbor servers add gpu1 https://gpu1-ocr.gpuharbor.xyz --key SECRET_FROM_INSTALL
```

### Reusing tunnels

When you destroy a Vast.ai instance and create a new one, just use the same `OCRHARBOR_TUNNEL_TOKEN` in `.env`. The new instance connects to the same tunnel automatically — no need to re-create the tunnel or update your config. `gpu1-ocr.gpuharbor.xyz` just points to the new machine.

## Configuration

Set these in a `.env` file before running `install.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OCRHARBOR_TUNNEL_TOKEN` | _(none)_ | Cloudflare tunnel token (from `ocrharbor tunnels create`) |
| `WORKER_SECRET` | _(generated)_ | Shared secret for API auth |
| `PORT` | `5001` | Listen port |
| `HF_HOME` | `/workspace/.cache/huggingface` | HuggingFace cache dir |
| `MAX_QUEUE_SIZE` | `500` | Max queued jobs |

## API

All endpoints (except `/health`) require `Authorization: Bearer <WORKER_SECRET>`.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/jobs` | Submit OCR job (multipart: file + form fields). Returns 202. |
| GET | `/jobs/{id}` | Get job status and result |
| DELETE | `/jobs/{id}` | Cancel a queued job |
| GET | `/jobs` | List all active jobs |
| GET | `/health` | GPU status, model loaded, queue depth |

## What's Included

```
ocrdoctotext_pkg/        # Bundled OCR engine library (pip-installable)
ocrharbor_worker/        # FastAPI worker application
  main.py                # App entry point, endpoints, lifespan
  config.py              # pydantic-settings
  auth.py                # Bearer token verification
  models.py              # Pydantic response schemas
  job_manager.py         # In-memory job queue + GPU runner
  ocr_bridge.py          # OCREngine singleton
  webhook.py             # Callback sender (optional)
install.sh               # One-command setup script
restart-server.sh        # Start/restart worker in background
requirements.txt         # Python dependencies
```

## Shared Cloudflare credentials

Both `gpuharbor` and `ocrharbor` share credentials stored at `~/.gpuharbor/cloudflare.yaml`. Run `gpuharbor tunnels init` once to set up for both.
