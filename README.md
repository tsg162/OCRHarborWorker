# OCR GPU Worker

Remote GPU OCR worker for the OCRServer system. Runs on a VAST.ai GPU instance, accepts image jobs via API, runs LightOnOCR-2-1B inference, and reports results back to the control node via polling.

## Quick Start on VAST.ai

### 1. Create Instance

- **Template: PyTorch (Vast)** — CUDA + PyTorch pre-installed
- GPU: 4+ GB VRAM (1B model in bfloat16 is ~2GB)

### 2. Install

```bash
ssh root@<vast-ip>
cd /workspace
git clone https://github.com/tsg162/OCRServer.git
cd OCRServer/worker-remote
bash install.sh                # default port 5001
bash install.sh --port 8080    # or pick a port
```

### 3. Start

```bash
python3 -m ocrharbor_worker.main
```

### 4. Register with Control Node

```bash
# On your local machine:
ocrharbor workers add gpu1 <tunnel-url> --key <secret-from-install>
```

### 5. Stop / Uninstall

```bash
pkill -f 'ocrharbor_worker.main'
```

## API

All endpoints (except `/health`) require `Authorization: Bearer <WORKER_SECRET>`.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/jobs` | Submit OCR job (multipart: file + form fields). Returns 202. |
| GET | `/jobs/{id}` | Get job status and result |
| DELETE | `/jobs/{id}` | Cancel a queued job |
| GET | `/jobs` | List all active jobs |
| GET | `/health` | GPU status, model loaded, queue depth |

## Configuration

Edit `.env` (created by install.sh):

| Variable | Required | Description |
|----------|----------|-------------|
| `WORKER_SECRET` | Yes | Shared secret for API auth |
| `PORT` | No | Listen port (default: 5001) |
| `HF_HOME` | No | HuggingFace cache dir (default: `/workspace/.cache/huggingface`) |
| `MAX_QUEUE_SIZE` | No | Max queued jobs (default: 500) |

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
requirements.txt         # Python dependencies
```
