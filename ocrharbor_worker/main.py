from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ocrharbor_worker.auth import verify_secret
from ocrharbor_worker.config import settings
from ocrharbor_worker.job_manager import Job, get_job_manager
from ocrharbor_worker.models import JobDetail, JobResponse, OCRResultPayload
from ocrharbor_worker.ocr_bridge import get_ocr_engine, is_engine_loaded


class ConfigUpdate(BaseModel):
    batch_size: Optional[int] = Field(None, ge=1, le=64)
    batch_wait_seconds: Optional[float] = Field(None, ge=0.0, le=30.0)
    max_queue_size: Optional[int] = Field(None, ge=1, le=10000)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_START_TIME = time.time()


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _job_detail(job: Job) -> JobDetail:
    result = None
    if job.status == "completed" and job.result_text is not None:
        result = OCRResultPayload(
            text=job.result_text,
            model=job.result_model or "",
            elapsed_seconds=job.result_elapsed or 0.0,
        )
    return JobDetail(
        job_id=job.id,
        status=job.status,
        filename=job.filename,
        created_at=_iso(job.created_at) or "",
        started_at=_iso(job.started_at),
        completed_at=_iso(job.completed_at),
        result=result,
        error=job.error,
    )


_cached_public_ip: str = "unknown"


async def _get_public_ip() -> str:
    global _cached_public_ip
    if _cached_public_ip != "unknown":
        return _cached_public_ip
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("https://api.ipify.org")
            _cached_public_ip = resp.text.strip()
            return _cached_public_ip
    except Exception:
        return "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    public_ip = await _get_public_ip()
    logger.info("Public IP: %s", public_ip)
    logger.info("Worker URL: http://%s:%d", public_ip, settings.PORT)
    logger.info("Health check: curl http://%s:%d/health", public_ip, settings.PORT)

    logger.info("Loading OCR model...")
    await asyncio.to_thread(get_ocr_engine)
    logger.info("Model loaded — ready to accept jobs")

    manager = get_job_manager()
    runner_task = asyncio.create_task(manager.start_runner())
    cleanup_task = asyncio.create_task(manager.start_cleanup())
    yield
    logger.info("Shutting down, finishing current job...")
    await manager.stop()
    runner_task.cancel()
    cleanup_task.cancel()


app = FastAPI(title="OCR GPU Worker", version="1.0.0", lifespan=lifespan)


@app.post("/jobs", status_code=202)
async def submit_job(
    file: UploadFile = File(...),
    job_id: str = Form(default=""),
    page: int = Form(default=1),
    task: str = Form(default="ocr"),
    callback_url: str = Form(default=""),
    _: None = Depends(verify_secret),
) -> JobResponse:
    manager = get_job_manager()
    if manager.queue_depth() >= manager.max_queue_size:
        raise HTTPException(
            429,
            "Queue full",
            headers={"Retry-After": "10"},
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")

    job = manager.submit(
        filename=file.filename or "unknown",
        image_data=raw,
        content_type=file.content_type or "",
        page=page,
        task=task,
        callback_url=callback_url,
    )
    return JobResponse(
        job_id=job.id,
        status=job.status,
        queue_position=manager.queue_depth(),
        created_at=_iso(job.created_at) or "",
    )


@app.get("/jobs/{job_id}")
async def get_job(job_id: str, _: None = Depends(verify_secret)) -> JobDetail:
    manager = get_job_manager()
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return _job_detail(job)


@app.delete("/jobs", status_code=200)
async def clear_queue(_: None = Depends(verify_secret)):
    manager = get_job_manager()
    cancelled = manager.clear_queue()
    return {"success": True, "cancelled": cancelled, "queue_depth": manager.queue_depth()}


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str, _: None = Depends(verify_secret)):
    manager = get_job_manager()
    if not manager.cancel(job_id):
        raise HTTPException(404, "Job not found or already finished")
    return {"success": True, "job_id": job_id, "status": "cancelled"}


@app.get("/jobs")
async def list_jobs(_: None = Depends(verify_secret)):
    manager = get_job_manager()
    jobs = manager.list_jobs()
    return {
        "jobs": [_job_detail(j) for j in jobs],
        "queue_depth": manager.queue_depth(),
    }


@app.put("/config")
async def update_config(body: ConfigUpdate, _: None = Depends(verify_secret)):
    manager = get_job_manager()
    applied: dict[str, object] = {}
    if body.batch_size is not None:
        manager.batch_size = body.batch_size
        applied["batch_size"] = body.batch_size
    if body.batch_wait_seconds is not None:
        manager.batch_wait_seconds = body.batch_wait_seconds
        applied["batch_wait_seconds"] = body.batch_wait_seconds
    if body.max_queue_size is not None:
        manager.max_queue_size = body.max_queue_size
        applied["max_queue_size"] = body.max_queue_size
    if not applied:
        raise HTTPException(400, "No config fields provided")
    logger.info("Config updated: %s", applied)
    return {"updated": applied, "config": _current_config(manager)}


def _current_config(manager=None):
    if manager is None:
        manager = get_job_manager()
    return {
        "batch_size": manager.batch_size,
        "batch_wait_seconds": manager.batch_wait_seconds,
        "max_queue_size": manager.max_queue_size,
    }


@app.get("/config")
async def get_config(_: None = Depends(verify_secret)):
    return _current_config()


@app.get("/health")
async def health():
    import os
    import torch
    manager = get_job_manager()
    public_ip = _cached_public_ip
    return {
        "status": "ok",
        "public_ip": public_ip,
        "worker_url": f"http://{public_ip}:{settings.PORT}",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_loaded": is_engine_loaded(),
        "queue_depth": manager.queue_depth(),
        "config": _current_config(manager),
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "vast_id": os.environ.get("VAST_CONTAINERLABEL"),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "ocrharbor_worker.main:app",
        host=settings.HOST,
        port=settings.PORT,
    )
