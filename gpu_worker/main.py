from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from gpu_worker.auth import verify_secret
from gpu_worker.config import settings
from gpu_worker.job_manager import Job, get_job_manager
from gpu_worker.models import JobDetail, JobResponse, OCRResultPayload
from gpu_worker.ocr_bridge import get_ocr_engine, is_engine_loaded

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


async def _get_public_ip() -> str:
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("https://api.ipify.org")
            return resp.text.strip()
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
    if manager.queue_depth() >= settings.MAX_QUEUE_SIZE:
        raise HTTPException(429, "Queue full")

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


@app.get("/health")
async def health():
    import torch
    public_ip = await _get_public_ip()
    return {
        "status": "ok",
        "public_ip": public_ip,
        "worker_url": f"http://{public_ip}:{settings.PORT}",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_loaded": is_engine_loaded(),
        "queue_depth": get_job_manager().queue_depth(),
        "uptime_seconds": round(time.time() - _START_TIME, 1),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "gpu_worker.main:app",
        host=settings.HOST,
        port=settings.PORT,
    )
