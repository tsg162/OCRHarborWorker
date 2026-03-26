from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO

from PIL import Image

from gpu_worker.config import settings
from gpu_worker.models import OCRResultPayload, WebhookPayload
from gpu_worker.ocr_bridge import get_ocr_engine
from gpu_worker.webhook import send_webhook

logger = logging.getLogger(__name__)


def _is_pdf(filename: str, content_type: str) -> bool:
    return filename.lower().endswith(".pdf") or "pdf" in content_type.lower()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    filename: str
    content_type: str
    image_data: bytes | None
    page: int = 1
    task: str = "ocr"
    callback_url: str = ""
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    result_text: str | None = None
    result_model: str | None = None
    result_elapsed: float | None = None
    error: str | None = None


class JobManager:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=settings.MAX_QUEUE_SIZE)
        self._jobs: dict[str, Job] = {}
        self._shutdown = asyncio.Event()

    def submit(
        self,
        filename: str,
        image_data: bytes,
        content_type: str,
        page: int = 1,
        task: str = "ocr",
        callback_url: str = "",
    ) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(
            id=job_id,
            filename=filename,
            content_type=content_type,
            image_data=image_data,
            page=page,
            task=task,
            callback_url=callback_url,
        )
        self._jobs[job_id] = job
        self._queue.put_nowait(job_id)
        logger.info("Job %s queued (%s)", job_id, filename)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.status in ("completed", "failed"):
            return False
        job.status = "cancelled"
        job.image_data = None
        return True

    def queue_depth(self) -> int:
        return self._queue.qsize()

    def list_jobs(self) -> list[Job]:
        return list(self._jobs.values())

    async def start_runner(self) -> None:
        logger.info("Job runner started")
        while not self._shutdown.is_set():
            try:
                job_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            job = self._jobs.get(job_id)
            if job is None or job.status == "cancelled":
                continue

            job.status = "processing"
            job.started_at = time.time()
            logger.info("Processing job %s (%s)", job.id, job.filename)

            try:
                result = await asyncio.to_thread(self._run_ocr, job)
                job.status = "completed"
                job.result_text = result.text
                job.result_model = result.model
                job.result_elapsed = result.elapsed_seconds
                logger.info("Job %s completed in %.1fs", job.id, result.elapsed_seconds)
            except Exception as exc:
                job.status = "failed"
                job.error = str(exc)
                logger.error("Job %s failed: %s", job.id, exc)
            finally:
                job.completed_at = time.time()
                job.image_data = None  # free memory

            await self._send_job_webhook(job)

    def _run_ocr(self, job: Job) -> object:
        engine = get_ocr_engine()
        if _is_pdf(job.filename, job.content_type):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(job.image_data)
                tmp_path = f.name
            try:
                results = engine.run_pdf(tmp_path, pages=[job.page])
                return results[0].ocr_result
            finally:
                os.unlink(tmp_path)
        else:
            image = Image.open(BytesIO(job.image_data)).convert("RGB")
            return engine.run(image, task=job.task)

    async def _send_job_webhook(self, job: Job) -> None:
        payload = WebhookPayload(
            job_id=job.id,
            worker_job_id=job.id,
            status=job.status,
            text=job.result_text,
            model=job.result_model,
            elapsed_seconds=job.result_elapsed,
            error=job.error,
        )
        await send_webhook(payload, callback_url=job.callback_url or None)

    async def start_cleanup(self) -> None:
        while not self._shutdown.is_set():
            await asyncio.sleep(60)
            cutoff = time.time() - settings.JOB_TTL_SECONDS
            expired = [
                jid for jid, job in self._jobs.items()
                if job.completed_at and job.completed_at < cutoff
            ]
            for jid in expired:
                del self._jobs[jid]
            if expired:
                logger.info("Cleaned up %d expired jobs", len(expired))

    async def stop(self) -> None:
        self._shutdown.set()


_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager
