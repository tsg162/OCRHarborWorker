from __future__ import annotations

from pydantic import BaseModel


class JobResponse(BaseModel):
    job_id: str
    status: str
    queue_position: int
    created_at: str


class OCRResultPayload(BaseModel):
    text: str
    model: str
    elapsed_seconds: float


class JobDetail(BaseModel):
    job_id: str
    status: str
    filename: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    result: OCRResultPayload | None = None
    error: str | None = None


class WebhookPayload(BaseModel):
    job_id: str
    worker_job_id: str
    status: str
    text: str | None = None
    model: str | None = None
    elapsed_seconds: float | None = None
    error: str | None = None
