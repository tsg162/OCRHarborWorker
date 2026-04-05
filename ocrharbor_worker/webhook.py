from __future__ import annotations

import asyncio
import logging

import httpx

from ocrharbor_worker.config import settings
from ocrharbor_worker.models import WebhookPayload

logger = logging.getLogger(__name__)


async def send_webhook(payload: WebhookPayload, callback_url: str | None = None) -> None:
    url = callback_url or settings.CALLBACK_URL
    if not url:
        logger.debug("No callback URL configured, skipping webhook for job %s", payload.job_id)
        return

    headers = {}
    if settings.CALLBACK_SECRET:
        headers["X-API-Key"] = settings.CALLBACK_SECRET

    for attempt in range(settings.WEBHOOK_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=settings.WEBHOOK_TIMEOUT) as client:
                resp = await client.post(url, json=payload.model_dump(), headers=headers)
                if resp.status_code < 400:
                    logger.info("Webhook delivered for job %s (attempt %d)", payload.job_id, attempt + 1)
                    return
                logger.warning(
                    "Webhook returned %d for job %s (attempt %d)",
                    resp.status_code, payload.job_id, attempt + 1,
                )
        except httpx.RequestError as exc:
            logger.warning(
                "Webhook request failed for job %s (attempt %d): %s",
                payload.job_id, attempt + 1, exc,
            )
        if attempt < settings.WEBHOOK_MAX_RETRIES - 1:
            await asyncio.sleep(2 ** attempt)

    logger.error("All webhook attempts exhausted for job %s", payload.job_id)
