import hmac

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from gpu_worker.config import settings

_bearer = HTTPBearer()


async def verify_secret(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> None:
    if not settings.WORKER_SECRET:
        return  # no secret configured — allow all (dev mode)
    if not hmac.compare_digest(credentials.credentials, settings.WORKER_SECRET):
        raise HTTPException(401, "Invalid secret")
