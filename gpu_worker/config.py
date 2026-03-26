from pydantic_settings import BaseSettings


class WorkerSettings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKER_SECRET: str = ""
    CALLBACK_URL: str = ""
    CALLBACK_SECRET: str = ""
    OCR_MODEL: str = "lightonai/LightOnOCR-2-1B"
    MAX_QUEUE_SIZE: int = 100
    JOB_TTL_SECONDS: int = 3600
    WEBHOOK_TIMEOUT: float = 10.0
    WEBHOOK_MAX_RETRIES: int = 3

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = WorkerSettings()
