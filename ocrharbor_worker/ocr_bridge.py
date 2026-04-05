import threading

from ocrharbor_worker.config import settings

_engine = None
_lock = threading.Lock()


def get_ocr_engine():
    global _engine
    if _engine is not None:
        return _engine
    with _lock:
        if _engine is not None:
            return _engine
        from ocrdoctotext import OCREngine
        _engine = OCREngine(settings.OCR_MODEL)
        _engine.load()
        return _engine


def is_engine_loaded() -> bool:
    return _engine is not None
