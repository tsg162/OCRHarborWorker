from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image

from .backends import resolve_backend
from .backends.base import BaseBackend
from .preprocessing import load_image, pdf_page_count, render_pdf_page
from .types import OCRResult, PageResult
from .utils import detect_device, select_dtype


class OCREngine:

    def __init__(
        self,
        model: str,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        max_new_tokens: int = 4096,
    ) -> None:
        self._model_slug = model
        self._device = detect_device(device)
        self._dtype = select_dtype(self._device, dtype)
        self._backend: BaseBackend = resolve_backend(model, max_new_tokens)
        self._loaded = False

    def load(self) -> OCREngine:
        if not self._loaded:
            self._backend.load(self._device, self._dtype)
            self._loaded = True
        return self

    def run(
        self,
        source: str | Image.Image,
        task: str = "ocr",
        page: int = 1,
    ) -> OCRResult:
        self.load()

        if isinstance(source, str) and source.lower().endswith(".pdf"):
            image = render_pdf_page(source, page)
        else:
            image = load_image(source)

        start = time.perf_counter()
        text = self._backend.run(image, task)
        elapsed = time.perf_counter() - start

        return OCRResult(
            text=text,
            model=self._model_slug,
            task=task,
            elapsed_seconds=round(elapsed, 3),
        )

    def run_batch(
        self,
        sources: list[str | Image.Image],
        task: str = "ocr",
    ) -> list[OCRResult]:
        self.load()

        images = [load_image(src) for src in sources]

        start = time.perf_counter()
        texts = self._backend.run_batch(images, task)
        elapsed = time.perf_counter() - start
        per_image = round(elapsed / len(images), 3) if images else 0

        return [
            OCRResult(
                text=text,
                model=self._model_slug,
                task=task,
                elapsed_seconds=per_image,
            )
            for text in texts
        ]

    def run_pdf(
        self,
        path: str | Path,
        pages: Iterable[int] | None = None,
        task: str = "ocr",
    ) -> list[PageResult]:
        self.load()
        path = str(path)
        total = pdf_page_count(path)

        if pages is None:
            page_numbers = range(1, total + 1)
        else:
            page_numbers = list(pages)

        results: list[PageResult] = []
        for num in page_numbers:
            image = render_pdf_page(path, num)
            start = time.perf_counter()
            text = self._backend.run(image, task)
            elapsed = time.perf_counter() - start
            ocr_result = OCRResult(
                text=text,
                model=self._model_slug,
                task=task,
                elapsed_seconds=round(elapsed, 3),
            )
            results.append(PageResult(page_number=num, ocr_result=ocr_result))

        return results

    def __repr__(self) -> str:
        return (
            f"OCREngine(model={self._model_slug!r}, "
            f"device={self._device}, dtype={self._dtype})"
        )
