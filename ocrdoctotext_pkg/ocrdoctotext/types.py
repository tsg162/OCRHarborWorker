from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OCRResult:
    text: str
    model: str
    task: str
    elapsed_seconds: float


@dataclass
class PageResult:
    page_number: int
    ocr_result: OCRResult

    @property
    def text(self) -> str:
        return self.ocr_result.text
