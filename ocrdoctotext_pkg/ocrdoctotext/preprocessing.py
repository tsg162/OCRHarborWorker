from __future__ import annotations

from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


def load_image(source: str | Image.Image) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGB")

    if source.startswith(("http://", "https://")):
        resp = requests.get(source, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")

    return Image.open(source).convert("RGB")


def render_pdf_page(path: str | Path, page_number: int, dpi: int = 200) -> Image.Image:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    page_index = page_number - 1
    if page_index < 0 or page_index >= len(pdf):
        raise ValueError(f"Page {page_number} out of range (PDF has {len(pdf)} pages)")
    page = pdf[page_index]
    scale = dpi / 72
    bitmap = page.render(scale=scale)
    image = bitmap.to_pil().convert("RGB")
    return image


def pdf_page_count(path: str | Path) -> int:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    return len(pdf)
