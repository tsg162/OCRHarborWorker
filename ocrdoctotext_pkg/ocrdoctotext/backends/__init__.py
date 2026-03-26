from __future__ import annotations

from typing import TYPE_CHECKING

from .lighton import LightOnBackend
from .paddleocrvl import PaddleOCRVLBackend

if TYPE_CHECKING:
    from .base import BaseBackend

# Maps a model slug prefix to its backend class.
# Lookup: try longest-prefix match so variants (e.g. LightOnOCR-2-1B-bbox) resolve correctly.
_BACKEND_PREFIXES: list[tuple[str, type[BaseBackend]]] = [
    ("lightonai/LightOnOCR", LightOnBackend),
    ("PaddlePaddle/PaddleOCR-VL", PaddleOCRVLBackend),
]


def resolve_backend(model_slug: str, max_new_tokens: int) -> BaseBackend:
    for prefix, cls in _BACKEND_PREFIXES:
        if model_slug.startswith(prefix):
            return cls(model_slug=model_slug, max_new_tokens=max_new_tokens)

    raise ValueError(
        f"No backend found for model {model_slug!r}. "
        f"Supported prefixes: {[p for p, _ in _BACKEND_PREFIXES]}"
    )
