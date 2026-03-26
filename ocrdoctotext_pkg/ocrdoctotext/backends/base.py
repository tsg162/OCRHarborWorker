from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from PIL import Image


class BaseBackend(ABC):

    @abstractmethod
    def load(self, device: torch.device, dtype: torch.dtype) -> None:
        """Download/load model and processor onto the given device."""

    @abstractmethod
    def run(self, image: Image.Image, task: str) -> str:
        """Run inference on a single PIL Image. Return extracted text."""

    @abstractmethod
    def model_id(self) -> str:
        """Return the HuggingFace model identifier this backend was loaded with."""
