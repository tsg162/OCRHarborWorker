from __future__ import annotations

import torch
from PIL import Image
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

from .base import BaseBackend


class LightOnBackend(BaseBackend):

    def __init__(self, model_slug: str, max_new_tokens: int = 4096) -> None:
        self._model_slug = model_slug
        self._max_new_tokens = max_new_tokens
        self._model: LightOnOcrForConditionalGeneration | None = None
        self._processor: LightOnOcrProcessor | None = None
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None

    def model_id(self) -> str:
        return self._model_slug

    def load(self, device: torch.device, dtype: torch.dtype) -> None:
        self._device = device
        self._dtype = dtype
        self._processor = LightOnOcrProcessor.from_pretrained(self._model_slug)
        self._model = LightOnOcrForConditionalGeneration.from_pretrained(
            self._model_slug, torch_dtype=dtype
        ).to(device)

    def run(self, image: Image.Image, task: str) -> str:
        assert self._model is not None and self._processor is not None, "Call load() first"

        conversation = [
            {"role": "user", "content": [{"type": "image", "image": image}]}
        ]

        inputs = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(device=self._device, dtype=self._dtype)
            if v.is_floating_point()
            else v.to(self._device)
            for k, v in inputs.items()
        }

        output_ids = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens)
        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        return self._processor.decode(generated_ids, skip_special_tokens=True)
