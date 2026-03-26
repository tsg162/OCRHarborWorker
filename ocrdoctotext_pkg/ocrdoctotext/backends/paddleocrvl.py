from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from .base import BaseBackend

TASK_PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}


class PaddleOCRVLBackend(BaseBackend):

    def __init__(self, model_slug: str, max_new_tokens: int = 4096) -> None:
        self._model_slug = model_slug
        self._max_new_tokens = max_new_tokens
        self._model: AutoModelForCausalLM | None = None
        self._processor: AutoProcessor | None = None
        self._device: torch.device | None = None

    def model_id(self) -> str:
        return self._model_slug

    def load(self, device: torch.device, dtype: torch.dtype) -> None:
        self._device = device
        self._processor = AutoProcessor.from_pretrained(
            self._model_slug, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_slug,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device).eval()

    def run(self, image: Image.Image, task: str) -> str:
        assert self._model is not None and self._processor is not None, "Call load() first"

        prompt = TASK_PROMPTS.get(task)
        if prompt is None:
            raise ValueError(
                f"Unsupported task {task!r} for PaddleOCR-VL. "
                f"Choose from: {list(TASK_PROMPTS)}"
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        return self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
