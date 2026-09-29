"""Qwen2.5-VL (and Qwen2-VL) element detector.

Loaded once per worker. Needs the `worker` extra (torch, transformers,
bitsandbytes) and a CUDA GPU for the quantised modes.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from prism.domain import Element

from .base import DetectionCanceled
from .parse import ParseResult, parse_elements

log = logging.getLogger(__name__)

PROMPT = (
    "Detect every user interface element in this screenshot. "
    'Output a JSON array only. Each item: {"bbox_2d": [x1, y1, x2, y2], '
    '"label": one of "button", "link", "input", "checkbox", "icon", "text", "image", '
    '"text": the visible text, or "" if none}. '
    "Use one item per element. Do not group several elements into one box."
)


@dataclass
class DetectionRun:
    parsed: ParseResult
    raw_text: str
    input_size: tuple[int, int]
    new_tokens: int
    seconds: float


class _StopWhen(StoppingCriteria):
    def __init__(self, should_stop: Callable[[], bool]) -> None:
        self._should_stop = should_stop

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any
    ) -> torch.BoolTensor:
        stop = self._should_stop()
        return torch.full((input_ids.shape[0],), stop, dtype=torch.bool, device=input_ids.device)  # type: ignore[return-value]


class QwenDetector:
    def __init__(
        self,
        model_id: str,
        *,
        quant: Literal["nf4", "int8", "none"] = "nf4",
        max_side: int = 1280,
        max_new_tokens: int = 2048,
    ) -> None:
        self.model_id = model_id
        self.quant = quant
        self.max_side = max_side
        self.max_new_tokens = max_new_tokens

        cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16
        if not cuda:
            dtype = torch.float32
        quant_config = None
        if quant == "nf4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        elif quant == "int8":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        if quant_config is not None and not cuda:
            raise RuntimeError(f"{quant} quantisation needs a CUDA GPU; use quant='none' on CPU")

        started = time.perf_counter()
        # We resize to max_side ourselves; the cap here just stops the processor
        # from scaling back up past it.
        self._processor = AutoProcessor.from_pretrained(model_id, max_pixels=max_side * max_side)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=dtype,
            quantization_config=quant_config,
            device_map="auto" if cuda else "cpu",
            attn_implementation="sdpa",
        ).eval()
        # Qwen2-VL writes boxes on a 0-1000 grid; Qwen2.5-VL uses pixels of its input.
        self._normalised_1000 = self._model.config.model_type == "qwen2_vl"
        log.info("loaded %s (%s) in %.1fs", model_id, quant, time.perf_counter() - started)

    @property
    def version(self) -> str:
        return f"{self.model_id.rsplit('/', 1)[-1]}:{self.quant}:{self.max_side}"

    def detect(self, image: Image.Image, should_stop: Callable[[], bool]) -> list[Element]:
        return self.run(image, should_stop).parsed.elements

    def run(
        self, image: Image.Image, should_stop: Callable[[], bool] = lambda: False
    ) -> DetectionRun:
        started = time.perf_counter()
        image = _fit(image.convert("RGB"), self.max_side)
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}
        ]
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[prompt], images=[image], return_tensors="pt").to(
            self._model.device
        )

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([_StopWhen(should_stop)]),
            )
        if should_stop():
            raise DetectionCanceled

        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        text = self._processor.decode(new_tokens, skip_special_tokens=True)

        # The processor resizes to a multiple of the patch size; boxes refer to that.
        _, grid_h, grid_w = (int(v) for v in inputs["image_grid_thw"][0])
        frame = (grid_w * 14, grid_h * 14)
        if self._normalised_1000:
            parsed = parse_elements(text, 1000, 1000)
        else:
            parsed = parse_elements(text, frame[0], frame[1])

        if parsed.truncated:
            log.warning(
                "model output hit max_new_tokens=%d, kept partial list", self.max_new_tokens
            )
        return DetectionRun(
            parsed=parsed,
            raw_text=text,
            input_size=frame,
            new_tokens=int(new_tokens.shape[0]),
            seconds=time.perf_counter() - started,
        )


def _fit(image: Image.Image, max_side: int) -> Image.Image:
    if max(image.size) <= max_side:
        return image
    scale = max_side / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)
