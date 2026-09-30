"""Text lines from an OCR model (PP-OCR through RapidOCR, ONNX on the CPU).

The VLMs miss a lot of small text (see eval/README.md); a dedicated text
detector finds nearly all of it in about half a second per page.
"""

import numpy as np
from PIL import Image
from rapidocr import RapidOCR
from rapidocr.utils.output import RapidOCROutput

from prism.domain import Box, Element, ElementKind

MIN_SIDE = 0.002  # same noise floor as the VLM parser


class OcrReader:
    def __init__(self, min_score: float = 0.5) -> None:
        self._engine = RapidOCR(
            params={
                "Global.use_cls": False,  # screenshots aren't rotated
                "Global.text_score": min_score,
                "Global.log_level": "error",
            }
        )

    def read(self, image: Image.Image) -> list[Element]:
        """One TEXT element per detected line, boxes in [0, 1] of the image."""
        rgb = np.asarray(image.convert("RGB"))
        result = self._engine(np.ascontiguousarray(rgb[:, :, ::-1]))  # engine expects BGR
        # Detection + recognition always returns this type; the union is for other modes.
        if not isinstance(result, RapidOCROutput):
            raise TypeError(f"unexpected OCR result {type(result).__name__}")
        if result.boxes is None or result.txts is None:
            return []

        w, h = image.size
        lines: list[Element] = []
        for quad, text in zip(result.boxes, result.txts, strict=True):
            x1 = max(0.0, float(quad[:, 0].min()) / w)
            y1 = max(0.0, float(quad[:, 1].min()) / h)
            x2 = min(1.0, float(quad[:, 0].max()) / w)
            y2 = min(1.0, float(quad[:, 1].max()) / h)
            if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
                continue
            lines.append(
                Element(
                    id=len(lines),
                    kind=ElementKind.TEXT,
                    box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                    text=text.strip()[:200] or None,
                )
            )
        return lines
