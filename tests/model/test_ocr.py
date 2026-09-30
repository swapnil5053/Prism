"""Runs the real OCR model (bundled with rapidocr, CPU, about a second)."""

import pytest
from PIL import Image, ImageDraw, ImageFont

pytest.importorskip("rapidocr")

from prism.domain import ElementKind
from prism.vision.ocr import OcrReader

pytestmark = pytest.mark.model


def test_reads_lines_with_normalised_boxes() -> None:
    img = Image.new("RGB", (480, 200), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=28)
    draw.text((30, 30), "Create account", fill="black", font=font)
    draw.text((30, 120), "Forgot password", fill=(90, 90, 90), font=font)

    lines = OcrReader().read(img)
    assert len(lines) == 2
    assert all(e.kind is ElementKind.TEXT for e in lines)
    first, second = sorted(lines, key=lambda e: e.box.y1)
    assert first.text is not None and "account" in first.text.lower()
    # Box within the image and around where the text was drawn.
    assert 0.0 <= first.box.x1 < 30 / 480 + 0.02
    assert first.box.y1 < 30 / 200 + 0.05 < second.box.y1


def test_blank_image_has_no_lines() -> None:
    assert OcrReader().read(Image.new("RGB", (200, 100), "white")) == []
