import pytest
from PIL import Image

from prism.domain import Box, Element, ElementKind
from prism.vision.base import DetectionCanceled
from prism.vision.hybrid import HybridDetector, merge


def el(kind: str, x1: float, y1: float, x2: float, y2: float, text: str | None = None) -> Element:
    return Element(id=0, kind=ElementKind(kind), box=Box(x1=x1, y1=y1, x2=x2, y2=y2), text=text)


def test_model_text_is_replaced_by_ocr() -> None:
    detected = [el("text", 0.1, 0.1, 0.3, 0.15, "Wrong"), el("button", 0.5, 0.5, 0.7, 0.6)]
    lines = [el("text", 0.1, 0.1, 0.3, 0.14, "Heading"), el("text", 0.1, 0.8, 0.4, 0.84, "Footer")]
    out = merge(detected, lines)
    assert [e.kind for e in out] == [ElementKind.BUTTON, ElementKind.TEXT, ElementKind.TEXT]
    assert [e.text for e in out[1:]] == ["Heading", "Footer"]
    assert [e.id for e in out] == [0, 1, 2]


def test_caption_inside_a_control_is_not_listed_again() -> None:
    button = el("button", 0.5, 0.5, 0.7, 0.6)
    link = el("link", 0.1, 0.02, 0.2, 0.05, "Home")
    lines = [
        el("text", 0.55, 0.53, 0.65, 0.57, "Save"),
        el("text", 0.1, 0.02, 0.2, 0.05, "Home page"),
    ]
    out = merge([button, link], lines)
    assert len(out) == 2
    assert out[0].text == "Save"  # filled from OCR when the model gave none
    assert out[1].text == "Home"  # the model's own text is kept


def test_label_beside_an_input_stays_separate() -> None:
    field = el("input", 0.3, 0.2, 0.8, 0.26)
    lines = [el("text", 0.05, 0.21, 0.25, 0.25, "Email"), el("text", 0.32, 0.21, 0.5, 0.25, "you@")]
    out = merge([field], lines)
    assert out[0].text == "you@"  # placeholder inside the field
    assert [(e.kind, e.text) for e in out[1:]] == [(ElementKind.TEXT, "Email")]


def test_text_on_an_image_is_kept() -> None:
    hero = el("image", 0.0, 0.0, 1.0, 0.5)
    out = merge([hero], [el("text", 0.2, 0.2, 0.6, 0.3, "Welcome")])
    assert [e.kind for e in out] == [ElementKind.IMAGE, ElementKind.TEXT]


def test_line_mostly_outside_a_control_is_text() -> None:
    button = el("button", 0.5, 0.5, 0.6, 0.6)
    out = merge([button], [el("text", 0.55, 0.52, 0.9, 0.58, "Long line")])
    assert len(out) == 2


def test_multi_line_caption_is_joined_in_reading_order() -> None:
    button = el("button", 0.1, 0.1, 0.5, 0.4)
    lines = [
        el("text", 0.15, 0.25, 0.45, 0.3, "account"),
        el("text", 0.15, 0.15, 0.45, 0.2, "Create"),
    ]
    assert merge([button], lines)[0].text == "Create account"


class FakeDetector:
    version = "fake:1"

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, image: Image.Image, should_stop: object) -> list[Element]:
        self.calls += 1
        return [el("button", 0.1, 0.1, 0.5, 0.5)]


class FakeReader:
    def read(self, image: Image.Image) -> list[Element]:
        return [el("text", 0.6, 0.6, 0.9, 0.7, "Hello")]


def test_hybrid_detector() -> None:
    inner = FakeDetector()
    hybrid = HybridDetector(inner, FakeReader())
    assert hybrid.version == "fake:1+ocr"
    out = hybrid.detect(Image.new("RGB", (10, 10)), lambda: False)
    assert [e.kind for e in out] == [ElementKind.BUTTON, ElementKind.TEXT]


def test_hybrid_detector_cancel_skips_the_model() -> None:
    inner = FakeDetector()
    with pytest.raises(DetectionCanceled):
        HybridDetector(inner, FakeReader()).detect(Image.new("RGB", (10, 10)), lambda: True)
    assert inner.calls == 0


def test_glyph_read_on_an_icon_is_not_extra_text() -> None:
    # A 7x11 px menu icon; OCR boxes the glyph more loosely and reads it as "三".
    icon = el("icon", 0.787, 0.033, 0.794, 0.052)
    glyph = el("text", 0.784, 0.030, 0.800, 0.058, "三")
    out = merge([icon], [glyph])
    assert [(e.kind, e.text) for e in out] == [(ElementKind.ICON, None)]


def test_lone_symbol_from_ocr_is_an_icon() -> None:
    # The model missed the menu icon; OCR found it but read the glyph as "三".
    lines = [
        el("text", 0.95, 0.03, 0.97, 0.06, "三"),
        el("text", 0.90, 0.03, 0.92, 0.06, "⚙"),
        el("text", 0.10, 0.50, 0.12, 0.53, "A"),  # a single letter is still text
    ]
    out = merge([], lines)
    assert [(e.kind, e.text) for e in out] == [
        (ElementKind.ICON, None),
        (ElementKind.ICON, None),
        (ElementKind.TEXT, "A"),
    ]
