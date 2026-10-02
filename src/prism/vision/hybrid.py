"""Controls from the VLM, text from OCR.

The VLM is good at saying what a thing is (button, link, input) and weak at
finding every piece of small text. OCR is the reverse. `merge` keeps the
VLM's controls and images, drops its text items, and adds each OCR line that
isn't the caption of a control.
"""

from collections.abc import Callable
from typing import Protocol

from PIL import Image

from prism.domain import Box, Element, ElementKind

from .base import DetectionCanceled, Detector

# Controls whose own words OCR also finds: a line mostly inside one is its caption.
_CAPTIONED = {
    ElementKind.BUTTON,
    ElementKind.LINK,
    ElementKind.INPUT,
    ElementKind.CHECKBOX,
    ElementKind.ICON,
}
CAPTION_OVERLAP = 0.5


class TextReader(Protocol):
    def read(self, image: Image.Image) -> list[Element]: ...


def merge(detected: list[Element], lines: list[Element]) -> list[Element]:
    controls = [e for e in detected if e.kind is not ElementKind.TEXT]
    captions: dict[int, list[Element]] = {}
    extra = []
    for line in lines:
        owner = _owner(line.box, controls)
        if owner is None and _is_glyph(line.text):
            extra.append(line.model_copy(update={"kind": ElementKind.ICON, "text": None}))
        elif owner is None:
            extra.append(line.model_copy(update={"kind": ElementKind.TEXT}))
        else:
            captions.setdefault(owner, []).append(line)

    out = []
    for i, control in enumerate(controls):
        # Prompts that don't ask the model for text still get it, from OCR.
        # Not for icons: what OCR reads on an icon is a glyph, not words.
        if control.text is None and i in captions and control.kind is not ElementKind.ICON:
            words = " ".join(line.text for line in _reading_order(captions[i]) if line.text)
            control = control.model_copy(update={"text": words[:200] or None})
        out.append(control)
    out += extra
    return [e.model_copy(update={"id": i}) for i, e in enumerate(out)]


def _owner(line: Box, controls: list[Element]) -> int | None:
    best, best_overlap = None, CAPTION_OVERLAP
    for i, c in enumerate(controls):
        if c.kind not in _CAPTIONED:
            continue
        overlap = _inside(line, c.box)
        # OCR reads some icon glyphs as characters (a menu icon came back as "三"),
        # and its box is looser than the model's box for a small icon, so the
        # line is mostly outside the icon. Count it if it covers most of the icon.
        if c.kind is ElementKind.ICON:
            overlap = max(overlap, _inside(c.box, line))
        if overlap >= best_overlap:
            best, best_overlap = i, overlap
    return best


def _is_glyph(text: str | None) -> bool:
    """A lone symbol (☰, ⚙, or a menu icon OCR read as "三") rather than a word."""
    t = (text or "").strip()
    return len(t) == 1 and not (t.isascii() and t.isalnum())


def _inside(inner: Box, outer: Box) -> float:
    """Fraction of inner's area that lies within outer."""
    ix = max(0.0, min(inner.x2, outer.x2) - max(inner.x1, outer.x1))
    iy = max(0.0, min(inner.y2, outer.y2) - max(inner.y1, outer.y1))
    return ix * iy / inner.area if inner.area > 0 else 0.0


def _reading_order(lines: list[Element]) -> list[Element]:
    return sorted(lines, key=lambda e: (round(e.box.y1, 2), e.box.x1))


class HybridDetector:
    def __init__(self, detector: Detector, reader: TextReader) -> None:
        self._detector = detector
        self._reader = reader

    @property
    def version(self) -> str:
        return f"{self._detector.version}+ocr"

    def detect(self, image: Image.Image, should_stop: Callable[[], bool]) -> list[Element]:
        # OCR first: it takes under a second, and a cancel during it is caught below.
        lines = self._reader.read(image)
        if should_stop():
            raise DetectionCanceled
        return merge(self._detector.detect(image, should_stop), lines)
