import time
from collections.abc import Callable

from PIL import Image

from prism.domain import Box, Element, ElementKind
from prism.vision.base import DetectionCanceled


class FakeDetector:
    """Stands in for the VLM in tests. Never used by the running app."""

    version = "fake-detector"

    def __init__(
        self,
        elements: list[Element] | None = None,
        *,
        delay_s: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self.elements = (
            elements
            if elements is not None
            else [
                Element(id=0, kind=ElementKind.BUTTON, box=Box(x1=0.1, y1=0.1, x2=0.3, y2=0.2)),
            ]
        )
        self.delay_s = delay_s
        self.error = error
        self.calls = 0

    def detect(self, image: Image.Image, should_stop: Callable[[], bool]) -> list[Element]:
        self.calls += 1
        deadline = time.monotonic() + self.delay_s
        while time.monotonic() < deadline:
            if should_stop():
                raise DetectionCanceled
            time.sleep(0.02)
        if self.error:
            raise self.error
        return self.elements
