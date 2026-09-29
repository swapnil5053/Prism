from collections.abc import Callable
from typing import Protocol

from PIL import Image

from prism.domain import Element


class DetectionCanceled(Exception):
    """Raised by a detector when should_stop() turned true mid-run."""


class Detector(Protocol):
    @property
    def version(self) -> str:
        """Identifies model + settings; stored with every result."""
        ...

    def detect(self, image: Image.Image, should_stop: Callable[[], bool]) -> list[Element]:
        """Blocking. Called from a worker thread, never on the event loop."""
        ...
