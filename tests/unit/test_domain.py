import pytest
from pydantic import ValidationError

from prism.domain import Box, ElementKind


def test_box_rejects_inverted() -> None:
    with pytest.raises(ValidationError):
        Box(x1=0.5, y1=0.1, x2=0.2, y2=0.3)


def test_iou() -> None:
    a = Box(x1=0, y1=0, x2=0.5, y2=0.5)
    b = Box(x1=0.25, y1=0, x2=0.75, y2=0.5)
    assert a.iou(a) == 1
    assert a.iou(b) == pytest.approx(1 / 3)
    assert a.iou(Box(x1=0.6, y1=0.6, x2=1, y2=1)) == 0


def test_to_pixels() -> None:
    assert Box(x1=0.1, y1=0.2, x2=0.5, y2=1).to_pixels(200, 100) == (20, 20, 100, 100)


def test_interactive_kinds() -> None:
    assert ElementKind.BUTTON.interactive
    assert not ElementKind.TEXT.interactive
