import pytest
from PIL import Image, ImageDraw, ImageFont

from prism.a11y import contrast
from prism.a11y.contrast import contrast_ratio, estimate
from prism.domain import Box, Element, ElementKind


@pytest.mark.parametrize(
    ("fg", "bg", "expected"),
    [
        ((0, 0, 0), (255, 255, 255), 21.0),
        ((255, 255, 255), (255, 255, 255), 1.0),
        ((0x77, 0x77, 0x77), (255, 255, 255), 4.48),  # the classic near-miss grey
        ((0x76, 0x76, 0x76), (255, 255, 255), 4.54),
        ((0xFF, 0x00, 0x00), (255, 255, 255), 4.0),
        ((0x00, 0x00, 0xFF), (255, 255, 255), 8.59),
    ],
)
def test_contrast_ratio_matches_wcag_reference(
    fg: tuple[int, int, int], bg: tuple[int, int, int], expected: float
) -> None:
    assert contrast_ratio(fg, bg) == pytest.approx(expected, abs=0.01)
    assert contrast_ratio(bg, fg) == pytest.approx(expected, abs=0.01)


def text_image(
    fg: tuple[int, int, int], bg: tuple[int, int, int], size: int = 32
) -> tuple[Image.Image, Box]:
    img = Image.new("RGB", (400, 120), bg)
    font = ImageFont.load_default(size=size)
    draw = ImageDraw.Draw(img)
    draw.text((20, 30), "Sign in to continue", fill=fg, font=font)
    x1, y1, x2, y2 = draw.textbbox((20, 30), "Sign in to continue", font=font)
    pad = 4
    box = Box(
        x1=(x1 - pad) / img.width,
        y1=(y1 - pad) / img.height,
        x2=(x2 + pad) / img.width,
        y2=(y2 + pad) / img.height,
    )
    return img, box


@pytest.mark.parametrize(
    ("fg", "bg"),
    [
        ((0, 0, 0), (255, 255, 255)),
        ((0x99, 0x99, 0x99), (255, 255, 255)),
        ((255, 255, 255), (0x1A, 0x73, 0xE8)),
        ((0xAA, 0xAA, 0xAA), (0x22, 0x22, 0x22)),
    ],
)
def test_estimate_close_to_true_ratio(fg: tuple[int, int, int], bg: tuple[int, int, int]) -> None:
    img, box = text_image(fg, bg)
    est = estimate(img, box)
    assert est is not None
    true = contrast_ratio(fg, bg)
    assert est.ratio == pytest.approx(true, rel=0.15)


def test_blank_box_has_nothing_to_measure() -> None:
    img = Image.new("RGB", (100, 100), "white")
    assert estimate(img, Box(x1=0.1, y1=0.1, x2=0.9, y2=0.9)) is None


def test_flags_low_contrast_text_only() -> None:
    low, box = text_image((0xBB, 0xBB, 0xBB), (255, 255, 255), size=14)
    ok, _ = text_image((0x33, 0x33, 0x33), (255, 255, 255), size=14)
    el = Element(id=3, kind=ElementKind.TEXT, box=box)

    [finding] = contrast.check(low, [el], dpr=1)
    assert finding.rule == "text-contrast"
    assert finding.element_id == 3
    assert finding.required == 4.5
    assert finding.measured is not None and finding.measured < 3
    assert finding.severity == "serious"

    assert contrast.check(ok, [el], dpr=1) == []


def test_large_text_uses_lower_threshold() -> None:
    # #949494 on white is about 3.0:1: fails for body text, passes for large text.
    img, box = text_image((0x90, 0x90, 0x90), (255, 255, 255), size=40)
    el = Element(id=0, kind=ElementKind.TEXT, box=box)
    assert contrast.check(img, [el], dpr=1) == []
    # Same pixels on a 2x screenshot are half the CSS size, so body-text rules apply.
    assert len(contrast.check(img, [el], dpr=2)) == 1


def test_ignores_non_text_elements() -> None:
    img, box = text_image((0xDD, 0xDD, 0xDD), (255, 255, 255))
    assert contrast.check(img, [Element(id=0, kind=ElementKind.IMAGE, box=box)], dpr=1) == []
