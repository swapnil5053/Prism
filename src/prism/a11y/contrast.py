"""Text contrast (WCAG 2.2 SC 1.4.3) estimated from pixels.

We only have a screenshot, not the DOM, so the text and background colours are
estimated: split the pixels in a text box into two clusters, call the bigger
one background, and take the strongest-contrast tail of the smaller one as the
text colour. Anti-aliased edges sit between the two, so averaging the whole
text cluster would understate contrast and flag text that is actually fine.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from prism.domain import Box, Element, ElementKind, Finding

NORMAL_TEXT = 4.5
LARGE_TEXT = 3.0
LARGE_TEXT_PX = 24  # 18pt. We can't see font weight, so the 14pt-bold case is ignored.

_MIN_TEXT_SHARE = 0.005  # fewer "text" pixels than this: nothing to measure
_MAX_SAMPLE_SIDE = 96
_ITERATIONS = 8
_INSET = 0.06


@dataclass(frozen=True)
class ContrastEstimate:
    text_rgb: tuple[int, int, int]
    background_rgb: tuple[int, int, int]
    ratio: float


def relative_luminance(
    rgb: NDArray[np.float64] | tuple[float, float, float],
) -> NDArray[np.float64]:
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    result: NDArray[np.float64] = linear @ np.array([0.2126, 0.7152, 0.0722])
    return result


def contrast_ratio(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    la, lb = float(relative_luminance(a)), float(relative_luminance(b))
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def estimate(image: Image.Image, box: Box) -> ContrastEstimate | None:
    x1, y1, x2, y2 = box.to_pixels(image.width, image.height)
    # Trim the edges: rounded corners and borders there aren't text or background.
    dx, dy = max(1, round((x2 - x1) * _INSET)), max(1, round((y2 - y1) * _INSET))
    crop = image.crop((x1 + dx, y1 + dy, x2 - dx, y2 - dy)).convert("RGB")
    if crop.width < 2 or crop.height < 2:
        return None
    crop.thumbnail((_MAX_SAMPLE_SIDE * 4, _MAX_SAMPLE_SIDE), Image.Resampling.NEAREST)
    pixels = np.asarray(crop, dtype=np.float64).reshape(-1, 3)
    lum = relative_luminance(pixels)

    labels = _two_means(pixels)
    big = 0 if (labels == 0).sum() >= (labels == 1).sum() else 1
    background = pixels[labels == big]
    text = pixels[labels != big]
    text_lum = lum[labels != big]
    if len(text) < _MIN_TEXT_SHARE * len(pixels):
        return None

    bg = background.mean(axis=0)
    bg_lum = float(relative_luminance(bg))
    # The 20% of text pixels furthest from the background in luminance: the
    # solid stroke centres rather than the blended edges.
    distance = np.abs(text_lum - bg_lum)
    core = text[distance >= np.quantile(distance, 0.8)]
    fg = core.mean(axis=0)

    fg_rgb = _to_rgb(fg)
    bg_rgb = _to_rgb(bg)
    return ContrastEstimate(fg_rgb, bg_rgb, round(contrast_ratio(fg_rgb, bg_rgb), 2))


def check(image: Image.Image, elements: list[Element], dpr: float) -> list[Finding]:
    findings = []
    for el in elements:
        if el.kind not in (ElementKind.TEXT, ElementKind.BUTTON, ElementKind.LINK):
            continue
        est = estimate(image, el.box)
        if est is None:
            continue
        height_css = (el.box.y2 - el.box.y1) * image.height / dpr
        # A single line box is roughly 1.2-1.4x the font size. Multi-line text
        # boxes would read as "large", so only trust this for short boxes.
        large = LARGE_TEXT_PX * 1.2 <= height_css <= LARGE_TEXT_PX * 3
        required = LARGE_TEXT if large else NORMAL_TEXT
        if est.ratio >= required:
            continue
        findings.append(
            Finding(
                rule="text-contrast",
                wcag="1.4.3",
                severity="serious" if est.ratio < 3.0 else "moderate",
                element_id=el.id,
                message=(
                    f"Text contrast is about {est.ratio}:1 "
                    f"({_hex(est.text_rgb)} on {_hex(est.background_rgb)}); "
                    f"needs {required}:1."
                ),
                measured=est.ratio,
                required=required,
            )
        )
    return findings


def _two_means(pixels: NDArray[np.float64]) -> NDArray[np.int_]:
    # Seed with the most common colour (almost always the background) and the
    # pixel furthest from it. Seeding with the darkest and lightest pixels
    # instead latched onto stray corner pixels on small buttons.
    _, inverse, counts = np.unique(
        (pixels // 8).astype(np.int_), axis=0, return_inverse=True, return_counts=True
    )
    common = pixels[inverse.ravel() == counts.argmax()].mean(axis=0)
    far = pixels[((pixels - common) ** 2).sum(axis=1).argmax()]
    centres = np.stack([common, far])
    labels = np.zeros(len(pixels), dtype=np.int_)
    for i in range(_ITERATIONS):
        dist = ((pixels[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        new = dist.argmin(axis=1)
        if i > 0 and (new == labels).all():
            break
        labels = new
        for k in (0, 1):
            if (labels == k).any():
                centres[k] = pixels[labels == k].mean(axis=0)
    return labels


def _to_rgb(c: NDArray[np.float64]) -> tuple[int, int, int]:
    r, g, b = (round(float(v)) for v in np.clip(c, 0, 255))
    return (r, g, b)


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)
