"""Accessibility checks on detected elements. Pixel heuristics, not a DOM audit."""

from PIL import Image

from prism.domain import AnalysisResult, Element, Finding

from . import contrast, labels, target_size
from .score import score

__all__ = ["audit"]


def audit(image: Image.Image, elements: list[Element], dpr: float = 1.0) -> AnalysisResult:
    findings: list[Finding] = [
        *contrast.check(image, elements, dpr),
        *target_size.check(elements, image.width, image.height, dpr),
        *labels.check(elements, image.width, image.height),
    ]
    findings.sort(key=lambda f: (f.element_id, f.rule))
    return AnalysisResult(elements=elements, findings=findings, score=score(elements, findings))
