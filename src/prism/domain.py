"""Core data types produced by detection and auditing.

Boxes are normalised to [0, 1] relative to the original image, so they don't
depend on whatever resize the model applied internally.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ElementKind(StrEnum):
    BUTTON = "button"
    LINK = "link"
    INPUT = "input"
    CHECKBOX = "checkbox"
    ICON = "icon"
    TEXT = "text"
    IMAGE = "image"

    @property
    def interactive(self) -> bool:
        return self in _INTERACTIVE


_INTERACTIVE = {
    ElementKind.BUTTON,
    ElementKind.LINK,
    ElementKind.INPUT,
    ElementKind.CHECKBOX,
    ElementKind.ICON,
}


class Box(BaseModel):
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _ordered(self) -> "Box":
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("box must have x1 < x2 and y1 < y2")
        return self

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            round(self.x1 * width),
            round(self.y1 * height),
            round(self.x2 * width),
            round(self.y2 * height),
        )

    def iou(self, other: "Box") -> float:
        ix = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        iy = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class Element(BaseModel):
    id: int
    kind: ElementKind
    box: Box
    text: str | None = Field(default=None, max_length=500)


Severity = Literal["minor", "moderate", "serious"]


class Finding(BaseModel):
    rule: str
    wcag: str
    severity: Severity
    element_id: int
    message: str
    measured: float | None = None
    required: float | None = None


class AnalysisResult(BaseModel):
    elements: list[Element]
    findings: list[Finding]
    score: float = Field(ge=0, le=100)
