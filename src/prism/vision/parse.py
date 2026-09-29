"""Turn raw model text into validated elements.

The model's output is untrusted: it can be wrapped in markdown fences, cut off
by the token limit, use labels we don't know, or contain boxes outside the
image. Anything that can't be repaired is dropped and counted, never guessed.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from prism.domain import Box, Element, ElementKind

MAX_ELEMENTS = 300
MIN_SIDE = 0.002  # boxes thinner than this (as a fraction of the image) are noise
DUPLICATE_IOU = 0.9

_FENCE = re.compile(r"```(?:json)?\s*(.*?)(?:```|$)", re.DOTALL)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

# What models actually say, mapped to our labels.
_SYNONYMS: dict[str, ElementKind] = {
    "button": ElementKind.BUTTON,
    "btn": ElementKind.BUTTON,
    "link": ElementKind.LINK,
    "hyperlink": ElementKind.LINK,
    "input": ElementKind.INPUT,
    "textbox": ElementKind.INPUT,
    "text box": ElementKind.INPUT,
    "text field": ElementKind.INPUT,
    "textfield": ElementKind.INPUT,
    "search bar": ElementKind.INPUT,
    "search box": ElementKind.INPUT,
    "dropdown": ElementKind.INPUT,
    "select": ElementKind.INPUT,
    "checkbox": ElementKind.CHECKBOX,
    "radio": ElementKind.CHECKBOX,
    "radio button": ElementKind.CHECKBOX,
    "switch": ElementKind.CHECKBOX,
    "toggle": ElementKind.CHECKBOX,
    "icon": ElementKind.ICON,
    "text": ElementKind.TEXT,
    "label": ElementKind.TEXT,
    "heading": ElementKind.TEXT,
    "title": ElementKind.TEXT,
    "paragraph": ElementKind.TEXT,
    "image": ElementKind.IMAGE,
    "img": ElementKind.IMAGE,
    "logo": ElementKind.IMAGE,
    "picture": ElementKind.IMAGE,
    "photo": ElementKind.IMAGE,
}


@dataclass
class ParseResult:
    elements: list[Element]
    items_seen: int = 0
    valid_json: bool = True
    truncated: bool = False
    dropped: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1


def parse_elements(text: str, frame_width: float, frame_height: float) -> ParseResult:
    """Parse a JSON list of {"bbox_2d": [x1, y1, x2, y2], "label": ..., "text": ...}.

    Coordinates are in a frame of frame_width x frame_height (the resized image
    the model saw, or 1000 x 1000 for models that normalise).
    """
    items, valid, truncated = _load_items(text)
    result = ParseResult(elements=[], valid_json=valid, truncated=truncated)
    result.items_seen = len(items)

    candidates: list[Element] = []
    for item in items:
        element = _to_element(item, len(candidates), frame_width, frame_height, result)
        if element is not None:
            candidates.append(element)

    for element in candidates:
        if len(result.elements) >= MAX_ELEMENTS:
            result.drop("over_limit")
            continue
        if any(
            e.kind == element.kind and e.box.iou(element.box) >= DUPLICATE_IOU
            for e in result.elements
        ):
            result.drop("duplicate")
            continue
        result.elements.append(element.model_copy(update={"id": len(result.elements)}))
    return result


def _load_items(text: str) -> tuple[list[Any], bool, bool]:
    fenced = _FENCE.search(text)
    body = fenced.group(1) if fenced else text
    start = body.find("[")
    if start == -1:
        return [], False, False
    body = body[start:]

    end = body.rfind("]")
    if end != -1:
        try:
            data = json.loads(body[: end + 1])
            return (data if isinstance(data, list) else []), isinstance(data, list), False
        except json.JSONDecodeError:
            pass

    # Cut off mid-array (token limit): keep every object that closed.
    cut = body.rfind("}")
    while cut != -1:
        try:
            data = json.loads(body[: cut + 1] + "]")
            return (data if isinstance(data, list) else []), False, True
        except json.JSONDecodeError:
            cut = body.rfind("}", 0, cut)
    return [], False, True


def _to_element(item: Any, idx: int, fw: float, fh: float, result: ParseResult) -> Element | None:
    if not isinstance(item, dict):
        result.drop("not_an_object")
        return None

    kind = _SYNONYMS.get(str(item.get("label", "")).strip().lower())
    if kind is None:
        result.drop("unknown_label")
        return None

    coords = item.get("bbox_2d", item.get("bbox"))
    if not (
        isinstance(coords, list)
        and len(coords) == 4
        and all(isinstance(c, int | float) and not isinstance(c, bool) for c in coords)
    ):
        result.drop("bad_box")
        return None

    x1, y1, x2, y2 = (float(c) for c in coords)
    x1, x2 = sorted((_clamp(x1 / fw), _clamp(x2 / fw)))
    y1, y2 = sorted((_clamp(y1 / fh), _clamp(y2 / fh)))
    if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
        result.drop("degenerate_box")
        return None

    raw_text = item.get("text")
    label_text = _CONTROL.sub("", raw_text).strip()[:200] if isinstance(raw_text, str) else None

    try:
        return Element(
            id=idx,
            kind=kind,
            box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
            text=label_text or None,
        )
    except ValidationError:
        result.drop("invalid")
        return None


def _clamp(v: float) -> float:
    return min(1.0, max(0.0, v))
