"""On-disk dataset format shared by the synthetic generator and any converted
real dataset: a folder of screenshots plus labels.jsonl, one page per line."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from prism.domain import Box, Element, ElementKind


@dataclass
class TruthElement:
    element: Element
    # Optional ground truth for the audit rules. None = unknown for this item.
    fg: tuple[int, int, int] | None = None
    bg: tuple[int, int, int] | None = None
    font_px: float | None = None
    bold: bool | None = None
    has_visible_label: bool | None = None


@dataclass
class Page:
    image: Path
    width: int
    height: int
    dpr: float
    elements: list[TruthElement]


def _rgb(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    v = value.lstrip("#")
    return (int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16))


def load(root: Path, limit: int | None = None) -> Iterator[Page]:
    with (root / "labels.jsonl").open(encoding="utf-8") as fh:
        for n, line in enumerate(fh):
            if limit is not None and n >= limit:
                return
            row = json.loads(line)
            elements = []
            for i, e in enumerate(row["elements"]):
                x1, y1, x2, y2 = e["box"]
                element = Element(
                    id=i,
                    kind=ElementKind(e["kind"]),
                    box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                    text=e.get("text") or None,
                )
                elements.append(
                    TruthElement(
                        element=element,
                        fg=_rgb(e.get("fg")),
                        bg=_rgb(e.get("bg")),
                        font_px=e.get("font_px"),
                        bold=e.get("bold"),
                        has_visible_label=e.get("has_visible_label"),
                    )
                )
            yield Page(
                image=root / row["image"],
                width=row["width"],
                height=row["height"],
                dpr=row.get("dpr", 1.0),
                elements=elements,
            )
