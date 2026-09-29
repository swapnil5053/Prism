"""Inputs without a visible label (WCAG 2.2 SC 3.3.2), by layout.

A label counts if a text element sits just left of the input on the same row,
or just above it and horizontally overlapping. Placeholder text inside the
input doesn't count as a label: it disappears as soon as someone types.
"""

from prism.domain import Element, ElementKind, Finding


def check(elements: list[Element], width: int, height: int) -> list[Finding]:
    texts = [e for e in elements if e.kind is ElementKind.TEXT]
    findings = []
    for el in elements:
        if el.kind not in (ElementKind.INPUT, ElementKind.CHECKBOX):
            continue
        if any(_labels(t, el, width, height) for t in texts):
            continue
        has_placeholder = el.kind is ElementKind.INPUT and bool(el.text)
        findings.append(
            Finding(
                rule="visible-label",
                wcag="3.3.2",
                severity="moderate" if has_placeholder else "serious",
                element_id=el.id,
                message=(
                    "Input has placeholder text but no visible label."
                    if has_placeholder
                    else "No visible label found for this control."
                ),
            )
        )
    return findings


def _labels(text: Element, control: Element, width: int, height: int) -> bool:
    t, c = text.box, control.box
    ch = (c.y2 - c.y1) * height
    # Same row, to the left (or right, for checkboxes and radios).
    v_overlap = min(t.y2, c.y2) - max(t.y1, c.y1)
    if v_overlap > 0:
        gap_left = (c.x1 - t.x2) * width
        gap_right = (t.x1 - c.x2) * width
        if 0 <= gap_left <= 2 * ch:
            return True
        if control.kind is ElementKind.CHECKBOX and 0 <= gap_right <= 2 * ch:
            return True
    # Directly above.
    h_overlap = min(t.x2, c.x2) - max(t.x1, c.x1)
    gap_above = (c.y1 - t.y2) * height
    return h_overlap > 0 and 0 <= gap_above <= ch
