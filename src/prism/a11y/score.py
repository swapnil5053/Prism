"""A 0-100 score: 100 minus the severity-weighted share of elements with problems.

Each element counts once, at its worst finding, so ten findings on one button
don't outweigh ten separate broken buttons. Elements no rule applies to (plain
images, icons that aren't targets) are left out of the denominator.
"""

from prism.domain import Element, ElementKind, Finding

WEIGHTS = {"serious": 1.0, "moderate": 0.5, "minor": 0.2}
_CHECKED = {
    ElementKind.TEXT,
    ElementKind.BUTTON,
    ElementKind.LINK,
    ElementKind.INPUT,
    ElementKind.CHECKBOX,
    ElementKind.ICON,
}


def score(elements: list[Element], findings: list[Finding]) -> float:
    checked = sum(1 for e in elements if e.kind in _CHECKED)
    if checked == 0:
        return 100.0
    worst: dict[int, float] = {}
    for f in findings:
        worst[f.element_id] = max(worst.get(f.element_id, 0.0), WEIGHTS[f.severity])
    penalty = sum(worst.values()) / checked
    return round(max(0.0, 100.0 * (1 - penalty)), 1)
