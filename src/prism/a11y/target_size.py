"""Target size (WCAG 2.2 SC 2.5.8, AA): interactive targets at least 24x24 CSS px.

The spacing exception (small targets with enough empty space around them) and
the inline-link exception aren't modelled; links get a lower severity because
many of them are inline.
"""

from prism.domain import Element, ElementKind, Finding

MIN_CSS_PX = 24
TINY_CSS_PX = 16


def check(elements: list[Element], width: int, height: int, dpr: float) -> list[Finding]:
    findings = []
    for el in elements:
        if not el.kind.interactive:
            continue
        w = (el.box.x2 - el.box.x1) * width / dpr
        h = (el.box.y2 - el.box.y1) * height / dpr
        smallest = min(w, h)
        if smallest >= MIN_CSS_PX:
            continue
        if el.kind is ElementKind.LINK:
            severity = "minor"
        elif smallest < TINY_CSS_PX:
            severity = "serious"
        else:
            severity = "moderate"
        findings.append(
            Finding(
                rule="target-size",
                wcag="2.5.8",
                severity=severity,
                element_id=el.id,
                message=f"Target is {w:.0f}x{h:.0f} CSS px; needs at least 24x24.",
                measured=round(smallest, 1),
                required=MIN_CSS_PX,
            )
        )
    return findings
