import pytest

from prism.a11y import audit, labels, target_size
from prism.a11y.score import score
from prism.domain import Box, Element, ElementKind, Finding

W, H = 1000, 800


def el(
    i: int, kind: ElementKind, x1: int, y1: int, x2: int, y2: int, text: str | None = None
) -> Element:
    return Element(id=i, kind=kind, box=Box(x1=x1 / W, y1=y1 / H, x2=x2 / W, y2=y2 / H), text=text)


class TestTargetSize:
    def test_big_enough(self) -> None:
        assert target_size.check([el(0, ElementKind.BUTTON, 0, 0, 100, 40)], W, H, 1) == []

    def test_small_icon_is_serious(self) -> None:
        [f] = target_size.check([el(0, ElementKind.ICON, 0, 0, 14, 14)], W, H, 1)
        assert f.severity == "serious"
        assert f.measured == 14

    def test_dpr_halves_css_size(self) -> None:
        button = el(0, ElementKind.BUTTON, 0, 0, 40, 40)
        assert target_size.check([button], W, H, 1) == []
        [f] = target_size.check([button], W, H, 2)
        assert f.measured == 20
        assert f.severity == "moderate"

    def test_links_are_minor(self) -> None:
        [f] = target_size.check([el(0, ElementKind.LINK, 0, 0, 80, 16)], W, H, 1)
        assert f.severity == "minor"

    def test_text_is_not_a_target(self) -> None:
        assert target_size.check([el(0, ElementKind.TEXT, 0, 0, 5, 5)], W, H, 1) == []


class TestLabels:
    def test_label_left_of_input(self) -> None:
        elements = [
            el(0, ElementKind.TEXT, 10, 100, 90, 130),
            el(1, ElementKind.INPUT, 100, 95, 400, 135),
        ]
        assert labels.check(elements, W, H) == []

    def test_label_above_input(self) -> None:
        elements = [
            el(0, ElementKind.TEXT, 100, 60, 180, 85),
            el(1, ElementKind.INPUT, 100, 95, 400, 135),
        ]
        assert labels.check(elements, W, H) == []

    def test_placeholder_only(self) -> None:
        [f] = labels.check([el(0, ElementKind.INPUT, 100, 95, 400, 135, text="Email")], W, H)
        assert f.severity == "moderate"
        assert "placeholder" in f.message

    def test_far_away_text_does_not_count(self) -> None:
        elements = [
            el(0, ElementKind.TEXT, 10, 500, 90, 530),
            el(1, ElementKind.INPUT, 100, 95, 400, 135),
        ]
        [f] = labels.check(elements, W, H)
        assert f.severity == "serious"

    def test_checkbox_label_on_the_right(self) -> None:
        elements = [
            el(0, ElementKind.CHECKBOX, 10, 100, 30, 120),
            el(1, ElementKind.TEXT, 40, 100, 200, 120),
        ]
        assert labels.check(elements, W, H) == []


class TestScore:
    def test_no_findings(self) -> None:
        assert score([el(0, ElementKind.BUTTON, 0, 0, 50, 50)], []) == 100

    def test_nothing_checkable(self) -> None:
        assert score([el(0, ElementKind.IMAGE, 0, 0, 50, 50)], []) == 100

    def test_worst_finding_per_element(self) -> None:
        elements = [el(i, ElementKind.BUTTON, 0, 0, 50, 50) for i in range(4)]
        findings = [
            Finding(rule="a", wcag="x", severity="serious", element_id=0, message=""),
            Finding(rule="b", wcag="x", severity="minor", element_id=0, message=""),
            Finding(rule="a", wcag="x", severity="moderate", element_id=1, message=""),
        ]
        # (1.0 + 0.5) / 4 elements
        assert score(elements, findings) == pytest.approx(62.5)


def test_audit_combines_rules() -> None:
    from PIL import Image

    img = Image.new("RGB", (W, H), "white")
    elements = [el(0, ElementKind.ICON, 0, 0, 12, 12), el(1, ElementKind.INPUT, 100, 95, 400, 135)]
    result = audit(img, elements, dpr=1)
    assert {f.rule for f in result.findings} == {"target-size", "visible-label"}
    assert 0 <= result.score < 100
