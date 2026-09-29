import pytest

from prism.domain import Box, Element, ElementKind
from prism.evaluation.metrics import Counts, match, percentile


def box(x: float, kind: ElementKind = ElementKind.BUTTON, i: int = 0) -> Element:
    return Element(id=i, kind=kind, box=Box(x1=x, y1=0.1, x2=x + 0.1, y2=0.2))


def test_perfect_match() -> None:
    truth = [box(0.1, i=0), box(0.5, i=1)]
    score, pairs = match(truth, truth)
    assert (score.overall.tp, score.overall.fp, score.overall.fn) == (2, 0, 0)
    assert sorted(pairs) == [(0, 0), (1, 1)]


def test_kind_must_agree() -> None:
    score, _ = match([box(0.1, ElementKind.LINK)], [box(0.1, ElementKind.BUTTON)])
    assert (score.overall.tp, score.overall.fp, score.overall.fn) == (0, 1, 1)


def test_one_prediction_matches_one_truth() -> None:
    score, _ = match([box(0.1, i=0)], [box(0.1, i=0), box(0.11, i=1)])
    assert (score.overall.tp, score.overall.fn) == (1, 1)


def test_low_iou_is_a_miss() -> None:
    score, _ = match([box(0.15)], [box(0.1)])  # IoU = 1/3
    assert score.overall.tp == 0


def test_counts() -> None:
    c = Counts(tp=3, fp=1, fn=2)
    assert c.precision == 0.75
    assert c.recall == 0.6
    assert c.f1 == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert Counts().f1 == 0


def test_percentile() -> None:
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([], 0.9) == 0
