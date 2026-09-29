from collections import defaultdict
from dataclasses import dataclass, field

from prism.domain import Element, ElementKind


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class DetectionScore:
    overall: Counts = field(default_factory=Counts)
    by_kind: dict[ElementKind, Counts] = field(default_factory=lambda: defaultdict(Counts))

    def add(self, other: "DetectionScore") -> None:
        self.overall.add(other.overall)
        for kind, counts in other.by_kind.items():
            self.by_kind[kind].add(counts)


def match(
    predicted: list[Element], truth: list[Element], iou_threshold: float = 0.5
) -> tuple[DetectionScore, list[tuple[int, int]]]:
    """Greedy one-to-one matching of same-kind boxes, highest IoU first.

    Returns the scores and the (predicted id, truth id) pairs that matched.
    """
    candidates = [
        (p.box.iou(t.box), pi, ti)
        for pi, p in enumerate(predicted)
        for ti, t in enumerate(truth)
        if p.kind == t.kind
    ]
    candidates.sort(reverse=True)
    used_p: set[int] = set()
    used_t: set[int] = set()
    pairs = []
    for iou, pi, ti in candidates:
        if iou < iou_threshold:
            break
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        pairs.append((predicted[pi].id, truth[ti].id))

    score = DetectionScore()
    for pi, p in enumerate(predicted):
        c = score.by_kind[p.kind]
        if pi in used_p:
            c.tp += 1
            score.overall.tp += 1
        else:
            c.fp += 1
            score.overall.fp += 1
    for ti, t in enumerate(truth):
        if ti not in used_t:
            score.by_kind[t.kind].fn += 1
            score.overall.fn += 1
    return score, pairs


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)
