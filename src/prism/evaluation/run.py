"""Benchmarks.

    prism-eval synth  --out eval/data/synth --count 300
    prism-eval oracle --data eval/data/synth           # audit rules on true boxes (CPU)
    prism-eval detect --data eval/data/synth --quant nf4 --max-side 1280   # needs a GPU

Every command prints a Markdown table and writes a JSON summary to --results.
"""

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from prism.a11y import audit
from prism.a11y.contrast import LARGE_TEXT, NORMAL_TEXT, contrast_ratio, estimate
from prism.a11y.target_size import MIN_CSS_PX
from prism.domain import Element, ElementKind, Finding

from .dataset import Page, load
from .metrics import Counts, DetectionScore, match, percentile

_TEXTUAL = {ElementKind.TEXT, ElementKind.BUTTON, ElementKind.LINK}


def true_findings(page: Page) -> set[tuple[str, int]]:
    """(rule, truth element id) pairs a perfect auditor would report."""
    out = set()
    css_w, css_h = page.width / page.dpr, page.height / page.dpr
    for t in page.elements:
        el = t.element
        if el.kind in _TEXTUAL and t.fg and t.bg and t.font_px:
            large = t.font_px >= 24 or (bool(t.bold) and t.font_px >= 18.66)
            if contrast_ratio(t.fg, t.bg) < (LARGE_TEXT if large else NORMAL_TEXT):
                out.add(("text-contrast", el.id))
        if el.kind.interactive:
            w = (el.box.x2 - el.box.x1) * css_w
            h = (el.box.y2 - el.box.y1) * css_h
            if min(w, h) < MIN_CSS_PX:
                out.add(("target-size", el.id))
        if el.kind in (ElementKind.INPUT, ElementKind.CHECKBOX) and t.has_visible_label is False:
            out.add(("visible-label", el.id))
    return out


def _finding_counts(
    predicted: list[Finding], truth: set[tuple[str, int]], to_truth: dict[int, int]
) -> dict[str, Counts]:
    counts: dict[str, Counts] = defaultdict(Counts)
    hits = set()
    for f in predicted:
        key = (f.rule, to_truth.get(f.element_id, -1))
        if key in truth:
            counts[f.rule].tp += 1
            hits.add(key)
        else:
            counts[f.rule].fp += 1
    for rule, _ in truth - hits:
        counts[rule].fn += 1
    return counts


def oracle(data: Path, limit: int | None) -> dict[str, Any]:
    errors: list[float] = []
    agree = total = clear_agree = clear_total = 0
    rules: dict[str, Counts] = defaultdict(Counts)
    for page in load(data, limit):
        with Image.open(page.image) as img:
            image = img.convert("RGB")
        truth_elements = [t.element for t in page.elements]
        result = audit(image, truth_elements, page.dpr)
        per_rule = _finding_counts(
            result.findings, true_findings(page), {e.id: e.id for e in truth_elements}
        )
        for rule, c in per_rule.items():
            rules[rule].add(c)

        for t in page.elements:
            if t.element.kind in _TEXTUAL and t.fg and t.bg:
                est = estimate(image, t.element.box)
                if est is None:
                    continue
                true = contrast_ratio(t.fg, t.bg)
                errors.append(abs(est.ratio - true) / true)
                same = (est.ratio >= NORMAL_TEXT) == (true >= NORMAL_TEXT)
                total += 1
                agree += same
                # Cases not within 10% of the threshold, where a human wouldn't hesitate.
                if abs(true - NORMAL_TEXT) > 0.1 * NORMAL_TEXT:
                    clear_total += 1
                    clear_agree += same

    return {
        "contrast": {
            "elements": total,
            "median_rel_error": round(statistics.median(errors), 3) if errors else None,
            "p90_rel_error": round(percentile(errors, 0.9), 3),
            "agreement_at_4.5": round(agree / total, 3) if total else None,
            "agreement_outside_10pct_band": (
                round(clear_agree / clear_total, 3) if clear_total else None
            ),
        },
        "rules": {r: _counts(c) for r, c in sorted(rules.items())},
    }


def detect(data: Path, limit: int | None, model: str, quant: str, max_side: int) -> dict[str, Any]:
    import torch

    from prism.vision.qwen import QwenDetector

    detector = QwenDetector(model, quant=quant, max_side=max_side)  # type: ignore[arg-type]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    det = DetectionScore()
    rules: dict[str, Counts] = defaultdict(Counts)
    latencies: list[float] = []
    valid = truncated = pages = 0
    for page in load(data, limit):
        with Image.open(page.image) as img:
            image = img.convert("RGB")
        started = time.perf_counter()
        run = detector.run(image)
        result = audit(image, run.parsed.elements, page.dpr)
        latencies.append(time.perf_counter() - started)
        pages += 1
        valid += run.parsed.valid_json
        truncated += run.parsed.truncated

        truth: list[Element] = [t.element for t in page.elements]
        score, pairs = match(run.parsed.elements, truth)
        det.add(score)
        per_rule = _finding_counts(result.findings, true_findings(page), dict(pairs))
        for rule, c in per_rule.items():
            rules[rule].add(c)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    return {
        "model": detector.version,
        "pages": pages,
        "valid_json_rate": valid / pages if pages else None,
        "truncated_rate": truncated / pages if pages else None,
        "latency_s": {"p50": percentile(latencies, 0.5), "p95": percentile(latencies, 0.95)},
        "peak_vram_gb": peak_gb,
        "detection": {
            "overall": _counts(det.overall),
            "by_kind": {k.value: _counts(c) for k, c in sorted(det.by_kind.items())},
        },
        "rules_end_to_end": {r: _counts(c) for r, c in sorted(rules.items())},
    }


def _counts(c: Counts) -> dict[str, float | int]:
    return {
        "tp": c.tp,
        "fp": c.fp,
        "fn": c.fn,
        "precision": round(c.precision, 3),
        "recall": round(c.recall, 3),
        "f1": round(c.f1, 3),
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = []
    for section, rows in summary.items():
        if not isinstance(rows, dict) or not rows:
            lines.append(f"- **{section}**: {rows}")
            continue
        first = next(iter(rows.values()))
        if isinstance(first, dict):
            cols = list(first.keys())
            lines += [
                f"\n**{section}**\n",
                "| | " + " | ".join(cols) + " |",
                "|---" * (len(cols) + 1) + "|",
            ]
            for name, vals in rows.items():
                lines.append(f"| {name} | " + " | ".join(str(vals.get(c, "")) for c in cols) + " |")
        else:
            lines.append(f"\n**{section}**: " + ", ".join(f"{k} = {v}" for k, v in rows.items()))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(prog="prism-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate a synthetic labelled dataset")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--count", type=int, default=300)
    s.add_argument("--seed", type=int, default=13)
    s.add_argument("--chromium", help="path to a Chromium binary, if Playwright's isn't installed")

    for name in ("oracle", "detect"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--limit", type=int)
        p.add_argument("--results", type=Path)
        if name == "detect":
            p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
            p.add_argument("--quant", choices=["nf4", "int8", "none"], default="nf4")
            p.add_argument("--max-side", type=int, default=1280)

    args = parser.parse_args()
    if args.cmd == "synth":
        from .synth import generate

        generate(args.out, args.count, args.seed, args.chromium)
        print(f"wrote {args.count} pages to {args.out}")
        return

    if args.cmd == "oracle":
        summary = oracle(args.data, args.limit)
    else:
        summary = detect(args.data, args.limit, args.model, args.quant, args.max_side)
    summary["dataset"] = str(args.data)
    print(_markdown(summary))
    if args.results:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
