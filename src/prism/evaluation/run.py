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


def detect(
    data: Path,
    limit: int | None,
    model: str,
    quant: str,
    max_side: int,
    pages_log: Path | None = None,
    max_new_tokens: int = 2048,
) -> dict[str, Any]:
    import torch

    from prism.vision.qwen import QwenDetector

    detector = QwenDetector(
        model,
        quant=quant,  # type: ignore[arg-type]
        max_side=max_side,
        max_new_tokens=max_new_tokens,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    det = DetectionScore()
    rules: dict[str, Counts] = defaultdict(Counts)
    latencies: list[float] = []
    valid = truncated = hit_limit = pages = errors = 0
    total = sum(1 for _ in load(data, limit))
    log = pages_log.open("w", encoding="utf-8") if pages_log else None
    try:
        for page in load(data, limit):
            with Image.open(page.image) as img:
                image = img.convert("RGB")
            started = time.perf_counter()
            try:
                run = detector.run(image)
            except torch.cuda.OutOfMemoryError:
                # One oversized page shouldn't end an unattended run.
                errors += 1
                torch.cuda.empty_cache()
                print(f"[{pages + errors}/{total}] {page.image.name}: out of memory, skipped")
                continue
            result = audit(image, run.parsed.elements, page.dpr)
            seconds = time.perf_counter() - started
            latencies.append(seconds)
            pages += 1
            valid += run.parsed.valid_json
            truncated += run.parsed.truncated
            hit_limit += run.new_tokens >= max_new_tokens

            truth: list[Element] = [t.element for t in page.elements]
            score, pairs = match(run.parsed.elements, truth)
            det.add(score)
            per_rule = _finding_counts(result.findings, true_findings(page), dict(pairs))
            for rule, c in per_rule.items():
                rules[rule].add(c)

            print(
                f"[{pages + errors}/{total}] {page.image.name}: {seconds:.1f}s, "
                f"{run.new_tokens} tokens, {len(run.parsed.elements)}/{len(truth)} elements, "
                f"running F1 {det.overall.f1:.3f}",
                flush=True,
            )
            if log is not None:
                record = {
                    "image": page.image.name,
                    "seconds": round(seconds, 2),
                    "input_size": run.input_size,
                    "new_tokens": run.new_tokens,
                    "valid_json": run.parsed.valid_json,
                    "truncated": run.parsed.truncated,
                    "dropped": run.parsed.dropped,
                    "matched": score.overall.tp,
                    "predicted": len(run.parsed.elements),
                    "truth": len(truth),
                    "raw": run.raw_text,
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
    finally:
        if log is not None:
            log.close()

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    return {
        "model": detector.version,
        "pages": pages,
        "pages_out_of_memory": errors,
        "valid_json_rate": round(valid / pages, 3) if pages else None,
        "truncated_rate": round(truncated / pages, 3) if pages else None,
        "hit_token_limit_rate": round(hit_limit / pages, 3) if pages else None,
        "latency_s": {
            "p50": round(percentile(latencies, 0.5), 2),
            "p95": round(percentile(latencies, 0.95), 2),
        },
        "peak_vram_gb": round(peak_gb, 2) if peak_gb is not None else None,
        "detection": {
            "overall": _counts(det.overall),
            **{k.value: _counts(c) for k, c in sorted(det.by_kind.items())},
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
            rows = {k: v for k, v in rows.items() if v}
            if not rows:
                continue
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
            p.add_argument("--max-new-tokens", type=int, default=2048)

    args = parser.parse_args()
    if args.cmd == "synth":
        from .synth import generate

        generate(args.out, args.count, args.seed, args.chromium)
        print(f"wrote {args.count} pages to {args.out}")
        return

    if args.cmd == "oracle":
        summary = oracle(args.data, args.limit)
    else:
        pages_log = args.results.with_suffix(".pages.jsonl") if args.results else None
        if pages_log:
            pages_log.parent.mkdir(parents=True, exist_ok=True)
        summary = detect(
            args.data,
            args.limit,
            args.model,
            args.quant,
            args.max_side,
            pages_log,
            args.max_new_tokens,
        )
    summary["dataset"] = str(args.data)
    print(_markdown(summary))
    if args.results:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
