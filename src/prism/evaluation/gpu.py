"""Benchmarks that load the model: `detect` and `timing`. Need the worker extra
and, for anything but the tiny test checkpoint, a GPU."""

import contextlib
import json
import shutil
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from prism.vision.hybrid import merge
from prism.vision.prompts import DEFAULT_PROMPT

from .dataset import load
from .metrics import percentile
from .run import PipelineScore, line_record, ocr_reader


def detect(
    data: Path,
    limit: int | None,
    model: str,
    quant: str,
    max_side: int,
    pages_log: Path | None = None,
    max_new_tokens: int = 2048,
    prompt: str = DEFAULT_PROMPT,
    text: str = "model",
) -> dict[str, Any]:
    import torch

    from prism.vision.qwen import QwenDetector

    reader = ocr_reader() if text == "ocr" else None

    detector = QwenDetector(
        model,
        quant=quant,  # type: ignore[arg-type]
        max_side=max_side,
        max_new_tokens=max_new_tokens,
        prompt=prompt,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    score = PipelineScore()
    latencies: list[float] = []
    valid = truncated = hit_limit = pages = errors = 0
    total = sum(1 for _ in load(data, limit))
    log = pages_log.open("w", encoding="utf-8") if pages_log else None
    try:
        for page in load(data, limit):
            with Image.open(page.image) as img:
                image = img.convert("RGB")
            started = time.perf_counter()
            lines = reader.read(image) if reader is not None else None
            try:
                run = detector.run(image)
            except torch.cuda.OutOfMemoryError:
                # One oversized page shouldn't end an unattended run.
                errors += 1
                torch.cuda.empty_cache()
                print(f"[{pages + errors}/{total}] {page.image.name}: out of memory, skipped")
                continue
            seconds = time.perf_counter() - started
            latencies.append(seconds)
            pages += 1
            valid += run.parsed.valid_json
            truncated += run.parsed.truncated
            hit_limit += run.new_tokens >= max_new_tokens
            elements = run.parsed.elements if lines is None else merge(run.parsed.elements, lines)
            strict = score.add(page, image, elements)

            print(
                f"[{pages + errors}/{total}] {page.image.name}: {seconds:.1f}s, "
                f"{run.new_tokens} tokens, {len(elements)}/{len(page.elements)} "
                f"elements, box F1 so far {score.boxes.f1:.3f}",
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
                    "matched": strict.overall.tp,
                    "predicted": len(elements),
                    "truth": len(page.elements),
                    "raw": run.raw_text,
                }
                if lines is not None:
                    record["ocr"] = [line_record(e) for e in lines]
                log.write(json.dumps(record) + "\n")
                log.flush()
    finally:
        if log is not None:
            log.close()

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    return {
        "model": detector.version,
        "text_source": text,
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
        **score.summary(),
    }


class _GpuSampler:
    """Polls nvidia-smi in the background while a page runs.

    Laptop GPUs change clocks with temperature and power mode, which is the
    first suspect when the same model runs at different speeds.
    """

    FIELDS = ("clocks.sm", "power.draw", "temperature.gpu")

    def __init__(self, interval_s: float = 0.5) -> None:
        self._exe = shutil.which("nvidia-smi")
        self._interval = interval_s
        self._samples: list[tuple[float, ...]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_GpuSampler":
        if self._exe is not None:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _poll(self) -> None:
        assert self._exe is not None
        query = [self._exe, f"--query-gpu={','.join(self.FIELDS)}", "--format=csv,noheader,nounits"]
        while not self._stop.is_set():
            # A slow or odd reading is skipped, never allowed to end the benchmark.
            with contextlib.suppress(ValueError, OSError, subprocess.SubprocessError):
                out = subprocess.run(query, capture_output=True, text=True, timeout=5, check=True)  # noqa: S603
                self._samples.append(tuple(float(v) for v in out.stdout.split(",")[:3]))
            self._stop.wait(self._interval)

    def means(self) -> dict[str, float] | None:
        if not self._samples:
            return None
        cols = zip(*self._samples, strict=True)
        names = ("sm_clock_mhz", "power_w", "temp_c")
        return {n: round(statistics.mean(c), 1) for n, c in zip(names, cols, strict=True)}


def timing(
    data: Path,
    n_pages: int,
    repeats: int,
    model: str,
    quant: str,
    max_side: int,
    prompt: str,
    text: str,
    max_new_tokens: int = 2048,
) -> dict[str, Any]:
    """Where the time goes: preprocessing, prefill, decoding, OCR.

    Runs the same pages several times after a warm-up, so differences between
    repeats come from the machine, not the input.
    """
    import torch
    import transformers

    from prism.vision.qwen import QwenDetector

    detector = QwenDetector(
        model,
        quant=quant,  # type: ignore[arg-type]
        max_side=max_side,
        max_new_tokens=max_new_tokens,
        prompt=prompt,
    )
    reader = ocr_reader() if text == "ocr" else None
    pages = list(load(data, n_pages))
    images = []
    for page in pages:
        with Image.open(page.image) as img:
            images.append(img.convert("RGB"))

    # First call pays for CUDA context setup, kernel loading and cuBLAS handles.
    detector.run(images[0], profile=True)
    if reader is not None:
        reader.read(images[0])

    rows: list[dict[str, Any]] = []
    for rep in range(repeats):
        for page, image in zip(pages, images, strict=True):
            ocr_s = None
            if reader is not None:
                t = time.perf_counter()
                reader.read(image)
                ocr_s = time.perf_counter() - t
            with _GpuSampler() as gpu:
                run = detector.run(image, profile=True)
            row = {
                "repeat": rep,
                "image": page.image.name,
                "prompt_tokens": run.prompt_tokens,
                "new_tokens": run.new_tokens,
                "preprocess_s": round(run.preprocess_s or 0.0, 3),
                "prefill_s": round(run.prefill_s or 0.0, 3),
                "decode_tokens_per_s": round(run.decode_tokens_per_s or 0.0, 2),
                "model_s": round(run.seconds, 2),
                "ocr_s": round(ocr_s, 3) if ocr_s is not None else None,
                "gpu": gpu.means(),
            }
            rows.append(row)
            print(
                f"[{rep + 1}/{repeats}] {page.image.name}: {run.seconds:.1f}s, "
                f"prefill {row['prefill_s']:.2f}s, {run.new_tokens} tokens at "
                f"{row['decode_tokens_per_s']:.1f}/s, gpu {row['gpu']}",
                flush=True,
            )

    def med(key: str) -> float:
        return round(float(statistics.median(r[key] for r in rows)), 3)

    speeds = [r["decode_tokens_per_s"] for r in rows]
    per_page: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        per_page[r["image"]].append(r["decode_tokens_per_s"])
    spread = [statistics.stdev(v) / statistics.mean(v) for v in per_page.values() if len(v) > 1]
    decode_share = [
        (r["new_tokens"] - 1) / r["decode_tokens_per_s"] / r["model_s"]
        for r in rows
        if r["decode_tokens_per_s"] > 0
    ]
    clocks = [(r["gpu"]["sm_clock_mhz"], r["decode_tokens_per_s"]) for r in rows if r["gpu"]]
    clock_corr = None
    if len(clocks) > 2 and len({c for c, _ in clocks}) > 1:
        clock_corr = round(statistics.correlation(*zip(*clocks, strict=True)), 2)

    gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    return {
        "model": detector.version,
        "text_source": text,
        "pages": len(pages),
        "repeats": repeats,
        "environment": {
            "gpu": gpu_name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
        },
        "median": {
            "prompt_tokens": med("prompt_tokens"),
            "new_tokens": med("new_tokens"),
            "preprocess_s": med("preprocess_s"),
            "prefill_s": med("prefill_s"),
            "decode_tokens_per_s": med("decode_tokens_per_s"),
            "model_s": med("model_s"),
            **({"ocr_s": med("ocr_s")} if reader is not None else {}),
        },
        "decode_share_of_model_time": round(statistics.median(decode_share), 3)
        if decode_share
        else None,
        "decode_tokens_per_s_range": [round(min(speeds), 1), round(max(speeds), 1)],
        "same_page_spread_cv": round(statistics.median(spread), 3) if spread else None,
        "sm_clock_vs_speed_correlation": clock_corr,
        "runs": rows,
    }
