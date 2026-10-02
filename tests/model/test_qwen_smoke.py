"""Runs the real QwenDetector code on a tiny random model (CPU, a few seconds)."""

from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("transformers")

from prism.vision.base import DetectionCanceled
from prism.vision.qwen import QwenDetector
from tests.model.tiny_qwen import build

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def detector(tmp_path_factory: pytest.TempPathFactory) -> QwenDetector:
    path = build(tmp_path_factory.mktemp("tiny-qwen"))
    return QwenDetector(str(path), quant="none", max_side=112, max_new_tokens=8)


def test_generates_and_parses(detector: QwenDetector) -> None:
    run = detector.run(Image.new("RGB", (300, 200), "white"))
    assert run.new_tokens <= 8
    assert run.input_size[0] <= 112
    assert run.parsed.elements == []  # random weights; the point is that nothing crashes


def test_downscales_to_max_side(detector: QwenDetector) -> None:
    run = detector.run(Image.new("RGB", (2400, 1200), "white"))
    assert max(run.input_size) <= 112


def test_stop_flag_cancels(detector: QwenDetector) -> None:
    with pytest.raises(DetectionCanceled):
        detector.detect(Image.new("RGB", (100, 100)), should_stop=lambda: True)


def test_version_names_model_and_settings(detector: QwenDetector) -> None:
    assert detector.version.endswith(":none:112:prompt-v2")


def test_quantisation_needs_gpu(tmp_path: Path) -> None:
    import torch

    if torch.cuda.is_available():
        pytest.skip("has a GPU")
    with pytest.raises(RuntimeError, match="CUDA"):
        QwenDetector(str(tmp_path), quant="nf4")


def test_profile_splits_prefill_and_decode(detector: QwenDetector) -> None:
    run = detector.run(Image.new("RGB", (200, 100), "white"), profile=True)
    assert run.prompt_tokens > 0
    assert run.preprocess_s is not None and run.prefill_s is not None
    if run.new_tokens > 1:
        assert run.decode_tokens_per_s is not None and run.decode_tokens_per_s > 0


def test_timing_benchmark_runs_end_to_end(tmp_path: Path) -> None:
    from prism.evaluation.gpu import timing
    from tests.unit.test_evaluation import write_page

    model = build(tmp_path / "tiny")
    data = tmp_path / "data"
    data.mkdir()
    write_page(data)
    summary = timing(data, 1, 2, str(model), "none", 112, "v4", "model", max_new_tokens=8)
    assert summary["repeats"] == 2 and len(summary["runs"]) == 2
    assert summary["median"]["prompt_tokens"] > 0
    assert summary["model"].endswith("prompt-v4")


def test_detect_benchmark_logs_pages_and_ocr(tmp_path: Path) -> None:
    import json

    pytest.importorskip("rapidocr")
    from prism.evaluation.gpu import detect
    from tests.unit.test_evaluation import write_page

    model = build(tmp_path / "tiny")
    data = tmp_path / "data"
    data.mkdir()
    write_page(data)
    log = tmp_path / "run.pages.jsonl"
    summary = detect(data, None, str(model), "none", 112, log, 8, "v4", "ocr")
    assert summary["pages"] == 1 and summary["text_source"] == "ocr"
    [record] = [json.loads(line) for line in log.read_text().splitlines()]
    # The random model finds nothing; OCR still finds the two lines of text, and
    # they reach the scorer (the fixture's hand-drawn boxes are loose, so not
    # every line clears IoU 0.5).
    assert len(record["ocr"]) == 2
    assert summary["detection"]["boxes_any_label"]["tp"] >= 1
