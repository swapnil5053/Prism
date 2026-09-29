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


def test_version_names_model_and_settings(detector: QwenDetector, tmp_path: Path) -> None:
    assert detector.version.endswith(":none:112")


def test_quantisation_needs_gpu(tmp_path: Path) -> None:
    import torch

    if torch.cuda.is_available():
        pytest.skip("has a GPU")
    with pytest.raises(RuntimeError, match="CUDA"):
        QwenDetector(str(tmp_path), quant="nf4")
