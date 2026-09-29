import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from prism.evaluation.dataset import load
from prism.evaluation.run import oracle, true_findings
from prism.evaluation.synth import build_page


def write_page(root: Path) -> None:
    img = Image.new("RGB", (400, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=16)
    draw.text((20, 20), "Low contrast", fill=(200, 200, 200), font=font)
    draw.text((20, 60), "Readable text", fill=(20, 20, 20), font=font)
    draw.rectangle((20, 120, 30, 130), fill=(0, 0, 0))  # a 10px icon
    img.save(root / "p.png")

    def box(x1: int, y1: int, x2: int, y2: int) -> list[float]:
        return [x1 / 400, y1 / 200, x2 / 400, y2 / 200]

    row = {
        "image": "p.png",
        "width": 400,
        "height": 200,
        "dpr": 1,
        "elements": [
            {
                "kind": "text",
                "box": box(16, 16, 140, 42),
                "fg": "#c8c8c8",
                "bg": "#ffffff",
                "font_px": 16,
                "bold": False,
            },
            {
                "kind": "text",
                "box": box(16, 56, 150, 82),
                "fg": "#141414",
                "bg": "#ffffff",
                "font_px": 16,
                "bold": False,
            },
            {"kind": "icon", "box": box(20, 120, 31, 131)},
            {"kind": "input", "box": box(200, 20, 380, 50), "has_visible_label": False},
        ],
    }
    (root / "labels.jsonl").write_text(json.dumps(row) + "\n")


def test_true_findings(tmp_path: Path) -> None:
    write_page(tmp_path)
    [page] = load(tmp_path)
    # The input is 180x30 CSS px, so only its missing label counts.
    assert true_findings(page) == {
        ("text-contrast", 0),
        ("target-size", 2),
        ("visible-label", 3),
    }


def test_oracle_scores_rules(tmp_path: Path) -> None:
    write_page(tmp_path)
    summary = oracle(tmp_path, limit=None)
    assert summary["rules"]["text-contrast"]["recall"] == 1.0
    assert summary["rules"]["target-size"]["tp"] == 1
    assert summary["rules"]["visible-label"]["tp"] == 1
    assert summary["contrast"]["elements"] == 2


def test_synthetic_pages_are_reproducible() -> None:
    a = build_page(random.Random(5))
    b = build_page(random.Random(5))
    c = build_page(random.Random(6))
    assert a == b
    assert a != c
    assert 'data-kind="button"' in a[0] or 'data-kind="link"' in a[0]
