import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from prism.evaluation.dataset import load
from prism.evaluation.run import compare, oracle, rescore, true_findings
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


def test_rescore_scores_boxes_and_labels_separately(tmp_path: Path) -> None:
    write_page(tmp_path)
    # The model finds the low-contrast text but calls it a link, and finds the icon.
    raw = json.dumps(
        [
            {"bbox_2d": [16, 16, 140, 42], "label": "link"},
            {"bbox_2d": [20, 120, 31, 131], "label": "icon"},
        ]
    )
    log = tmp_path / "run.pages.jsonl"
    log.write_text(
        json.dumps(
            {
                "image": "p.png",
                "seconds": 2.0,
                "new_tokens": 40,
                "input_size": [400, 200],
                "raw": raw,
            }
        )
        + "\n"
    )
    summary = rescore(tmp_path, log, frame="pixels")
    det = summary["detection"]
    assert det["boxes_any_label"]["tp"] == 2
    assert det["boxes_and_labels"]["tp"] == 1
    assert summary["confusion_truth_to_predicted"]["text"]["link"] == 1
    # The contrast finding lands on the right element despite the wrong label.
    assert summary["rules_end_to_end"]["text-contrast"]["tp"] == 1
    assert summary["tokens_per_second"] == 20.0


def test_rescore_merges_saved_ocr_lines(tmp_path: Path) -> None:
    write_page(tmp_path)
    # The model found only the icon; OCR found both text lines. No OCR engine needed:
    # the lines saved in the log are used.
    raw = json.dumps([{"bbox_2d": [20, 120, 31, 131], "label": "icon"}])
    ocr = [[0.04, 0.08, 0.35, 0.21, "Low contrast"], [0.04, 0.28, 0.375, 0.41, "Readable text"]]
    log = tmp_path / "run.pages.jsonl"
    record = {"image": "p.png", "seconds": 1.0, "new_tokens": 10, "input_size": [400, 200]}
    log.write_text(json.dumps({**record, "raw": raw, "ocr": ocr}) + "\n")

    without = rescore(tmp_path, log, frame="pixels")
    with_ocr = rescore(tmp_path, log, frame="pixels", text="ocr")
    assert without["detection"]["boxes_any_label"]["tp"] == 1
    assert with_ocr["detection"]["boxes_any_label"]["tp"] == 3
    assert with_ocr["rules_end_to_end"]["text-contrast"]["tp"] == 1
    assert with_ocr["text_source"] == "ocr"


def test_load_skip_and_limit_select_a_slice(tmp_path: Path) -> None:
    write_page(tmp_path)
    row = (tmp_path / "labels.jsonl").read_text()
    (tmp_path / "labels.jsonl").write_text(
        "".join(row.replace('"p.png"', f'"p{i}.png"') for i in range(5))
    )
    assert [p.image.name for p in load(tmp_path, limit=2, skip=3)] == ["p3.png", "p4.png"]
    assert [p.image.name for p in load(tmp_path, skip=4)] == ["p4.png"]


def test_compare_is_paired_on_shared_pages(tmp_path: Path) -> None:
    write_page(tmp_path)
    record = {"image": "p.png", "seconds": 1.0, "input_size": [400, 200]}
    icon = {"bbox_2d": [20, 120, 31, 131], "label": "icon"}
    low_contrast = {"bbox_2d": [16, 16, 140, 42], "label": "text"}
    a = tmp_path / "a.pages.jsonl"
    b = tmp_path / "b.pages.jsonl"
    a.write_text(json.dumps({**record, "new_tokens": 30, "raw": json.dumps([icon])}) + "\n")
    # b also logged a page a never ran; it must be left out of the comparison.
    b.write_text(
        json.dumps({**record, "new_tokens": 50, "raw": json.dumps([icon, low_contrast])})
        + "\n"
        + json.dumps({**record, "image": "other.png", "new_tokens": 9, "raw": "[]"})
        + "\n"
    )
    result = compare(tmp_path, [a], [b], resamples=50)
    assert result["pages"] == 1
    assert result["median_new_tokens"] == {"a": 30, "b": 50}
    contrast = result["f1"]["text-contrast"]
    assert (contrast["a_f1"], contrast["b_f1"]) == (0.0, 1.0)
    # One page: every resample is that page, so the interval is the difference itself.
    assert contrast["ci95_low"] == contrast["ci95_high"] == contrast["difference"] == 1.0
