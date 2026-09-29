import json

import pytest

from prism.domain import ElementKind
from prism.vision.parse import MAX_ELEMENTS, parse_elements


def item(box: list[float], label: str = "button", text: str = "OK") -> dict[str, object]:
    return {"bbox_2d": box, "label": label, "text": text}


def test_plain_json_in_pixel_frame() -> None:
    raw = json.dumps([item([100, 50, 300, 100])])
    result = parse_elements(raw, frame_width=1000, frame_height=500)
    assert result.valid_json and not result.truncated
    [el] = result.elements
    assert el.kind is ElementKind.BUTTON
    assert el.text == "OK"
    assert (el.box.x1, el.box.y1, el.box.x2, el.box.y2) == (0.1, 0.1, 0.3, 0.2)


def test_markdown_fence_and_chatter() -> None:
    raw = (
        'Sure! Here are the elements:\n```json\n[{"bbox_2d": [0, 0, 10, 10], "label": "icon"}]\n```'
    )
    assert [e.kind for e in parse_elements(raw, 100, 100).elements] == [ElementKind.ICON]


def test_truncated_output_keeps_complete_items() -> None:
    raw = json.dumps([item([0, 0, 10, 10]), item([20, 20, 40, 40])])[:-30]
    result = parse_elements(raw, 100, 100)
    assert result.truncated
    assert len(result.elements) == 1


def test_no_json_at_all() -> None:
    result = parse_elements("I cannot see any interface elements.", 100, 100)
    assert result.elements == []
    assert not result.valid_json


def test_label_synonyms() -> None:
    raw = json.dumps(
        [
            item([0, 0, 10, 10], "Text Field"),
            item([20, 0, 30, 10], "hyperlink"),
            item([40, 0, 50, 10], "toggle"),
            item([60, 0, 70, 10], "logo"),
        ]
    )
    kinds = [e.kind for e in parse_elements(raw, 100, 100).elements]
    assert kinds == [ElementKind.INPUT, ElementKind.LINK, ElementKind.CHECKBOX, ElementKind.IMAGE]


def test_unknown_label_dropped_and_counted() -> None:
    result = parse_elements(json.dumps([item([0, 0, 10, 10], "carousel")]), 100, 100)
    assert result.elements == []
    assert result.dropped == {"unknown_label": 1}


@pytest.mark.parametrize(
    "box",
    [[0, 0, 10], [0, 0, "10", 10], None, [True, 0, 10, 10], [5, 5, 5, 5]],
)
def test_bad_boxes_dropped(box: object) -> None:
    raw = json.dumps([{"bbox_2d": box, "label": "button"}])
    assert parse_elements(raw, 100, 100).elements == []


def test_out_of_range_and_swapped_boxes_are_fixed() -> None:
    raw = json.dumps([item([150, 80, -20, 10])])
    [el] = parse_elements(raw, 100, 100).elements
    assert (el.box.x1, el.box.y1, el.box.x2, el.box.y2) == (0.0, 0.1, 1.0, 0.8)


def test_duplicates_removed_and_ids_renumbered() -> None:
    raw = json.dumps(
        [
            item([0, 0, 50, 50]),
            item([1, 1, 50, 50]),  # same button again
            item([1, 1, 50, 50], "text"),  # same place, different kind: kept
            item([60, 60, 90, 90]),
        ]
    )
    result = parse_elements(raw, 100, 100)
    assert [e.id for e in result.elements] == [0, 1, 2]
    assert result.dropped == {"duplicate": 1}


def test_element_cap() -> None:
    raw = json.dumps([item([2 * i, 0, 2 * i + 2, 1]) for i in range(MAX_ELEMENTS + 5)])
    result = parse_elements(raw, 700, 1)
    assert len(result.elements) == MAX_ELEMENTS


def test_text_is_cleaned_not_trusted() -> None:
    raw = json.dumps([item([0, 0, 10, 10], text="<img src=x onerror=alert(1)>\x00" + "a" * 500)])
    [el] = parse_elements(raw, 100, 100).elements
    # Stored as plain data; the client renders it with textContent.
    assert el.text is not None
    assert "\x00" not in el.text
    assert len(el.text) == 200


def test_non_list_json() -> None:
    assert parse_elements('{"bbox_2d": [0, 0, 1, 1]}', 100, 100).elements == []
