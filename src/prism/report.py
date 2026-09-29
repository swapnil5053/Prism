"""Standalone HTML report: one file with the annotated screenshot inlined."""

import base64
from html import escape
from io import BytesIO

from PIL import Image, ImageDraw

from prism.domain import AnalysisResult

_COLOURS = {"serious": (200, 30, 30), "moderate": (215, 120, 0), "minor": (90, 90, 90)}
_RANK = {"minor": 0, "moderate": 1, "serious": 2}

_CSS = """
body { font: 15px/1.5 system-ui, sans-serif; color: #1b1b1b; max-width: 960px;
       margin: 2rem auto; padding: 0 1rem; }
h1 { font-size: 1.4rem; margin-bottom: .25rem; }
.meta { color: #555; margin-top: 0; }
img { max-width: 100%; border: 1px solid #ccc; }
table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
th, td { text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #ddd;
         vertical-align: top; }
th { background: #f4f4f4; }
.serious { color: #b3261e; font-weight: 600; }
.moderate { color: #9a5b00; }
.minor { color: #555; }
"""


def annotate(image: Image.Image, result: AnalysisResult) -> bytes:
    """Outline every element that has a finding, coloured by its worst severity."""
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    worst: dict[int, str] = {}
    for f in result.findings:
        if _RANK[f.severity] >= _RANK.get(worst.get(f.element_id, "minor"), 0):
            worst[f.element_id] = f.severity
    width = max(2, round(max(img.size) / 500))
    by_id = {e.id: e for e in result.elements}
    for element_id, severity in worst.items():
        el = by_id.get(element_id)
        if el is not None:
            x1, y1, x2, y2 = el.box.to_pixels(img.width, img.height)
            draw.rectangle((x1, y1, x2, y2), outline=_COLOURS[severity], width=width)
            draw.text((x1 + 2, max(0, y1 - 12)), str(element_id), fill=_COLOURS[severity])
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_html(
    title: str, created: str, result: AnalysisResult, image: Image.Image, model: str | None
) -> str:
    png = base64.b64encode(annotate(image, result)).decode()
    by_id = {e.id: e for e in result.elements}
    rows = []
    for f in sorted(result.findings, key=lambda f: (-_RANK[f.severity], f.element_id)):
        el = by_id.get(f.element_id)
        what = f"{el.kind.value}" + (f" “{el.text}”" if el and el.text else "") if el else "?"
        rows.append(
            "<tr>"
            f"<td>{f.element_id}</td>"
            f"<td>{escape(what)}</td>"
            f'<td class="{f.severity}">{f.severity}</td>'
            f"<td>{escape(f.wcag)}</td>"
            f"<td>{escape(f.message)}</td>"
            "</tr>"
        )
    table = (
        "<table><thead><tr><th>#</th><th>Element</th><th>Severity</th><th>WCAG</th>"
        f"<th>Problem</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
        if rows
        else "<p>No problems found by the automated checks.</p>"
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>Prism report: {escape(title)}</title><style>{_CSS}</style></head><body>"
        f"<h1>{escape(title)}</h1>"
        f"<p class='meta'>Score {result.score:.0f}/100 · {len(result.elements)} elements · "
        f"{len(result.findings)} findings · {escape(created)}"
        f"{' · ' + escape(model) if model else ''}</p>"
        f"<img alt='Screenshot with problem areas outlined' src='data:image/png;base64,{png}'>"
        f"{table}"
        "<p class='meta'>Automated checks estimate contrast and sizes from pixels. "
        "They don't replace testing with real assistive technology.</p>"
        "</body></html>"
    )
