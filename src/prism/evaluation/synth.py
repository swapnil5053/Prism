"""Synthetic UI screenshots with exact labels.

Random pages are built from a handful of components (nav bars, forms, cards),
rendered in headless Chromium, and labelled from the DOM: every box, colour,
font size and label relationship is known exactly, so both the detector and
the pixel-based audit rules can be scored against real ground truth.
"""

import json
import random
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from prism.a11y.contrast import contrast_ratio

WORDS = [
    "account",
    "billing",
    "cancel",
    "continue",
    "dashboard",
    "email",
    "export",
    "files",
    "help",
    "home",
    "invite",
    "language",
    "members",
    "notifications",
    "overview",
    "password",
    "payment",
    "plan",
    "privacy",
    "profile",
    "projects",
    "reports",
    "save",
    "search",
    "security",
    "settings",
    "share",
    "sign",
    "start",
    "storage",
    "team",
    "update",
    "upload",
    "usage",
    "workspace",
]
ICONS = ["✕", "☰", "⚙", "?", "+", "⋯", "✎", "↻"]
CONTRAST_TARGETS = [1.6, 2.2, 2.8, 3.3, 4.0, 4.7, 6.0, 9.0, 14.0]


@dataclass
class Palette:
    page: tuple[int, int, int]
    nav: tuple[int, int, int]
    accent: tuple[int, int, int]


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _pick_fg(rng: random.Random, bg: tuple[int, int, int]) -> tuple[int, int, int]:
    """A text colour whose contrast with bg is near a randomly chosen target."""
    target = rng.choice(CONTRAST_TARGETS)
    hue = rng.choice([(1, 1, 1), (1, 0.6, 0.5), (0.5, 0.7, 1), (0.6, 1, 0.7)])
    best, best_err = (0, 0, 0), float("inf")
    for level in range(0, 256, 3):
        rgb = (
            min(255, round(level * hue[0])),
            min(255, round(level * hue[1])),
            min(255, round(level * hue[2])),
        )
        err = abs(contrast_ratio(rgb, bg) - target)
        if err < best_err:
            best, best_err = rgb, err
    return best


def _words(rng: random.Random, lo: int, hi: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(lo, hi))).capitalize()


def _palette(rng: random.Random) -> Palette:
    dark = rng.random() < 0.25
    page = (
        (rng.randint(15, 40),) * 3 if dark else (rng.randint(235, 255), rng.randint(235, 255), 255)
    )
    nav = (rng.randint(20, 60), rng.randint(20, 80), rng.randint(60, 140))
    accent = rng.choice([(26, 115, 232), (0, 120, 90), (180, 40, 60), (110, 60, 200)])
    return Palette(page=page, nav=nav, accent=accent)


def _text(rng: random.Random, bg: tuple[int, int, int], size: int, content: str) -> str:
    fg = _pick_fg(rng, bg)
    weight = 700 if rng.random() < 0.3 else 400
    return (
        f'<div data-kind="text" style="color:{_hex(fg)};font-size:{size}px;'
        f'font-weight:{weight};white-space:nowrap;display:inline-block;margin:6px 0">'
        f"{escape(content)}</div>"
    )


def _button(rng: random.Random, p: Palette) -> str:
    pad_v, pad_h = rng.choice([(2, 6), (4, 10), (8, 16), (12, 24)])
    size = rng.choice([11, 13, 14, 16])
    fg = _pick_fg(rng, p.accent)
    return (
        f'<button data-kind="button" style="background:{_hex(p.accent)};color:{_hex(fg)};'
        f"border:0;border-radius:6px;padding:{pad_v}px {pad_h}px;font-size:{size}px;"
        f'margin:8px 8px 8px 0">{escape(_words(rng, 1, 2))}</button>'
    )


def _icon(rng: random.Random, bg: tuple[int, int, int]) -> str:
    side = rng.choice([14, 18, 22, 28, 36, 44])
    fg = _pick_fg(rng, bg)
    return (
        f'<button data-kind="icon" aria-label="icon" style="width:{side}px;height:{side}px;'
        f"padding:0;border:0;background:transparent;color:{_hex(fg)};"
        f'font-size:{max(10, side - 6)}px;line-height:1;margin:0 6px">{rng.choice(ICONS)}</button>'
    )


def _nav(rng: random.Random, p: Palette) -> str:
    links = "".join(
        f'<a data-kind="link" href="#" style="color:{_hex(_pick_fg(rng, p.nav))};'
        f'margin:0 12px;font-size:{rng.choice([13, 14, 16])}px;text-decoration:none">'
        f"{escape(_words(rng, 1, 1))}</a>"
        for _ in range(rng.randint(2, 5))
    )
    logo = (
        f'<div data-kind="image" style="width:36px;height:36px;border-radius:8px;'
        f'background:{_hex(p.accent)};margin-right:16px"></div>'
    )
    return (
        f'<nav style="display:flex;align-items:center;padding:12px 20px;'
        f'background:{_hex(p.nav)}">{logo}{links}'
        f'<span style="flex:1"></span>{_icon(rng, p.nav)}</nav>'
    )


def _field(rng: random.Random, p: Palette, n: int) -> str:
    labelled = rng.random() < 0.75
    placeholder = _words(rng, 1, 2) if (not labelled or rng.random() < 0.4) else ""
    height = rng.choice([20, 28, 36, 44])
    label = _text(rng, p.page, 14, _words(rng, 1, 2)) + "<br>" if labelled else ""
    return (
        f'<div style="margin:10px 0">{label}'
        f'<input data-kind="input" data-labelled="{str(labelled).lower()}" '
        f'placeholder="{escape(placeholder)}" style="width:{rng.randint(180, 360)}px;'
        f'height:{height}px;border:1px solid #888;border-radius:4px;padding:0 8px"></div>'
    )


def _checkbox(rng: random.Random, p: Palette) -> str:
    labelled = rng.random() < 0.8
    label = (
        f'<span style="margin-left:8px">{_text(rng, p.page, 14, _words(rng, 2, 4))}</span>'
        if labelled
        else ""
    )
    return (
        f'<div style="display:flex;align-items:center;margin:8px 0">'
        f'<input type="checkbox" data-kind="checkbox" data-labelled="{str(labelled).lower()}" '
        f'style="width:{rng.choice([13, 18, 24])}px;height:{rng.choice([13, 18, 24])}px;margin:0">'
        f"{label}</div>"
    )


def _form(rng: random.Random, p: Palette) -> str:
    parts = [_text(rng, p.page, rng.choice([20, 24, 28]), _words(rng, 2, 3)), "<br>"]
    parts += [_field(rng, p, i) for i in range(rng.randint(1, 3))]
    if rng.random() < 0.6:
        parts.append(_checkbox(rng, p))
    parts.append(_button(rng, p))
    return f'<section style="padding:16px 24px">{"".join(parts)}</section>'


def _cards(rng: random.Random, p: Palette) -> str:
    cards = []
    for _ in range(rng.randint(2, 3)):
        r, g, b = (min(255, c + rng.randint(-10, 10)) for c in p.page)
        card_bg = (r, g, b)
        cards.append(
            f'<div style="background:{_hex(card_bg)};border:1px solid #ccc;border-radius:8px;'
            f'padding:12px;margin:8px;width:220px">'
            f'<div data-kind="image" style="height:{rng.randint(60, 100)}px;'
            f'background:{_hex(p.accent)};opacity:.35;border-radius:4px"></div>'
            f"{_text(rng, card_bg, rng.choice([13, 15, 18]), _words(rng, 2, 3))}<br>"
            f"{_button(rng, p)}</div>"
        )
    return (
        f'<section style="display:flex;flex-wrap:wrap;padding:8px 16px">{"".join(cards)}</section>'
    )


def build_page(rng: random.Random) -> tuple[str, int, float]:
    """Return (html, viewport width, device pixel ratio)."""
    p = _palette(rng)
    mobile = rng.random() < 0.35
    body = _nav(rng, p)
    sections = [_form, _cards] if rng.random() < 0.5 else [_cards, _form]
    for make in sections[: rng.randint(1, 2)]:
        body += make(rng, p)
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><style>"
        "body{margin:0;font-family:'DejaVu Sans',Arial,sans-serif}</style></head>"
        f"<body style='background:{_hex(p.page)}'>{body}</body></html>"
    )
    return html, (390 if mobile else 1024), (2.0 if mobile else 1.0)


_EXTRACT_JS = """
() => {
  const W = document.documentElement.scrollWidth, H = document.documentElement.scrollHeight;
  const bgOf = (el) => {
    for (let n = el; n; n = n.parentElement) {
      const c = getComputedStyle(n).backgroundColor;
      if (c && !c.endsWith(', 0)') && c !== 'transparent') return c;
    }
    return 'rgb(255, 255, 255)';
  };
  return {W, H, elements: [...document.querySelectorAll('[data-kind]')].map(el => {
    const r = el.getBoundingClientRect(), s = getComputedStyle(el);
    return {
      kind: el.dataset.kind,
      box: [r.left / W, r.top / H, r.right / W, r.bottom / H],
      text: el.tagName === 'INPUT' ? (el.placeholder || '') : el.innerText.trim(),
      fg: s.color, bg: bgOf(el.tagName === 'INPUT' ? el : el.parentElement || el),
      own_bg: s.backgroundColor,
      font_px: parseFloat(s.fontSize), bold: parseInt(s.fontWeight) >= 700,
      labelled: el.dataset.labelled === undefined ? null : el.dataset.labelled === 'true',
    };
  })};
}
"""


def _css_rgb(value: str) -> str | None:
    nums = value[value.find("(") + 1 : value.find(")")].split(",")
    if len(nums) < 3:
        return None
    if len(nums) == 4 and float(nums[3]) == 0:
        return None
    r, g, b = (round(float(n)) for n in nums[:3])
    return _hex((r, g, b))


def generate(out: Path, count: int, seed: int, chromium: str | None = None) -> None:
    from playwright.sync_api import sync_playwright

    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with sync_playwright() as pw, (out / "labels.jsonl").open("w", encoding="utf-8") as labels:
        browser = pw.chromium.launch(executable_path=chromium)
        contexts: dict[tuple[int, float], Any] = {}
        for i in range(count):
            html, width, dpr = build_page(rng)
            key = (width, dpr)
            if key not in contexts:
                contexts[key] = browser.new_context(
                    viewport={"width": width, "height": 600}, device_scale_factor=dpr
                )
            page = contexts[key].new_page()
            page.set_content(html)
            data = page.evaluate(_EXTRACT_JS)
            name = f"{i:04d}.png"
            page.screenshot(path=str(out / name), full_page=True)
            page.close()

            elements = []
            for e in data["elements"]:
                x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in e["box"])
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    continue
                # A filled button's text sits on the button's own colour.
                bg = _css_rgb(e["own_bg"]) if e["kind"] == "button" else None
                elements.append(
                    {
                        "kind": e["kind"],
                        "box": [x1, y1, x2, y2],
                        "text": e["text"],
                        "fg": _css_rgb(e["fg"]),
                        "bg": bg or _css_rgb(e["bg"]),
                        "font_px": e["font_px"],
                        "bold": e["bold"],
                        "has_visible_label": e["labelled"],
                    }
                )
            row = {
                "image": name,
                "width": round(data["W"] * dpr),
                "height": round(data["H"] * dpr),
                "dpr": dpr,
                "elements": elements,
            }
            labels.write(json.dumps(row) + "\n")
        browser.close()
