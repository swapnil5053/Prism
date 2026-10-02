// Pure helpers behind the page: no DOM, so they're unit-tested on their own.

export const RULES = {
  "1.4.3": {
    name: "Text contrast",
    url: "https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum",
  },
  "2.5.8": {
    name: "Target size",
    url: "https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum",
  },
  "3.3.2": {
    name: "Visible label",
    url: "https://www.w3.org/WAI/WCAG22/Understanding/labels-or-instructions",
  },
};

export const RANK = { serious: 0, moderate: 1, minor: 2 };

/** The API's {x1, y1, x2, y2} box (0-1 of the image) as {x, y, w, h}. */
export function toBox(b) {
  return { x: b.x1, y: b.y1, w: b.x2 - b.x1, h: b.y2 - b.y1 };
}

/** Page state for an analysis or a progress event. */
export function stateOf(a) {
  switch (a.status) {
    case "queued":
      return "queued";
    case "running":
      return a.stage === "auditing" ? "checking" : "detecting";
    case "completed":
      return a.result && a.result.findings.length === 0 ? "clean" : "done";
    case "failed":
    case "canceled":
      return a.status;
    default:
      return "empty";
  }
}

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

export function statusText(state, ctx = {}) {
  switch (state) {
    case "empty":
      return "Choose a screenshot, drop one on the page, or try the example.";
    case "uploading":
      return `Uploading ${ctx.name || "file"}: ${ctx.pct || 0}%`;
    case "queued":
      return "Queued. Waiting for an analysis worker.";
    case "detecting":
      return "Detecting elements in the screenshot.";
    case "checking":
      return "Checking the elements against 3 rules.";
    case "done":
      return `Done. ${plural(ctx.findings, "finding")} in ${ctx.elements} elements.`;
    case "clean":
      return ctx.elements
        ? `Done. Nothing found in ${ctx.elements} elements.`
        : "Done, but no elements were detected, so nothing was checked.";
    case "failed":
      return `Failed. ${ctx.error || "The server could not process this image."}`;
    case "canceled":
      return "Canceled. The analysis stopped before it finished.";
    case "offline":
      return "Connection lost. Reload or retry to check on this analysis.";
    default:
      return "";
  }
}

/**
 * Contrast details for a finding: the backend's fields when present, otherwise
 * parsed from the message (results stored before the fields existed).
 */
export function contrastInfo(f) {
  if (f.fg && f.bg && f.measured != null && f.required != null) {
    return { fg: f.fg, bg: f.bg, ratio: f.measured, required: f.required };
  }
  const m =
    /([\d.]+)\s*:\s*1\s*\((#[0-9a-f]{6})\s+on\s+(#[0-9a-f]{6})\)\s*;?\s*needs\s*([\d.]+)\s*:\s*1/i.exec(
      f.message || "",
    );
  return m ? { ratio: +m[1], fg: m[2], bg: m[3], required: +m[4] } : null;
}

/**
 * Findings joined to their elements, grouped by WCAG criterion, most severe
 * first, and numbered 1..n in display order (the numbers on the screenshot).
 */
export function groupFindings(result) {
  const byId = new Map(result.elements.map((e) => [e.id, e]));
  const groups = new Map();
  for (const f of result.findings) {
    const element = byId.get(f.element_id);
    if (!element) continue;
    if (!groups.has(f.wcag)) groups.set(f.wcag, []);
    groups.get(f.wcag).push({
      ...f,
      element: { kind: element.kind, text: element.text, box: toBox(element.box) },
    });
  }
  const list = [...groups.entries()].map(([wcag, items]) => ({
    wcag,
    items: items.sort((a, b) => RANK[a.severity] - RANK[b.severity]),
  }));
  list.sort(
    (a, b) =>
      RANK[a.items[0].severity] - RANK[b.items[0].severity] || b.items.length - a.items.length,
  );
  let n = 0;
  for (const g of list) {
    for (const item of g.items) {
      item.i = n;
      item.n = ++n;
    }
  }
  return list;
}

export function severityCounts(items) {
  const counts = { serious: 0, moderate: 0, minor: 0 };
  for (const it of items) counts[it.severity] += 1;
  return counts;
}
