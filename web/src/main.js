import { cancelAnalysis, getAnalysis, listAnalyses, reportUrl, uploadScreenshot } from "./api.js";
import { followAnalysis } from "./events.js";
import {
  RULES,
  contrastInfo,
  groupFindings,
  severityCounts,
  stateOf,
  statusText,
} from "./overlay.js";

const $ = (id) => document.getElementById(id);
const body = document.body;
const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
const behavior = () => (motion.matches ? "auto" : "smooth");

const TYPES = ["image/png", "image/jpeg", "image/webp"];
const MAX_BYTES = 10 * 1024 * 1024;
const BUSY = ["uploading", "queued", "detecting", "checking"];
const STOPPED = ["failed", "canceled", "offline"];
const TERMINAL = ["completed", "failed", "canceled"];
const STEP_AT = { uploading: 0, queued: 1, detecting: 2, checking: 3, done: 4, clean: 4 };
const LABEL = {
  empty: "Ready",
  uploading: "Uploading",
  queued: "Queued",
  detecting: "Detecting",
  checking: "Checking",
  done: "Done",
  clean: "Done",
  failed: "Failed",
  canceled: "Canceled",
  offline: "Offline",
};
const BUSY_LABEL = {
  queued: "Queued",
  detecting: "Detecting elements…",
  checking: "Checking rules…",
};
const STAGE_MSG = {
  failed: [
    "Analysis failed",
    (c) => c.error || "The server could not process this image.",
    "Try again",
  ],
  canceled: [
    "Analysis canceled",
    () => "Nothing was checked. Run it again or choose another file.",
    "Run again",
  ],
  offline: [
    "Connection lost",
    () => "Prism lost contact with the server. The analysis may still be running.",
    "Retry now",
  ],
};

const ui = {
  upload: $("upload"),
  file: $("file"),
  fileName: $("file-name"),
  ratio: $("ratio"),
  stateLabel: $("state-label"),
  currentName: $("current-name"),
  steps: $("steps"),
  status: $("status"),
  progress: $("progress"),
  cancel: $("cancel"),
  downloads: $("downloads"),
  reportHtml: $("report-html"),
  reportJson: $("report-json"),
  scroller: $("stage-scroll"),
  figure: $("figure"),
  shot: $("shot"),
  boxes: $("boxes"),
  busy: $("stage-busy"),
  shotMeta: $("shot-meta"),
  stageTitle: $("stage-title"),
  stageText: $("stage-text"),
  stageAction: $("stage-action"),
  panel: $("panel"),
  summary: $("summary"),
  findings: $("findings"),
  history: $("history"),
  example: $("example"),
  toggleBoxes: $("toggle-boxes"),
};

// The analysis on screen, the file it came from (for "Try again"), and its live connections.
const cur = {
  analysis: null,
  name: null,
  file: null,
  dpr: 1,
  stopFollowing: null,
  abortUpload: null,
  items: [],
  selected: -1,
  lastStep: -1,
};

/* ---------- helpers ---------- */

// Everything from the server, model output included, goes in as text, never HTML.
function h(tag, attrs, ...kids) {
  const n = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? "" : String(v));
    }
  }
  for (const c of kids.flat()) if (c != null && c !== false) n.append(c);
  return n;
}

const SEV_PATH = {
  serious: "M6 0.5 11.5 6 6 11.5 0.5 6Z",
  moderate: "M6 1 11.5 11H0.5Z",
  minor: "M6 1.5a4.5 4.5 0 1 0 0 9 4.5 4.5 0 0 0 0-9Z",
};
function sevIcon(s) {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("class", "sev-icon");
  svg.setAttribute("viewBox", "0 0 12 12");
  svg.setAttribute("aria-hidden", "true");
  const p = document.createElementNS(ns, "path");
  p.setAttribute("d", SEV_PATH[s] || SEV_PATH.minor);
  svg.append(p);
  return svg;
}
const fmt = (n) => String(+(+n).toFixed(2));
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
const fmtTime = (t) => new Date(t).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
const isHex = (s) => /^#[0-9a-f]{6}$/i.test(s);

/* ---------- page state ---------- */

function render(state, ctx = {}) {
  body.dataset.state = state;
  ui.stateLabel.textContent = LABEL[state];
  ui.currentName.textContent = cur.name || "No file";

  let at = -1;
  if (state in STEP_AT) at = cur.lastStep = STEP_AT[state];
  else if (STOPPED.includes(state)) at = cur.lastStep;
  else cur.lastStep = -1;
  [...ui.steps.children].forEach((li, i) => {
    li.classList.toggle("is-done", i < at);
    li.classList.toggle("is-current", i === at);
    li.classList.toggle("is-stopped", i === at && STOPPED.includes(state));
    if (i === at && at < 4) li.setAttribute("aria-current", "step");
    else li.removeAttribute("aria-current");
  });

  ui.status.textContent = statusText(state, { ...ctx, name: cur.name });
  const busy = BUSY.includes(state);
  ui.progress.hidden = !busy;
  if (state === "uploading") ui.progress.value = ctx.pct || 0;
  else ui.progress.removeAttribute("value");
  ui.cancel.hidden = !busy;
  ui.downloads.hidden = !(state === "done" || state === "clean");
  ui.busy.textContent =
    state === "uploading" ? `Uploading ${ctx.pct || 0}%` : BUSY_LABEL[state] || "";

  if (STAGE_MSG[state]) {
    const [title, text, action] = STAGE_MSG[state];
    ui.stageTitle.textContent = title;
    ui.stageText.textContent = text(ctx);
    ui.stageAction.textContent = action;
    // Retrying needs the file (failed/canceled) or an analysis to reconnect to (offline).
    ui.stageAction.hidden = state === "offline" ? !cur.analysis : !cur.file;
  }

  if (state !== "done" && state !== "clean") {
    ui.summary.replaceChildren();
    ui.findings.replaceChildren();
    ui.boxes.replaceChildren();
    if (busy) {
      ui.summary.append(
        h("p", { class: "summary-empty", text: "Results appear here when checking finishes." }),
      );
    } else if (STOPPED.includes(state)) {
      ui.summary.append(h("p", { class: "summary-empty", text: "No results for this analysis." }));
    }
  }
}

/* ---------- image ---------- */

function showImage(src) {
  ui.boxes.replaceChildren();
  ui.shotMeta.textContent = "";
  if (!src) {
    ui.shot.removeAttribute("src");
    ui.shot.alt = "";
    return;
  }
  ui.shot.alt = `Screenshot ${cur.name || ""}`.trim();
  if (ui.shot.getAttribute("src") !== src) ui.shot.src = src;
  else if (ui.shot.complete) onImageLoad();
}

function onImageLoad() {
  if (!ui.shot.naturalWidth) return;
  ui.shotMeta.textContent = `${ui.shot.naturalWidth} × ${ui.shot.naturalHeight} px · ${cur.dpr}×`;
  applyZoom();
}
ui.shot.addEventListener("load", onImageLoad);

let zoom = "fit";
function applyZoom() {
  const actual = zoom === "actual";
  ui.figure.classList.toggle("is-actual", actual);
  // 100% means CSS pixels: a 2x phone screenshot shows at half its pixel width.
  ui.shot.style.width =
    actual && ui.shot.naturalWidth ? `${ui.shot.naturalWidth / (cur.dpr || 1)}px` : "";
  placeMarkers();
}

/* ---------- results ---------- */

function showResult(analysis) {
  const result = analysis.result;
  const groups = groupFindings(result);
  cur.items = groups.flatMap((g) => g.items);
  cur.selected = -1;
  const ctx = { findings: cur.items.length, elements: result.elements.length };
  render(cur.items.length ? "done" : "clean", ctx);
  renderSummary(result);
  renderFindings(groups, result);
  renderBoxes();
  setDownloads(analysis);
}

function renderSummary(result) {
  const counts = severityCounts(cur.items);
  ui.summary.replaceChildren(
    h(
      "div",
      null,
      h("span", { class: "score-label", text: "Score" }),
      h(
        "p",
        { class: "score-value" },
        h("span", { class: "score-num", text: String(Math.round(result.score)) }),
        h("span", { class: "score-max", text: "/100" }),
      ),
    ),
    h(
      "div",
      null,
      h("p", {
        class: "tally",
        text: `${plural(cur.items.length, "finding")} in ${result.elements.length} elements`,
      }),
      h(
        "ul",
        { class: "sev-tally" },
        Object.entries(counts).map(([s, c]) =>
          h("li", { class: `sev-${s}${c ? "" : " is-zero"}` }, sevIcon(s), `${c} ${s}`),
        ),
      ),
    ),
    h("p", { class: "score-note", text: "Weighted by severity; 100 means nothing found." }),
  );
}

function swatch(hex, role) {
  const chip = h("span", { class: "chip", "aria-hidden": "true" });
  chip.style.backgroundColor = hex;
  return h(
    "span",
    { class: "sw" },
    chip,
    h("span", { text: hex }),
    h("span", { class: "vh", text: ` ${role}` }),
  );
}

function findingEl(it) {
  const c = it.wcag === "1.4.3" ? contrastInfo(it) : null;
  const el = it.element;
  let detail;
  if (c && isHex(c.fg) && isHex(c.bg)) {
    const sample = h("span", { class: "sample", "aria-hidden": "true", text: "Aa" });
    sample.style.color = c.fg;
    sample.style.backgroundColor = c.bg;
    detail = h(
      "span",
      { class: "contrast" },
      sample,
      h(
        "span",
        { class: "pair" },
        swatch(c.fg, "text"),
        h("span", { class: "on", text: "on" }),
        swatch(c.bg, "background"),
      ),
      h(
        "span",
        { class: "ratio" },
        h("strong", { text: `${fmt(c.ratio)} : 1` }),
        h("span", { class: "needs", text: ` needs ${fmt(c.required)} : 1` }),
      ),
    );
  } else {
    detail = h("span", { class: "f-msg", text: it.message });
  }
  return h(
    "li",
    null,
    h(
      "button",
      {
        type: "button",
        class: `finding sev-${it.severity}`,
        id: `f-${it.i}`,
        "data-i": it.i,
        "aria-pressed": "false",
      },
      h("span", { class: "num", "aria-hidden": "true", text: String(it.n) }),
      h(
        "span",
        { class: "f-body" },
        h(
          "span",
          { class: "f-top" },
          h("span", { class: "vh", text: `Finding ${it.n}, ` }),
          h("span", { class: "sev" }, sevIcon(it.severity), it.severity),
          h("span", { class: "f-el", text: el.text ? `${el.kind} “${el.text}”` : el.kind }),
        ),
        detail,
      ),
    ),
  );
}

function renderFindings(groups, result) {
  ui.findings.replaceChildren();
  if (!groups.length) {
    ui.findings.append(
      h(
        "div",
        { class: "clean" },
        result.elements.length
          ? h("p", { class: "clean-title", text: "No problems found." })
          : h("p", { class: "clean-title is-empty", text: "Nothing to check." }),
        h("p", {
          text: result.elements.length
            ? `Prism checked ${result.elements.length} elements for text contrast, target size and visible labels.`
            : "No controls or text were detected in this screenshot. That usually means the image isn't a UI, or the detector missed everything; the score doesn't mean the page passes.",
        }),
      ),
    );
    return;
  }
  for (const g of groups) {
    const rule = RULES[g.wcag] || { name: g.items[0].rule, url: null };
    const id = `g-${g.wcag.replace(/\./g, "-")}`;
    ui.findings.append(
      h(
        "section",
        { class: "group", "aria-labelledby": id },
        h(
          "h3",
          { class: "group-head", id },
          h("span", { text: rule.name }),
          rule.url
            ? h(
                "a",
                { class: "group-wcag", href: rule.url, target: "_blank", rel: "noopener" },
                `WCAG ${g.wcag}`,
              )
            : h("span", { class: "group-wcag", text: `WCAG ${g.wcag}` }),
          h(
            "span",
            { class: "group-count" },
            String(g.items.length),
            h("span", { class: "vh", text: g.items.length === 1 ? " finding" : " findings" }),
          ),
        ),
        h("ol", { class: "group-list" }, g.items.map(findingEl)),
      ),
    );
  }
}

function renderBoxes() {
  ui.boxes.replaceChildren();
  ui.boxes.classList.remove("has-selection");
  const seen = new Map();
  for (const it of cur.items) {
    const bx = it.element.box;
    const box = h("div", {
      class: `box sev-${it.severity}`,
      id: `b-${it.i}`,
      "aria-hidden": "true",
    });
    Object.assign(box.style, {
      left: `${bx.x * 100}%`,
      top: `${bx.y * 100}%`,
      width: `${bx.w * 100}%`,
      height: `${bx.h * 100}%`,
    });
    // Several findings on one element: offset their markers so all stay clickable.
    const key = [bx.x, bx.y].map((v) => v.toFixed(3)).join();
    const k = seen.get(key) || 0;
    seen.set(key, k + 1);
    const el = it.element;
    const marker = h("button", {
      type: "button",
      class: `marker sev-${it.severity}`,
      id: `m-${it.i}`,
      "data-i": it.i,
      "data-k": k,
      "aria-label": `Finding ${it.n}, ${it.severity}: ${it.message} (${el.kind}${el.text ? ` “${el.text}”` : ""})`,
      text: String(it.n),
    });
    ui.boxes.append(box, marker);
  }
  placeMarkers();
}

function placeMarkers() {
  const W = ui.shot.clientWidth;
  const H = ui.shot.clientHeight;
  if (!W) return;
  for (const mk of ui.boxes.querySelectorAll(".marker")) {
    const bx = cur.items[+mk.dataset.i].element.box;
    const k = +mk.dataset.k;
    const below = bx.y * H < 28;
    const right = bx.x * W + 26 * (k + 1) > W;
    mk.classList.toggle("is-below", below);
    mk.classList.toggle("is-right", right);
    mk.style.left = `${(right ? bx.x + bx.w : bx.x) * 100}%`;
    mk.style.top = `${(below ? bx.y + bx.h : bx.y) * 100}%`;
    mk.style.marginLeft = k ? `${(right ? -26 : 26) * k}px` : "";
  }
}

/* ---------- selection ---------- */

function hover(i, on) {
  for (const p of ["b-", "m-", "f-"]) {
    const n = $(p + i);
    if (n) n.classList.toggle("is-hover", on);
  }
}

function reveal(container, target) {
  const c = container.getBoundingClientRect();
  const t = target.getBoundingClientRect();
  const scrollable =
    container.scrollHeight > container.clientHeight + 1 ||
    container.scrollWidth > container.clientWidth + 1;
  if (scrollable && getComputedStyle(container).overflowY !== "visible") {
    container.scrollTo({
      top: container.scrollTop + (t.top - c.top) - Math.max(16, (c.height - t.height) / 2),
      left: container.scrollLeft + (t.left - c.left) - Math.max(16, (c.width - t.width) / 2),
      behavior: behavior(),
    });
    if (c.top < 0 || c.top > window.innerHeight - 80) {
      window.scrollBy({ top: c.top - 8, behavior: behavior() });
    }
  } else if (t.top < 8 || t.bottom > window.innerHeight - 8) {
    window.scrollBy({ top: t.top - window.innerHeight / 3, behavior: behavior() });
  }
}

function select(i, from) {
  if (cur.selected === i) i = -1;
  cur.selected = i;
  document.querySelectorAll(".is-selected").forEach((n) => n.classList.remove("is-selected"));
  ui.findings
    .querySelectorAll(".finding")
    .forEach((n) => n.setAttribute("aria-pressed", String(+n.dataset.i === i)));
  ui.boxes.classList.toggle("has-selection", i >= 0);
  if (i < 0) return;
  const box = $(`b-${i}`);
  const mk = $(`m-${i}`);
  const fd = $(`f-${i}`);
  [box, mk, fd].forEach((n) => n && n.classList.add("is-selected"));
  if (from === "marker") {
    reveal(ui.panel, fd);
    fd.focus({ preventScroll: true });
  } else if (box) {
    reveal(ui.scroller, box);
  }
}

function on(container, selector, handlers) {
  for (const [type, fn] of Object.entries(handlers)) {
    container.addEventListener(type, (e) => {
      const target = e.target.closest(selector);
      if (target) fn(+target.dataset.i, e);
    });
  }
}
on(ui.findings, ".finding", {
  click: (i) => select(i, "list"),
  mouseover: (i) => hover(i, true),
  mouseout: (i) => hover(i, false),
  focusin: (i) => hover(i, true),
  focusout: (i) => hover(i, false),
});
on(ui.boxes, ".marker", {
  click: (i) => select(i, "marker"),
  mouseover: (i) => hover(i, true),
  mouseout: (i) => hover(i, false),
  focusin: (i) => hover(i, true),
  focusout: (i) => hover(i, false),
});
ui.findings.addEventListener("keydown", (e) => {
  const list = [...ui.findings.querySelectorAll(".finding")];
  const at = list.indexOf(document.activeElement);
  if (at < 0) return;
  const to = { ArrowDown: at + 1, ArrowUp: at - 1, Home: 0, End: list.length - 1 }[e.key];
  if (to == null) return;
  e.preventDefault();
  list[Math.max(0, Math.min(list.length - 1, to))].focus();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && cur.selected >= 0) select(cur.selected);
});

document.querySelectorAll("[data-zoom]").forEach((btn) =>
  btn.addEventListener("click", () => {
    zoom = btn.dataset.zoom;
    document
      .querySelectorAll("[data-zoom]")
      .forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
    applyZoom();
  }),
);
ui.toggleBoxes.addEventListener("change", () =>
  ui.figure.classList.toggle("hide-boxes", !ui.toggleBoxes.checked),
);
if ("ResizeObserver" in window) new ResizeObserver(placeMarkers).observe(ui.shot);

/* ---------- downloads ---------- */

function setDownloads(analysis) {
  const base = (cur.name || "screenshot").replace(/\.[^.]+$/, "");
  ui.reportHtml.href = reportUrl(analysis.id, "html");
  ui.reportHtml.download = `${base}-report.html`;
  ui.reportJson.href = reportUrl(analysis.id, "json");
  ui.reportJson.download = `${base}-report.json`;
}

/* ---------- history ---------- */

async function refreshHistory() {
  let items;
  try {
    items = (await listAnalyses(10)).items;
  } catch {
    ui.history.replaceChildren(
      h("tr", null, h("td", { colspan: 5, class: "recent-empty", text: "Couldn't load history." })),
    );
    return;
  }
  if (!items.length) {
    ui.history.replaceChildren(
      h("tr", null, h("td", { colspan: 5, class: "recent-empty", text: "No analyses yet." })),
    );
    return;
  }
  ui.history.replaceChildren(
    ...items.map((a) =>
      h(
        "tr",
        null,
        h(
          "th",
          { scope: "row" },
          h("button", {
            type: "button",
            class: "link-btn",
            text: a.original_filename || "Screenshot",
            onclick: () => {
              location.hash = a.id;
            },
          }),
        ),
        h("td", null, h("span", { class: `hstatus-${a.status}`, text: a.status })),
        h("td", { class: "num-cell", text: a.finding_count ?? "—" }),
        h("td", { class: "num-cell", text: a.score == null ? "—" : String(Math.round(a.score)) }),
        h("td", null, h("time", { datetime: a.created_at, text: fmtTime(a.created_at) })),
      ),
    ),
  );
}

/* ---------- following an analysis ---------- */

function stopActivity() {
  cur.stopFollowing?.();
  cur.stopFollowing = null;
  cur.abortUpload?.();
  cur.abortUpload = null;
}

function finish(analysis) {
  cur.analysis = analysis;
  if (analysis.status === "completed") showResult(analysis);
  else render(stateOf(analysis), { error: analysis.error });
  refreshHistory();
}

function open(analysis) {
  stopActivity();
  cur.analysis = analysis;
  cur.name = analysis.original_filename || "Screenshot";
  cur.dpr = analysis.device_pixel_ratio || 1;
  showImage(analysis.image_url);
  if (TERMINAL.includes(analysis.status)) {
    finish(analysis);
    return;
  }
  render(stateOf(analysis));
  cur.stopFollowing = followAnalysis(analysis.id, {
    onEvent: async (event) => {
      if (!TERMINAL.includes(event.status)) {
        render(stateOf(event));
        return;
      }
      cur.stopFollowing = null;
      try {
        finish(event.snapshot ?? (await getAnalysis(analysis.id)));
      } catch (err) {
        render("failed", { error: err.message });
      }
    },
    onGiveUp: () => {
      cur.stopFollowing = null;
      render("offline");
    },
  });
}

async function openId(id) {
  try {
    open(await getAnalysis(id));
  } catch (err) {
    cur.name = null;
    showImage(null);
    render("failed", { error: err.status === 404 ? "That analysis doesn't exist." : err.message });
  }
}

function openFromHash() {
  const id = location.hash.slice(1);
  if (/^[0-9a-f-]{36}$/.test(id) && id !== cur.analysis?.id) openId(id);
}

/* ---------- starting an analysis ---------- */

async function startFile(file) {
  if (!file) return;
  stopActivity();
  cur.analysis = null;
  cur.file = file;
  cur.name = file.name;
  cur.dpr = +ui.ratio.value || 1;
  ui.fileName.textContent = file.name;
  if (!TYPES.includes(file.type)) {
    showImage(null);
    render("failed", { error: "This file type isn't supported. Use PNG, JPEG or WebP." });
    return;
  }
  if (file.size > MAX_BYTES) {
    showImage(null);
    render("failed", { error: "This file is larger than 10 MB." });
    return;
  }
  // Show the screenshot straight away; the server's copy replaces it once uploaded.
  showImage(URL.createObjectURL(file));
  render("uploading", { pct: 0 });
  const upload = uploadScreenshot(file, cur.dpr, (pct) => {
    if (body.dataset.state === "uploading") render("uploading", { pct });
  });
  cur.abortUpload = upload.abort;
  let analysis;
  try {
    analysis = await upload.promise;
  } catch (err) {
    cur.abortUpload = null;
    if (err.name === "AbortError") render("canceled");
    else render("failed", { error: err.message });
    return;
  }
  cur.abortUpload = null;
  history.replaceState(null, "", `#${analysis.id}`);
  open(analysis);
  refreshHistory();
}

// The header's picker waits for Analyze (so the pixel ratio can be set first). The
// "Choose file" buttons on the stage have no Analyze button next to them, so a file
// picked there starts right away.
let startOnPick = false;
document.querySelectorAll(".stage-scroll label[for='file']").forEach((label) =>
  label.addEventListener("click", () => {
    startOnPick = true;
  }),
);
document.querySelector(".appbar label[for='file']").addEventListener("click", () => {
  startOnPick = false;
});
ui.file.addEventListener("change", () => {
  const file = ui.file.files[0];
  ui.fileName.textContent = file ? file.name : "No file chosen";
  if (file && startOnPick) startFile(file);
  startOnPick = false;
});
ui.upload.addEventListener("submit", (e) => {
  e.preventDefault();
  const file = ui.file.files[0];
  if (!file) {
    ui.status.textContent = "Choose a screenshot first.";
    ui.file.focus();
    return;
  }
  startFile(file);
});
ui.example.addEventListener("click", async () => {
  ui.ratio.value = "1";
  try {
    const res = await fetch("/example.png");
    if (!res.ok) throw new Error("The example image is missing.");
    startFile(new File([await res.blob()], "example.png", { type: "image/png" }));
  } catch (err) {
    render("failed", { error: err.message });
  }
});
ui.cancel.addEventListener("click", () => {
  if (cur.abortUpload) {
    cur.abortUpload();
  } else if (cur.analysis) {
    // The server confirms with a "canceled" event, which ends the run on screen.
    cancelAnalysis(cur.analysis.id).catch((err) => {
      ui.status.textContent = err.message;
    });
  }
});
ui.stageAction.addEventListener("click", () => {
  if (body.dataset.state === "offline" && cur.analysis) openId(cur.analysis.id);
  else if (cur.file) startFile(cur.file);
});

/* ---------- drag & drop ---------- */

let depth = 0;
const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
window.addEventListener("dragenter", (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  depth += 1;
  body.classList.add("is-dragging");
});
window.addEventListener("dragover", (e) => {
  if (hasFiles(e)) e.preventDefault();
});
window.addEventListener("dragleave", (e) => {
  if (!hasFiles(e)) return;
  depth = Math.max(0, depth - 1);
  if (!depth) body.classList.remove("is-dragging");
});
window.addEventListener("drop", (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  depth = 0;
  body.classList.remove("is-dragging");
  startFile(e.dataTransfer.files[0]);
});

/* ---------- start ---------- */

window.addEventListener("hashchange", openFromHash);
render("empty");
refreshHistory();
openFromHash();
