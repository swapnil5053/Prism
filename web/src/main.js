import { cancelAnalysis, getAnalysis, listAnalyses, reportUrl, uploadScreenshot } from "./api.js";
import { followAnalysis } from "./events.js";
import { boxStyle, describeStatus, sortFindings, worstSeverity } from "./overlay.js";

const $ = (id) => document.getElementById(id);
let stopFollowing = null;

// Everything from the server (including model output) goes in via textContent.
function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function setStatus(text) {
  $("status").textContent = text;
}

function showResult(analysis) {
  const { result } = analysis;
  const boxes = $("boxes");
  const list = $("findings");
  boxes.replaceChildren();
  list.replaceChildren();
  if (!result) return;

  const worst = worstSeverity(result.findings);
  const byId = new Map(result.elements.map((e) => [e.id, e]));
  const boxNodes = new Map();
  for (const [id, severity] of worst) {
    const element = byId.get(id);
    if (!element) continue;
    const node = el("div", undefined, `box ${severity}`);
    Object.assign(node.style, boxStyle(element.box));
    boxes.append(node);
    boxNodes.set(id, node);
  }

  $("summary").textContent =
    `Score ${Math.round(result.score)}/100 · ${result.findings.length} findings ` +
    `in ${result.elements.length} elements`;

  for (const f of sortFindings(result.findings)) {
    const item = el("li");
    const button = el("button");
    button.type = "button";
    const element = byId.get(f.element_id);
    button.append(
      el("span", f.severity, `sev ${f.severity}`),
      el("span", f.message),
      el(
        "div",
        `${element?.kind ?? "element"}${element?.text ? ` "${element.text}"` : ""} · WCAG ${f.wcag}`,
        "meta",
      ),
    );
    button.addEventListener("click", () => {
      for (const node of boxNodes.values()) node.classList.remove("active");
      boxNodes.get(f.element_id)?.classList.add("active");
    });
    item.append(button);
    list.append(item);
  }
  if (result.elements.length === 0) {
    $("summary").textContent = "No elements detected";
  } else if (result.findings.length === 0) {
    $("summary").textContent = `No problems found in ${result.elements.length} elements`;
  }
}

async function open(analysis) {
  stopFollowing?.();
  $("current").hidden = false;
  $("shot").src = analysis.image_url;
  $("shot").alt = `Screenshot: ${analysis.original_filename ?? "upload"}`;
  $("findings").replaceChildren();
  $("boxes").replaceChildren();
  $("summary").textContent = "";

  const finished = (a) => {
    setStatus(describeStatus({ status: a.status, message: a.error }));
    $("cancel").hidden = true;
    const done = a.status === "completed";
    $("downloads").hidden = !done;
    if (done) {
      $("report-html").href = reportUrl(a.id, "html");
      $("report-json").href = reportUrl(a.id, "json");
      showResult(a);
    }
    refreshHistory();
  };

  if (["completed", "failed", "canceled"].includes(analysis.status)) {
    finished(analysis);
    return;
  }

  setStatus(describeStatus(analysis));
  $("downloads").hidden = true;
  $("cancel").hidden = false;
  $("cancel").onclick = () => cancelAnalysis(analysis.id).catch((e) => setStatus(e.message));

  stopFollowing = followAnalysis(analysis.id, {
    onEvent: async (event) => {
      setStatus(describeStatus(event));
      if (["completed", "failed", "canceled"].includes(event.status)) {
        finished(event.snapshot ?? (await getAnalysis(analysis.id)));
      }
    },
    onGiveUp: () => setStatus("Lost connection. Reload the page to check on this analysis."),
  });
}

async function refreshHistory() {
  const list = $("history");
  try {
    const page = await listAnalyses(10);
    list.replaceChildren(
      ...page.items.map((a) => {
        const item = el("li");
        const link = el("a", a.original_filename ?? "Screenshot");
        link.href = `#${a.id}`;
        item.append(
          link,
          el("span", ` · ${a.status} · ${new Date(a.created_at).toLocaleString()}`, "meta"),
        );
        return item;
      }),
    );
    if (page.items.length === 0) list.append(el("li", "Nothing yet.", "meta"));
  } catch {
    list.replaceChildren(el("li", "Couldn't load history.", "meta"));
  }
}

$("upload").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = new FormData(e.target);
  const file = form.get("file");
  if (!(file instanceof File) || file.size === 0) return;
  const button = e.target.querySelector("button");
  button.disabled = true;
  setStatus("Uploading…");
  $("current").hidden = false;
  try {
    const analysis = await uploadScreenshot(file, Number(form.get("dpr")));
    history.replaceState(null, "", `#${analysis.id}`);
    await open(analysis);
  } catch (err) {
    setStatus(err.message);
  } finally {
    button.disabled = false;
  }
});

async function openFromHash() {
  const id = location.hash.slice(1);
  if (!/^[0-9a-f-]{36}$/.test(id)) return;
  try {
    await open(await getAnalysis(id));
  } catch (err) {
    setStatus(err.message);
  }
}

window.addEventListener("hashchange", openFromHash);
refreshHistory();
openFromHash();
