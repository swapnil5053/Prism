const RANK = { minor: 0, moderate: 1, serious: 2 };

/** Position a normalised [0, 1] box over an image of any displayed size. */
export function boxStyle(box) {
  const pct = (v) => `${(v * 100).toFixed(3)}%`;
  return {
    left: pct(box.x1),
    top: pct(box.y1),
    width: pct(box.x2 - box.x1),
    height: pct(box.y2 - box.y1),
  };
}

/** Worst severity per element id, for colouring boxes. */
export function worstSeverity(findings) {
  const worst = new Map();
  for (const f of findings) {
    const current = worst.get(f.element_id);
    if (current === undefined || RANK[f.severity] > RANK[current]) {
      worst.set(f.element_id, f.severity);
    }
  }
  return worst;
}

/** Findings, most severe first, then by element. */
export function sortFindings(findings) {
  return [...findings].sort(
    (a, b) => RANK[b.severity] - RANK[a.severity] || a.element_id - b.element_id,
  );
}

export function describeStatus(event) {
  if (event.status === "running") {
    return event.stage === "auditing" ? "Checking accessibility…" : "Detecting elements…";
  }
  return (
    {
      queued: "Waiting for the worker…",
      completed: "Done.",
      failed: event.message || "Analysis failed.",
      canceled: "Canceled.",
    }[event.status] ?? event.status
  );
}
