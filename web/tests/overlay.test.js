import { describe, expect, it } from "vitest";
import {
  contrastInfo,
  groupFindings,
  severityCounts,
  stateOf,
  statusText,
  toBox,
} from "../src/overlay.js";

const box = (x1, y1, x2, y2) => ({ x1, y1, x2, y2 });

describe("toBox", () => {
  it("turns corners into position and size", () => {
    expect(toBox(box(0.1, 0.25, 0.5, 0.75))).toEqual({ x: 0.1, y: 0.25, w: 0.4, h: 0.5 });
  });
});

describe("stateOf", () => {
  it("maps statuses and stages to page states", () => {
    expect(stateOf({ status: "queued" })).toBe("queued");
    expect(stateOf({ status: "running", stage: "detecting" })).toBe("detecting");
    expect(stateOf({ status: "running", stage: "auditing" })).toBe("checking");
    expect(stateOf({ status: "running" })).toBe("detecting");
    expect(stateOf({ status: "failed" })).toBe("failed");
    expect(stateOf({ status: "canceled" })).toBe("canceled");
  });

  it("tells a clean result from one with findings", () => {
    expect(stateOf({ status: "completed", result: { findings: [] } })).toBe("clean");
    expect(stateOf({ status: "completed", result: { findings: [{}] } })).toBe("done");
  });
});

describe("statusText", () => {
  it("pluralises and shows the server's error", () => {
    expect(statusText("done", { findings: 1, elements: 9 })).toBe("Done. 1 finding in 9 elements.");
    expect(statusText("done", { findings: 3, elements: 9 })).toMatch(/3 findings/);
    expect(statusText("failed", { error: "Queue down" })).toBe("Failed. Queue down");
    expect(statusText("clean", { elements: 0 })).toMatch(/nothing was checked/);
  });
});

describe("contrastInfo", () => {
  it("prefers the structured fields", () => {
    const f = { fg: "#777777", bg: "#ffffff", measured: 4.48, required: 4.5, message: "x" };
    expect(contrastInfo(f)).toEqual({ fg: "#777777", bg: "#ffffff", ratio: 4.48, required: 4.5 });
  });

  it("falls back to the message for older results", () => {
    const f = { message: "Text contrast is about 1.59:1 (#65a5a3 on #1c74e6); needs 3.0:1." };
    expect(contrastInfo(f)).toEqual({ ratio: 1.59, fg: "#65a5a3", bg: "#1c74e6", required: 3 });
  });

  it("returns null when there is nothing to show", () => {
    expect(contrastInfo({ message: "Target is 7x11 CSS px" })).toBeNull();
  });
});

describe("groupFindings", () => {
  const result = {
    elements: [
      { id: 0, kind: "link", text: "Home", box: box(0.1, 0.1, 0.2, 0.15) },
      { id: 1, kind: "icon", text: null, box: box(0.9, 0.1, 0.92, 0.12) },
      { id: 2, kind: "text", text: "Fine print", box: box(0.1, 0.8, 0.5, 0.85) },
    ],
    findings: [
      { wcag: "2.5.8", severity: "minor", element_id: 0, message: "small link" },
      { wcag: "1.4.3", severity: "moderate", element_id: 2, message: "grey text" },
      { wcag: "2.5.8", severity: "serious", element_id: 1, message: "tiny icon" },
      { wcag: "2.5.8", severity: "serious", element_id: 99, message: "unknown element" },
    ],
  };

  it("groups by criterion, worst group first, and numbers in display order", () => {
    const groups = groupFindings(result);
    expect(groups.map((g) => g.wcag)).toEqual(["2.5.8", "1.4.3"]);
    const items = groups.flatMap((g) => g.items);
    expect(items.map((it) => [it.n, it.message])).toEqual([
      [1, "tiny icon"],
      [2, "small link"],
      [3, "grey text"],
    ]);
    expect(items.map((it) => it.i)).toEqual([0, 1, 2]);
  });

  it("attaches the element with a position-and-size box", () => {
    const [first] = groupFindings(result)[0].items;
    expect(first.element.kind).toBe("icon");
    expect(first.element.box.x).toBeCloseTo(0.9);
    expect(first.element.box.w).toBeCloseTo(0.02);
  });

  it("skips findings whose element is missing", () => {
    const items = groupFindings(result).flatMap((g) => g.items);
    expect(items.some((it) => it.element_id === 99)).toBe(false);
  });

  it("counts by severity", () => {
    const items = groupFindings(result).flatMap((g) => g.items);
    expect(severityCounts(items)).toEqual({ serious: 1, moderate: 1, minor: 1 });
  });
});
