import { describe, expect, it } from "vitest";
import { boxStyle, describeStatus, sortFindings, worstSeverity } from "../src/overlay.js";

describe("boxStyle", () => {
  it("converts a normalised box to percentages", () => {
    expect(boxStyle({ x1: 0.1, y1: 0.25, x2: 0.5, y2: 0.75 })).toEqual({
      left: "10.000%",
      top: "25.000%",
      width: "40.000%",
      height: "50.000%",
    });
  });
});

describe("worstSeverity", () => {
  it("keeps the most severe finding per element", () => {
    const worst = worstSeverity([
      { element_id: 1, severity: "minor" },
      { element_id: 1, severity: "serious" },
      { element_id: 1, severity: "moderate" },
      { element_id: 2, severity: "moderate" },
    ]);
    expect(Object.fromEntries(worst)).toEqual({ 1: "serious", 2: "moderate" });
  });
});

describe("sortFindings", () => {
  it("orders by severity then element", () => {
    const sorted = sortFindings([
      { element_id: 3, severity: "minor" },
      { element_id: 2, severity: "serious" },
      { element_id: 1, severity: "serious" },
    ]);
    expect(sorted.map((f) => f.element_id)).toEqual([1, 2, 3]);
  });
});

describe("describeStatus", () => {
  it("shows the failure message from the server", () => {
    expect(describeStatus({ status: "failed", message: "Queue down" })).toBe("Queue down");
    expect(describeStatus({ status: "running", stage: "auditing" })).toMatch(/accessibility/);
  });
});
