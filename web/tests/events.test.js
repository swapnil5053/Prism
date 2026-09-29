import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { followAnalysis } from "../src/events.js";

class FakeSocket {
  static all = [];
  constructor(url) {
    this.url = url;
    FakeSocket.all.push(this);
  }
  send() {}
  close() {
    this.closed = true;
  }
  emit(event) {
    this.onmessage({ data: JSON.stringify(event) });
  }
  drop(code = 1006) {
    this.onclose({ code });
  }
}

const loc = { protocol: "http:", host: "localhost:5173" };

describe("followAnalysis", () => {
  beforeEach(() => {
    FakeSocket.all = [];
    vi.useFakeTimers();
  });
  afterEach(() => vi.useRealTimers());

  it("resumes from the last event after a drop", () => {
    const seen = [];
    followAnalysis(
      "abc",
      { onEvent: (e) => seen.push(e.stage) },
      {
        WebSocket: FakeSocket,
        location: loc,
      },
    );
    const first = FakeSocket.all[0];
    expect(first.url).toBe("ws://localhost:5173/api/v1/analyses/abc/events?after=0");

    first.emit({ id: "5-0", status: "running", stage: "detecting" });
    first.drop();
    vi.advanceTimersByTime(500);

    const second = FakeSocket.all[1];
    expect(second.url).toContain("after=5-0");
    second.emit({ id: "6-0", status: "completed" });
    expect(second.closed).toBe(true);
    expect(seen).toEqual(["detecting", undefined]);
  });

  it("gives up after too many failures", () => {
    const onGiveUp = vi.fn();
    followAnalysis(
      "abc",
      { onEvent: () => {}, onGiveUp },
      {
        WebSocket: FakeSocket,
        location: loc,
        maxRetries: 2,
      },
    );
    for (let i = 0; i < 3; i++) {
      FakeSocket.all.at(-1).drop();
      vi.advanceTimersByTime(10_000);
    }
    expect(onGiveUp).toHaveBeenCalledOnce();
    expect(FakeSocket.all).toHaveLength(3);
  });

  it("does not retry when the server refuses access", () => {
    const onGiveUp = vi.fn();
    followAnalysis(
      "abc",
      { onEvent: () => {}, onGiveUp },
      { WebSocket: FakeSocket, location: loc },
    );
    FakeSocket.all[0].drop(1008);
    vi.advanceTimersByTime(10_000);
    expect(FakeSocket.all).toHaveLength(1);
    expect(onGiveUp).toHaveBeenCalledOnce();
  });

  it("uses wss on https pages", () => {
    followAnalysis(
      "abc",
      { onEvent: () => {} },
      {
        WebSocket: FakeSocket,
        location: { protocol: "https:", host: "prism.example" },
      },
    );
    expect(FakeSocket.all[0].url.startsWith("wss://prism.example/")).toBe(true);
  });
});
