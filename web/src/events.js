import { eventsUrl } from "./api.js";

const TERMINAL = new Set(["completed", "failed", "canceled"]);

/**
 * Follow an analysis' progress events. Reconnects with backoff and resumes
 * from the last event id, so nothing is missed or repeated.
 *
 * @param {string} id
 * @param {{onEvent: (e: object) => void, onGiveUp?: () => void}} handlers
 * @returns {() => void} stop
 */
export function followAnalysis(id, { onEvent, onGiveUp }, opts = {}) {
  const WS = opts.WebSocket ?? globalThis.WebSocket;
  const maxRetries = opts.maxRetries ?? 6;
  const baseDelay = opts.baseDelayMs ?? 500;
  let lastId = "0";
  let retries = 0;
  let stopped = false;
  let socket = null;
  let timer = null;

  const connect = () => {
    socket = new WS(eventsUrl(id, lastId, opts.location));
    socket.onmessage = (msg) => {
      retries = 0;
      const event = JSON.parse(msg.data);
      if (event.id) lastId = event.id;
      onEvent(event);
      if (TERMINAL.has(event.status)) stop();
    };
    socket.onclose = (e) => {
      if (stopped) return;
      // 1008 = not ours / not allowed. Retrying won't help.
      if (e.code === 1008 || retries >= maxRetries) {
        stopped = true;
        onGiveUp?.();
        return;
      }
      const delay = Math.min(baseDelay * 2 ** retries, 10_000);
      retries += 1;
      timer = setTimeout(connect, delay);
    };
  };

  function stop() {
    stopped = true;
    clearTimeout(timer);
    socket?.close();
  }

  connect();
  return stop;
}
