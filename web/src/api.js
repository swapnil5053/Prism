const BASE = "/api/v1/analyses";

export class ApiError extends Error {
  constructor(status, detail) {
    super(detail || `Request failed (${status})`);
    this.status = status;
  }
}

async function request(url, options = {}) {
  const res = await fetch(url, { credentials: "same-origin", ...options });
  const body = res.headers.get("content-type")?.includes("json") ? await res.json() : null;
  // 503 on upload still returns the (failed) analysis; let the caller show it.
  if (!res.ok && !(res.status === 503 && body?.id)) {
    const detail = typeof body?.detail === "string" ? body.detail : undefined;
    throw new ApiError(res.status, detail);
  }
  return body;
}

export function uploadScreenshot(file, devicePixelRatio = 1) {
  const form = new FormData();
  form.append("file", file);
  form.append("device_pixel_ratio", String(devicePixelRatio));
  return request(BASE, { method: "POST", body: form });
}

export const getAnalysis = (id) => request(`${BASE}/${id}`);
export const listAnalyses = (limit = 20) => request(`${BASE}?limit=${limit}`);
export const cancelAnalysis = (id) => request(`${BASE}/${id}/cancel`, { method: "POST" });
export const reportUrl = (id, format = "html") => `${BASE}/${id}/report?format=${format}`;

export function eventsUrl(id, after, loc = window.location) {
  const proto = loc.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${loc.host}${BASE}/${id}/events?after=${encodeURIComponent(after)}`;
}
