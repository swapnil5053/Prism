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

/**
 * Upload with progress. fetch() can't report upload progress, so this uses XHR.
 * Returns {promise, abort}; abort() rejects the promise with an AbortError.
 */
export function uploadScreenshot(file, devicePixelRatio = 1, onProgress = () => {}) {
  const form = new FormData();
  form.append("file", file);
  form.append("device_pixel_ratio", String(devicePixelRatio));
  const xhr = new XMLHttpRequest();
  const promise = new Promise((resolve, reject) => {
    xhr.open("POST", BASE);
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onload = () => {
      const body = xhr.response;
      // 503 still returns the (failed) analysis; let the caller show it.
      if (xhr.status < 300 || (xhr.status === 503 && body?.id)) resolve(body);
      else
        reject(
          new ApiError(xhr.status, typeof body?.detail === "string" ? body.detail : undefined),
        );
    };
    xhr.onerror = () => reject(new ApiError(0, "Couldn't reach the server."));
    xhr.onabort = () => reject(new DOMException("Upload canceled", "AbortError"));
    xhr.send(form);
  });
  return { promise, abort: () => xhr.abort() };
}

export const getAnalysis = (id) => request(`${BASE}/${id}`);
export const listAnalyses = (limit = 20) => request(`${BASE}?limit=${limit}`);
export const cancelAnalysis = (id) => request(`${BASE}/${id}/cancel`, { method: "POST" });
export const reportUrl = (id, format = "html") => `${BASE}/${id}/report?format=${format}`;

export function eventsUrl(id, after, loc = window.location) {
  const proto = loc.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${loc.host}${BASE}/${id}/events?after=${encodeURIComponent(after)}`;
}
