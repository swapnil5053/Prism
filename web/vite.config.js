import { defineConfig } from "vite";

// In dev, proxy the API so the page and API share an origin (and the cookie).
export default defineConfig({
  server: {
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", ws: true },
    },
  },
  test: {
    environment: "jsdom",
  },
});
