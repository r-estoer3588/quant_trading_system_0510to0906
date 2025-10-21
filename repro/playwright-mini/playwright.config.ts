import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./",
  fullyParallel: true,
  timeout: 60 * 1000,
  retries: 0,
  use: {
    baseURL: "http://localhost:8501",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    viewport: { width: 1280, height: 800 },
    locale: "ja-JP",
    timezoneId: "Asia/Tokyo",
    testIdAttribute: "data-testid",
  },
});
