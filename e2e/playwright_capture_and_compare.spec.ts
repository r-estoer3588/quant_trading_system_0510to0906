import { test } from "@playwright/test";
import { execSync } from "child_process";

// Simple example: navigate to the Streamlit app, wait briefly for UI to settle,
// then invoke the Python comparator (which will look up the newest snapshot if
// --new is omitted). Adjust URLs and waits to your environment.

test("capture UI metrics and compare to baseline", async ({ page }) => {
  await page.goto("http://localhost:8501");
  await page.waitForSelector("text=本日のシグナル", { timeout: 30_000 });
  // Click the main "Generate Signals" button to run the pipeline which
  // will trigger finalize and write the UI metrics snapshot.
  const runButton = page.locator("text=Generate Signals");
  if ((await runButton.count()) > 0) {
    await runButton.click();
    // wait for completion success message
    await page.waitForSelector("text=Signals generation complete", {
      timeout: 120_000,
    });
  } else {
    // fallback: wait briefly for app to maybe auto-run (rare)
    await page.waitForTimeout(2000);
  }

  try {
    execSync(
      "python tools/playwright_snapshot_check.py --baseline results_csv/ui_metrics_baseline.json",
      {
        stdio: "inherit",
      }
    );
  } catch (err) {
    // bubble up so Playwright marks the test as failed
    throw err;
  }
});
