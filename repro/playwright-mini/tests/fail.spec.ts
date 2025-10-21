import { expect, test } from "@playwright/test";

// Demonstrates a flaky pattern: fixed timeout shorter than the actual async work
test("flaky: fixed wait causes intermittent failure", async ({ page }) => {
  await page.setContent(`
    <button id="run">Run</button>
    <div id="result" style="display:none">Done</div>
    <script>
      document.getElementById('run').addEventListener('click', () => {
        // Simulate variable delay (1.5s)
        setTimeout(()=>{ document.getElementById('result').style.display='block'; }, 1500);
      });
    </script>
  `);

  await page.click("#run");
  // Bad: fixed sleep that may be too short
  await page.waitForTimeout(1000);
  await expect(page.locator("#result")).toBeVisible();
});
