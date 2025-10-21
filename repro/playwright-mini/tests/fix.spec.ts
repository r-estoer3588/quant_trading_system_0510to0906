import { expect, test } from "@playwright/test";

// Fix: use conditional wait instead of fixed timeout
test("fixed->conditional: replace waitForTimeout with waitForSelector", async ({
  page,
}) => {
  await page.setContent(`
    <button id="run">Run</button>
    <div id="result" style="display:none">Done</div>
    <script>
      document.getElementById('run').addEventListener('click', () => {
        setTimeout(()=>{ document.getElementById('result').style.display='block'; }, 1500);
      });
    </script>
  `);

  await page.click("#run");
  // Good: wait for element to appear
  await page.waitForSelector("#result", { state: "visible", timeout: 3000 });
  await expect(page.locator("#result")).toBeVisible();
});
