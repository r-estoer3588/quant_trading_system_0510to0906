import { expect, test } from "@playwright/test";

// Smaller, self-contained repro for the run-today-signals flow
test("smoke: run today signals quick", async ({ page }) => {
  await page.goto("http://localhost:8501", { timeout: 120000 });
  await page.waitForLoadState("networkidle");

  const runBtn = page.getByRole("button", { name: /本日のシグナル/i });
  if ((await runBtn.count()) > 0) {
    await runBtn.click();
    await page.waitForSelector("table", { state: "visible", timeout: 120000 });
    const tables = await page.locator("table").count();
    expect(tables).toBeGreaterThan(0);
  } else {
    // If the button is not found, assert that the main heading appears
    const bodyText = await page.textContent("body");
    expect(bodyText).toContain("本日のシグナル");
  }
});
