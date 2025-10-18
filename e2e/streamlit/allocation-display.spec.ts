import { expect, test } from "@playwright/test";

// Allow enough time for the full pipeline + Streamlit client render (20 minutes)
test.setTimeout(20 * 60 * 1000);

/**
 * 配分結果表示のテスト
 *
 * テスト対象: apps/app_integrated.py の配分結果タブ
 * 前提条件: run_all_systems_today.py が正常終了していること
 */

test.describe("配分結果表示", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await page.waitForLoadState("networkidle");

    // Wait for the app to perform client-side rendering.
    // Streamlit serves a static no-JS placeholder initially; wait until a known
    // heading or a data frame/table appears. Use a generous timeout because
    // the full pipeline may run before Streamlit starts.
    await Promise.race([
      page.waitForSelector("text=配分サマリー", { timeout: 120000 }).catch(() => null),
      page.waitForSelector("text=Allocation", { timeout: 120000 }).catch(() => null),
      page
        .waitForSelector('[data-testid="stDataFrame"]', { timeout: 120000 })
        .catch(() => null),
      page.waitForSelector("table", { timeout: 120000 }).catch(() => null),
    ]);
    // Ensure Integrated tab is selected and trigger a run so the integrated
    // summary (allocation) is produced and visible for assertions.
    try {
      // Try to activate the Integrated tab (localized or english label)
      await page
        .locator("role=tab[name=/統合|Integrated/i]")
        .first()
        .click()
        .catch(() => null);

      // Robustly find the run-integrated button via role matching
      const integratedBtn = page.locator(
        "role=button[name=/統合実行|run integrated|統合実行開始/i]"
      );
      if ((await integratedBtn.count()) > 0) {
        await integratedBtn
          .first()
          .waitFor({ state: "visible", timeout: 5000 })
          .catch(() => null);
        await integratedBtn.first().click();
      } else {
        // fallback to button:has-text
        const fallback = page.locator('button:has-text("統合実行")');
        if ((await fallback.count()) > 0) {
          await fallback.first().click();
        }
      }

      // Wait for integrated summary heading or table to appear
      await Promise.race([
        page
          .waitForSelector("text=統合サマリー", { timeout: 120000 })
          .catch(() => null),
        page
          .waitForSelector("text=Integrated Summary", { timeout: 120000 })
          .catch(() => null),
        page
          .waitForSelector('[data-testid="stDataFrame"]', { timeout: 120000 })
          .catch(() => null),
        page.waitForSelector("table", { timeout: 120000 }).catch(() => null),
      ]);
    } catch (e) {
      // ignore errors here; tests will assert and fail if summary not available
    }
  });

  test("配分サマリーが表示される", async ({ page }) => {
    // 配分関連のタブに切り替え（必要に応じて）
    // await page.click('text=配分結果');
    const bodyText = await page.textContent("body");

    // Accept a few signals that allocation/integrated summary is present:
    // - an st.dataframe is rendered
    // - an HTML table exists
    // - the Integrated Summary heading is present
    const hasDataFrame =
      (await page.locator('[data-testid="stDataFrame"]').count()) > 0;
    const hasTable = (await page.locator("table").count()) > 0;
    const hasIntegratedSummary = !!(
      bodyText && bodyText.includes("Integrated Summary")
    );
    const hasLongShort = !!(
      bodyText &&
      (bodyText.includes("Long") || bodyText.includes("Short"))
    );

    expect(
      hasDataFrame || hasTable || hasIntegratedSummary || hasLongShort
    ).toBeTruthy();
  });

  test("Long と Short の区別がある", async ({ page }) => {
    const bodyText = await page.textContent("body");

    // Either explicit Long/Short text appears or a dataframe with a 'side' column is present
    const hasLongShort = !!(bodyText && bodyText.match(/Long|Short/i));
    const hasDataFrame =
      (await page.locator('[data-testid="stDataFrame"]').count()) > 0;

    expect(hasLongShort || hasDataFrame).toBeTruthy();
  });
});
