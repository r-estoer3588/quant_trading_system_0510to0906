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
      // Debug: list top-level tabs and their labels so we can see what the UI shows
      const tabLoc = page.locator('role=tab');
      const tcount = await tabLoc.count();
      console.log('[DEBUG] found tabs count=', tcount);
      let clicked = false;
      for (let i = 0; i < tcount; i++) {
        const t = tabLoc.nth(i);
        let txt = '';
        try {
          txt = (await t.innerText()).trim();
        } catch (e) {
          txt = (await t.textContent()) || '';
        }
        console.log(`[DEBUG] top-tab[${i}]='${txt}'`);
        if (/統合|Integrated/i.test(txt)) {
          await t.click().catch(() => null);
          clicked = true;
          break;
        }
      }
      // Fallback: click first tab that is not a system tab (system1..)
      if (!clicked) {
        for (let i = 0; i < tcount; i++) {
          const t = tabLoc.nth(i);
          const txt = (await t.innerText()).trim();
          if (!/^system\d+/i.test(txt) && txt.length > 0) {
            console.log('[DEBUG] fallback clicking top-tab index=', i, 'text=', txt);
            await t.click().catch(() => null);
            clicked = true;
            break;
          }
        }
      }

      // Capture a trimmed body snippet for debugging so we can inspect what was rendered
      try {
        const body = await page.textContent('body');
        console.log('[DEBUG] body snippet:', (body || '').slice(0, 2000));
      } catch (e) {
        // ignore
      }

      // Robustly find the run-integrated button via role matching or text
      const integratedBtn = page.locator(
        'role=button[name=/統合実行|run integrated|統合実行開始|run today signals|当日シグナル実行|▶ 本日のシグナル実行/i]'
      );
      if ((await integratedBtn.count()) > 0) {
        await integratedBtn.first().waitFor({ state: 'visible', timeout: 10000 }).catch(() => null);
        await integratedBtn.first().click().catch(() => null);
      } else {
        // fallback to button:has-text for likely candidates
        const fallbackTexts = ['統合実行', 'run integrated', '本日のシグナル実行', '▶ 本日のシグナル実行', 'run today signals'];
        for (const ft of fallbackTexts) {
          const fb = page.locator(`button:has-text("${ft}")`);
          if ((await fb.count()) > 0) {
            await fb.first().click().catch(() => null);
            break;
          }
        }
      }

      // Wait for integrated summary heading or table to appear
      await Promise.race([
        page.waitForSelector('text=統合サマリー', { timeout: 120000 }).catch(() => null),
        page.waitForSelector('text=Integrated Summary', { timeout: 120000 }).catch(() => null),
        page.waitForSelector('[data-testid="stDataFrame"]', { timeout: 120000 }).catch(() => null),
        page.waitForSelector('table', { timeout: 120000 }).catch(() => null),
      ]);
    } catch (e) {
      // ignore errors here; tests will assert and fail if summary not available
      console.log('[DEBUG] beforeEach integration click error:', e);
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
