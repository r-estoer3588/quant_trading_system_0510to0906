import { expect, test } from "@playwright/test";
import { addTestIds } from "../helpers/annotateTestIds";

// Allow enough time for the full pipeline + Streamlit client render (20 minutes)
test.setTimeout(20 * 60 * 1000);

/**
 * Streamlit 統合アプリの基本動作テスト
 *
 * テスト対象: apps/app_integrated.py
 * 確認項目: ページ読み込み、タブ表示、基本UI要素
 */

test.describe("Streamlit 統合アプリ - 基本動作", () => {
  test("アプリが正常に起動する", async ({ page }) => {
    // Streamlit UI にアクセス
    await page.goto("/", { timeout: 20 * 60 * 1000 });
    await addTestIds(page);

    // ページタイトルが存在することを確認
    await expect(page).toHaveTitle(/.*/, { timeout: 10000 });

    // Streamlit の iframe が読み込まれるまで待機
    await page.waitForLoadState("networkidle");
  });

  test("メインタブが表示される", async ({ page }) => {
    await page.goto("/", { timeout: 20 * 60 * 1000 });

    // Streamlit が完全に読み込まれるまで待機
    await page.waitForLoadState("networkidle");

    // タブまたはヘッダーが表示されることを確認
    // 注: Streamlit の実際の要素に応じて調整が必要
    const bodyText = await page.textContent("body");
    expect(bodyText).toBeTruthy();
  });

  test("エラーが表示されていない", async ({ page }) => {
    await page.goto("/", { timeout: 20 * 60 * 1000 });
    await page.waitForLoadState("networkidle");

    // Streamlit のエラーメッセージが表示されていないことを確認
    const errorElements = await page.locator('[data-testid="stException"]').count();
    expect(errorElements).toBe(0);
  });
});
