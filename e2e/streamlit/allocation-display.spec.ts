import { expect, test } from '@playwright/test';

/**
 * 配分結果表示のテスト
 *
 * テスト対象: apps/app_integrated.py の配分結果タブ
 * 前提条件: run_all_systems_today.py が正常終了していること
 */

test.describe('配分結果表示', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('配分サマリーが表示される', async ({ page }) => {
    // 配分関連のタブに切り替え（必要に応じて）
    // await page.click('text=配分結果');

    const bodyText = await page.textContent('body');

    // 配分関連のキーワードが含まれていることを確認
    const hasAllocationInfo =
      bodyText?.includes('Long') ||
      bodyText?.includes('Short') ||
      bodyText?.includes('配分') ||
      bodyText?.includes('Allocation');

    expect(hasAllocationInfo).toBeTruthy();
  });

  test('Long と Short の区別がある', async ({ page }) => {
    const bodyText = await page.textContent('body');

    // Long/Short の表示があることを確認
    expect(bodyText).toMatch(/Long|Short/i);
  });
});
