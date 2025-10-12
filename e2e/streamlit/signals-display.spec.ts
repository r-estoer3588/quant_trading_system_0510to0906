import { expect, test } from '@playwright/test';

/**
 * 当日シグナル表示のテスト
 *
 * テスト対象: apps/app_integrated.py の当日シグナルタブ
 * 前提条件: scripts/run_all_systems_today.py が実行済みでシグナルCSVが存在すること
 */

test.describe('当日シグナル表示', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('シグナルデータが表示される', async ({ page }) => {
    // タブの切り替えが必要な場合はここで実装
    // 例: await page.click('text=当日シグナル');

    // データフレームまたはテーブルが存在することを確認
    const hasDataFrame = await page.locator('[data-testid="stDataFrame"]').count();
    const hasTable = await page.locator('table').count();

    expect(hasDataFrame + hasTable).toBeGreaterThan(0);
  });

  test('複数システムのシグナルが存在する', async ({ page }) => {
    // System1〜7 のいずれかが表示されていることを確認
    const bodyText = await page.textContent('body');

    const hasSystemSignals =
      bodyText?.includes('System1') ||
      bodyText?.includes('System2') ||
      bodyText?.includes('System3') ||
      bodyText?.includes('System4') ||
      bodyText?.includes('System5') ||
      bodyText?.includes('System6') ||
      bodyText?.includes('System7');

    expect(hasSystemSignals).toBeTruthy();
  });
});
