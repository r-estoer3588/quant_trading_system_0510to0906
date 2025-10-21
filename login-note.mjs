import { chromium } from 'playwright';
import fs from 'fs';

const STATE_PATH = './note-state.json';

// 手動ログインのため環境変数は不要

const wait = (ms) => new Promise(r => setTimeout(r, ms));

(async () => {
  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext();
  const page = await context.newPage();

  await page.goto('https://note.com/login');

  console.log('手動でログインしてください。ログイン完了を自動検知します...');

  // ログイン完了を自動検知（note.comのトップページに遷移するまで待機）
  try {
    await page.waitForURL(/note\.com\/?$/, { timeout: 300000 }); // 5分待機
    console.log('ログイン完了を検知しました！');
  } catch (error) {
    console.log('ログイン完了の検知に失敗しました。手動でEnterキーを押してください。');
    await new Promise(resolve => {
      process.stdin.once('data', () => {
        resolve();
      });
    });
  }

  console.log('ログイン状態を保存中...');

  // 保存
  await context.storageState({ path: STATE_PATH });
  console.log('Saved:', STATE_PATH);

  await browser.close();
})();
