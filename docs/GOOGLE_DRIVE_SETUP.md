# Google Drive Setup Guide

このガイドでは、月次レポートを Google Drive に自動アップロードするためのセットアップ方法を説明します。

## 前提条件

- Google アカウント
- Google Cloud Platform へのアクセス

## セットアップ手順

### 1. Google Cloud Project の作成

1. [Google Cloud Console](https://console.cloud.google.com/) にアクセス
2. 新しいプロジェクトを作成（例: "Quant Trading Reports"）

### 2. Google Drive API の有効化

1. プロジェクトダッシュボードで「API とサービス」→「ライブラリ」
2. "Google Drive API" を検索
3. 「有効にする」をクリック

### 3. サービスアカウントの作成

1. 「API とサービス」→「認証情報」
2. 「認証情報を作成」→「サービス アカウント」
3. サービスアカウント名を入力（例: "quant-trading-uploader"）
4. 「作成して続行」
5. ロールは不要（スキップ）
6. 「完了」をクリック

### 4. JSON キーのダウンロード

1. 作成したサービスアカウントをクリック
2. 「キー」タブ → 「鍵を追加」→「新しい鍵を作成」
3. キーのタイプ: **JSON**
4. 「作成」をクリック
5. ダウンロードされた JSON ファイルを以下の場所に保存：
   ```
   c:\Repos\quant_trading_system\data\google_service_account.json
   ```

### 5. Google Drive フォルダの準備

1. Google Drive で新しいフォルダを作成（例: "Trading Reports"）
2. フォルダを右クリック → 「共有」
3. サービスアカウントのメールアドレスを追加
   - メールアドレスは `google_service_account.json` ファイル内の `client_email` フィールド
   - 例: `quant-trading-uploader@project-id.iam.gserviceaccount.com`
4. 権限: 「編集者」
5. フォルダの ID をコピー
   - フォルダを開いた際の URL から取得
   - URL: `https://drive.google.com/drive/folders/FOLDER_ID_HERE`
   - `FOLDER_ID_HERE` の部分がフォルダ ID

### 6. 環境変数の設定

`.env` ファイルに以下を追加：

```
GOOGLE_DRIVE_FOLDER_ID=your_folder_id_here
```

### 7. 必要なライブラリのインストール

```powershell
.\venv\Scripts\Activate.ps1
pip install google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2
```

## テスト

```powershell
python -c "from common.google_drive_uploader import upload_to_drive; print(upload_to_drive('reports/monthly_report_202512.xlsx'))"
```

成功すると、Google Drive の URL が表示されます。

## トラブルシューティング

### エラー: "Service account credentials not found"

- `google_service_account.json` が正しい場所にあるか確認
- パスを確認: `c:\Repos\quant_trading_system\data\google_service_account.json`

### エラー: "Permission denied"

- サービスアカウントにフォルダへのアクセス権限があるか確認
- フォルダ共有設定を再確認

### エラー: "Google Drive libraries not installed"

- `pip install google-api-python-client google-auth` を実行

## セキュリティ

⚠️ **重要**: `google_service_account.json` は機密情報です！

- Git にコミットしない（`.gitignore` に追加済み）
- 他人と共有しない
- 定期的にキーをローテーション

## 完了

セットアップ完了後、月次レポートは自動的に Google Drive にアップロードされ、
Slack にダウンロードリンク Get 通知されます。
