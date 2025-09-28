# Codacy CI/CD Integration Setup

このプロジェクトでは GitHub Actions を使用して Codacy 分析を自動実行しています。

## セットアップ手順

### 1. Codacy プロジェクトトークンの設定

1. [Codacy](https://www.codacy.com/)にログインし、プロジェクトを選択
2. Settings > Integrations > Project API で Project Token を取得
3. GitHub リポジトリの Settings > Secrets and variables > Actions で以下のシークレットを追加：
   - Name: `CODACY_PROJECT_TOKEN`
   - Value: Codacy から取得したプロジェクトトークン

### 2. ワークフローの自動実行

以下のイベントで自動的に分析が実行されます：

- `main`, `develop`, `branch0906` ブランチへのプッシュ
- `main`, `develop` ブランチへのプルリクエスト

### 3. 分析内容

#### Codacy Security Scan

- セキュリティ脆弱性の検出
- コード品質の問題の特定
- SARIF 形式でのレポート生成
- GitHub Security タブでの結果表示

#### Python Code Quality Check

- Ruff による静的解析
- Black によるコードフォーマットチェック
- Python 構文チェック

## 結果の確認

1. **GitHub Actions**: リポジトリの Actions タブで実行状況を確認
2. **Security**: リポジトリの Security タブで脆弱性レポートを確認
3. **Pull Request**: PR ページで自動チェック結果を確認

## ローカルでの事前チェック

プッシュ前に以下のコマンドでローカルチェックを実行できます：

```powershell
# Ruffによる静的解析
ruff check .

# Blackによるフォーマットチェック
black --check .

# 基本的な構文チェック
python -m py_compile apps/app_today_signals.py
```

## トラブルシューティング

- **トークンエラー**: Codacy プロジェクトトークンが正しく設定されているか確認
- **権限エラー**: GitHub リポジトリの Settings > Actions で必要な権限が設定されているか確認
- **分析失敗**: Actions タブでログを確認し、依存関係やコード品質の問題を修正
