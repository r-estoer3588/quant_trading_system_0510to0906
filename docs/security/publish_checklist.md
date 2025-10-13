# 公開前チェックリスト（note 掲載向け）

このリポジトリの URL を note 等に掲載する前に、以下を確認してください。

## 1. 機密情報の漏えい防止

- `.env` / `*.env` がコミットされていない（`.gitignore` で除外済み）
- API キー・トークンを平文で含むファイルがない（`EODHD_API_KEY`, `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, `SLACK_BOT_TOKEN` など）
- もし過去コミットに含めた可能性がある場合は、即時ローテーション＋履歴削除（`git filter-repo` 等）

## 2. GitHub リポジトリ設定

- ブランチ保護（default branch）を有効化
  - 直 push 禁止（PR 必須）
  - 必須レビュー（最低 1 名）
  - 必須ステータスチェック（CI）
  - force-push とブランチ削除を禁止
- GitHub Actions
  - ワークフロー `permissions: contents: read` を明示
  - フォーク由来の PR では secrets を渡さない（既定）
  - 必要に応じて「ワークフロー実行の承認」を有効化

## 3. 運用に関する注意

- 大容量の生データや個人情報を含めない
- ライセンス（`LICENSE`）を明記し、利用条件を示す
- Issue/PR テンプレートを用意してスパム抑止・意図の明確化

## 4. 代替公開方法（任意）

- 公開用ミラーを作り、最小構成のみ公開
- リポジトリをアーカイブ（読み取り専用）
- Release ZIP を配布（PR/Issue を受け付けない形）

---

実施ログ（任意）:

- [ ] 検索で機密情報なし（`Select-String` / VSCode 検索）
- [ ] ブランチ保護設定スクショ保存
- [ ] Actions の permissions 確認
- [ ] テンプレート・CODEOWNERS 確認
