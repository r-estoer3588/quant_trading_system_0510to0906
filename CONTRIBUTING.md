# Contributing & Formatting Guidelines

このリポジトリに変更を加える際の最低限の手順と、Windows 環境で発生しやすいファイルロック/整形フローの再発防止策をまとめます。

目的:

- 事前にローカルで自動整形（isort/black/ruff）を実行しておくことで pre-commit フックによるコミットブロックを防ぐ
- Windows やエディタによるファイルロックで isort 等が失敗するケースの回避方法を示す
- CI での品質ゲートを明文化して、フォーマットの不一致による PR 差戻しを削減する

必須前提

- Python 仮想環境をアクティベートして作業する（例: `python -m venv venv` → `venv\Scripts\Activate.ps1`）
- `pre-commit` がインストールされていること（`pip install pre-commit`）

推奨ワークフロー（コミット前）

1. エディタやテストランナー、Streamlit 等、リポジトリのファイルをロックするプロセスは閉じる
   - VS Code を使う場合は必ずファイル保存とウィンドウの最小化ではなく、ワークスペースを完全に閉じることを推奨します
2. 仮想環境をアクティベート
3. インポート順序を整理
   - `python -m isort .`
4. コード整形（Black）
   - `python -m black .`
5. Ruff のフォーマット
   - `python -m ruff format .`
6. pre-commit を実行して差分をチェック（ローカルで事前に通す）
   - `pre-commit run --all-files`
7. 変更をステージ → commit（`git add -A && git commit -m "..."`）

Windows 特有のトラブルシュート（ファイルロック）

- エラー例:
  - `Permission denied` / `アクセスが拒否されました` / `process is using the file`
- 対処手順（順序）:
  1. エディタ（VS Code）を完全に閉じる
  2. PowerShell で該当プロセスを確認して終了する（例: `Get-Process -Name python` → `Stop-Process -Id <PID>`）
  3. それでも解消しない場合は Windows を再起動してロックを解除する
  4. 特定のファイルだけロックされる場合、`handle.exe`（Sysinternals）でハンドルを確認して強制的に解放する手順を検討

ステージング/コミットで pre-commit が自動修正を行った場合

- pre-commit はコミットの前に自動修正を施すことがあります。自動修正が行われた場合は修正後の変更を再ステージして再度コミットしてください。

CI（推奨）

- GitHub Actions（例）に `pre-commit` を導入して、PR のマージ前に `pre-commit run --all-files` を必須チェックにしてください。
- また、`black` と `ruff` のバージョンを `pyproject.toml` や `requirements.txt` で固定することを推奨します。

短いチェックリスト（コミット前）

- [ ] VS Code 等の編集プロセスを閉じた
- [ ] `python -m isort .` を実行した
- [ ] `python -m black .` を実行した
- [ ] `python -m ruff format .` を実行した
- [ ] `pre-commit run --all-files` を通した

サポート

- 自動整形で行き詰まったり、pre-commit がコミットをブロックして解除できない場合は、詳細ログを共有してください。こちらで順に原因を調べます。

---

短い背景: このファイルは自動フォーマットや pre-commit によるコミットブロックの再発を防ぐための最低限のドキュメントです。必要に応じて CI の example ワークフローやより詳しい Powershell スクリプト（`tools/auto_format_converge.ps1` の使い方）を追加します。
