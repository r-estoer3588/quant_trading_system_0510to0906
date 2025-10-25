Title: Harden controlled systems test + CI workflow

Summary
-------
この PR は以下を行います：

- `tests/test_systems_controlled_all.py` の堅牢化（diagnostics キーの存在チェック、明確なエラーメッセージ、Bグループの検証）
- テスト実行ラッパー `scripts/run_controlled_tests.py` を追加/整備（既出）
- CI ワークフロー `.github/workflows/controlled-tests.yml` を追加
- `docs/README.md` にテスト実行手順を追記

目的
----
このテストは systems 1–6 のランキング段階（top-10）と最終候補の整合性を決定論的に検証します。
CI でこのテストを実行することで、ランキングロジックや UI 表示の回帰を早期に検出できます。

検証
----
ローカルで以下を実行して確認済みです：

```
python scripts/run_controlled_tests.py
# -> 6 passed
```

ワークフローを必須チェックにする手順（管理者が実行）
--------------------------------------------
ワークフロー自体はこの PR に含めましたが、GitHub のブランチ保護ルールで「必須チェック」に設定する操作はリポジトリ管理者が行う必要があります。手順は次の通りです。

UI 手順（推奨、最も簡単）
1. GitHub リポジトリに移動 → Settings → Branches
2. Add rule（または既存の rule を編集）
3. Branch name pattern に `main` を指定
4. Require status checks to pass before merging にチェックを入れる
5. リストから `controlled-tests`（ワークフロー名により表示されるチェック名）を選択して追加
6. Save changes

gh CLI（管理者トークンがある場合の参考例）
```
# 例: ワークフロー名は実行後に表示されるチェック名に合わせてください
gh api repos/:owner/:repo/branches/main/protection -X PUT -F required_status_checks='{"strict":true,"contexts":["controlled-tests"]}'
```

Notes / Caveats
----------------
- ワークフロー名（チェック名）は GitHub Actions が最初に成功した後に表示される名前と一致させる必要があります。PR 作成後に最初の実行を確認してから保護ルールを追加すると確実です。
- ブランチ保護の操作は管理者権限が必要です。私の側で PR を作成しますが、保護ルールの有効化は管理者にお願いしてください。

もし PR を確認したら、管理者に上記の UI 手順で「controlled-tests」を必須にしてもらうよう依頼してください。
