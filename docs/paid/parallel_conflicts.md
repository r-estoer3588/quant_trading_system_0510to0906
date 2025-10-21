# 並列競合（Parallel Conflict）対処ガイド（有料章）

目的: CI や複数ジョブで同一リポジトリを同時に動かしたときに起きる「ファイル競合」「一時データの上書き」「ログ破損」などを再現・予防・解決するための実践手順集です。

想定される問題点（短く）

- 結果 CSV やログを同じパスへ同時に書き込んで上書きが発生する
- テストやワークフローが同一のキャッシュ／ロックなし I/O を共有して相互に干渉する
- レポート生成や検証が他ランの出力を読み込んで誤判定する

設計方針（なぜこれをするのか）

1. 書き込み先を分離する（名前空間/サブディレクトリ） — もっとも単純で効果的
2. それでも共有リソースが残る箇所には排他を入れる（軽量ランロック）
3. レース条件を避けるために、書き込みは原子的に行う（tmp->rename など）
4. テストは可能な限り副作用が小さい形で実装し、必要なら `serial` に分離

実装（このリポジトリで既に組み込んだ内容）

1. 環境変数で制御

- PIPELINE_USE_RUN_LOCK=1
  - 保存フェーズで RunLock（ディレクトリ作成ベース）を取得し、並列でのクリティカルセクションを防ぎます。
- PIPELINE_USE_RUN_SUBDIR=1
  - 出力先を `results_csv/run_<NAMESPACE>/` のようなラン毎サブディレクトリに分離します。
- RUN_NAMESPACE=xxx
  - 任意のラン識別子。CI では一意な値（run_id / GUID）を指定してください。

2. CLI/スクリプトからの使い方

- CLI 例（手動実行）

  $ RUN_NAMESPACE=ci_1234 \
  PIPELINE_USE_RUN_SUBDIR=1 \
  PIPELINE_USE_RUN_LOCK=1 \
   python scripts/run_all_systems_today.py --save-csv --run-namespace ci_1234

  または

  $ python scripts/run_all_systems_today.py --save-csv --run-namespace ci_1234

  # start_playwright_server.ps1 は CI 用に自動で RUN_NAMESPACE を生成します

3. コードでの排他制御 (RunLock)

- 既に追加済み: `common/run_lock.py`

使い方（簡潔）

```py
from common.run_lock import RunLock

with RunLock("today_signals", timeout=300):
    # 保存処理 / 共有リソースへアクセスするクリティカルセクション
    save_results()
```

RunLock の設計ポイント（実装上の妥協）

- ディレクトリの原子作成 (os.mkdir) をロックの取得に使うことで外部ライブラリを不要にしました。
- ステールロックを検出して一定時間を過ぎたら best-effort で削除します（CI の強制終了対策）。
- 完全に堅牢なロックを求める場合はファイルロックライブラリ（portalocker）などを検討してください。

4. 保存先をランごとに分離する

- 実行時に `ctx.run_namespace` が指定されていると、保存フェーズは
  `results_csv/run_<run_namespace>/signals_final_<suffix>.csv` のように書き出します。
- CI では `RUN_NAMESPACE` に `github.run_id` や `run_number` を埋めると便利です。サンプルワークフローは `docs/playwright_mini_repro.zip` に含みます。

5. Playwright / E2E テスト側の対策

- Web サーバ（Streamlit）をテストランで立ち上げる際に `RUN_NAMESPACE` を付与し、Web アプリ側が生成する成果物を分離します（start_playwright_server.ps1 は CI で自動生成する仕組みにしました）。
- Playwright のテスト自体は以下を検討してください:
  - 共有リソースにアクセスするテストは `test.describe.serial` を使って直列化
  - 可能なら各テストが自分のラン名前空間を要求するようにする

6. GitHub Actions への組み込み（例）

```yaml
jobs:
  e2e:
    runs-on: ubuntu-latest
    env:
      PIPELINE_USE_RUN_SUBDIR: "1"
      PIPELINE_USE_RUN_LOCK: "1"
      RUN_NAMESPACE: "${{ github.run_id }}-${{ github.run_number }}"
```

これにより同リポジトリ上で複数のランが同時に動いても、出力ファイルはそれぞれ別フォルダへ保存され、かつ保存フェーズで短時間の排他取得を行うことで衝突を低減します。

7. トラブルシュート（短め）

- 競合が発生している疑いがある場合は: `ls locks/` を確認してロックが残っていないかを確認
- `locks/<name>.lock/owner.txt` を見てどのプロセスが最後に取得しているか、タイムスタンプを確認
- GitHub Actions の実行ログで `RUN_NAMESPACE` を表示するステップを挟むと追跡が楽になる

8. 追加の改善案（優先度: 高 → 低）

- 永続データベース（SQLite）を利用する場合は WAL モード + テーブル名に名前空間を付ける
- テスト用の一時 S3 バケットや GCS プレフィックスを使う（CI の並列化を最大化する場合）
- ファイルシステムのロックではなく OS/DB レベルのロックに移行するとより堅牢

---

付録: 変更点の早見

- 追加: `common/run_lock.py`（軽量ディレクトリロック）
- CLI: `scripts/run_all_systems_today.py --run-namespace` を追加
- 保存フェーズ: `RUN_NAMESPACE` / `PIPELINE_USE_RUN_SUBDIR` を使って出力を分離
- start_playwright_server.ps1: CI 実行時に RUN_NAMESPACE を自動生成し、サブディレクトリ + ロックを有効化
- テスト: `e2e/helpers/annotateTestIds.ts` などでテスト側分離も支援

この章を読み終えたら、次は「並列ジョブでの S3/Cloud 出力の分離」「DB スキーマのテスト時分離」を有料追加できます。希望するトピックを教えてください。
