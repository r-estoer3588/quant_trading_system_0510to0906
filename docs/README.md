# Quant Trading System - ドキュメント総覧

このドキュメントは、7 つの売買システムを統合したクオンツトレーディングシステムの包括的なガイドです。

## 🌟 はじめに

**このドキュメントの使い方**:

- **初めて使う方**: [📘 初心者向けガイド](#-初心者向けガイド) から始めてください（環境構築 →UI 起動 → 基本操作の順）
- **開発者の方**: [🔧 開発者向けガイド](#-開発者向けガイド) で技術詳細・テスト・カスタマイズ方法を確認してください
- **特定の課題を解決したい**: 目次から該当セクションに直接ジャンプできます

---

## 📘 初心者向けガイド

このセクションは、システムを初めて使う方向けの基本的な操作ガイドです。

### 🚀 クイックスタート

1. **環境構築**: [セットアップ](../README.md#セットアップ) で Python 環境と依存パッケージをインストール
2. **UI 起動**: [基本実行](../README.md#実行例) で Streamlit アプリを起動
3. **動作確認**: UI から「Generate Signals」ボタンをクリックしてシグナル生成を試す

### 📊 システム概要

- [システム構成と資産配分](#システム構成と資産配分) - 7 つのシステムの役割と配分比率
- [各システム詳細](./systems/INDEX.md) - System1-7 の個別仕様（ロング/ショート、エントリー条件など）
- [パフォーマンス指標](#kpi) - バックテストの見方と評価基準

### 🏃 運用ガイド

- [運用ガイド一覧](./operations/INDEX.md) - 自動実行・通知・監視設定
- [文字化け対策ガイド（Windows）](./operations/mojibake_guide.md) - UTF-8/NO_EMOJI の設定

---

## 🔧 開発者向けガイド

このセクションは、コードを変更・拡張・テストする開発者向けの技術資料です。

### 🧪 テストと検証

- [テスト実行](./testing.md) - システム動作確認の基本

  - **統合制御テスト**（systems 1-6）: `tests/test_systems_controlled_all.py` を実行する短い検証。開発中や変更適用後に素早くランク付けと最終エントリ数の整合を確認するために使います。

    **Windows (PowerShell):**

    ```powershell
    python scripts/run_controlled_tests.py
    # または直接 pytest を使う場合
    python -m pytest -q tests/test_systems_controlled_all.py
    ```

    **Unix/Linux/Mac:**

    ```bash
    python3 scripts/run_controlled_tests.py
    # または直接 pytest を使う場合
    python3 -m pytest -q tests/test_systems_controlled_all.py
    # または Makefile を使う場合
    make test-controlled
    ```

### 技術文書

- [技術文書一覧](./technical/INDEX.md) - 指標・実装・仕様の詳細資料
- [キャッシュインデックス要件](./technical/cache_index_requirements.md) - Feather 形式の制約と日付インデックス変換
- [候補数ゼロガイド](./technical/zero_candidates_guide.md) - System6 等で候補が出ない理由（正常動作）
- [環境変数一覧](./technical/environment_variables.md) - 既定値と用途
- [SPY/取引日ユーティリティ](./technical/spy_utils.md) - 営業日ヘルパの仕様
- [Playwright E2E テスト統合](./technical/playwright_integration.md) - Streamlit UI 自動テスト

### 📖 Context Note 形式

このプロジェクトでは、各ソースファイルの先頭に **「Context Note」** という設計意図・注意点をコメント形式で記載しています。以下の場面で参照してください：

- **コード変更前**: ファイルの役割・前提条件・禁止事項を確認して、設計をズレさせない
- **Copilot と相談するとき**: Context Note で Copilot が文脈を把握し、より正確な提案を得られる
- **レビュー・デバッグ時**: README.md の抽象情報を補足する具体的なガイドとして機能

---

## 📚 共通リファレンス

以下のセクションは、初心者・開発者共通で参照する詳細資料です。

### 📖 Context Note 形式

このプロジェクトでは、各ソースファイルの先頭に **「Context Note」** という設計意図・注意点をコメント形式で記載しています。以下の場面で参照してください：

- **コード変更前**: ファイルの役割・前提条件・禁止事項を確認して、設計をズレさせない
- **Copilot と相談するとき**: Context Note で Copilot が文脈を把握し、より正確な提案を得られる
- **レビュー・デバッグ時**: README.md の抽象情報を補足する具体的なガイドとして機能

#### 📋 Context Note の構成

```python
# 🧠 Context Note
# このファイルは【役割・責務】
#
# 前提条件：
#   - 【設計前提1】
#   - 【設計前提2】
#
# ロジック単位：
#   function_name() → 【役割】
#
# Copilot へ：
#   → 【重点領域・禁止事項】
```

#### 📂 主要ファイルの Context Note 一覧

| ファイル                           | 役割                                                     |
| ---------------------------------- | -------------------------------------------------------- |
| `core/systemX.py`                  | エントリー・ランキング・フィルタロジック（ロジック重視） |
| `strategies/systemX_strategy.py`   | core のラッパー層（UI 連携）                             |
| `scripts/run_all_systems_today.py` | 当日パイプライン全体（フロー管理）                       |
| `common/cache_manager.py`          | キャッシュ層（I/O 統一、直接アクセス禁止）               |
| `apps/app_integrated.py`           | Streamlit UI（進捗表示・タブ管理）                       |

詳細は各ファイルの先頭 Context Note を参照してください。

**新規ファイル追加・リファクタリング時の Context Note 記載ガイド**: [Context Note 追記ガイド](./context_note_guide.md)

### 🔗 関連リンク

- [メイン README](../README.md) - プロジェクト全体概要
- [変更履歴](../CHANGELOG.md) - リリースノート
- [GitHub Instructions](../.github/copilot-instructions.md) - AI 開発ガイド

---

## システム構成と資産配分

4 つの買いシステム

- システム 1 ーロング・トレンド・ハイ・モメンタム（トレード資産の 25%を配分）
- システム 4 ーロング・トレンド・ロー・ボラティリティ（トレード資産の 25%を配分）
- システム 3 ーロング・ミーン・リバージョン・セルオフ（トレード資産の 25%を配分）
- システム 5 ーロング・ミーン・リバージョン・ハイ ADX・リバーサル（トレード資産の 25%を配分）

3 つの売りシステム

- システム 2 ーショート RSI スラスト（トレード資産の 40%を配分）
- システム 6 ーショート・ミーン・リバージョン・ハイ・シックスデイサージ（トレード資産の 40%を配分）
- システム 7 ーカタストロフィーヘッジ（トレード資産の 20%を配分）

### モニタリング（daily_metrics.csv）

- 出力先: `results_csv/daily_metrics.csv`
- 生成タイミング: `scripts/run_all_systems_today.py` 実行時（当日シグナル抽出の終盤）
- カラム:
  - `date`: NYSE 最新営業日
  - `system`: `system1`〜`system7`
  - `prefilter_pass`: 事前フィルター通過銘柄数（system7 は SPY の有無で 1/0）
  - `candidates`: 当日候補数（最終スコアリング前のシステム別集計）
- 用途: 事前フィルターの通過数と候補数の推移を日次で可視化し、データ品質やシグナル強度の変動を監視する。

#### UI: Metrics タブ

- `app_integrated.py` のタブに `Metrics` を追加。`results_csv/daily_metrics.csv` を読み込み、システム別に `prefilter_pass` と `candidates` の推移をライン／バーで表示できる。

#### 検証レポート（任意）

- `tools/build_metrics_report.py` が最新日のメトリクスと各システムのシグナル CSV（`signals_systemX_YYYY-MM-DD.csv`）を突き合わせ、`results_csv/daily_metrics_report.csv` を生成する。件数の齟齬チェックやサンプル銘柄の目視確認に使う。

---

## 半自動の検証ループ運用

AI と一緒にコード修正を進めるときは、テストと画像確認を毎回セットで行います。ここでは必ず確認すべき流れをまとめます。

1. **初回の基準づくり**: `make verify` もしくは PowerShell で `./tools/verify.ps1` を実行して、テストとスナップショットと画像差分をそろえます。画像が別フォルダに出力される場合は `make verify IMGDIR=outputs/images` のように指定します。
2. **半自動ループの開始**: `python tools/auto_refine_loop.py` を実行すると、テスト → スナップショット → 画像差分まで自動で進みます。差分が残った場合は Copilot に貼り付けるプロンプトが表示され、AI が修正案を示します。
3. **人の判断ポイント**: AI が提案した修正を確認し、適用するかスキップするかをコンソールで選びます。適用後はファイルを保存し、Enter を押すだけで次のサイクルへ進みます。
4. **終了条件**: 差分がなくなると自動で終了します。回数を制限したい場合は `python tools/auto_refine_loop.py --max-iterations 3` のように指定します。PowerShell の補助スクリプト `tools/auto_refine.ps1` でも同じ指定が可能です。
5. **成果の保管**: スナップショットは `snapshots/<timestamp>/` に保存され、`imgdiff_report.html` で差分の有無を振り返れます。後から比較したいときは `tools/imgdiff.py --snap-a ... --snap-b ...` を手動で呼び出しても構いません。

完全自動ではありませんが、AI が差分を読み取り、自律的に提案を出してくれるので、開発者は提案の良し悪しを判断するだけでループを回せます。差分が減らない場合はフォントや乱数など再現性の要因を疑い、必要であれば `common/testing.py::set_test_determinism()` で固定化します。

### チャット用プロンプト名

ループ操作を会話で共有するときは「検証ループプロンプト」という呼び名を使います。以下の定型文をチャットに貼れば、チーム全員が同じ流れで対応できます。

```
[検証ループプロンプト]
目的: 変更コードをテストし、スナップショットと画像差分で確認する
手順:
1. make verify （または .\tools\verify.ps1）
2. python tools/auto_refine_loop.py
3. 差分が無ければ終了。差分があれば Copilot への提案と判断を繰り返す
出力: snapshots/<timestamp>/ と imgdiff_report.html を保管
```

### UI スクリーンショット取得（自動実行対応）

**重要**: 現在のスナップショット機能は、ファイルシステム上の成果物（CSV・ログ・画像ファイル）のみを対象としており、**Streamlit UI の画面キャプチャは手動操作が必要でした**。しかし、Playwright を使えば **ボタンクリックから完了画面の撮影まで完全自動化** できます。

#### 1. Playwright セットアップ（初回のみ）

**前提条件:** 仮想環境（venv）を有効化してから実行してください。

**Windows (PowerShell):**

```powershell
# 仮想環境の有効化
.\venv\Scripts\Activate.ps1

# Playwright インストール（約5分、300MBダウンロード）
pip install playwright
playwright install chromium
```

**Unix/Linux/Mac:**

```bash
# 仮想環境の有効化
source venv/bin/activate

# Playwright インストール（約5分、300MBダウンロード）
pip install playwright
playwright install chromium

# ネットワーク制限がある場合（システムライブラリも一緒にインストール）
playwright install chromium --with-deps
```

**トラブルシューティング:**

- インストールが失敗する場合: [Playwright 公式ドキュメント](https://playwright.dev/python/docs/intro)
- プロキシ環境の場合: `HTTPS_PROXY` 環境変数を設定してください

**VSCode 拡張機能（推奨）**:

1. VSCode で拡張機能 "Playwright Test for VSCode" をインストール
2. UI テストのデバッグ、レコーディング、インスペクターが使えるようになります
3. `tools/capture_ui_screenshot.py` のステップ実行やブレークポイントが可能

**拡張機能の利点**:

- **Test Explorer**: VSCode のサイドバーからテストを実行
- **Pick Locator**: UI 要素を選択して Playwright セレクターを自動生成
- **Trace Viewer**: 実行履歴をタイムライン表示
- **Codegen**: ブラウザ操作を自動コード生成

詳細: [Playwright VSCode Extension](https://playwright.dev/docs/getting-started-vscode)

#### 2. 自動実行（ボタンクリック + スクリーンショット + スナップショット）

**推奨**: 統合スクリプトを使う

**Windows (PowerShell):**

```powershell
# PowerShell スクリプト（ワンコマンド）
.\tools\run_and_snapshot.ps1

# オプション: 実行完了まで60秒待機
.\tools\run_and_snapshot.ps1 -WaitAfterClick 60

# オプション: スクリーンショットのみ（スナップショット作成をスキップ）
.\tools\run_and_snapshot.ps1 -SkipSnapshot
```

**Unix/Linux/Mac:**

```bash
# Python スクリプトを直接実行
python3 tools/capture_ui_screenshot.py \
    --url http://localhost:8501 \
    --output results_images/today_signals_complete.png \
    --click-button "Generate Signals" \
    --wait-after-click 30

# Makefile を使う場合
make run-and-snapshot
```

**内部動作**:

1. Playwright が Streamlit アプリ (`http://localhost:8501`) を開く
2. 「Generate Signals」ボタンを自動クリック
3. 実行完了を待機（デフォルト 30 秒、進行状況バーの消失を検出）
4. フルページスクリーンショットを `results_images/today_signals_complete.png` に保存
   - **ディレクトリ自動作成**: 出力先ディレクトリ（`results_images/`）が存在しない場合は自動的に作成されます
5. `results_csv`, `logs`, `results_images` をスナップショット

**画像パスのガイドライン**:

- スクリーンショット出力先は `results_images/` または `screenshots/` を推奨（`.gitignore` で管理済み）
- カスタムパスを指定する場合: スクリプトが親ディレクトリを自動作成するため、事前準備は不要です
- 例: `--output custom_dir/subdir/image.png` → `custom_dir/subdir/` が自動作成される

#### 3. 手動実行（カスタマイズが必要な場合）

**Windows (PowerShell):**

```powershell
# ボタンクリック + スクリーンショット
python tools/capture_ui_screenshot.py `
    --url http://localhost:8501 `
    --output results_images/today_signals_complete.png `
    --click-button "Generate Signals" `
    --wait-after-click 30

# スクリーンショットのみ（ボタンクリックなし）
python tools/capture_ui_screenshot.py --url http://localhost:8501 --output screenshots/ui_snapshot.png

# カスタムディレクトリに保存（親ディレクトリが自動作成される）
python tools/capture_ui_screenshot.py --url http://localhost:8501 --output my_reports/2024-11/ui_final.png
```

**Unix/Linux/Mac:**

```bash
# ボタンクリック + スクリーンショット
python3 tools/capture_ui_screenshot.py \
    --url http://localhost:8501 \
    --output results_images/today_signals_complete.png \
    --click-button "Generate Signals" \
    --wait-after-click 30

# スクリーンショットのみ（ボタンクリックなし）
python3 tools/capture_ui_screenshot.py --url http://localhost:8501 --output screenshots/ui_snapshot.png

# カスタムディレクトリに保存（親ディレクトリが自動作成される）
python3 tools/capture_ui_screenshot.py --url http://localhost:8501 --output my_reports/2024-11/ui_final.png
```

**注意事項**:

- `results_images/` と `screenshots/` は `.gitignore` で除外されているため、Git に追跡されません
- PR やレポート用の画像は手動で管理するか、別の専用ディレクトリ（例: `docs/images/`）にコピーしてください

**利用可能なオプション**:

- `--url`: Streamlit アプリの URL（デフォルト: `http://localhost:8501`）
- `--output`: 保存先パス（プロジェクトルートからの相対パス）
  -- `--click-button`: クリックするボタンのテキスト（例: `"Generate Signals"`）
- `--wait-after-click`: ボタンクリック後の待機時間（秒）（デフォルト: 15）
- `--no-scroll`: 最下部へのスクロールを無効化（デフォルトは有効）
- `--wait`: ページ読み込み後の待機時間（秒）（デフォルト: 3）

#### 4. チャットから指定（AI コーディング時）

会話で「検証ループプロンプト」と言えば、AI が自動的に以下を実行します：

```
[検証ループプロンプト - UI版]
目的: Streamlit UIの完了画面を自動撮影し、スナップショット比較
手順:
1. .\tools\run_and_snapshot.ps1 でボタンクリック→撮影→スナップショット
2. python tools/imgdiff.py で差分確認
3. 差分があれば Copilot に修正提案を依頼
完全自動: ボタンクリック、待機、スクロール、フルページ撮影まですべて自動
```

#### 5. 注意事項

- **事前条件**: Streamlit アプリが `http://localhost:8501` で起動していること
- **フルページ**: デフォルトで最下部までスクロール後、全体を撮影
- **待機時間**: 処理が重い場合は `--wait-after-click` を延長（例: 60 秒）
  -- **ボタン名**: デフォルトは英語表記 `"Generate Signals"`。日本語 UI を使う場合は絵文字を含めた日本語表記（例: `"▶ 本日のシグナル実行"`）を指定してください。
- **環境依存**: フォント・レンダリングの微差が発生する可能性あり（決定性は `common/testing.py::set_test_determinism()` で対応）
- **Git 管理**: `snapshots/`, `results_images/`, `screenshots/` は `.gitignore` で除外されています（テスト用一時ファイルのため）

#### 6. Makefile ターゲット（Unix/Linux/Mac）

```bash
make run-and-snapshot
```

内部的に `tools/capture_ui_screenshot.py` と `tools/snapshot.py` を連続実行します。

#### 7. Image Diff Report（ダークモード対応）

`tools/imgdiff.py` で生成される HTML レポートは **ダークモード** で表示されます：

**特徴**:

- VSCode スタイルの配色（背景: `#1e1e1e`, テキスト: `#d4d4d4`）
- 画像ホバー時の拡大表示（スムーズなトランジション）
- 差分がない場合は緑色のチェックマーク付きメッセージ
- レスポンシブデザイン（モバイル対応）

**生成場所**:

- `snapshots/<最新>/imgdiff_report.html`
- ブラウザで開くと自動的にダークモードで表示されます

**カスタマイズ**:
配色を変更したい場合は `tools/imgdiff.py` の `_build_report()` 関数内の CSS を編集してください。
