# Quant Trading System - ドキュメント総覧

このドキュメントは、7 つの売買システムを統合したクオンツトレーディングシステムの包括的なガイドです。

## 📚 ドキュメント構成

### 🚀 [クイックスタート](#quick-start)

- [セットアップ](../README.md#セットアップ) - 初回環境構築
- [基本実行](../README.md#実行例) - UI 起動と基本操作
- [テスト実行](./testing.md) - システム動作確認

### 📊 [システム概要](#trading-systems)

- [システム構成と資産配分](#システム構成と資産配分)
- [各システム詳細](./systems/) - System1-7 の個別仕様
- [パフォーマンス指標](#kpi)

### 🔧 [技術文書](#technical-docs)

- [キャッシュインデックス要件](./technical/cache_index_requirements.md) - Feather 形式の制約と日付インデックス変換
- [候補数ゼロガイド](./technical/zero_candidates_guide.md) - System6 等で候補が出ない理由(正常動作)
- [環境変数一覧](./technical/environment_variables.md) - 既定値と用途
- [SPY/取引日ユーティリティ](./technical/spy_utils.md) - 営業日ヘルパの仕様

### 🏃 [運用ガイド](#operations)

- [自動実行設定](./schedule_quick_start.md) - Windows Task Scheduler
- [通知設定](./NOTIFICATIONS.md) - Slack/Discord 連携
- [UI メトリクス](./today_signals_ui_metrics.md) - ダッシュボード
- [Bulk API 品質ガイド](./operations/bulk_api_quality_guide.md) - データ品質検証と設定

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

**Python パッケージとブラウザのインストール**:

```powershell
# venv 環境で実行
pip install playwright
playwright install chromium  # 約300MB
```

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

```powershell
# PowerShell スクリプト（ワンコマンド）
.\tools\run_and_snapshot.ps1

# オプション: 実行完了まで60秒待機
.\tools\run_and_snapshot.ps1 -WaitAfterClick 60

# オプション: スクリーンショットのみ（スナップショット作成をスキップ）
.\tools\run_and_snapshot.ps1 -SkipSnapshot
```

**内部動作**:

1. Playwright が Streamlit アプリ (`http://localhost:8501`) を開く
2. 「▶ 本日のシグナル実行」ボタンを自動クリック
3. 実行完了を待機（デフォルト 30 秒、進行状況バーの消失を検出）
4. フルページスクリーンショットを `results_images/today_signals_complete.png` に保存
5. `results_csv`, `logs`, `results_images` をスナップショット

#### 3. 手動実行（カスタマイズが必要な場合）

**ボタンクリック + スクリーンショット**:

```powershell
python tools/capture_ui_screenshot.py `
    --url http://localhost:8501 `
    --output results_images/today_signals_complete.png `
    --click-button "▶ 本日のシグナル実行" `
    --wait-after-click 30
```

**スクリーンショットのみ（ボタンクリックなし）**:

```powershell
python tools/capture_ui_screenshot.py --url http://localhost:8501 --output screenshots/ui_snapshot.png
```

**利用可能なオプション**:

- `--url`: Streamlit アプリの URL（デフォルト: `http://localhost:8501`）
- `--output`: 保存先パス（プロジェクトルートからの相対パス）
- `--click-button`: クリックするボタンのテキスト（例: `"▶ 本日のシグナル実行"`）
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
- **ボタン名**: UI が日本語の場合、絵文字も含めて正確に指定（例: `"▶ 本日のシグナル実行"`）
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
