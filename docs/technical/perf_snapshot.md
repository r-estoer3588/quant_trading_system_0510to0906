# パフォーマンススナップショット (Perf Snapshot)

当日パイプライン / システム実行の軽量なランタイム計測を JSON で記録する仕組みです。`--perf-snapshot` フラグを `scripts/run_all_systems_today.py` に付与すると、処理完了時に `logs/perf_snapshots/` 配下へ 1 ファイル書き出します。

## 目的

- latest_only/フル実行の所要時間差を継続観測
- システム別の相対的ボトルネック把握
- キャッシュ I/O（feather/csv）の増減トラッキング
- 候補生成件数 (candidate_count) の推移監視

## 出力ファイル例

```
logs/perf_snapshots/perf_2025-10-03_142501_latest.json
logs/perf_snapshots/perf_2025-10-03_142732_full.json
```

プレフィックス `perf_YYYY-MM-DD_HHMMSS_{latest|full}.json`

## JSON スキーマ

| キー                               | 型            | 説明                                  |
| ---------------------------------- | ------------- | ------------------------------------- |
| schema_version                     | int           | スキーマ版数 (2 から開始)             |
| timestamp                          | str (ISO8601) | 計測完了時刻                          |
| latest_only                        | bool          | latest_only モードか                  |
| total_time_sec                     | float         | 全体経過秒数                          |
| cache_io                           | object        | キャッシュ I/O カウンタ               |
| cache_io.read_feather              | int           | Feather 読み込み回数                  |
| cache_io.read_csv                  | int           | CSV 読み込み回数 (フォールバック)     |
| cache_io.write_feather             | int           | Feather 書き込み回数                  |
| cache_io.write_csv                 | int           | CSV 書き込み回数 (フォールバック)     |
| per_system                         | object        | システム別計測結果 (system1..system7) |
| per_system.systemX.elapsed_sec     | float         | システム候補生成に要した秒数          |
| per_system.systemX.symbol_count    | int           | 候補生成呼び出し時に渡したシンボル数  |
| per_system.systemX.candidate_count | int or null   | 生成された候補件数（解釈不能時 null） |

### candidate_count の定義

`generate_candidates` の戻り値型に応じて以下の規則で推定します。

- `dict`: `len(dict)`
- `list` / `set` / `tuple`: `len(collection)`
- `(dict, DataFrame|None)` 形式 (System6 想定): 先頭要素が `dict` の場合その `len`
- 上記以外・例外発生: `null` （不明扱い）を記録

内部的には `StrategyBase._compute_candidate_count` ヘルパで一元化しています。

## 実装ポイント

- 標準ライブラリのみ使用 (`time`, `json`, `pathlib`, `datetime`)
- グローバルシングルトン: `common.perf_snapshot.enable_global_perf()` で有効化
- 各 strategy の `generate_candidates` 内で `mark_system_start/end` を呼び出し
- 例外は握りつぶし: サービスロジックへ影響しないフェイルセーフ設計
- キャッシュ I/O カウントは `common/cache_file_io.py` など低レイヤでインクリメント

## 使い方

1. 通常実行

```
python scripts/run_all_systems_today.py --perf-snapshot --latest-only
```

2. 生成された JSON を開く

```
less logs/perf_snapshots/perf_2025-10-03_142501_latest.json
```

## 比較ユーティリティ

`scripts/compare_perf_snapshots.py` で 2 つの JSON を比較し、所要時間や I/O の差分と % を表示します。

### 例

```
python scripts/compare_perf_snapshots.py logs/perf_snapshots/perf_2025-10-03_142501_latest.json \
    logs/perf_snapshots/perf_2025-10-03_142732_full.json
```

出力（例）:

```
Overall:
  total_time_sec: 12.34 -> 20.11 (+63.0%)
Cache IO:
  read_feather: 120 -> 180 (+50.0%)
Per System:
  system1 elapsed_sec: 1.20 -> 1.80 (+50.0%) candidates: 15 -> 15 (0.0%)
  ...
```

### CI / 回帰チェックでの簡易利用例

高速モード (latest_only) とフルモードをそれぞれ 1 回走らせて保存後、閾値判定で落とす簡易スクリプト例:

```python
import json, sys, subprocess, glob
subprocess.run(["python", "scripts/run_all_systems_today.py", "--perf-snapshot", "--latest-only"], check=True)
subprocess.run(["python", "scripts/run_all_systems_today.py", "--perf-snapshot"], check=True)
files = sorted(glob.glob("logs/perf_snapshots/perf_*_latest.json"))
latest = files[-1]
full = sorted(glob.glob("logs/perf_snapshots/perf_*_full.json"))[-1]
with open(latest) as f: a = json.load(f)
with open(full) as f: b = json.load(f)
ratio = b["total_time_sec"]/max(a["total_time_sec"],1e-6)
if ratio > 2.5:
    print(f"WARN: full run slower than expected ratio={ratio:.2f}")
    sys.exit(1)
```

## データ活用の小さなレシピ

直近のスナップショットをまとめて読み込み、システム別の時間や候補数を表形式のデータ（DataFrame）に整える例です。集計用の素朴な入り口として使えます。

```python
from pathlib import Path
import json
import pandas as pd

snapshot_dir = Path("logs/perf_snapshots")
rows = []
for path in sorted(snapshot_dir.glob("perf_*.json"))[-20:]:
  data = json.loads(path.read_text(encoding="utf-8"))
  for system, metrics in data["per_system"].items():
    rows.append({
      "file": path.name,
      "mode": "latest" if "_latest" in path.stem else "full",
      "system": system,
      "elapsed_sec": metrics["elapsed_sec"],
      "symbol_count": metrics["symbol_count"],
      "candidate_count": metrics["candidate_count"],
      "total_time_sec": data["total_time_sec"],
      "timestamp": data["timestamp"],
    })

df = pd.DataFrame(rows)
print(df.groupby(["mode", "system"])["elapsed_sec"].median())
```

## 既知の制約と注意点

- v2 以降: `run()` 開始時にシステム状態と I/O カウンタをリセットするため、同一プロセス内で連続実行しても値が累積しません。
- 1 実行内で同一システムの `mark_system_start` を複数回呼ぶケース（再入）は考慮していません。必要になった場合は複数セクション ID を導入してください。

## 運用のヒント

- 直近 N 実行の中央値比較でノイズ除去（外部 I/O の揺らぎ対策）
- `candidate_count` 急減: フィルタ分岐やデータ欠損の早期検知に活用
- `symbol_count` が不自然に減った場合は前段フェーズ（シンボルロード / フィルタ）を確認

## 変更履歴

- v1: 初版 (elapsed, symbol_count, cache I/O)
- v2: `candidate_count` 追加 / 戦略側ロジックをヘルパ関数へ集約 / `schema_version` 導入 / 不明候補件数は null 記録
