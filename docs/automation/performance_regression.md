# Performance Regression Detection

## 🎯 目的

コード変更によるパフォーマンス劣化を自動検出。

## 📋 実装プラン

### Phase 1: ベンチマーク自動実行

```python
# tools/auto_benchmark.py
"""
コミット前にベンチマークを実行し、パフォーマンス劣化を検出
"""
import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime

BENCHMARK_HISTORY = Path("benchmarks/history.jsonl")

def run_benchmark():
    """ベンチマーク実行"""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_all_systems_today.py",
            "--test-mode", "mini",
            "--skip-external",
            "--benchmark"
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("❌ Benchmark failed")
        return None

    # ベンチマーク結果を読み込み
    benchmark_files = list(Path("results_csv_test").glob("benchmark_*.json"))
    if not benchmark_files:
        print("⚠️  No benchmark file found")
        return None

    latest_benchmark = max(benchmark_files, key=lambda p: p.stat().st_mtime)
    with open(latest_benchmark) as f:
        data = json.load(f)

    return data

def compare_with_baseline(current: dict, baseline: dict, threshold: float = 0.10):
    """ベースラインと比較（10%以上の劣化を検出）"""
    regressions = []

    for phase, current_time in current.get('phase_times', {}).items():
        baseline_time = baseline.get('phase_times', {}).get(phase, 0)

        if baseline_time > 0:
            regression_pct = (current_time - baseline_time) / baseline_time

            if regression_pct > threshold:
                regressions.append({
                    'phase': phase,
                    'baseline': baseline_time,
                    'current': current_time,
                    'regression_pct': regression_pct
                })

    return regressions

def get_baseline():
    """過去7日間の中央値をベースラインとして使用"""
    if not BENCHMARK_HISTORY.exists():
        return None

    import pandas as pd

    # 履歴読み込み
    history = []
    with open(BENCHMARK_HISTORY) as f:
        for line in f:
            history.append(json.loads(line))

    if len(history) < 3:
        return None

    # 直近7日間
    recent = history[-7:]

    # フェーズごとの中央値を計算
    baseline = {'phase_times': {}}

    df = pd.DataFrame([h['phase_times'] for h in recent])
    for col in df.columns:
        baseline['phase_times'][col] = df[col].median()

    return baseline

def save_benchmark(data: dict):
    """ベンチマーク結果を履歴に保存"""
    BENCHMARK_HISTORY.parent.mkdir(parents=True, exist_ok=True)

    record = {
        'timestamp': datetime.now().isoformat(),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
        'phase_times': data.get('phase_times', {}),
        'total_time': data.get('total_time', 0)
    }

    with open(BENCHMARK_HISTORY, 'a') as f:
        f.write(json.dumps(record) + '\n')

def main():
    print("🔍 Running performance benchmark...")

    # ベンチマーク実行
    current = run_benchmark()
    if not current:
        return 1

    # ベースライン取得
    baseline = get_baseline()

    if baseline:
        # 回帰検出
        regressions = compare_with_baseline(current, baseline, threshold=0.10)

        if regressions:
            print("\n⚠️  Performance regressions detected:\n")
            for reg in regressions:
                print(f"  • {reg['phase']}: {reg['regression_pct']:.1%} slower")
                print(f"    Baseline: {reg['baseline']:.2f}s → Current: {reg['current']:.2f}s")

            print("\n❌ Performance degraded by >10%. Review changes.")

            response = input("\nContinue with commit? (y/N): ")
            if response.lower() != 'y':
                return 1
        else:
            print("✅ No performance regressions detected")
    else:
        print("ℹ️  No baseline available (need 3+ historical runs)")

    # 結果を履歴に保存
    save_benchmark(current)

    return 0

if __name__ == "__main__":
    sys.exit(main())
```

### Phase 2: pre-commit 統合

```yaml
# .pre-commit-config.yaml に追加
- repo: local
  hooks:
    - id: performance-check
      name: Performance Regression Check
      entry: python tools/auto_benchmark.py
      language: system
      pass_filenames: false
      files: ^(core|common|strategies)/.*\.py$
      stages: [pre-push]
```

### Phase 3: ベンチマーク履歴可視化

```python
# tools/visualize_benchmarks.py
"""
ベンチマーク履歴をグラフ化
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    history_file = Path("benchmarks/history.jsonl")

    if not history_file.exists():
        print("No benchmark history found")
        return

    # 履歴読み込み
    history = []
    with open(history_file) as f:
        for line in f:
            history.append(json.loads(line))

    # DataFrame 変換
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # フェーズごとのタイムライン
    fig, ax = plt.subplots(figsize=(12, 6))

    phase_times = pd.DataFrame(df['phase_times'].tolist(), index=df['timestamp'])

    for col in phase_times.columns:
        ax.plot(phase_times.index, phase_times[col], marker='o', label=col)

    ax.set_xlabel('Date')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Performance Benchmark History')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    output = Path("benchmarks/performance_trend.png")
    plt.savefig(output, dpi=150)
    print(f"✅ Chart saved: {output}")

if __name__ == "__main__":
    main()
```

## 🔧 使い方

### 自動実行（pre-push）

```powershell
git push origin branch0906

# → 自動でベンチマーク実行 & 比較
```

### 手動実行

```powershell
# ベンチマーク実行
python tools/auto_benchmark.py

# 履歴可視化
python tools/visualize_benchmarks.py
```

## 📈 メリット

- ✅ パフォーマンス劣化の早期検出
- ✅ 最適化の効果測定
- ✅ 長期的なパフォーマンストレンド把握
- ✅ ボトルネック特定の補助
