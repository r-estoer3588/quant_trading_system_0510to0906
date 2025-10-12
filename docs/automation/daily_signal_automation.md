# Daily Signal Generation Automation

## 🎯 目的

毎日決まった時間にシグナル生成を自動実行し、結果を Slack/メールで通知。

## 📋 実装プラン

### Phase 1: Windows タスクスケジューラ統合

```powershell
# tools/schedule_daily_signals.ps1
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument `
    "-NoProfile -ExecutionPolicy Bypass -File C:\Repos\quant_trading_system\scripts\daily_auto_run.ps1"

$Trigger = New-ScheduledTaskTrigger -Daily -At "16:30" # NYSE 終了後

$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable

Register-ScheduledTask -TaskName "Quant_DailySignals" -Action $Action -Trigger $Trigger -Settings $Settings
```

### Phase 2: 実行スクリプト

```powershell
# scripts/daily_auto_run.ps1
param(
    [switch]$DryRun = $false
)

$ErrorActionPreference = "Stop"
$LogFile = "logs/auto_run_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

try {
    # 1. Python 環境アクティベート
    & "C:\Repos\quant_trading_system\venv\Scripts\Activate.ps1"

    # 2. シグナル生成
    Write-Output "[$([DateTime]::Now)] Starting signal generation..." | Tee-Object -FilePath $LogFile -Append
    python scripts/run_all_systems_today.py --parallel --save-csv 2>&1 | Tee-Object -FilePath $LogFile -Append

    if ($LASTEXITCODE -ne 0) {
        throw "Signal generation failed with exit code $LASTEXITCODE"
    }

    # 3. 結果サマリー生成
    $LatestCSV = Get-ChildItem "data_cache\signals\signals_final_*.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $SignalCount = (Import-Csv $LatestCSV.FullName).Count

    # 4. Slack 通知
    if (-not $DryRun) {
        python scripts/notify_results.py --signals $SignalCount --log $LogFile
    }

    Write-Output "[$([DateTime]::Now)] ✅ Auto-run completed: $SignalCount signals" | Tee-Object -FilePath $LogFile -Append

} catch {
    Write-Error "[$([DateTime]::Now)] ❌ Auto-run failed: $_" | Tee-Object -FilePath $LogFile -Append

    # エラー通知
    if (-not $DryRun) {
        python scripts/notify_error.py --error "$_" --log $LogFile
    }

    exit 1
}
```

### Phase 3: Slack 通知スクリプト

```python
# scripts/notify_results.py
import os
import sys
import argparse
from pathlib import Path
from common.notification import send_slack_message

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", type=int, required=True)
    parser.add_argument("--log", type=str, required=True)
    args = parser.parse_args()

    # サマリー作成
    message = f"""
📊 *Daily Signal Generation Complete*
• Signals: {args.signals}
• Log: `{args.log}`
• Status: ✅ Success
    """.strip()

    send_slack_message(message, channel="#trading-signals")

if __name__ == "__main__":
    main()
```

## 🔧 セットアップ手順

1. スケジューラ登録

   ```powershell
   .\tools\schedule_daily_signals.ps1
   ```

2. ドライラン確認

   ```powershell
   .\scripts\daily_auto_run.ps1 -DryRun
   ```

3. 手動実行テスト
   ```powershell
   Start-ScheduledTask -TaskName "Quant_DailySignals"
   ```

## 📈 メリット

- ✅ 毎日 16:30 に自動実行（NYSE 終了後）
- ✅ 結果を Slack で即座に確認
- ✅ エラー時も通知で気づける
- ✅ ログ自動保存で履歴追跡

## 🔄 GitHub Actions 版（クラウド実行）

```yaml
# .github/workflows/daily-signals.yml
name: Daily Signal Generation

on:
  schedule:
    - cron: "30 20 * * 1-5" # UTC 20:30 = EST 16:30 (平日のみ)
  workflow_dispatch:

jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run signal generation
        run: |
          python scripts/run_all_systems_today.py --parallel --save-csv
        env:
          COMPACT_TODAY_LOGS: "1"

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: daily-signals-${{ github.run_number }}
          path: data_cache/signals/signals_final_*.csv

      - name: Notify Slack
        if: always()
        uses: slackapi/slack-github-action@v1
        with:
          channel-id: "trading-signals"
          slack-message: "Daily signals: ${{ job.status }}"
        env:
          SLACK_BOT_TOKEN: ${{ secrets.SLACK_BOT_TOKEN }}
```
