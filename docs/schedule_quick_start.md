Quick start — schedule `run_auto_rule`

1. Verify your virtualenv and environment variables (ALPACA keys, optional SLACK/Discord tokens).
2. Validate the script works manually:

```powershell
python .\scripts\run_auto_rule.py --dry-run
```

3. Register the task (example): open an elevated PowerShell and run:

```powershell
# register via helper script
powershell -ExecutionPolicy Bypass -File "C:\Repos\quant_trading_system\docs\register_task_examples.ps1"
```

4. Confirm the task exists and test-run it:

```powershell
Start-ScheduledTask -TaskName AutoRuleDaily
Get-ScheduledTask -TaskName AutoRuleDaily | Format-List *
```

Notes

- The example registers a weekday (Mon–Fri) 08:00 run with `-DryRun` (no actual orders). Remove `-DryRun` / add `-Paper` when ready for live testing.
- If your environment uses a virtualenv in `.venv`, make sure `scripts/run_auto_rule.ps1` can activate it or adjust the wrapper accordingly.
