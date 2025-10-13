@echo off
echo ========================================
echo フル実行（6000+銘柄）開始
echo ========================================
echo.

set ENABLE_PROGRESS_EVENTS=1
set EXPORT_DIAGNOSTICS_SNAPSHOT_ALWAYS=1

echo 環境変数設定:
echo   ENABLE_PROGRESS_EVENTS=%ENABLE_PROGRESS_EVENTS%
echo   EXPORT_DIAGNOSTICS_SNAPSHOT_ALWAYS=%EXPORT_DIAGNOSTICS_SNAPSHOT_ALWAYS%
echo.

echo 仮想環境を有効化中...
call venv\Scripts\activate.bat

echo.
echo Pythonスクリプト実行中...
echo.

python scripts/run_all_systems_today.py --parallel --save-csv

echo.
echo ========================================
echo 実行完了
echo ========================================
pause
