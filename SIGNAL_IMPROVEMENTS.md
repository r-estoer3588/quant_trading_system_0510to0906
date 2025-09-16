# Signal Extraction System Improvements

## Overview
This document outlines the improvements made to address signal extraction issues identified in the problem statement.

## Issues Addressed

### 1. System5/6 Skip Aggregation (②)
**Problem**: System6 showed confusing skip aggregation like "🧪 スキップ内訳: insufficient_rows: 209, 209 件: 1"

**Solution**: Modified `core/system6.py` to avoid mixing skip_callback aggregation with log summary output.

### 2. Insufficient Rows Logic (①)
**Problem**: System5 showed "insufficient_rows" even when rolling CSV had 240+ rows.

**Solution**: Enhanced diagnostics in `core/system5.py`:
- Added detailed error messages showing row reduction: `insufficient_rows_after_dropna_75_from_240`
- Tracks original row count vs. post-dropna count
- The issue occurs because SMA100 requires 100 clean rows after removing missing High/Low/Close data

### 3. Shortable Stock Checking (③)
**Problem**: Need to check if stocks are borrowable for short selling.

**Solution**: Implemented shortable checking for System2 and System6:
- Added `check_shortable_stocks()` function in `common/broker_alpaca.py`
- Integrated checks in `strategies/system2_strategy.py` and `strategies/system6_strategy.py`
- Controlled by `settings.risk.enable_shortable_check` configuration
- Logs excluded symbols: "🚫 System2: WLDS - ショート不可のため除外"

### 4. System6 Date Display (④)
**Problem**: All entries showing same date (9/15) despite different entry dates.

**Solution**: Added diagnostic logging in `common/today_signals.py`:
- Logs search dates, candidate dates, and selected date
- Format: "🗓️ System6日付選択: 今日=2024-09-16, 検索順=[09-16, 09-15], 候補日=[09-15], 選択=09-15"

### 5. UI Improvements (⑤⑥)
**Problem**: Unnecessary CSV download buttons and order message text.

**Solution**: 
- Removed system-specific CSV download buttons in `common/ui_components.py`
- Order message already updated to: "約定反映後の資金余力でLong/Shortを再設定しました:"

### 6. JSON Status Display (⑷)
**Problem**: Unwanted JSON status output in UI.

**Solution**: Already addressed - JSON display replaced with simple notification: "注文状況を更新しました（詳細はログ参照）"

### 7. Long/Short Display (⑧)
**Problem**: Need "-" display for non-applicable systems.

**Solution**: Modified `app_today_signals.py` to show separate Long_Position and Short_Position columns:
- System1,3,4,5 (Long): Show symbol in Long_Position, "-" in Short_Position
- System2,6,7 (Short): Show symbol in Short_Position, "-" in Long_Position

### 8. System Classification Fix
**Problem**: System4 was incorrectly classified as short system.

**Solution**: Corrected `common/today_signals.py`:
- LONG_SYSTEMS: {system1, system3, system4, system5}
- SHORT_SYSTEMS: {system2, system6, system7}

### 9. Skip Reason Export (⑨)
**Problem**: Need better visibility into skipped symbols.

**Solution**: Added CSV export functionality:
- Exports to `results_dir/skipped_symbols_{system}_{YYYYMMDD}.csv`
- Includes symbol, skip_reason, reason_total_count, sample_order
- Logs: "📄 スキップ詳細をCSV出力: {path}"

## Configuration Options

### Shortable Checking
```yaml
risk:
  enable_shortable_check: false  # Set to true to enable shortable validation
```

### Diagnostic Output
- Enhanced error messages automatically included
- Skip reason CSV export automatically generated when skips occur
- Date validation logging automatically enabled for System6

## Testing
All improvements have been validated:
- ✓ System5/6 import and syntax check
- ✓ Long/Short system classification consistency
- ✓ Shortable check function availability
- ✓ Enhanced diagnostic message format

## Files Modified
- `core/system5.py` - Enhanced diagnostics
- `core/system6.py` - Fixed skip aggregation  
- `strategies/system2_strategy.py` - Added shortable checks
- `strategies/system6_strategy.py` - Added shortable checks
- `common/broker_alpaca.py` - Added shortable check function
- `common/today_signals.py` - Fixed system classification, added date validation, skip export
- `common/ui_components.py` - Removed unnecessary download buttons
- `app_today_signals.py` - Added Long/Short display with "-" markers