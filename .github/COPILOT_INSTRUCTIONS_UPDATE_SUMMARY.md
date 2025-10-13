# Copilot Instructions Update Summary

**Updated**: 2025-10-13  
**Status**: ✅ Successfully updated

## Overview

The `.github/copilot-instructions.md` file has been analyzed and updated to better guide AI coding agents working in this quant trading system codebase. The existing documentation was already comprehensive, and this update adds recent improvements and best practices discovered through codebase analysis.

## Key Updates Made

### 1. **VS Code Tasks Integration** (New Section)

- Added explicit mention of VS Code task runner integration
- Highlighted quick commands via `Ctrl+Shift+P` → "Run Task"
- Documented "Safe: 〜" tasks that use `tools/safe_exec.ps1` for UTF-8/error handling

**Why**: Developers can now leverage pre-configured tasks for common workflows instead of typing long commands.

### 2. **Test Determinism Enhancements** (Expanded)

- Clarified that `conftest.py` provides automatic test determinism via fixtures
- Emphasized that manual `set_test_determinism()` calls are unnecessary in new tests
- Added specific details: freezegun time, random seeds, pytest-xdist compatibility

**Why**: Prevents confusion about when/how to achieve deterministic tests; the framework handles it automatically.

### 3. **E2E Testing with Playwright** (New Section)

- Complete Playwright setup instructions (pip install + browser installation)
- VSCode extension recommendations for UI test development
- Multiple execution modes (headless, headed, UI, quick)
- UI screenshot automation workflow with `tools/capture_ui_screenshot.py`

**Why**: UI testing was mentioned in docs but not in the main instructions file; now AI agents know how to test Streamlit UI changes.

### 4. **Recent Architectural Improvements** (New Section)

Added documentation for October 2025 improvements:

- **Bulk API Quality Management**: Volume tolerance relaxation (0.5% → 5.0%)
- **Setup Predicates Integration**: DRY principle applied to all systems
- **Cache Index Requirements**: Feather format DatetimeIndex requirement
- **System7 Coverage**: 52% → 66% improvement
- **Pytest-cov Compatibility**: `.reindex()` pattern for NumPy compatibility

**Why**: AI agents need to understand recent architectural decisions to maintain consistency.

### 5. **Enhanced Checklist** (Expanded)

Added two new checklist items:

- ✅ Test determinism verification (automatic via conftest.py)
- ✅ VS Code task definition for new scripts (optional but recommended)

**Why**: Ensures AI agents consider these aspects when making changes.

## What Was Preserved

The following valuable content was intentionally kept unchanged:

1. **Mandatory Documentation Reference** (Section 0): Links to `docs/README.md` hierarchy
2. **Core Architecture** (Sections 1-3): Entry points, cache hierarchy, system characteristics
3. **Configuration Management** (Section 4): Priority order (JSON > YAML > .env)
4. **Development Workflow** (Section 5): Essential commands and test strategies
5. **Guardrails** (Section 6): Breaking change prohibitions
6. **Implementation Patterns** (Section 7): Two-phase, logging, strategy/core separation
7. **Code Quality** (Section 8): Style guides, pre-commit hooks, mypy usage

## Architecture Highlights Discovered

During analysis, the following architectural patterns were confirmed and documented:

### Data Flow Architecture

```
External APIs → full_backup/ (原本) → base/ (指標付与) → rolling/ (直近300日)
                                    ↓
                              Feather優先、CSV自動フォールバック
```

### Two-Phase Signal Processing

```
Phase 2: Filter列生成 (today_filters.py)
    ↓
Phase 6: Setup Predicates適用 (system_setup_predicates.py)
    ↓
Ranking & Allocation (core/systemX.py)
```

### Strategy Pattern

```
core/systemX.py (純粋ロジック)
    ↓
strategies/systemX_strategy.py (StrategyBase継承ラッパ)
    ↓
common/integrated_backtest.py (統合バックテスト)
```

## Codebase Quality Metrics

Discovered during analysis:

| Metric                  | Value     | Notes                            |
| ----------------------- | --------- | -------------------------------- |
| Test Coverage (System7) | 66%       | ✅ Target achieved (was 52%)     |
| Fast Test Mode          | 2 seconds | `--test-mode mini` on 10 symbols |
| Cache Size Reduction    | 74%       | Feather vs CSV format            |
| Column Deduplication    | 40%       | 58→35 columns (OHLCV PascalCase) |
| Pre-commit Checks       | 5 stages  | ruff/black/isort/basic/pre-push  |

## Critical Patterns for AI Agents

### ✅ DO

- Use `CacheManager` for ALL data access
- Use `get_env_config()` for environment variables
- Preserve `DEFAULT_LONG_ALLOCATIONS` / `DEFAULT_SHORT_ALLOCATIONS`
- Keep System7 fixed to SPY only
- Run `--test-mode mini --skip-external` before committing
- Use `SystemLogger` for unified logging

### ❌ DON'T

- Direct `pd.read_csv("data_cache/...")` access
- `os.environ.get()` direct usage
- Breaking changes to CLI flags or public APIs
- Adding network calls to test paths without `--skip-external` compatibility
- Manual `set_test_determinism()` in new tests (handled by conftest.py)

## Testing Integration Points

### Unit Tests

- `pytest -q`: Fast deterministic tests
- `pytest --cov=core --cov-report=term-missing`: Coverage measurement
- Automatic determinism via `conftest.py` fixtures

### Integration Tests

- `--test-mode mini`: 10 symbols, 2 seconds
- `--test-mode quick`: 50 symbols
- `--test-mode sample`: 100 symbols
- `--skip-external`: Bypass NASDAQ Trader / market calendars

### E2E Tests

- `npm test`: Playwright headless
- `npm run test:headed`: Browser visible
- `tools/run_and_snapshot.ps1`: UI screenshot automation

## Environment Variables (Type-Safe Access)

The codebase uses `config/environment.py::EnvironmentConfig` for type-safe access:

```python
from config.environment import get_env_config

env = get_env_config()  # シングルトン
if env.compact_logs:
    logger.setLevel(logging.WARNING)

# Validation (危険な設定を検出)
errors = env.validate()
if errors:
    for err in errors:
        logger.warning(err)
```

**All 40+ environment variables** are documented in `docs/technical/environment_variables.md`.

## Documentation Structure

The project has exceptional documentation organization:

```
docs/
├── README.md (統合目次 - ALWAYS START HERE)
├── systems/ (System1-7仕様書)
├── technical/ (アーキテクチャ・指標計算・処理フロー)
├── operations/ (自動実行・通知・監視)
└── today_signal_scan/ (8フェーズ詳細)
```

## Feedback Requested

Please review the updated `.github/copilot-instructions.md` and provide feedback on:

1. **Completeness**: Are there any critical patterns or workflows missing?
2. **Clarity**: Are the instructions clear and actionable for AI agents?
3. **Accuracy**: Do the technical details accurately reflect the codebase?
4. **Priorities**: Should any sections be reordered or emphasized differently?

## Next Steps

Consider these optional enhancements:

1. **Add Examples Section**: Concrete code snippets for common tasks
2. **Quick Reference Card**: One-page cheat sheet for most frequent operations
3. **Troubleshooting Guide**: Common errors and their solutions
4. **Architecture Diagrams**: Visual representations of data flow and component interactions

---

**Generated by**: GitHub Copilot Code Review
**Review Status**: Ready for human review
**Confidence**: High (based on comprehensive codebase analysis)
