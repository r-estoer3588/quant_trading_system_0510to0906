# core.system1

System1 core logic (Long ROC200 momentum).

ROC200-based momentum strategy:
- Indicators: ROC200, SMA200, DollarVolume20 (precomputed only)
- Setup conditions: Close>5, DollarVolume20>25M, Close>SMA200, ROC200>0
- Candidate generation: ROC200 descending ranking by date, extract top_n
- Optimization: Removed all indicator calculations, using precomputed indicators only

---

## クラス一覧

## クラス: `System1Diagnostics`

Lightweight diagnostics payload for System1 candidate generation.

### メソッド

#### `__init__(self, mode: 'str' = 'full_scan', top_n: 'int' = 20, symbols_total: 'int' = 0, symbols_with_data: 'int' = 0, total_symbols: 'int' = 0, filter_pass: 'int' = 0, setup_flag_true: 'int' = 0, fallback_pass: 'int' = 0, roc200_positive: 'int' = 0, final_pass: 'int' = 0, setup_predicate_count: 'int' = 0, ranked_top_n_count: 'int' = 0, predicate_only_pass_count: 'int' = 0, mismatch_flag: 'int' = 0, date_fallback_count: 'int' = 0, ranking_source: 'str | None' = None, exclude_reasons: 'DefaultDict[str, int]' = <factory>) -> None`

Initialize self.  See help(type(self)) for accurate signature.

#### `as_dict(self) -> 'dict[str, Any]'`



## 関数一覧

### `generate_candidates_system1(prepared_dict: 'dict[str, pd.DataFrame] | None', *, top_n: 'int | None' = None, progress_callback: 'Callable[[str], None] | None' = None, log_callback: 'Callable[[str], None] | None' = None, batch_size: 'int | None' = None, latest_only: 'bool' = False, include_diagnostics: 'bool' = False, diagnostics: 'System1Diagnostics | Mapping[str, Any] | None' = None, **kwargs: 'object') -> 'tuple[dict[pd.Timestamp, object], pd.DataFrame | None] | tuple[dict[pd.Timestamp, object], pd.DataFrame | None, dict[str, object]]'`

System1 candidate generation (ROC200 descending ranking).

Returns a tuple of (per-date candidates, merged dataframe,
diagnostics when requested).

### `generate_roc200_ranking_system1(data_dict: 'dict[str, pd.DataFrame]', date: 'str', top_n: 'int' = 20, log_callback: 'Callable[[str], None] | None' = None) -> 'list[dict]'`

Generate ROC200-based ranking for a specific date.

Args:
    data_dict: Dictionary of prepared data
    date: Target date (YYYY-MM-DD format)
    top_n: Number of top candidates to return
    log_callback: Optional logging callback

Returns:
    List of candidate dictionaries with symbol, ROC200, and other metrics

### `get_total_days_system1(data_dict: 'dict[str, pd.DataFrame]') -> 'int'`

Get total days count for System1 data.

Args:
    data_dict: Data dictionary

Returns:
    Maximum day count

### `prepare_data_vectorized_system1(raw_data_dict: 'dict[str, pd.DataFrame] | None', *, progress_callback: 'Callable[[str], None] | None' = None, log_callback: 'Callable[[str], None] | None' = None, skip_callback: 'Callable[[str, str], None] | None' = None, batch_size: 'int | None' = None, reuse_indicators: 'bool' = True, symbols: 'list[str] | None' = None, use_process_pool: 'bool' = False, max_workers: 'int | None' = None, **_kwargs: 'object') -> 'dict[str, pd.DataFrame]'`

System1 data preparation processing (ROC200 momentum strategy).

Execute high-speed processing using precomputed indicators.

Args:
    raw_data_dict: Raw data dictionary (None to fetch from cache)
    progress_callback: Progress reporting callback
    log_callback: Log output callback
    skip_callback: Error skip callback
    batch_size: Batch size
    reuse_indicators: Reuse existing indicators (for speed)
    symbols: Target symbol list
    use_process_pool: Process pool usage flag
    max_workers: Maximum worker count

Returns:
    Processed data dictionary

### `summarize_system1_diagnostics(diag: 'Mapping[str, Any] | None', *, max_reasons: 'int' = 3) -> 'dict[str, Any]'`

Normalize raw diagnostics payload for display/log output.

Args:
    diag: Raw diagnostics mapping emitted by ``generate_candidates_system1``.
    max_reasons: Maximum number of exclusion reasons to keep.

Returns:
    Dictionary containing integer-normalized counters and (optionally)
    a trimmed ``exclude_reasons`` mapping sorted by descending count.

### `system1_row_passes_setup(row: 'pd.Series', *, allow_fallback: 'bool' = True) -> 'tuple[bool, dict[str, bool], str | None]'`

System1 setup evaluation using SMA trend and ROC200.

Conditions:
- SMA trend: SMA25 > SMA50 (individual stock condition)
- ROC200 > 0 (momentum confirmation)

Note: Market condition (SPY > SMA100) is checked at orchestrator level,
not within this function. Phase 2 filter (Price>=5, DV20>=50M) is assumed
to have already passed for rows reaching this function.

Args:
    row: DataFrame row containing indicators
    allow_fallback: Legacy parameter for backward compatibility (ignored)

Returns:
    (passes, flags, reason) tuple where:
    - passes: True if all conditions met
    - flags: dict with individual condition results
    - reason: exclusion reason if passes is False
