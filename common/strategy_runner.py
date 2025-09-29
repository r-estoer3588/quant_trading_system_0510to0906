"""戦略実行ハーネス - System1-7の並列実行・進捗管理・エラーハンドリング

run_all_systems_today.py から戦略実行の責務を分離:
  - 戦略の並列/直列実行制御
  - 進捗レポート・UI通知機能
  - プロセスプール/フォールバック処理
  - 例外処理・ログ集約機能

注意: 公開 API は run_all_systems_today.py と互換。
      依存: ThreadPoolExecutor, multiprocessing, pandas, threading
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any

import pandas as pd

from common.stage_metrics import GLOBAL_STAGE_METRICS
from config.settings import get_settings

__all__ = [
    "StrategyRunner",
    "run_strategies_parallel",
    "run_strategies_serial",
    "_run_single_strategy",
]


class StrategyRunner:
    """戦略実行の統合クラス - 並列/直列実行・進捗管理"""

    def __init__(
        self,
        log_callback: Callable[[str], None] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        per_system_progress: Callable[[str, str], None] | None = None,
    ):
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self.per_system_progress = per_system_progress

    def run_strategies(
        self,
        strategies: dict[str, Any],
        basic_data: dict[str, pd.DataFrame],
        raw_data_sets: dict[str, dict[str, pd.DataFrame]],
        spy_df: pd.DataFrame | None = None,
        today: pd.Timestamp | None = None,
        parallel: bool = False,
    ) -> dict[str, tuple[pd.DataFrame, str, list[str]]]:
        """戦略群の実行（並列/直列選択可能）

        Args:
            strategies: System名 -> 戦略オブジェクトのマッピング
            basic_data: 基礎データ（全シンボル）
            raw_data_sets: システム別フィルター済みデータセット
            spy_df: SPYデータ（System4用）
            today: 対象日
            parallel: True=並列実行、False=直列実行

        Returns:
            System名 -> (結果DataFrame, メッセージ, ログリスト) のマッピング
        """
        if parallel:
            return run_strategies_parallel(
                strategies,
                basic_data,
                raw_data_sets,
                spy_df,
                today,
                self.log_callback,
                self.per_system_progress,
            )
        else:
            return run_strategies_serial(
                strategies,
                basic_data,
                raw_data_sets,
                spy_df,
                today,
                self.log_callback,
            )


def run_strategies_parallel(
    strategies: dict[str, Any],
    basic_data: dict[str, pd.DataFrame],
    raw_data_sets: dict[str, dict[str, pd.DataFrame]],
    spy_df: pd.DataFrame | None = None,
    today: pd.Timestamp | None = None,
    log_callback: Callable[[str], None] | None = None,
    per_system_progress: Callable[[str, str], None] | None = None,
) -> dict[str, tuple[pd.DataFrame, str, list[str]]]:
    """戦略群の並列実行"""
    results: dict[str, tuple[pd.DataFrame, str, list[str]]] = {}

    with ThreadPoolExecutor() as executor:
        futures: dict[Future, str] = {}

        # 全戦略を並列開始
        for name, stg in strategies.items():
            if per_system_progress:
                try:
                    per_system_progress(name, "start")
                except Exception:
                    pass

            # CLI専用: 各システム開始を即時表示
            if log_callback:
                try:
                    log_callback(f"▶ {name} 開始")
                except Exception:
                    pass

            fut = executor.submit(
                _run_single_strategy,
                name,
                stg,
                basic_data,
                raw_data_sets,
                spy_df,
                today,
                log_callback,
            )
            futures[fut] = name

        # 完了待ち・逐次処理（継続的なドレイン統合）
        pending: set[Future] = set(futures.keys())
        completed_count = 0

        while pending:
            done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)

            # 0.2秒タイムアウト中またはタスク完了時にドレイン実行
            # リアルタイムUI同期の向上
            try:
                from scripts.run_all_systems_today import _drain_stage_event_queue

                _drain_stage_event_queue()
            except (ImportError, AttributeError):
                pass

            for future in done:
                system_name = futures[future]
                completed_count += 1

                try:
                    df, msg, logs = future.result()
                    results[system_name] = (df, msg, logs)

                    if per_system_progress:
                        per_system_progress(system_name, "done")

                    if log_callback:
                        log_callback(f"✅ {system_name} 完了: {msg}")
                        # ワーカーログを出力
                        for log_line in logs:
                            log_callback(f"[{system_name}] {log_line}")

                except Exception as e:
                    results[system_name] = (
                        pd.DataFrame(),
                        f"❌ {system_name}: エラー",
                        [],
                    )
                    if log_callback:
                        log_callback(f"❌ {system_name} 失敗: {e}")

    return results


def run_strategies_serial(
    strategies: dict[str, Any],
    basic_data: dict[str, pd.DataFrame],
    raw_data_sets: dict[str, dict[str, pd.DataFrame]],
    spy_df: pd.DataFrame | None = None,
    today: pd.Timestamp | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> dict[str, tuple[pd.DataFrame, str, list[str]]]:
    """戦略群の直列実行"""
    results: dict[str, tuple[pd.DataFrame, str, list[str]]] = {}

    for name, stg in strategies.items():
        if log_callback:
            log_callback(f"▶ {name} 開始")

        try:
            df, msg, logs = _run_single_strategy(
                name, stg, basic_data, raw_data_sets, spy_df, today, log_callback
            )
            results[name] = (df, msg, logs)

            if log_callback:
                log_callback(f"✅ {name} 完了: {msg}")
                # ログ出力
                for log_line in logs:
                    log_callback(f"[{name}] {log_line}")

        except Exception as e:
            results[name] = (pd.DataFrame(), f"❌ {name}: エラー", [])
            if log_callback:
                log_callback(f"❌ {name} 失敗: {e}")

        # 各戦略完了後にドレイン実行（直列実行でもリアルタイム同期）
        try:
            from scripts.run_all_systems_today import _drain_stage_event_queue

            _drain_stage_event_queue()
        except (ImportError, AttributeError):
            pass

    return results


def _run_single_strategy(
    name: str,
    stg: Any,
    basic_data: dict[str, pd.DataFrame],
    raw_data_sets: dict[str, dict[str, pd.DataFrame]],
    spy_df: pd.DataFrame | None = None,
    today: pd.Timestamp | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, str, list[str]]:
    """単一戦略の実行

    Args:
        name: System名（例: "system1"）
        stg: 戦略オブジェクト
        basic_data: 全体の基礎データ
        raw_data_sets: システム別フィルター済みデータセット
        spy_df: SPYデータ（System4用）
        today: 対象日
        log_callback: ログコールバック

    Returns:
        (結果DataFrame, メッセージ, ログリスト)
    """
    logs: list[str] = []

    def _local_log(message: str) -> None:
        logs.append(str(message))
        # メインスレッドでのみUI通知、ワーカースレッドではローカルログのみ
        try:
            is_main = threading.current_thread() is threading.main_thread()
        except Exception:
            is_main = False

        if log_callback and is_main:
            try:
                log_callback(f"[{name}] {message}")
            except Exception:
                pass
        else:
            try:
                print(f"[{name}] {message}", flush=True)
            except Exception:
                pass

    # システム別データセット選択
    if name in raw_data_sets:
        base = raw_data_sets[name]
    elif name == "system7":
        base = {"SPY": basic_data.get("SPY")}
    else:
        base = basic_data

    # System4 SPY依存チェック
    if name == "system4" and spy_df is None:
        _local_log(
            "⚠️ System4 は SPY 指標が必要ですが SPY データがありません。スキップします。"
        )
        return pd.DataFrame(), f"❌ {name}: 0 件 🚫", logs

    _local_log(f"🔎 {name}: シグナル抽出を開始")

    # 進捗コールバック設定
    def _stage_callback(
        progress: int,
        filter_count: int | None = None,
        setup_count: int | None = None,
        candidate_count: int | None = None,
        final_count: int | None = None,
    ) -> None:
        try:
            GLOBAL_STAGE_METRICS.record_stage(
                name, progress, filter_count, setup_count, candidate_count, final_count
            )
        except Exception:
            pass

    # 実行パラメータ設定
    use_process_pool = _should_use_process_pool()
    max_workers = _get_max_workers()
    lookback_days = _get_lookback_days(name, stg, base)

    if use_process_pool:
        _local_log(
            f"⚙️ {name}: プロセスプール実行を開始 (workers={max_workers or 'auto'})"
        )

    # 戦略実行
    df = pd.DataFrame()
    pool_outcome: str | None = None
    _t0 = time.time()

    try:
        df = stg.get_today_signals(
            base,
            market_df=spy_df,
            today=today,
            progress_callback=None,
            log_callback=_local_log if not use_process_pool else None,
            stage_progress=_stage_callback,
            use_process_pool=use_process_pool,
            max_workers=max_workers,
            lookback_days=lookback_days,
        )

        if use_process_pool:
            pool_outcome = "success"

        _elapsed = int(max(0, time.time() - _t0))
        _m, _s = divmod(_elapsed, 60)
        _local_log(f"⏱️ {name}: 経過 {_m}分{_s}秒")

    except Exception as e:
        _local_log(f"⚠️ {name}: シグナル抽出に失敗しました: {e}")

        # プロセスプール異常時はフォールバック再試行
        needs_fallback = use_process_pool and _should_fallback(str(e))
        if needs_fallback:
            _local_log("🛟 フォールバック再試行: プロセスプール無効化で実行します")
            try:
                _t0b = time.time()
                df = stg.get_today_signals(
                    base,
                    market_df=spy_df,
                    today=today,
                    progress_callback=None,
                    log_callback=_local_log,
                    stage_progress=None,
                    use_process_pool=False,
                    max_workers=None,
                    lookback_days=lookback_days,
                )
                _elapsed_b = int(max(0, time.time() - _t0b))
                _m2, _s2 = divmod(_elapsed_b, 60)
                _local_log(f"⏱️ {name} (fallback): 経過 {_m2}分{_s2}秒")
                pool_outcome = "fallback"
            except Exception as e2:
                _local_log(f"❌ {name}: フォールバックも失敗: {e2}")
                pool_outcome = "error"
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()
            pool_outcome = "error"

    finally:
        if use_process_pool:
            if pool_outcome == "success":
                _local_log(f"🏁 {name}: プロセスプール実行が完了しました")
            elif pool_outcome == "fallback":
                _local_log(
                    f"🏁 {name}: プロセスプール実行を終了（フォールバック実行済み）"
                )
            else:
                _local_log(f"🏁 {name}: プロセスプール実行を終了（結果: 失敗）")

    # 結果後処理
    if not df.empty:
        df = _post_process_results(df)

    # メッセージ生成
    if df is not None and not df.empty:
        msg = f"📊 {name}: {len(df)} 件"
    else:
        msg = f"❌ {name}: 0 件 🚫"

    _local_log(msg)
    return df, msg, logs


# ----- Helper Functions ----- #


def _should_use_process_pool() -> bool:
    """環境変数に基づくプロセスプール使用判定"""
    import os

    env_pp = os.environ.get("USE_PROCESS_POOL", "").strip().lower()
    return env_pp in {"1", "true", "yes", "on"}


def _get_max_workers() -> int | None:
    """ワーカー数の決定（環境変数 > 設定 > None）"""
    import os

    try:
        env_workers = os.environ.get("PROCESS_POOL_WORKERS", "").strip()
        if env_workers:
            return int(env_workers) or None
    except Exception:
        pass

    try:
        settings = get_settings(create_dirs=False)
        return int(getattr(settings, "THREADS_DEFAULT", 8)) or None
    except Exception:
        return None


def _get_lookback_days(name: str, stg: Any, base: dict[str, pd.DataFrame]) -> int:
    """戦略別ルックバック日数の決定"""
    import os

    # デフォルトルックバック設定
    try:
        settings = get_settings(create_dirs=True)
        lb_default = int(
            settings.cache.rolling.base_lookback_days
            + settings.cache.rolling.buffer_days
        )
    except Exception:
        lb_default = 300

    # システム別必要日数マップ
    try:
        margin = float(os.environ.get("LOOKBACK_MARGIN", "0.15"))
    except Exception:
        margin = 0.15

    need_map: dict[str, int] = {
        "system1": int(220 * (1 + margin)),
        "system2": int(120 * (1 + margin)),
        "system3": int(170 * (1 + margin)),  # SMA150用
        "system4": int(220 * (1 + margin)),  # SMA200用
        "system5": int(140 * (1 + margin)),
        "system6": int(80 * (1 + margin)),
        "system7": int(80 * (1 + margin)),
    }

    # 戦略カスタム日数（get_total_days メソッド）
    custom_need = None
    try:
        fn = getattr(stg, "get_total_days", None)
        if callable(fn):
            val = fn(base)
            if isinstance(val, int | float):
                custom_need = int(val)
            elif isinstance(val, str):
                custom_need = int(float(val))
    except Exception:
        custom_need = None

    # 最終決定
    try:
        min_floor = int(os.environ.get("LOOKBACK_MIN_DAYS", "80"))
    except Exception:
        min_floor = 80

    min_required = custom_need or need_map.get(name, lb_default)
    return min(lb_default, max(min_floor, int(min_required)))


def _should_fallback(error_msg: str) -> bool:
    """エラーメッセージからフォールバック要否を判定"""
    msg = error_msg.lower()
    fallback_keywords = [
        "process pool",
        "a child process terminated",
        "terminated abruptly",
        "forkserver",
        "__main__",
    ]
    return any(keyword in msg for keyword in fallback_keywords)


def _post_process_results(df: pd.DataFrame) -> pd.DataFrame:
    """結果DataFrameの後処理（スコア順ソート等）"""
    if df.empty:
        return df

    # スコアキー取得・ソート方向決定
    if "score_key" in df.columns and len(df):
        first_key = df["score_key"].iloc[0]
    else:
        first_key = None

    asc = _asc_by_score_key(first_key)
    df = df.sort_values("score", ascending=asc, na_position="last")
    return df.reset_index(drop=True)


def _asc_by_score_key(score_key: str | None) -> bool:
    """スコアキーに基づく昇順/降順の決定"""
    # system1/3/4/5: ROC/高数値系 = 降順
    # system2/6: 低数値系 = 昇順
    # 他: デフォルト降順
    if score_key is None:
        return False  # 降順

    key_lower = score_key.lower()
    if any(k in key_lower for k in ["rsi", "adx_low"]):
        return True  # 昇順（低い方が良い）
    else:
        return False  # 降順（高い方が良い）
