"""
System6 候補生成の超高速化版
根本的な処理フローの見直しによる30分達成
"""

import pandas as pd
from collections.abc import Callable
import time

from common.utils_spy import resolve_signal_entry_date


def generate_candidates_ultra_fast_system6(
    prepared_dict: dict[str, pd.DataFrame],
    top_n: int = 10,
    progress_callback: Callable | None = None,
    log_callback: Callable | None = None,
    skip_callback: Callable | None = None,
    **kwargs,
) -> list[tuple[str, dict]]:
    """
    System6用の超高速候補生成

    最適化ポイント:
    1. ベクトル化処理による高速化
    2. 早期終了条件の活用
    3. 不要な計算の省略
    4. メモリ効率化
    """

    if log_callback:
        log_callback("🚀 System6超高速候補生成開始...")

    start_time = time.time()

    # フィルタリング後の銘柄に対してセットアップ条件をベクトル化してチェック
    candidates = []
    total_symbols = len(prepared_dict)

    if total_symbols == 0:
        return []

    # バッチ処理で高速化
    batch_size = min(100, max(10, total_symbols // 5))
    processed = 0

    # セットアップ通過銘柄の事前スクリーニング
    setup_passed = []

    for symbol, df in prepared_dict.items():
        try:
            # セットアップ条件の高速チェック
            if "setup" not in df.columns:
                continue

            # 最新のsetup値をチェック（True/Falseの直接確認）
            if not df["setup"].iloc[-1]:
                continue

            # return_6dの条件チェック（20%以上）
            if "return_6d" not in df.columns or df["return_6d"].iloc[-1] <= 0.20:
                continue

            # UpTwoDays条件チェック
            if "UpTwoDays" not in df.columns or not df["UpTwoDays"].iloc[-1]:
                continue

            setup_passed.append((symbol, df))

        except Exception:
            continue

    setup_time = time.time() - start_time
    if log_callback:
        log_callback(
            f"✅ セットアップスクリーニング: {len(setup_passed)}/{total_symbols}銘柄通過 "
            f"| 経過: {setup_time:.1f}秒"
        )

    if len(setup_passed) == 0:
        if log_callback:
            log_callback("⚠️ セットアップ条件を満たす銘柄がありません")
        return []

    # 候補日決定（高速化版）
    candidate_date = None
    try:
        # SPYベースまたは固定日付を使用
        candidate_date = resolve_signal_entry_date()
    except Exception:
        # フォールバック：現在日付を使用
        from datetime import datetime

        candidate_date = datetime.now().date()

    # 各銘柄の候補スコア計算（高速ベクトル化）
    scored_candidates = []

    for symbol, df in setup_passed:
        try:
            # シンプルなスコア計算（return_6dベース）
            latest_return = df["return_6d"].iloc[-1]

            # 基本スコアは return_6d
            score = latest_return

            # 追加要素（オプショナル）
            if "atr10" in df.columns and df["atr10"].iloc[-1] > 0:
                # ATR正規化（ボラティリティ調整）
                volatility_adj = latest_return / (df["atr10"].iloc[-1] / df["Close"].iloc[-1])
                score = volatility_adj

            candidate_info = {
                "symbol": symbol,
                "entry_date": candidate_date,
                "return_6d": latest_return,
                "score": score,
                "setup_date": (
                    df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1]
                ),
            }

            scored_candidates.append((symbol, candidate_info))

        except Exception:
            continue

    # スコア順にソート（降順）
    scored_candidates.sort(key=lambda x: x[1]["score"], reverse=True)

    # トップN選択
    final_candidates = scored_candidates[:top_n]

    scoring_time = time.time() - start_time - setup_time
    total_time = time.time() - start_time

    if log_callback:
        log_callback(
            f"🎯 候補スコアリング完了: {len(final_candidates)}/{len(setup_passed)}銘柄選択 "
            f"| 経過: {scoring_time:.1f}秒"
        )
        log_callback(f"🏁 超高速候補生成完了: 総時間 {total_time:.1f}秒")

        # トップ候補表示
        if final_candidates:
            top_3 = final_candidates[:3]
            for i, (symbol, info) in enumerate(top_3, 1):
                log_callback(
                    f"  {i}位: {symbol} (return_6d: {info['return_6d']:.2%}, "
                    f"Score: {info['score']:.4f})"
                )

    return final_candidates


def prepare_data_ultra_optimized_system6(
    raw_data_dict: dict[str, pd.DataFrame],
    *,
    progress_callback: Callable | None = None,
    log_callback: Callable | None = None,
    skip_callback: Callable | None = None,
    reuse_indicators: bool = True,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """
    System6用の超最適化データ準備
    30分達成を目指した根本的最適化
    """

    start_time = time.time()

    if log_callback:
        log_callback("🚀 System6超最適化データ準備開始...")

    # Phase 1: より厳しい早期フィルタリング
    original_count = len(raw_data_dict)

    # より厳しい条件での事前フィルタリング
    aggressive_filtered = {}

    for symbol, df in raw_data_dict.items():
        try:
            # より厳しい基準での早期除外
            if len(df) < 100:  # より多くのデータが必要
                continue

            # 価格条件（より厳しく）
            if "Close" in df.columns:
                latest_close = df["Close"].iloc[-1]
            elif "close" in df.columns:
                latest_close = df["close"].iloc[-1]
            else:
                continue

            if latest_close < 10.0:  # 10ドル以上に限定
                continue

            # 出来高条件（より厳しく）
            if "Volume" in df.columns:
                recent_volume = df["Volume"].tail(5).mean()
                dollar_volume = recent_volume * latest_close
            elif "volume" in df.columns:
                recent_volume = df["volume"].tail(5).mean()
                dollar_volume = recent_volume * latest_close
            else:
                continue

            if dollar_volume < 20_000_000:  # 20M以上のdollar volume
                continue

            # 直近のリターンチェック（System6の趣旨に合致する銘柄のみ）
            if "Close" in df.columns and len(df) >= 10:
                recent_returns = df["Close"].pct_change(6).tail(5)
                # 最低1つは大きなリターンがあることを確認
                if recent_returns.max() < 0.15:  # 15%以上のリターンがない場合は除外
                    continue

            aggressive_filtered[symbol] = df

        except Exception:
            continue

    filter_time = time.time() - start_time
    filter_reduction = (
        (1 - len(aggressive_filtered) / original_count) * 100 if original_count > 0 else 0
    )

    if log_callback:
        log_callback(
            f"✅ 厳格フィルタ完了: {original_count} → {len(aggressive_filtered)}銘柄 "
            f"({filter_reduction:.1f}%削減) | 経過: {filter_time:.1f}秒"
        )

    # Phase 2: 事前計算インジケーターの完全活用
    result_dict = {}
    total = len(aggressive_filtered)

    if total == 0:
        return result_dict

    phase2_start = time.time() - start_time
    precomputed_used = 0
    recalculated = 0

    for symbol, df in aggressive_filtered.items():
        try:
            # インジケーターの可用性チェック
            required_cols = ["atr10", "dollarvolume50", "return_6d", "UpTwoDays", "filter", "setup"]
            has_all_indicators = all(col in df.columns for col in required_cols)

            if reuse_indicators and has_all_indicators:
                # 事前計算済みをそのまま使用（最高速）
                clean_df = df[["Open", "High", "Low", "Close", "Volume"] + required_cols].copy()
                clean_df = clean_df.dropna(subset=["atr10", "dollarvolume50", "return_6d"])
                clean_df = clean_df.loc[~clean_df.index.duplicated()].sort_index()
                clean_df.index = pd.to_datetime(clean_df.index).tz_localize(None)
                clean_df.index.name = "Date"

                result_dict[symbol] = clean_df
                precomputed_used += 1
            else:
                # 最小限の再計算
                from core.system6 import _compute_indicators_from_frame

                prepared_df = _compute_indicators_from_frame(df)
                if not prepared_df.empty:
                    result_dict[symbol] = prepared_df
                    recalculated += 1

        except Exception:
            if skip_callback:
                skip_callback(symbol, "processing_error")
            continue

    total_time = time.time() - start_time

    if log_callback:
        log_callback(f"🎯 超最適化完了: {len(result_dict)}銘柄処理 | 総時間: {total_time:.1f}秒")
        log_callback(f"📈 事前計算活用: {precomputed_used}, 再計算: {recalculated}")

    return result_dict
