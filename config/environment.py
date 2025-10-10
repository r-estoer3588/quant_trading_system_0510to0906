"""環境変数の統一管理モジュール。

全プロジェクトで使用される環境変数を EnvironmentConfig クラスで一元管理。
デフォルト値、型変換、バリデーションを提供。

使用例:
    >>> from config.environment import get_env_config
    >>> env = get_env_config()
    >>> if env.compact_logs:
    ...     print("Compact mode enabled")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import os


def _get_bool_env(key: str, default: bool = False) -> bool:
    """環境変数をboolとして取得。

    Args:
        key: 環境変数のキー
        default: デフォルト値

    Returns:
        環境変数の真偽値
    """
    val = os.environ.get(key, "").strip().lower()
    if not val:
        return default
    return val in {"1", "true", "yes", "on"}


def _get_int_env(key: str, default: int) -> int:
    """環境変数をintとして取得。

    Args:
        key: 環境変数のキー
        default: デフォルト値

    Returns:
        環境変数の整数値（変換失敗時はデフォルト）
    """
    val = os.environ.get(key, "").strip()
    if not val:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _get_float_env(key: str, default: float) -> float:
    """環境変数をfloatとして取得。

    Args:
        key: 環境変数のキー
        default: デフォルト値

    Returns:
        環境変数の浮動小数点値（変換失敗時はデフォルト）
    """
    val = os.environ.get(key, "").strip()
    if not val:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _get_str_env(key: str, default: str = "") -> str:
    """環境変数を文字列として取得。

    Args:
        key: 環境変数のキー
        default: デフォルト値

    Returns:
        環境変数の文字列値
    """
    return os.environ.get(key, default).strip()


@dataclass
class EnvironmentConfig:
    """環境変数の統一管理クラス。

    全環境変数をプロパティとして提供。
    型安全・デフォルト値・自動変換を保証。
    """

    # ===== 1. ログ制御 =====
    compact_logs: bool = field(default_factory=lambda: _get_bool_env("COMPACT_TODAY_LOGS", False))
    """DEBUGログを抑制し、簡潔なログ出力。本番推奨。"""

    enable_progress_events: bool = field(
        default_factory=lambda: _get_bool_env("ENABLE_PROGRESS_EVENTS", False)
    )
    """進捗イベントをlogs/progress_today.jsonlに出力。"""

    today_signals_log_mode: str = field(
        default_factory=lambda: _get_str_env("TODAY_SIGNALS_LOG_MODE", "")
    )
    """シグナルログモード: 'compact' / 'verbose' / 'single' / 'dated'"""

    structured_ui_logs: bool = field(
        default_factory=lambda: _get_bool_env("STRUCTURED_UI_LOGS", False)
    )
    """UI向け構造化ログ（JSON）を出力。"""

    structured_log_ndjson: bool = field(
        default_factory=lambda: _get_bool_env("STRUCTURED_LOG_NDJSON", False)
    )
    """NDJSON形式の構造化ログ出力。"""

    show_indicator_logs: bool = field(
        default_factory=lambda: _get_bool_env("SHOW_INDICATOR_LOGS", False)
    )
    """指標計算進捗ログを表示。"""

    enable_step_timings: bool = field(
        default_factory=lambda: _get_bool_env("ENABLE_STEP_TIMINGS", False)
    )
    """各処理ステップの実行時間を測定・出力。"""

    enable_substep_logs: bool = field(
        default_factory=lambda: _get_bool_env("ENABLE_SUBSTEP_LOGS", False)
    )
    """サブステップの詳細ログを有効化。"""

    trd_log_ok: bool = field(default_factory=lambda: _get_bool_env("TRD_LOG_OK", False))
    """TRDlist生成成功時のログを表示。"""

    rolling_issues_verbose_head: int = field(
        default_factory=lambda: _get_int_env("ROLLING_ISSUES_VERBOSE_HEAD", 5)
    )
    """rollingキャッシュ問題の詳細表示行数。"""

    no_emoji: bool = field(
        default_factory=lambda: _get_bool_env("NO_EMOJI", False)
        or _get_bool_env("DISABLE_EMOJI", False)
    )
    """ログからEmoji（絵文字）を削除。CI/CD用。"""

    # ===== 2. System3固有（テスト用閾値） =====
    min_drop3d_for_test: float | None = field(
        default_factory=lambda: (
            _get_float_env("MIN_DROP3D_FOR_TEST", float("nan"))
            if "MIN_DROP3D_FOR_TEST" in os.environ
            else None
        )
    )
    """⚠️テスト専用。System3の3日間下落率閾値を上書き。本番環境では絶対に設定しないこと。"""

    min_atr_ratio_for_test: float | None = field(
        default_factory=lambda: (
            _get_float_env("MIN_ATR_RATIO_FOR_TEST", float("nan"))
            if "MIN_ATR_RATIO_FOR_TEST" in os.environ
            else None
        )
    )
    """⚠️テスト専用。System3のATR比率閾値を上書き。本番環境では絶対に設定しないこと。"""

    # ===== 3. パフォーマンス・並列処理 =====
    use_process_pool: bool = field(default_factory=lambda: _get_bool_env("USE_PROCESS_POOL", False))
    """プロセスプールでの並列処理を有効化。"""

    process_pool_workers: int | None = field(
        default_factory=lambda: _get_int_env("PROCESS_POOL_WORKERS", 0) or None
    )
    """プロセスプールのワーカー数。Noneで自動決定。"""

    system6_use_process_pool: bool = field(
        default_factory=lambda: _get_bool_env("SYSTEM6_USE_PROCESS_POOL", False)
    )
    """System6専用のプロセスプール使用フラグ。"""

    basic_data_parallel: bool | None = field(
        default_factory=lambda: (
            True
            if _get_str_env("BASIC_DATA_PARALLEL", "").lower() == "1"
            else (False if _get_str_env("BASIC_DATA_PARALLEL", "").lower() == "0" else None)
        )
    )
    """基本データ読み込みの並列処理。True=強制並列、False=強制直列、None=自動。"""

    basic_data_parallel_threshold: int = field(
        default_factory=lambda: _get_int_env("BASIC_DATA_PARALLEL_THRESHOLD", 200)
    )
    """並列処理を開始する銘柄数の閾値。"""

    basic_data_max_workers: int | None = field(
        default_factory=lambda: _get_int_env("BASIC_DATA_MAX_WORKERS", 0) or None
    )
    """基本データ読み込みの最大ワーカー数。Noneで自動決定。"""

    lookback_margin: float = field(default_factory=lambda: _get_float_env("LOOKBACK_MARGIN", 0.15))
    """ルックバック期間のマージン（既定15%）。"""

    lookback_min_days: int = field(default_factory=lambda: _get_int_env("LOOKBACK_MIN_DAYS", 80))
    """ルックバック期間の最小日数。"""

    # ===== 4. テスト・デバッグ =====
    validate_setup_predicate: bool = field(
        default_factory=lambda: _get_bool_env("VALIDATE_SETUP_PREDICATE", False)
    )
    """setup列とpredicate関数の一致検証を有効化。開発・デバッグ用。"""

    streamlit_server_enabled: bool = field(
        default_factory=lambda: _get_bool_env("STREAMLIT_SERVER_ENABLED", False)
    )
    """Streamlitサーバーモード実行中かどうか。"""

    today_symbol_limit: int | None = field(
        default_factory=lambda: _get_int_env("TODAY_SYMBOL_LIMIT", 0) or None
    )
    """当日シグナルスキャンの対象銘柄数を制限。テスト・デバッグ用。"""

    basic_data_test_freshness_tolerance: int = field(
        default_factory=lambda: _get_int_env("BASIC_DATA_TEST_FRESHNESS_TOLERANCE", 365)
    )
    """テストモード時のデータ鮮度許容日数。"""

    full_scan_today: bool = field(default_factory=lambda: _get_bool_env("FULL_SCAN_TODAY", False))
    """全履歴をスキャン（latest_only=False）。デバッグ用。"""

    allow_critical_changes: bool = field(
        default_factory=lambda: _get_bool_env("ALLOW_CRITICAL_CHANGES", False)
    )
    """重要ファイルの変更を許可。通常は設定しない。"""

    run_planned_exits: bool = field(
        default_factory=lambda: _get_bool_env("RUN_PLANNED_EXITS", False)
    )
    """計画的エグジット処理を実行。"""

    max_verbose_lines: int | None = field(
        default_factory=lambda: _get_int_env("MAX_VERBOSE_LINES", 0) or None
    )
    """ログ圧縮検証時の最大行数（verbose）。"""

    max_compact_lines: int | None = field(
        default_factory=lambda: _get_int_env("MAX_COMPACT_LINES", 0) or None
    )
    """ログ圧縮検証時の最大行数（compact）。"""

    # ===== 5. API認証（機密情報） =====
    apca_api_key_id: str = field(default_factory=lambda: _get_str_env("APCA_API_KEY_ID", ""))
    """⚠️機密情報。Alpaca APIキーID。.envで管理。"""

    apca_api_secret_key: str = field(
        default_factory=lambda: _get_str_env("APCA_API_SECRET_KEY", "")
    )
    """⚠️機密情報。Alpaca APIシークレットキー。.envで管理。"""

    alpaca_api_base_url: str = field(
        default_factory=lambda: _get_str_env("ALPACA_API_BASE_URL", "")
    )
    """Alpaca APIのベースURL。"""

    alpaca_paper: bool = field(default_factory=lambda: _get_bool_env("ALPACA_PAPER", True))
    """ペーパートレーディングモード。本番ではfalse。"""

    slack_bot_token: str = field(default_factory=lambda: _get_str_env("SLACK_BOT_TOKEN", ""))
    """⚠️機密情報。Slack Bot Token。.envで管理。"""

    slack_channel_logs: str = field(default_factory=lambda: _get_str_env("SLACK_CHANNEL_LOGS", ""))
    """Slack通知先チャンネル（ログ）。"""

    slack_channel_equity: str = field(
        default_factory=lambda: _get_str_env("SLACK_CHANNEL_EQUITY", "")
    )
    """Slack通知先チャンネル（エクイティ）。"""

    slack_channel_signals: str = field(
        default_factory=lambda: _get_str_env("SLACK_CHANNEL_SIGNALS", "")
    )
    """Slack通知先チャンネル（シグナル）。"""

    discord_webhook_url: str = field(
        default_factory=lambda: _get_str_env("DISCORD_WEBHOOK_URL", "")
    )
    """⚠️機密情報。Discord Webhook URL。.envで管理。"""

    eodhd_api_key: str = field(default_factory=lambda: _get_str_env("EODHD_API_KEY", ""))
    """⚠️機密情報。EODHD APIキー。.envで管理。"""

    # ===== 6. 通知・ダッシュボード =====
    notify_use_rich: bool = field(default_factory=lambda: _get_bool_env("NOTIFY_USE_RICH", False))
    """通知をリッチカード形式で送信。"""

    cache_health_silent: bool = field(
        default_factory=lambda: _get_bool_env("CACHE_HEALTH_SILENT", False)
    )
    """キャッシュ健康診断のCLI通知を抑制。"""

    # ===== 7. その他 =====
    scheduler_workers: int = field(default_factory=lambda: _get_int_env("SCHEDULER_WORKERS", 4))
    """スケジューラーのワーカー数。"""

    bulk_update_workers: int = field(default_factory=lambda: _get_int_env("BULK_UPDATE_WORKERS", 4))
    """バルク更新のワーカー数。"""

    data_cache_dir: str = field(
        default_factory=lambda: _get_str_env("DATA_CACHE_DIR", "data_cache")
    )
    """データキャッシュディレクトリ。"""

    results_dir: str = field(default_factory=lambda: _get_str_env("RESULTS_DIR", "results_csv"))
    """結果CSV出力先ディレクトリ。"""

    logs_dir: str = field(default_factory=lambda: _get_str_env("LOGS_DIR", "logs"))
    """ログファイル出力先ディレクトリ。"""

    def validate(self) -> list[str]:
        """環境変数の妥当性を検証。

        Returns:
            エラーメッセージのリスト（空なら妥当）
        """
        errors: list[str] = []

        # テスト用閾値が本番環境で設定されていないかチェック
        if not self.compact_logs:  # 本番環境の可能性
            if self.min_drop3d_for_test is not None:
                errors.append(
                    "⚠️ MIN_DROP3D_FOR_TEST が設定されています。"
                    "本番環境では絶対に使用しないでください。"
                )
            if self.min_atr_ratio_for_test is not None:
                errors.append(
                    "⚠️ MIN_ATR_RATIO_FOR_TEST が設定されています。"
                    "本番環境では絶対に使用しないでください。"
                )

        # Alpacaペーパートレード警告
        if not self.alpaca_paper and self.apca_api_key_id:
            # 本番取引モード
            errors.append(
                "⚠️ ALPACA_PAPER=false（本番取引モード）が設定されています。"
                "本番取引を実行する場合のみ使用してください。"
            )

        return errors

    def is_production(self) -> bool:
        """本番環境かどうかを判定。

        Returns:
            本番環境の場合True
        """
        return self.compact_logs and not self.alpaca_paper

    def is_test_mode(self) -> bool:
        """テストモードかどうかを判定。

        Returns:
            テストモードの場合True
        """
        return (
            self.min_drop3d_for_test is not None
            or self.min_atr_ratio_for_test is not None
            or self.today_symbol_limit is not None
        )


@lru_cache(maxsize=1)
def get_env_config() -> EnvironmentConfig:
    """環境変数設定をシングルトンとして取得。

    初回呼び出し時に EnvironmentConfig を生成し、
    以降は同じインスタンスを返す。

    Returns:
        環境変数設定オブジェクト

    Examples:
        >>> env = get_env_config()
        >>> if env.compact_logs:
        ...     print("Compact mode")
        >>> if env.is_production():
        ...     print("Running in production")
    """
    return EnvironmentConfig()


def print_env_summary() -> None:
    """環境変数設定のサマリーを出力（デバッグ用）。"""
    env = get_env_config()
    errors = env.validate()

    print("=" * 60)
    print("環境変数設定サマリー")
    print("=" * 60)
    mode = "本番" if env.is_production() else ("テスト" if env.is_test_mode() else "開発")
    print(f"モード: {mode}")
    print(f"Compact Logs: {env.compact_logs}")
    print(f"Alpaca Paper Trading: {env.alpaca_paper}")
    print(f"Process Pool: {env.use_process_pool}")
    print(f"Validate Setup Predicate: {env.validate_setup_predicate}")
    print("=" * 60)

    if errors:
        print("⚠️ 警告:")
        for err in errors:
            print(f"  - {err}")
        print("=" * 60)


__all__ = [
    "EnvironmentConfig",
    "get_env_config",
    "print_env_summary",
]


if __name__ == "__main__":
    # デバッグ実行: python -m config.environment
    print_env_summary()
