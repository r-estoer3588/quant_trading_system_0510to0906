from abc import ABC, abstractmethod
import logging
import pandas as pd


class StrategyBase(ABC):
    """
    全戦略共通の抽象基底クラス。
    各戦略はこのクラスを継承し、必要メソッドを実装する。
    また、YAML のリスク設定とシステム固有設定を self.config に取り込む。
    """

    def __init__(self) -> None:
        logger = logging.getLogger(__name__)
        try:
            from config.settings import get_settings, get_system_params
        except ImportError as exc:
            logger.error("設定モジュールの読み込みに失敗: %s", exc)
            self.config = {}
            return

        try:
            settings = get_settings(create_dirs=True)
        except Exception as exc:
            logger.error("設定の初期化に失敗: %s", exc)
            self.config = {}
            return

        module = getattr(self.__class__, "__module__", "")
        sys_name = getattr(self, "SYSTEM_NAME", None)
        if not sys_name:
            parts = module.split('.')
            cand = next((p for p in parts if p.startswith('system') and any(ch.isdigit() for ch in p)), None)
            sys_name = cand or ""

        system_params = {}
        if sys_name:
            try:
                system_params = get_system_params(sys_name)
            except Exception as exc:
                logger.warning("システム固有パラメータ取得失敗: %s", exc)

        cfg = {
            "risk_pct": settings.risk.risk_pct,
            "max_positions": settings.risk.max_positions,
            "max_pct": settings.risk.max_pct,
        }
        try:
            cfg.update(system_params or {})
        except Exception as exc:
            logger.warning("config update failed: %s", exc)
        self.config = cfg

    @abstractmethod
    def prepare_data(self, raw_data_dict: dict, **kwargs) -> dict:
        """生データからインジケーターやシグナルを計算"""
        pass

    @abstractmethod
    def generate_candidates(self, data_dict: dict, market_df: pd.DataFrame, **kwargs):
        """日別仕掛け候補を生成"""
        pass

    @abstractmethod
    def run_backtest(
        self, data_dict: dict, candidates_by_date: dict, capital: float, **kwargs
    ) -> pd.DataFrame:
        """仕掛け候補に基づくバックテストを実施"""
        pass

    # ============================================================
    # 共通ユーティリティ: 資金管理 & ポジションサイズ計算
    # ============================================================
    def update_capital_with_exits(
        self, capital: float, active_positions: list, current_date
    ):
        """
        exit_date == current_date のポジションを決済して損益を反映。
        戻り値: (更新後capital, 未決済active_positions)
        """
        realized_pnl = sum(
            p["pnl"] for p in active_positions if p["exit_date"] == current_date
        )
        capital += realized_pnl
        active_positions = [
            p for p in active_positions if p["exit_date"] > current_date
        ]
        return capital, active_positions

    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_price: float,
        risk_pct: float = 0.02,
        max_pct: float = 0.10,
    ) -> int:
        """
        共通のポジションサイズ計算（System1〜7 共通）
        - capital: 現在資金
        - entry_price: エントリー価格
        - stop_price: 損切り価格
        - risk_pct: 1トレードのリスク割合（デフォルト2%）
        - max_pct: 1トレードの最大資金割合（デフォルト10%）
        """
        risk_per_trade = risk_pct * capital
        max_position_value = max_pct * capital

        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            return 0

        shares = min(risk_per_trade / risk_per_share, max_position_value / entry_price)
        return int(shares)

    # ============================================================
    # 当日シグナル抽出（共通関数）
    # ============================================================
    def get_today_signals(
        self,
        raw_data_dict: dict,
        *,
        market_df: pd.DataFrame | None = None,
        today: pd.Timestamp | None = None,
        progress_callback=None,
        log_callback=None,
    ) -> pd.DataFrame:
        """
        各 strategy の `prepare_data`/`generate_candidates` を流用し、
        最新営業日のみのシグナルを DataFrame で返す。

        戻り値カラム: symbol, system, side, signal_type, entry_date, entry_price, stop_price, score_key, score
        """
        from common.today_signals import get_today_signals_for_strategy

        return get_today_signals_for_strategy(
            self,
            raw_data_dict,
            market_df=market_df,
            today=today,
            progress_callback=progress_callback,
            log_callback=log_callback,
        )
