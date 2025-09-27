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
            settings = get_settings(create_dirs=False)
        except Exception as exc:
            logger.error("設定の初期化に失敗: %s", exc)
            self.config = {}
            return

        module = getattr(self.__class__, "__module__", "")
        sys_name = getattr(self, "SYSTEM_NAME", None)
        if not sys_name:
            parts = module.split(".")
            cand = next(
                (p for p in parts if p.startswith("system") and any(ch.isdigit() for ch in p)),
                None,
            )
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
    def prepare_data(
        self,
        raw_data_or_symbols: dict,
        reuse_indicators: bool | None = None,
        **kwargs,
    ) -> dict:
        """生データからインジケーターやシグナルを計算"""
        pass

    @abstractmethod
    def generate_candidates(
        self, data_dict: dict, market_df: pd.DataFrame | None = None, **kwargs
    ) -> tuple[dict, pd.DataFrame | None]:
        """日別仕掛け候補を生成し、(candidates_by_date, market_df) を返す"""
        pass

    @abstractmethod
    def run_backtest(
        self, data_dict: dict, candidates_by_date: dict, capital: float, **kwargs
    ) -> pd.DataFrame:
        """仕掛け候補に基づくバックテストを実施"""
        pass

    # ----------------------------
    # 共通ユーティリティ: 資金管理 & ポジションサイズ計算
    # ----------------------------
    def update_capital_with_exits(self, capital: float, active_positions: list, current_date):
        """
        exit_date == current_date のポジションを決済して損益を反映。
        戻り値: (更新後capital, 未決済active_positions)
        """
        realized_pnl = sum(p["pnl"] for p in active_positions if p["exit_date"] == current_date)
        capital += realized_pnl
        active_positions = [p for p in active_positions if p["exit_date"] > current_date]
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

    # ----------------------------
    # 当日シグナル抽出（共通関数）
    # ----------------------------
    def get_today_signals(
        self,
        raw_data_dict: dict,
        *,
        market_df: pd.DataFrame | None = None,
        today: pd.Timestamp | None = None,
        progress_callback=None,
        log_callback=None,
        stage_progress=None,
        use_process_pool: bool = False,
        max_workers: int | None = None,
        lookback_days: int | None = None,
    ) -> pd.DataFrame:
        """
        各 strategy の `prepare_data`/`generate_candidates` を流用し、
        最新営業日のみのシグナルを DataFrame で返す。

        戻り値カラム: symbol, system, side, signal_type, entry_date, entry_price,
        stop_price, score_key, score
        """
        from common.today_signals import get_today_signals_for_strategy

        return get_today_signals_for_strategy(
            self,
            raw_data_dict,
            market_df=market_df,
            today=today,
            progress_callback=progress_callback,
            log_callback=log_callback,
            stage_progress=stage_progress,
            use_process_pool=use_process_pool,
            max_workers=max_workers,
            lookback_days=lookback_days,
        )

    # ----------------------------
    # リファクタリング用共通メソッド（追加）
    # ----------------------------

    def _resolve_data_params(self, raw_data_or_symbols, use_process_pool=False, **kwargs):
        """
        データパラメータの共通解決処理
        戻り値: (symbols, raw_dict)
        """
        if isinstance(raw_data_or_symbols, dict):
            symbols = list(raw_data_or_symbols.keys())
            raw_dict = None if use_process_pool else raw_data_or_symbols
        else:
            symbols = list(raw_data_or_symbols)
            raw_dict = None
        return symbols, raw_dict

    def _get_top_n_setting(self, top_n_override=None):
        """
        top_n設定の共通取得処理
        """
        if top_n_override is not None:
            try:
                return max(0, int(top_n_override))
            except Exception:
                return 10

        try:
            from config.settings import get_settings

            return get_settings(create_dirs=False).backtest.top_n_rank
        except Exception:
            return 10

    def _get_market_df(self, data_dict, market_df=None):
        """
        Market DataFrame（SPY）の共通取得処理
        """
        if market_df is None:
            market_df = data_dict.get("SPY")
            if market_df is None:
                raise ValueError("SPY data not found in data_dict.")
        return market_df

    def _get_batch_size_setting(self, data_size):
        """
        batch_size設定の共通取得処理
        """
        try:
            from config.settings import get_settings
            from common.utils import resolve_batch_size

            batch_size = get_settings(create_dirs=False).data.batch_size
            return resolve_batch_size(data_size, batch_size)
        except Exception:
            return min(100, max(10, data_size // 10))

    def _compute_entry_common(self, df, candidate, atr_column="ATR20", stop_multiplier=None):
        """
        共通エントリー計算（ATRベースのストップロス）
        戻り値: (entry_price, stop_price, entry_idx) または None
        """
        try:
            entry_ts = pd.to_datetime(candidate["entry_date"]).normalize()
            # Use get_indexer to tolerate non-unique or non-exact match behavior
            idxer = pd.to_datetime(df.index).normalize().get_indexer([entry_ts])
            entry_idx = int(idxer[0]) if idxer.size > 0 else -1
        except Exception:
            return None

        if entry_idx <= 0 or entry_idx >= len(df):
            return None

        entry_price = float(df.iloc[entry_idx]["Open"])

        try:
            atr = float(df.iloc[entry_idx - 1][atr_column])
        except Exception:
            return None

        if stop_multiplier is None:
            stop_multiplier = float(self.config.get("stop_atr_multiple", 3.0))

        stop_price = entry_price - stop_multiplier * atr
        if entry_price - stop_price <= 0:
            return None

        return entry_price, stop_price, entry_idx

    def _prepare_data_with_fallback(self, core_prepare_func, raw_dict, symbols, **kwargs):
        """
        System1用のフォールバック付きprepare_data処理
        """
        try:
            return core_prepare_func(
                raw_dict,
                symbols=symbols,
                **kwargs,
            )
        except Exception as e:
            # フォールバック: プロセスプールを使わず・指標を再計算
            log_callback = kwargs.get("log_callback")
            if log_callback:
                try:
                    log_callback(
                        f"⚠️ {getattr(self, 'SYSTEM_NAME', 'unknown')}: prepare_data 失敗のためフォールバック再試行"
                        "（非プール・再計算）: "
                        f"{e}"
                    )
                except Exception:
                    pass
            fb_kwargs = dict(kwargs)
            fb_kwargs["reuse_indicators"] = False
            fb_kwargs["use_process_pool"] = False
            return core_prepare_func(
                raw_dict,
                symbols=symbols,
                **fb_kwargs,
            )

    def _prepare_data_template(self, raw_data_or_symbols, core_prepare_func, **kwargs):
        """
        prepare_data の共通テンプレートメソッド
        """
        symbols, raw_dict = self._resolve_data_params(
            raw_data_or_symbols, kwargs.get("use_process_pool", False)
        )

        # batch_size の設定（use_process_pool=Falseの時のみ）
        use_process_pool = kwargs.get("use_process_pool", False)
        if not use_process_pool and raw_dict is not None and kwargs.get("batch_size") is None:
            kwargs["batch_size"] = self._get_batch_size_setting(len(raw_dict))

        # System1のフォールバック処理
        if hasattr(self, "SYSTEM_NAME") and getattr(self, "SYSTEM_NAME", None) == "system1":
            return self._prepare_data_with_fallback(core_prepare_func, raw_dict, symbols, **kwargs)
        else:
            return core_prepare_func(raw_dict, symbols=symbols, **kwargs)
