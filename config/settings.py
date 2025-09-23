from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    # optional import for validation
    from .schemas import validate_config_dict  # type: ignore
except Exception:  # pragma: no cover
    validate_config_dict = None  # type: ignore

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # PyYAML が未導入でも動くように（後述のガードで対応）

# プロジェクトルート推定
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# .env を読み込み（存在すれば）
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)


# -----------------------------
# Dataclasses (セクション別)
# -----------------------------
@dataclass(frozen=True)
class RiskConfig:
    risk_pct: float = 0.02
    max_positions: int = 10
    max_pct: float = 0.10


@dataclass(frozen=True)
class DataConfig:
    vendor: str = "EODHD"
    eodhd_base: str = "https://eodhistoricaldata.com"
    api_key_env: str = "EODHD_API_KEY"
    cache_dir: Path = Path("data_cache")
    cache_recent_dir: Path = Path("data_cache_recent")
    max_workers: int = 20
    batch_size: int = 100
    request_timeout: int = 10
    download_retries: int = 3
    api_throttle_seconds: float = 1.5


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str = "2018-01-01"
    end_date: str = "2024-12-31"
    max_symbols: int = 500
    top_n_rank: int = 50
    initial_capital: int = 100000


@dataclass(frozen=True)
class OutputConfig:
    results_csv_dir: Path = Path("results_csv")
    logs_dir: Path = Path("logs")
    signals_dir: Path = Path("data_cache") / "signals"


@dataclass(frozen=True)
class CacheRollingConfig:
    base_lookback_days: int = 300
    buffer_days: int = 30
    max_staleness_days: int = 2
    prune_chunk_days: int = 30
    meta_file: str = "_meta.json"
    max_stale_days: int = 2
    max_symbols: int | None = None
    # 小数丸め桁数: None で無効、整数で有効
    round_decimals: int | None = 4
    # 並列ワーカー数: None で無効（スクリプト側の既定を使用）
    workers: int | None = 4
    # 適応制御パラメータ
    adaptive_window_count: int = 8
    adaptive_increase_threshold: float = 1.02
    adaptive_decrease_threshold: float = 0.98
    adaptive_step: int = 1
    adaptive_min_workers: int = 1
    # None の場合はスクリプト側の上限（CPU×2 など）を使用
    adaptive_max_workers: int | None = None
    # 進捗ステータスを書き出す間隔（秒）
    adaptive_report_seconds: int = 10
    # CSV ロケール設定: 小数点文字, 千位区切り, フィールド区切り
    csv_decimal_point: str = "."
    csv_thousands_sep: str | None = None
    csv_field_sep: str = ","


@dataclass(frozen=True)
class CacheConfig:
    full_dir: Path = Path("data_cache/full_backup")
    rolling_dir: Path = Path("data_cache/rolling")
    file_format: str = "auto"
    rolling: CacheRollingConfig = CacheRollingConfig()
    # キャッシュ書き出し時の丸め桁数: None で無効
    round_decimals: int | None = 4
    # CSV locale defaults (applies when writing CSV files)
    csv_decimal_point: str = "."
    csv_thousands_sep: str | None = None
    csv_field_sep: str = ","


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    rotation: str = "daily"
    filename: str = "app.log"


@dataclass(frozen=True)
class SchedulerJob:
    name: str
    cron: str
    task: str


@dataclass(frozen=True)
class SchedulerConfig:
    timezone: str = "America/New_York"
    jobs: tuple[SchedulerJob, ...] = tuple()


@dataclass(frozen=True)
class UIConfig:
    default_capital: int = 100000
    # 長短の既定資金配分比率（long の割合 0.0〜1.0）
    default_long_ratio: float = 0.5
    # システム別配分（long/short で独立に定義可能）。値は比率で合計1とは限らない（使用側で正規化）。
    long_allocations: dict[str, float] = field(
        default_factory=lambda: {
            "system1": 0.25,
            "system3": 0.25,
            "system4": 0.25,
            "system5": 0.25,
        }
    )
    short_allocations: dict[str, float] = field(
        default_factory=lambda: {
            "system2": 0.40,
            "system6": 0.40,
            "system7": 0.20,
        }
    )
    auto_tickers: tuple[str, ...] = tuple()
    debug_mode: bool = False
    show_download_buttons: bool = True


@dataclass(frozen=True)
class Settings:
    """アプリ全体で共有する設定値（YAML + .env を統合）
    優先度: .env > YAML > 既定値
    create_dirs=True で主要ディレクトリを作成
    """

    PROJECT_ROOT: Path

    # 既存互換（従来フィールド）
    DATA_CACHE_DIR: Path
    DATA_CACHE_RECENT_DIR: Path
    RESULTS_DIR: Path
    LOGS_DIR: Path

    # API/ネットワーク
    API_EODHD_BASE: str
    EODHD_API_KEY: str | None
    REQUEST_TIMEOUT: int
    DOWNLOAD_RETRIES: int
    API_THROTTLE_SECONDS: float

    # 実行パラメータ
    THREADS_DEFAULT: int
    MARKET_CAL_TZ: str

    # 新構成（YAMLセクション）
    risk: RiskConfig
    data: DataConfig
    cache: CacheConfig
    backtest: BacktestConfig
    outputs: OutputConfig
    logging: LoggingConfig
    scheduler: SchedulerConfig
    ui: UIConfig

    # システムごとの戦略パラメータ
    strategies: Mapping[str, Mapping[str, Any]]


# -----------------------------
# 内部ユーティリティ
# -----------------------------


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def _positive_int_or_none(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _as_path(base: Path, p: str | os.PathLike) -> Path:
    pth = Path(p)
    return pth if pth.is_absolute() else (base / pth)


def _load_config_generic(env_var: str, default_path: Path, loader) -> dict[str, Any]:
    cfg_path_env = os.getenv(env_var, "")
    cfg_path = Path(cfg_path_env) if cfg_path_env else default_path
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, encoding="utf-8") as f:
            return loader(f) or {}
    except Exception:
        return {}


def _load_yaml_config(project_root: Path) -> dict[str, Any]:
    """config/config.yaml を読み込んで辞書を返す。未存在なら空。"""
    if yaml is None:
        raise RuntimeError(
            "PyYAML が見つかりません。requirements.txt に PyYAML を追加しインストールしてください。"
        )
    default_path = project_root / "config" / "config.yaml"
    return _load_config_generic("APP_CONFIG", default_path, lambda f: yaml.safe_load(f))


def _load_yaml_config_validated(project_root: Path) -> dict[str, Any]:
    """YAMLを読み込み、可能ならPydanticで検証・正規化して返す。"""
    data = _load_yaml_config(project_root)
    if not data:
        return data
    if validate_config_dict is None:
        return data
    try:
        model = validate_config_dict(data)  # type: ignore
        return model.model_dump()  # type: ignore[attr-defined]
    except Exception:
        return data


def _load_json_config(project_root: Path) -> dict[str, Any]:
    """config/config.json を読み込んで辞書として返す。存在しなければ {}。"""
    default_path = project_root / "config" / "config.json"
    return _load_config_generic("APP_CONFIG_JSON", default_path, lambda f: json.load(f))


def _load_config_json_or_yaml_validated(project_root: Path) -> dict[str, Any]:
    """JSON > YAML の優先順位で読み込み、可能ならPydanticで検証して返す。"""
    data = _load_json_config(project_root)
    if not data:
        data = _load_yaml_config(project_root)
    if not data:
        return data
    if validate_config_dict is None:
        return data
    try:
        model = validate_config_dict(data)  # type: ignore
        return model.model_dump()  # type: ignore[attr-defined]
    except Exception:
        return data


def _build_scheduler(cfg: dict[str, Any]) -> SchedulerConfig:
    tz = cfg.get("timezone", "America/New_York")
    jobs_raw = cfg.get("jobs", []) or []
    jobs: list[SchedulerJob] = []
    for j in jobs_raw:
        try:
            jobs.append(SchedulerJob(name=j["name"], cron=j["cron"], task=j["task"]))
        except Exception:
            continue
    return SchedulerConfig(timezone=tz, jobs=tuple(jobs))


# -----------------------------
# 公開 API
# -----------------------------


def get_settings(create_dirs: bool = False) -> Settings:
    """設定を生成して返す。必要に応じて出力系ディレクトリを作成。"""
    root = PROJECT_ROOT

    # YAML 読み込み
    try:
        cfg = _load_config_json_or_yaml_validated(root)  # type: ignore[name-defined]
    except Exception:
        cfg = _load_yaml_config_validated(root)

    # YAML: セクション取得（なければ空）
    risk_cfg = cfg.get("risk", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    cache_cfg = cfg.get("cache")
    if not cache_cfg and isinstance(data_cfg, dict):
        cache_cfg = data_cfg.get("cache")
    cache_cfg = cache_cfg or {}
    rolling_cfg = cache_cfg.get("rolling", {}) or {}
    backtest_cfg = cfg.get("backtest", {}) or {}
    outputs_cfg = cfg.get("outputs", {}) or {}
    logging_cfg = cfg.get("logging", {}) or {}
    scheduler_cfg = cfg.get("scheduler", {}) or {}
    ui_cfg = cfg.get("ui", {}) or {}
    strategies_cfg = cfg.get("strategies", {}) or {}

    # YAML -> dataclass 変換 + .env 上書き
    risk = RiskConfig(
        risk_pct=float(os.getenv("RISK_PCT", risk_cfg.get("risk_pct", 0.02))),
        max_positions=int(os.getenv("MAX_POSITIONS", risk_cfg.get("max_positions", 10))),
        max_pct=float(os.getenv("MAX_PCT", risk_cfg.get("max_pct", 0.10))),
    )

    data = DataConfig(
        vendor=str(data_cfg.get("vendor", "EODHD")),
        eodhd_base=str(
            os.getenv(
                "API_EODHD_BASE",
                data_cfg.get("eodhd_base", "https://eodhistoricaldata.com"),
            )
        ),
        api_key_env=str(data_cfg.get("api_key_env", "EODHD_API_KEY")),
        cache_dir=_as_path(
            root, os.getenv("DATA_CACHE_DIR", data_cfg.get("cache_dir", "data_cache"))
        ),
        cache_recent_dir=_as_path(
            root,
            os.getenv(
                "DATA_CACHE_RECENT_DIR",
                data_cfg.get("cache_recent_dir", "data_cache_recent"),
            ),
        ),
        max_workers=_env_int("THREADS_DEFAULT", int(data_cfg.get("max_workers", 8))),
        batch_size=_env_int("BATCH_SIZE", int(data_cfg.get("batch_size", 100))),
        request_timeout=_env_int(
            "REQUEST_TIMEOUT",
            int(data_cfg.get("request_timeout", 10)),
        ),
        download_retries=_env_int(
            "DOWNLOAD_RETRIES",
            int(data_cfg.get("download_retries", 3)),
        ),
        api_throttle_seconds=_env_float(
            "API_THROTTLE_SECONDS",
            float(data_cfg.get("api_throttle_seconds", 1.5)),
        ),
    )

    max_symbols_cfg = _positive_int_or_none(rolling_cfg.get("max_symbols"))
    env_override_raw = os.getenv("ROLLING_MAX_SYMBOLS")
    if env_override_raw is not None:
        override_val = _positive_int_or_none(env_override_raw)
        if override_val is None and env_override_raw.strip() not in {"", "0"}:
            pass
        else:
            max_symbols_cfg = override_val

    stale_days_cfg = int(rolling_cfg.get("max_stale_days", 2))
    staleness_days_cfg = int(rolling_cfg.get("max_staleness_days", stale_days_cfg))

    cache = CacheConfig(
        full_dir=_as_path(root, cache_cfg.get("full_dir", "data_cache/full_backup")),
        rolling_dir=_as_path(root, cache_cfg.get("rolling_dir", "data_cache/rolling")),
        file_format=str(cache_cfg.get("file_format", "auto")),
        rolling=CacheRollingConfig(
            base_lookback_days=int(rolling_cfg.get("base_lookback_days", 300)),
            buffer_days=int(rolling_cfg.get("buffer_days", 30)),
            workers=_positive_int_or_none(rolling_cfg.get("workers")),
            max_staleness_days=staleness_days_cfg,
            prune_chunk_days=int(rolling_cfg.get("prune_chunk_days", 30)),
            meta_file=str(rolling_cfg.get("meta_file", "_meta.json")),
            max_stale_days=stale_days_cfg,
            max_symbols=max_symbols_cfg,
            round_decimals=_positive_int_or_none(rolling_cfg.get("round_decimals")),
            adaptive_window_count=int(rolling_cfg.get("adaptive_window_count", 8)),
            adaptive_increase_threshold=float(rolling_cfg.get("adaptive_increase_threshold", 1.02)),
            adaptive_decrease_threshold=float(rolling_cfg.get("adaptive_decrease_threshold", 0.98)),
            adaptive_step=int(rolling_cfg.get("adaptive_step", 1)),
            adaptive_min_workers=int(rolling_cfg.get("adaptive_min_workers", 1)),
            adaptive_max_workers=_positive_int_or_none(rolling_cfg.get("adaptive_max_workers")),
            adaptive_report_seconds=int(rolling_cfg.get("adaptive_report_seconds", 10)),
            csv_decimal_point=str(rolling_cfg.get("csv_decimal_point", ".")),
            csv_thousands_sep=(
                None
                if rolling_cfg.get("csv_thousands_sep") is None
                else str(rolling_cfg.get("csv_thousands_sep"))
            ),
            csv_field_sep=str(rolling_cfg.get("csv_field_sep", ",")),
        ),
        round_decimals=_positive_int_or_none(cache_cfg.get("round_decimals")),
        csv_decimal_point=str(cache_cfg.get("csv_decimal_point", ".")),
        csv_thousands_sep=(
            None
            if cache_cfg.get("csv_thousands_sep") is None
            else str(cache_cfg.get("csv_thousands_sep"))
        ),
        csv_field_sep=str(cache_cfg.get("csv_field_sep", ",")),
    )

    # 環境変数による丸め桁数の上書き (優先度: env > YAML)
    env_cache_round = os.getenv("CACHE_ROUND_DECIMALS")
    if env_cache_round is not None:
        cache = CacheConfig(
            full_dir=cache.full_dir,
            rolling_dir=cache.rolling_dir,
            file_format=cache.file_format,
            rolling=CacheRollingConfig(
                base_lookback_days=cache.rolling.base_lookback_days,
                buffer_days=cache.rolling.buffer_days,
                max_staleness_days=cache.rolling.max_staleness_days,
                prune_chunk_days=cache.rolling.prune_chunk_days,
                meta_file=cache.rolling.meta_file,
                max_stale_days=cache.rolling.max_stale_days,
                max_symbols=cache.rolling.max_symbols,
                round_decimals=_positive_int_or_none(env_cache_round),
            ),
            round_decimals=_positive_int_or_none(env_cache_round),
        )
    env_roll_round = os.getenv("ROLLING_CACHE_ROUND_DECIMALS")
    if env_roll_round is not None:
        cache = CacheConfig(
            full_dir=cache.full_dir,
            rolling_dir=cache.rolling_dir,
            file_format=cache.file_format,
            rolling=CacheRollingConfig(
                base_lookback_days=cache.rolling.base_lookback_days,
                buffer_days=cache.rolling.buffer_days,
                max_staleness_days=cache.rolling.max_staleness_days,
                prune_chunk_days=cache.rolling.prune_chunk_days,
                meta_file=cache.rolling.meta_file,
                max_stale_days=cache.rolling.max_stale_days,
                max_symbols=cache.rolling.max_symbols,
                round_decimals=_positive_int_or_none(env_roll_round),
            ),
            round_decimals=cache.round_decimals,
        )

    backtest = BacktestConfig(
        start_date=str(backtest_cfg.get("start_date", "2018-01-01")),
        end_date=str(backtest_cfg.get("end_date", "2024-12-31")),
        max_symbols=int(backtest_cfg.get("max_symbols", 500)),
        top_n_rank=int(backtest_cfg.get("top_n_rank", 50)),
        initial_capital=int(
            os.getenv("DEFAULT_CAPITAL", backtest_cfg.get("initial_capital", 100000))
        ),
    )

    outputs = OutputConfig(
        results_csv_dir=_as_path(
            root,
            os.getenv("RESULTS_DIR", outputs_cfg.get("results_csv_dir", "results_csv")),
        ),
        logs_dir=_as_path(
            root,
            os.getenv("LOGS_DIR", outputs_cfg.get("logs_dir", "logs")),
        ),
        signals_dir=_as_path(
            root,
            outputs_cfg.get("signals_dir", "data_cache/signals"),
        ),
    )

    logging = LoggingConfig(
        level=str(os.getenv("LOG_LEVEL", logging_cfg.get("level", "INFO"))).upper(),
        rotation=str(logging_cfg.get("rotation", "daily")),
        filename=str(os.getenv("LOG_FILENAME", logging_cfg.get("filename", "app.log"))),
    )

    scheduler = _build_scheduler(scheduler_cfg)

    # 長短比率の読み込み（0〜1にクランプ）
    try:
        _dlr_raw = os.getenv(
            "DEFAULT_LONG_RATIO",
            str(ui_cfg.get("default_long_ratio", 0.5)),
        )
        _dlr = float(_dlr_raw)
    except Exception:
        _dlr = 0.5
    _dlr = 0.0 if _dlr < 0.0 else 1.0 if _dlr > 1.0 else _dlr

    # システム別配分の読み込み（辞書が不正ならデフォルト）
    def _as_alloc_map(obj: Any, default_map: dict[str, float]) -> dict[str, float]:
        try:
            if not isinstance(obj, dict):
                return default_map
            out: dict[str, float] = {}
            for k, v in obj.items():
                try:
                    out[str(k)] = float(v)
                except Exception:
                    continue
            return out or default_map
        except Exception:
            return default_map

    ui = UIConfig(
        default_capital=int(os.getenv("DEFAULT_CAPITAL", ui_cfg.get("default_capital", 100000))),
        default_long_ratio=_dlr,
        long_allocations=_as_alloc_map(
            ui_cfg.get("long_allocations", {}),
            {
                "system1": 0.25,
                "system3": 0.25,
                "system4": 0.25,
                "system5": 0.25,
            },
        ),
        short_allocations=_as_alloc_map(
            ui_cfg.get("short_allocations", {}),
            {
                "system2": 0.40,
                "system6": 0.40,
                "system7": 0.20,
            },
        ),
        auto_tickers=tuple(ui_cfg.get("auto_tickers", ()) or ()),
        debug_mode=bool(
            os.getenv("DEBUG_MODE", str(ui_cfg.get("debug_mode", False))).lower()
            in ("1", "true", "yes")
        ),
        show_download_buttons=bool(
            os.getenv(
                "SHOW_DOWNLOAD_BUTTONS",
                str(ui_cfg.get("show_download_buttons", True)),
            ).lower()
            in ("1", "true", "yes")
        ),
    )

    # 既存互換フィールド（Settings 直下）
    data_cache = data.cache_dir
    data_cache_recent = data.cache_recent_dir
    results_dir = outputs.results_csv_dir
    logs_dir = outputs.logs_dir

    # EODHD API キー（env 優先）
    api_key = os.getenv(data.api_key_env)

    settings = Settings(
        PROJECT_ROOT=root,
        DATA_CACHE_DIR=data_cache,
        DATA_CACHE_RECENT_DIR=data_cache_recent,
        RESULTS_DIR=results_dir,
        LOGS_DIR=logs_dir,
        API_EODHD_BASE=data.eodhd_base,
        EODHD_API_KEY=api_key,
        REQUEST_TIMEOUT=data.request_timeout,
        DOWNLOAD_RETRIES=data.download_retries,
        API_THROTTLE_SECONDS=data.api_throttle_seconds,
        THREADS_DEFAULT=data.max_workers,
        MARKET_CAL_TZ=os.getenv("MARKET_CAL_TZ", "America/New_York"),
        risk=risk,
        data=data,
        cache=cache,
        backtest=backtest,
        outputs=outputs,
        logging=logging,
        scheduler=scheduler,
        ui=ui,
        strategies=strategies_cfg,
    )

    if create_dirs:
        for p in (
            settings.DATA_CACHE_DIR,
            settings.DATA_CACHE_RECENT_DIR,
            settings.RESULTS_DIR,
            settings.LOGS_DIR,
            settings.outputs.signals_dir,
        ):
            try:
                Path(p).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    return settings


# 便利関数: システム固有パラメータ取得


def get_system_params(system_name: str) -> Mapping[str, Any]:
    """YAML の strategies.<system_name> を返す。未定義なら空辞書。"""
    s = get_settings(create_dirs=False)
    return s.strategies.get(system_name, {})


__all__ = [
    "Settings",
    "RiskConfig",
    "DataConfig",
    "CacheConfig",
    "CacheRollingConfig",
    "BacktestConfig",
    "OutputConfig",
    "LoggingConfig",
    "SchedulerConfig",
    "SchedulerJob",
    "UIConfig",
    "get_settings",
    "get_system_params",
]
