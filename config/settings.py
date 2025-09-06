from __future__ import annotations

import os
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

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
    max_workers: int = 8
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
    RESULTS_DIR: Path
    LOGS_DIR: Path

    # API/ネットワーク
    API_EODHD_BASE: str
    EODHD_API_KEY: Optional[str]
    REQUEST_TIMEOUT: int
    DOWNLOAD_RETRIES: int
    API_THROTTLE_SECONDS: float

    # 実行パラメータ
    THREADS_DEFAULT: int
    MARKET_CAL_TZ: str

    # 新構成（YAMLセクション）
    risk: RiskConfig
    data: DataConfig
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


def _as_path(base: Path, p: str | os.PathLike) -> Path:
    pth = Path(p)
    return pth if pth.is_absolute() else (base / pth)


def _load_config_generic(env_var: str, default_path: Path, loader) -> Dict[str, Any]:
    cfg_path_env = os.getenv(env_var, "")
    cfg_path = Path(cfg_path_env) if cfg_path_env else default_path
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return loader(f) or {}
    except Exception:
        return {}


def _load_yaml_config(project_root: Path) -> Dict[str, Any]:
    """config/config.yaml を読み込んで辞書を返す。未存在なら空。"""
    if yaml is None:
        raise RuntimeError(
            "PyYAML が見つかりません。requirements.txt に PyYAML を追加しインストールしてください。"
        )
    default_path = project_root / "config" / "config.yaml"
    return _load_config_generic("APP_CONFIG", default_path, lambda f: yaml.safe_load(f))


def _load_yaml_config_validated(project_root: Path) -> Dict[str, Any]:
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


def _load_json_config(project_root: Path) -> Dict[str, Any]:
    """config/config.json を読み込んで辞書として返す。存在しなければ {}。"""
    default_path = project_root / "config" / "config.json"
    return _load_config_generic("APP_CONFIG_JSON", default_path, lambda f: json.load(f))


def _load_config_json_or_yaml_validated(project_root: Path) -> Dict[str, Any]:
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


def _build_scheduler(cfg: Dict[str, Any]) -> SchedulerConfig:
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
        eodhd_base=str(os.getenv("API_EODHD_BASE", data_cfg.get("eodhd_base", "https://eodhistoricaldata.com"))),
        api_key_env=str(data_cfg.get("api_key_env", "EODHD_API_KEY")),
        cache_dir=_as_path(root, os.getenv("DATA_CACHE_DIR", data_cfg.get("cache_dir", "data_cache"))),
        max_workers=_env_int("THREADS_DEFAULT", int(data_cfg.get("max_workers", 8))),
        request_timeout=_env_int("REQUEST_TIMEOUT", int(data_cfg.get("request_timeout", 10))),
        download_retries=_env_int("DOWNLOAD_RETRIES", int(data_cfg.get("download_retries", 3))),
        api_throttle_seconds=_env_float("API_THROTTLE_SECONDS", float(data_cfg.get("api_throttle_seconds", 1.5))),
    )

    backtest = BacktestConfig(
        start_date=str(backtest_cfg.get("start_date", "2018-01-01")),
        end_date=str(backtest_cfg.get("end_date", "2024-12-31")),
        max_symbols=int(backtest_cfg.get("max_symbols", 500)),
        top_n_rank=int(backtest_cfg.get("top_n_rank", 50)),
        initial_capital=int(os.getenv("DEFAULT_CAPITAL", backtest_cfg.get("initial_capital", 100000))),
    )

    outputs = OutputConfig(
        results_csv_dir=_as_path(root, os.getenv("RESULTS_DIR", outputs_cfg.get("results_csv_dir", "results_csv"))),
        logs_dir=_as_path(root, os.getenv("LOGS_DIR", outputs_cfg.get("logs_dir", "logs"))),
        signals_dir=_as_path(root, outputs_cfg.get("signals_dir", "data_cache/signals")),
    )

    logging = LoggingConfig(
        level=str(os.getenv("LOG_LEVEL", logging_cfg.get("level", "INFO"))).upper(),
        rotation=str(logging_cfg.get("rotation", "daily")),
        filename=str(os.getenv("LOG_FILENAME", logging_cfg.get("filename", "app.log"))),
    )

    scheduler = _build_scheduler(scheduler_cfg)

    ui = UIConfig(
        default_capital=int(os.getenv("DEFAULT_CAPITAL", ui_cfg.get("default_capital", 100000))),
        auto_tickers=tuple(ui_cfg.get("auto_tickers", ()) or ()),
        debug_mode=bool(os.getenv("DEBUG_MODE", str(ui_cfg.get("debug_mode", False))).lower() in ("1", "true", "yes")),
        show_download_buttons=bool(os.getenv("SHOW_DOWNLOAD_BUTTONS", str(ui_cfg.get("show_download_buttons", True))).lower() in ("1", "true", "yes")),
    )

    # 既存互換フィールド（Settings 直下）
    data_cache = data.cache_dir
    results_dir = outputs.results_csv_dir
    logs_dir = outputs.logs_dir

    # EODHD API キー（env 優先）
    api_key = os.getenv(data.api_key_env)

    settings = Settings(
        PROJECT_ROOT=root,
        DATA_CACHE_DIR=data_cache,
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
        backtest=backtest,
        outputs=outputs,
        logging=logging,
        scheduler=scheduler,
        ui=ui,
        strategies=strategies_cfg,
    )

    if create_dirs:
        for p in (settings.DATA_CACHE_DIR, settings.RESULTS_DIR, settings.LOGS_DIR, settings.outputs.signals_dir):
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
    "BacktestConfig",
    "OutputConfig",
    "LoggingConfig",
    "SchedulerConfig",
    "SchedulerJob",
    "UIConfig",
    "get_settings",
    "get_system_params",
]
