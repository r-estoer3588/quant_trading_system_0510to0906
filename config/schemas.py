from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, Field, field_validator

# mypy: disable-error-code=call-arg


class RiskModel(BaseModel):
    risk_pct: float = Field(0.02, ge=0, lt=1)
    max_positions: int = Field(10, ge=0)
    max_pct: float = Field(0.10, ge=0, lt=1)


class DataModel(BaseModel):
    vendor: str = "EODHD"
    eodhd_base: str = "https://eodhistoricaldata.com"
    api_key_env: str = "EODHD_API_KEY"
    cache_dir: str = "data_cache"
    cache_recent_dir: str = "data_cache_recent"
    max_workers: int = Field(8, ge=1)
    request_timeout: int = Field(10, ge=1)
    download_retries: int = Field(3, ge=0)
    api_throttle_seconds: float = Field(1.5, ge=0)


class BacktestModel(BaseModel):
    start_date: str = "2018-01-01"
    end_date: str = "2024-12-31"
    max_symbols: int = Field(500, ge=1)
    top_n_rank: int = Field(50, ge=1)
    initial_capital: int = Field(100000, ge=0)


class OutputsModel(BaseModel):
    results_csv_dir: str = "results_csv"
    logs_dir: str = "logs"
    signals_dir: str = "data_cache/signals"


class LoggingModel(BaseModel):
    level: str = "INFO"
    rotation: str = "daily"
    filename: str = "app.log"


class SchedulerJobModel(BaseModel):
    name: str
    cron: str
    task: str


class SchedulerModel(BaseModel):
    timezone: str = "America/New_York"
    jobs: list[SchedulerJobModel] = []


class UIModel(BaseModel):
    default_capital: int = 100000
    auto_tickers: list[str] = []
    debug_mode: bool = False
    show_download_buttons: bool = True


class AppConfigModel(BaseModel):
    # NOTE: pydantic plugin 未使用環境での mypy 誤検出 (call-arg) を抑制するため、
    # デフォルトインスタンス行に限定して理由付き ignore を付与。
    risk: RiskModel = RiskModel()
    data: DataModel = DataModel()
    backtest: BacktestModel = BacktestModel()
    outputs: OutputsModel = OutputsModel()
    logging: LoggingModel = LoggingModel()
    scheduler: SchedulerModel = SchedulerModel()
    ui: UIModel = UIModel()
    strategies: Mapping[str, Mapping[str, object]] = Field(default_factory=dict)

    @field_validator("logging")
    @classmethod
    def _normalize_logging(cls, v: LoggingModel):
        v.level = v.level.upper()
        return v


def validate_config_dict(d: Mapping[str, object]) -> AppConfigModel:
    """YAML辞書をPydanticで検証し、正規化したモデルを返す。"""
    model: AppConfigModel = AppConfigModel.model_validate(d)
    return model
