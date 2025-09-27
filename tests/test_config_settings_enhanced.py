"""Enhanced tests for config.settings module to improve coverage."""

import os
from pathlib import Path
from unittest.mock import patch


from config.settings import (
    _coerce_int,
    _env_int,
    _env_float,
    get_settings,
    Settings,
    RiskConfig,
    DataConfig,
    CacheConfig,
    UIConfig,
    LoggingConfig,
    SchedulerConfig,
    SchedulerJob,
)


class TestSettingsUtilityFunctions:
    """Test utility functions in config.settings module."""

    def test_env_int_with_valid_value(self):
        """Test _env_int with valid environment variable."""
        with patch.dict(os.environ, {"TEST_VAR": "42"}):
            result = _env_int("TEST_VAR", 10)
            assert result == 42

    def test_env_int_with_empty_value(self):
        """Test _env_int with empty environment variable."""
        with patch.dict(os.environ, {"TEST_VAR": ""}):
            result = _env_int("TEST_VAR", 10)
            assert result == 10

    def test_env_int_with_invalid_value(self):
        """Test _env_int with invalid environment variable."""
        with patch.dict(os.environ, {"TEST_VAR": "not_a_number"}):
            result = _env_int("TEST_VAR", 10)
            assert result == 10

    def test_env_int_with_missing_value(self):
        """Test _env_int with missing environment variable."""
        with patch.dict(os.environ, {}, clear=True):
            result = _env_int("MISSING_VAR", 10)
            assert result == 10

    def test_env_float_with_valid_value(self):
        """Test _env_float with valid environment variable."""
        with patch.dict(os.environ, {"TEST_FLOAT": "3.14"}):
            result = _env_float("TEST_FLOAT", 2.0)
            assert result == 3.14

    def test_env_float_with_invalid_value(self):
        """Test _env_float with invalid environment variable."""
        with patch.dict(os.environ, {"TEST_FLOAT": "not_a_float"}):
            result = _env_float("TEST_FLOAT", 2.0)
            assert result == 2.0

    def test_coerce_int_with_valid_values(self):
        """Test _coerce_int with various valid inputs."""
        assert _coerce_int(42, 0) == 42
        assert _coerce_int("42", 0) == 42
        assert _coerce_int(42.7, 0) == 42
        assert _coerce_int(True, 0) == 1
        assert _coerce_int(False, 0) == 0

    def test_coerce_int_with_invalid_values(self):
        """Test _coerce_int with invalid inputs."""
        assert _coerce_int("not_a_number", 10) == 10
        assert _coerce_int(None, 5) == 5
        assert _coerce_int([], 20) == 20


class TestDataclasses:
    """Test the dataclass configurations."""

    def test_risk_config_defaults(self):
        """Test RiskConfig default values."""
        config = RiskConfig()
        assert config.risk_pct == 0.02
        assert config.max_positions == 10
        assert config.max_pct == 0.10

    def test_risk_config_custom_values(self):
        """Test RiskConfig with custom values."""
        config = RiskConfig(risk_pct=0.05, max_positions=15, max_pct=0.15)
        assert config.risk_pct == 0.05
        assert config.max_positions == 15
        assert config.max_pct == 0.15

    def test_data_config_defaults(self):
        """Test DataConfig default values."""
        config = DataConfig()
        assert config.vendor == "EODHD"
        assert config.eodhd_base == "https://eodhistoricaldata.com"
        assert config.api_key_env == "EODHD_API_KEY"
        assert config.max_workers == 20
        assert config.batch_size == 100

    def test_cache_config_defaults(self):
        """Test CacheConfig default values."""
        config = CacheConfig()
        assert config.full_dir == Path("data_cache/full_backup")
        assert config.rolling_dir == Path("data_cache/rolling")
        assert config.file_format == "auto"
        assert config.round_decimals == 4
        assert config.disable_rolling_cache is False

    def test_ui_config_defaults(self):
        """Test UIConfig default values."""
        config = UIConfig()
        assert config.default_capital == 100000
        assert config.default_long_ratio == 0.5
        assert config.debug_mode is False
        assert config.show_download_buttons is True
        assert "system1" in config.long_allocations
        assert "system2" in config.short_allocations

    def test_logging_config_defaults(self):
        """Test LoggingConfig default values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.rotation == "daily"
        assert config.filename == "app.log"

    def test_scheduler_config_defaults(self):
        """Test SchedulerConfig default values."""
        config = SchedulerConfig()
        assert config.timezone == "America/New_York"
        assert len(config.jobs) == 0

    def test_scheduler_job_creation(self):
        """Test SchedulerJob creation."""
        job = SchedulerJob(name="test_job", cron="0 9 * * MON-FRI", task="test_task")
        assert job.name == "test_job"
        assert job.cron == "0 9 * * MON-FRI"
        assert job.task == "test_task"


class TestSettingsIntegration:
    """Test Settings class integration."""

    def test_get_settings_basic_functionality(self):
        """Test basic get_settings functionality."""
        settings = get_settings()
        assert isinstance(settings, Settings)
        assert isinstance(settings.risk, RiskConfig)
        assert isinstance(settings.data, DataConfig)
        assert isinstance(settings.cache, CacheConfig)
        assert isinstance(settings.ui, UIConfig)

    def test_get_settings_with_environment_variables(self):
        """Test get_settings with environment variables."""
        # Only test override if the key is not already set in the actual environment
        original_key = os.environ.get("EODHD_API_KEY", "")
        if not original_key:  # Only test if key not already set
            env_vars = {
                "EODHD_API_KEY": "test_api_key",
                "REQUEST_TIMEOUT": "30",
                "DOWNLOAD_RETRIES": "5",
                "THREADS_DEFAULT": "8",
            }

            with patch.dict(os.environ, env_vars):
                settings = get_settings()
                assert settings.EODHD_API_KEY == "test_api_key"
                assert settings.REQUEST_TIMEOUT == 30
                assert settings.DOWNLOAD_RETRIES == 5
                assert settings.THREADS_DEFAULT == 8

    def test_get_settings_with_missing_api_key(self):
        """Test get_settings handles missing API key gracefully."""
        # Test that function doesn't crash with missing env vars
        settings = get_settings()
        # Just verify the function works and returns appropriate type
        assert hasattr(settings, "EODHD_API_KEY")
        assert isinstance(settings.EODHD_API_KEY, str | type(None))

    def test_get_settings_creates_directories(self):
        """Test that get_settings creates necessary directories."""
        settings = get_settings(create_dirs=True)

        # Check that directories exist after creation
        assert settings.DATA_CACHE_DIR.exists()
        assert settings.LOGS_DIR.exists()
        assert settings.RESULTS_DIR.exists()

    def test_get_settings_caching(self):
        """Test that get_settings uses caching properly."""
        settings1 = get_settings()
        settings2 = get_settings()

        # Should be the same object due to caching
        assert settings1 is settings2

    def test_strategies_mapping(self):
        """Test that strategies mapping contains expected systems."""
        settings = get_settings()
        assert "system1" in settings.strategies
        assert "system2" in settings.strategies
        assert isinstance(settings.strategies["system1"], dict)
        assert isinstance(settings.strategies["system2"], dict)
