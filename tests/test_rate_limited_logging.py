"""
Tests for rate-limited logging functionality.
"""

import time
from unittest.mock import Mock

from common.rate_limited_logging import RateLimitedLogger, create_rate_limited_logger


class TestRateLimitedLogger:
    """Rate limited logger tests."""

    def test_rate_limiting_works(self):
        """Test that messages are rate limited correctly."""
        mock_logger = Mock()
        rate_logger = RateLimitedLogger(mock_logger, default_interval=1.0)

        # First call should go through
        rate_logger.debug_rate_limited("test message")
        assert mock_logger.debug.call_count == 1

        # Immediate second call should be blocked
        rate_logger.debug_rate_limited("test message")
        assert mock_logger.debug.call_count == 1

        # After interval, should go through again
        time.sleep(1.1)
        rate_logger.debug_rate_limited("test message")
        assert mock_logger.debug.call_count == 2

    def test_different_message_keys(self):
        """Test that different message keys are tracked separately."""
        mock_logger = Mock()
        rate_logger = RateLimitedLogger(mock_logger, default_interval=1.0)

        rate_logger.debug_rate_limited("message1", message_key="key1")
        rate_logger.debug_rate_limited("message2", message_key="key2")

        # Both should go through as they have different keys
        assert mock_logger.debug.call_count == 2

        # Same keys should be blocked
        rate_logger.debug_rate_limited("message1", message_key="key1")
        rate_logger.debug_rate_limited("message2", message_key="key2")

        assert mock_logger.debug.call_count == 2

    def test_custom_interval(self):
        """Test that custom intervals work correctly."""
        mock_logger = Mock()
        rate_logger = RateLimitedLogger(mock_logger, default_interval=5.0)

        rate_logger.info_rate_limited("test", interval=0.1)
        assert mock_logger.info.call_count == 1

        # Should be blocked immediately
        rate_logger.info_rate_limited("test", interval=0.1)
        assert mock_logger.info.call_count == 1

        # Should go through after short interval
        time.sleep(0.2)
        rate_logger.info_rate_limited("test", interval=0.1)
        assert mock_logger.info.call_count == 2

    def test_clear_history(self):
        """Test that history clearing works."""
        mock_logger = Mock()
        rate_logger = RateLimitedLogger(mock_logger, default_interval=1.0)

        rate_logger.debug_rate_limited("test message")
        assert mock_logger.debug.call_count == 1

        # Should be blocked
        rate_logger.debug_rate_limited("test message")
        assert mock_logger.debug.call_count == 1

        # Clear history
        rate_logger.clear_history()

        # Should go through again
        rate_logger.debug_rate_limited("test message")
        assert mock_logger.debug.call_count == 2

    def test_all_log_levels(self):
        """Test that all log levels work correctly."""
        mock_logger = Mock()
        rate_logger = RateLimitedLogger(mock_logger, default_interval=0.1)

        rate_logger.debug_rate_limited("debug msg", message_key="debug")
        rate_logger.info_rate_limited("info msg", message_key="info")
        rate_logger.warning_rate_limited("warning msg", message_key="warning")

        assert mock_logger.debug.call_count == 1
        assert mock_logger.info.call_count == 1
        assert mock_logger.warning.call_count == 1


def test_create_rate_limited_logger():
    """Test factory function."""
    rate_logger = create_rate_limited_logger("test_logger", 2.0)

    assert isinstance(rate_logger, RateLimitedLogger)
    assert rate_logger.default_interval == 2.0
    assert rate_logger.logger.name == "test_logger"
