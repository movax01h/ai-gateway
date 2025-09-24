"""Integration test for verbose AI logs functionality in AI Gateway."""

from unittest.mock import patch

from ai_gateway.structured_logging import can_log_request_data
from lib.verbose_ai_logs import current_verbose_ai_logs_context


class TestVerboseAiLogsIntegration:
    """Integration tests for verbose AI logs functionality."""

    def setup_method(self):
        """Reset context before each test."""
        current_verbose_ai_logs_context.set(False)

    def teardown_method(self):
        """Clean up context after each test."""
        current_verbose_ai_logs_context.set(False)

    def test_can_log_request_data_with_verbose_ai_logs_enabled(self):
        """Test can_log_request_data when verbose AI logs are enabled via shared context."""
        # Mock environment variables to simulate self-hosted mode
        with (
            patch("ai_gateway.structured_logging.CUSTOM_MODELS_ENABLED", True),
            patch("ai_gateway.structured_logging.ENABLE_REQUEST_LOGGING", False),
        ):

            # Set verbose AI logs enabled in shared context
            current_verbose_ai_logs_context.set(True)

            # Should return True because CUSTOM_MODELS_ENABLED=True and enabled_instance_verbose_ai_logs()=True
            assert can_log_request_data() is True

    def test_can_log_request_data_with_verbose_ai_logs_disabled(self):
        """Test can_log_request_data when verbose AI logs are disabled via shared context."""
        # Mock environment variables to simulate self-hosted mode
        with (
            patch("ai_gateway.structured_logging.CUSTOM_MODELS_ENABLED", True),
            patch("ai_gateway.structured_logging.ENABLE_REQUEST_LOGGING", False),
            patch(
                "ai_gateway.structured_logging.is_feature_enabled", return_value=False
            ),
        ):

            # Set verbose AI logs disabled in shared context
            current_verbose_ai_logs_context.set(False)

            # Should return False because CUSTOM_MODELS_ENABLED=True but enabled_instance_verbose_ai_logs()=False
            # and ENABLE_REQUEST_LOGGING=False and EXPANDED_AI_LOGGING feature flag=False
            assert can_log_request_data() is False

    def test_can_log_request_data_with_enable_request_logging_override(self):
        """Test can_log_request_data when ENABLE_REQUEST_LOGGING is True (should always return True)."""
        with patch("ai_gateway.structured_logging.ENABLE_REQUEST_LOGGING", True):

            # Even with verbose AI logs disabled, should return True due to ENABLE_REQUEST_LOGGING=True
            current_verbose_ai_logs_context.set(False)

            assert can_log_request_data() is True

    def test_can_log_request_data_saas_mode_with_feature_flag(self):
        """Test can_log_request_data in SaaS mode with EXPANDED_AI_LOGGING feature flag."""
        with (
            patch("ai_gateway.structured_logging.CUSTOM_MODELS_ENABLED", False),
            patch("ai_gateway.structured_logging.ENABLE_REQUEST_LOGGING", False),
            patch(
                "ai_gateway.structured_logging.is_feature_enabled", return_value=True
            ),
        ):

            # In SaaS mode, verbose AI logs context shouldn't matter
            current_verbose_ai_logs_context.set(False)

            # Should return True because not CUSTOM_MODELS_ENABLED and EXPANDED_AI_LOGGING feature flag=True
            assert can_log_request_data() is True
