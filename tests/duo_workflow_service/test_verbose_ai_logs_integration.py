"""End-to-end test for verbose AI logs functionality in DWS."""

# pylint: disable=file-naming-for-tests

import pytest

from ai_gateway.structured_logging import can_log_request_data
from lib.verbose_ai_logs import current_verbose_ai_logs_context


class TestVerboseAiLogsEndToEnd:
    """Test that verbose AI logs work end-to-end in DWS context."""

    @pytest.fixture(autouse=True)
    def setup_logging_like_dws(self, monkeypatch):
        """Initialize logging like DWS does (which now includes AI Gateway logging)."""
        # pylint: disable=import-outside-toplevel
        import ai_gateway.structured_logging as aigw_logging
        from duo_workflow_service.structured_logging import setup_logging
        from lib.verbose_ai_logs import enabled_instance_verbose_ai_logs

        # Store original values
        original_enable_request_logging = getattr(
            aigw_logging, "ENABLE_REQUEST_LOGGING", None
        )
        original_custom_models_enabled = getattr(
            aigw_logging, "CUSTOM_MODELS_ENABLED", None
        )
        original_enabled_instance_verbose_ai_logs = getattr(
            aigw_logging, "enabled_instance_verbose_ai_logs", None
        )

        # Mock environment variables to ensure consistent test behavior
        monkeypatch.setenv("AIGW_CUSTOM_MODELS__ENABLED", "true")
        monkeypatch.setenv("AIGW_LOGGING__ENABLE_REQUEST_LOGGING", "false")
        monkeypatch.setenv("AIGW_LOGGING__ENABLE_LITELLM_LOGGING", "false")

        # Setup logging (which reads the env vars we just mocked)
        setup_logging()

        # Ensure we're using the real enabled_instance_verbose_ai_logs function
        aigw_logging.enabled_instance_verbose_ai_logs = enabled_instance_verbose_ai_logs

        yield

        # Restore original values
        if original_enable_request_logging is not None:
            aigw_logging.ENABLE_REQUEST_LOGGING = original_enable_request_logging
        if original_custom_models_enabled is not None:
            aigw_logging.CUSTOM_MODELS_ENABLED = original_custom_models_enabled
        if original_enabled_instance_verbose_ai_logs is not None:
            aigw_logging.enabled_instance_verbose_ai_logs = (
                original_enabled_instance_verbose_ai_logs
            )

        current_verbose_ai_logs_context.set(False)

    def test_can_log_request_data_without_verbose_ai_logs(self):
        current_verbose_ai_logs_context.set(False)

        assert can_log_request_data() is False

    def test_can_log_request_data_with_verbose_ai_logs_enabled(self):
        current_verbose_ai_logs_context.set(True)

        assert can_log_request_data() is True

    def test_verbose_ai_logs_context_affects_logging_decision(self):
        current_verbose_ai_logs_context.set(False)
        disabled_result = can_log_request_data()

        current_verbose_ai_logs_context.set(True)
        enabled_result = can_log_request_data()

        assert disabled_result != enabled_result
        assert enabled_result is True
