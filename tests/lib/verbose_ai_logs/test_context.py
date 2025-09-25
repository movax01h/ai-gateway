"""Tests for the shared verbose AI logs context."""

from lib.verbose_ai_logs import (
    current_verbose_ai_logs_context,
    enabled_instance_verbose_ai_logs,
)


class TestVerboseAiLogsContext:
    """Test the shared verbose AI logs context functionality."""

    def test_enabled_instance_verbose_ai_logs_default(self):
        """Test that enabled_instance_verbose_ai_logs returns False by default."""

        assert enabled_instance_verbose_ai_logs() is False

    def test_enabled_instance_verbose_ai_logs_when_enabled(self):
        """Test that enabled_instance_verbose_ai_logs returns True when context is set to True."""
        current_verbose_ai_logs_context.set(True)

        assert enabled_instance_verbose_ai_logs() is True

        current_verbose_ai_logs_context.set(False)

    def test_enabled_instance_verbose_ai_logs_when_disabled(self):
        """Test that enabled_instance_verbose_ai_logs returns False when context is set to False."""
        current_verbose_ai_logs_context.set(False)

        assert enabled_instance_verbose_ai_logs() is False
