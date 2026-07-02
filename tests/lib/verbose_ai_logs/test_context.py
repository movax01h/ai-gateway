"""Tests for the shared verbose AI logs context."""

import pytest

from lib.verbose_ai_logs import (
    X_GITLAB_EXTENDED_LOGGING_HEADER,
    current_verbose_ai_logs_context,
    enabled_instance_verbose_ai_logs,
    extended_logging_context,
    is_extended_logging_enabled,
)


@pytest.mark.parametrize(
    "context_var,check_fn",
    [
        (current_verbose_ai_logs_context, enabled_instance_verbose_ai_logs),
        (extended_logging_context, is_extended_logging_enabled),
    ],
    ids=["instance_verbose_ai_logs", "extended_logging"],
)
class TestLoggingContextVars:
    """Test that each logging context var and its helper function behave consistently."""

    @pytest.fixture(autouse=True)
    def reset_context(self):
        """Reset both context vars to False after each test."""
        yield
        current_verbose_ai_logs_context.set(False)
        extended_logging_context.set(False)

    def test_default_is_false(self, context_var, check_fn):  # pylint: disable=unused-argument
        """Helper returns False when the context var has not been set."""
        assert check_fn() is False

    def test_returns_true_when_enabled(self, context_var, check_fn):
        """Helper returns True when the context var is set to True."""
        context_var.set(True)

        assert check_fn() is True

    def test_returns_false_when_disabled(self, context_var, check_fn):
        """Helper returns False when the context var is explicitly set to False."""
        context_var.set(False)

        assert check_fn() is False


class TestExtendedLoggingContext:
    """Additional tests specific to the extended logging context."""

    @pytest.fixture(autouse=True)
    def reset_contexts(self):
        """Reset both context vars to False after each test."""
        yield
        current_verbose_ai_logs_context.set(False)
        extended_logging_context.set(False)

    def test_independent_from_verbose_ai_logs(self):
        """extended_logging_context is independent of current_verbose_ai_logs_context."""
        current_verbose_ai_logs_context.set(True)
        extended_logging_context.set(False)

        assert enabled_instance_verbose_ai_logs() is True
        assert is_extended_logging_enabled() is False

        current_verbose_ai_logs_context.set(False)
        extended_logging_context.set(True)

        assert enabled_instance_verbose_ai_logs() is False
        assert is_extended_logging_enabled() is True

    def test_x_gitlab_extended_logging_header_value(self):
        """The header constant has the expected value."""
        assert X_GITLAB_EXTENDED_LOGGING_HEADER == "x-gitlab-extended-logging"
