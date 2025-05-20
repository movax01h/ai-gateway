"""
Tests for the slash_commands.error_handler module.
"""

# pylint: disable=file-naming-for-tests,unused-import

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from duo_workflow_service.entities.state import MessageTypeEnum, ToolStatus
from duo_workflow_service.slash_commands.error_handler import (
    SlashCommandConfigError,
    SlashCommandError,
    SlashCommandTemplateError,
    SlashCommandValidationError,
    create_error_ui_chat_log,
    format_error_response,
    log_command_error,
)


class TestSlashCommandExceptions:
    """Tests for slash command exception classes."""

    def test_slash_command_error(self):
        """Test the base SlashCommandError class."""
        error = SlashCommandError("Test error")
        assert str(error) == "Test error"

    def test_slash_command_config_error(self):
        """Test the SlashCommandConfigError class."""
        error = SlashCommandConfigError("Invalid config")
        assert str(error) == "Invalid config"
        assert isinstance(error, SlashCommandError)

    def test_slash_command_template_error(self):
        """Test the SlashCommandTemplateError class."""
        error = SlashCommandTemplateError("Template expansion failed")
        assert str(error) == "Template expansion failed"
        assert isinstance(error, SlashCommandError)

    def test_slash_command_validation_error(self):
        """Test the SlashCommandValidationError class."""
        error = SlashCommandValidationError("Required parameter missing")
        assert str(error) == "Required parameter missing"
        assert isinstance(error, SlashCommandError)


@patch("duo_workflow_service.slash_commands.error_handler.datetime")
def test_create_error_ui_chat_log(mock_datetime):
    """Test creating an error log entry."""
    mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc)
    mock_datetime.timezone = timezone

    error_log = create_error_ui_chat_log("Something went wrong")

    assert error_log["message_type"] == MessageTypeEnum.TOOL
    assert error_log["content"] == "Slash command error: Something went wrong"
    assert error_log["timestamp"] == "2023-01-01T12:00:00+00:00"
    assert error_log["status"] == ToolStatus.FAILURE
    assert error_log["correlation_id"] is None
    assert error_log["tool_info"] is None


def test_format_error_response_config_error():
    """Test formatting a configuration error."""
    error = SlashCommandConfigError("Missing required field")
    formatted = format_error_response(error)
    assert formatted == "Configuration error: Missing required field"


def test_format_error_response_template_error():
    """Test formatting a template error."""
    error = SlashCommandTemplateError("Invalid template variable")
    formatted = format_error_response(error)
    assert formatted == "Template error: Invalid template variable"


def test_format_error_response_validation_error():
    """Test formatting a validation error."""
    error = SlashCommandValidationError("Invalid parameter type")
    formatted = format_error_response(error)
    assert formatted == "Validation error: Invalid parameter type"


def test_format_error_response_generic_error():
    """Test formatting a generic error."""
    error = SlashCommandError("Something went wrong")
    formatted = format_error_response(error)
    assert formatted == "Error processing slash command: Something went wrong"


@patch("duo_workflow_service.slash_commands.error_handler.log")
def test_log_command_error_with_name(mock_logger):
    """Test logging an error with a command name."""
    error = ValueError("Invalid value")
    log_command_error("explain", error)

    mock_logger.error.assert_called_once()
    # Check the error message and context
    args, kwargs = mock_logger.error.call_args
    assert "Slash command error: Invalid value" in args
    assert kwargs["command"] == "explain"
    assert kwargs["error_type"] == "ValueError"


@patch("duo_workflow_service.slash_commands.error_handler.log")
def test_log_command_error_without_name(mock_logger):
    """Test logging an error without a command name."""
    error = RuntimeError("Unexpected error")
    log_command_error(None, error)

    mock_logger.error.assert_called_once()
    # Check the error message and context
    args, kwargs = mock_logger.error.call_args
    assert "Slash command error: Unexpected error" in args
    assert kwargs["command"] == "unknown"
    assert kwargs["error_type"] == "RuntimeError"


@patch("duo_workflow_service.slash_commands.error_handler.log")
def test_log_command_error_with_context(mock_logger):
    """Test logging an error with additional context."""
    error = Exception("Something went wrong")
    context = {"user_id": "123", "input": "/explain code"}
    log_command_error("explain", error, context)

    mock_logger.error.assert_called_once()
    # Check the error message and context
    args, kwargs = mock_logger.error.call_args
    assert "Slash command error: Something went wrong" in args
    assert kwargs["command"] == "explain"
    assert kwargs["error_type"] == "Exception"
    assert kwargs["user_id"] == "123"
    assert kwargs["input"] == "/explain code"
