"""
Error handling for slash commands.

This module provides error handling mechanisms for slash commands,
helping to report errors without crashing the service.
"""

from datetime import datetime, timezone
from typing import Dict, Optional

import structlog

from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    SlashCommandStatus,
    UiChatLog,
)

# Setup logging
log = structlog.stdlib.get_logger("slash_commands")


class SlashCommandError(Exception):
    """Base exception for slash command errors."""


class SlashCommandConfigError(SlashCommandError):
    """Error when a slash command configuration is invalid."""


class SlashCommandTemplateError(SlashCommandError):
    """Error when a template cannot be expanded."""


class SlashCommandValidationError(SlashCommandError):
    """Error when input validation fails."""


def create_error_ui_chat_log(error_message: str) -> UiChatLog:
    """
    Create a UI chat log entry for a slash command error.

    Args:
        error_message: The error message to display

    Returns:
        UiChatLog entry with the error information
    """
    return UiChatLog(
        message_type=MessageTypeEnum.TOOL,
        content=f"Slash command error: {error_message}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        status=SlashCommandStatus.FAILURE,
        correlation_id=None,
        tool_info=None,
        context_elements=None,
    )


def format_error_response(error: SlashCommandError) -> str:
    """
    Format a slash command error into a user-friendly error message.

    Args:
        error: The slash command error

    Returns:
        Formatted error message
    """
    if isinstance(error, SlashCommandConfigError):
        return f"Configuration error: {str(error)}"
    if isinstance(error, SlashCommandTemplateError):
        return f"Template error: {str(error)}"
    if isinstance(error, SlashCommandValidationError):
        return f"Validation error: {str(error)}"

    return f"Error processing slash command: {str(error)}"


def log_command_error(
    command_name: Optional[str],
    error: Exception,
    context: Optional[Dict] = None,
) -> None:
    """
    Log a slash command error with appropriate context.

    Args:
        command_name: The name of the command that caused the error, if known
        error: The exception that was raised
        context: Additional context information
    """
    if context is None:
        context = {}

    log_context = {
        "command": command_name or "unknown",
        "error_type": type(error).__name__,
        **context,
    }

    log.error(f"Slash command error: {str(error)}", **log_context)
