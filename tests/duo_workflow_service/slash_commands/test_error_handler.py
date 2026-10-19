"""Tests for the slash_commands.error_handler module."""

from duo_workflow_service.slash_commands.error_handler import (
    SlashCommandError,
    SlashCommandValidationError,
)


class TestSlashCommandExceptions:
    """Tests for slash command exception classes."""

    def test_slash_command_error(self):
        """Test the base SlashCommandError class."""
        error = SlashCommandError("Test error")
        assert str(error) == "Test error"

    def test_slash_command_validation_error(self):
        """Test the SlashCommandValidationError class."""
        error = SlashCommandValidationError("Required parameter missing")
        assert str(error) == "Required parameter missing"
        assert isinstance(error, SlashCommandError)
