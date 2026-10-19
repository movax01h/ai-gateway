"""Errors for slash commands."""


class SlashCommandError(Exception):
    """Base exception for slash command errors."""


class SlashCommandValidationError(SlashCommandError):
    """Error when input validation fails."""
