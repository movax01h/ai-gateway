"""Slash Commands Module for Duo Workflow Service.

This module handles processing messages that begin with a slash (/) character, mapping them to predefined commands
configured in YAML files.
"""

from duo_workflow_service.slash_commands.processor import SlashCommandsProcessor

__all__ = ["SlashCommandsProcessor"]
