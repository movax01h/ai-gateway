"""Configuration module for slash commands.

This module is responsible for loading and parsing YAML configuration files for slash commands from the
config/slash_commands directory.
"""

from pathlib import Path
from typing import Any, Dict

import structlog
from pydantic import BaseModel, Field

SLASH_COMMANDS_CONFIG_DIR = Path(__file__).parents[1] / "config" / "slash_commands"

log = structlog.stdlib.get_logger("slash_commands")


class SlashCommandDefinition(BaseModel):
    """Class for extracting slash command configuration from the YAML files."""

    name: str = ""
    description: str = ""
    system_prompt: str = ""
    goal: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)

    def __repr__(self) -> str:
        """String representation of the slash command config."""
        return f"SlashCommandDefinition(name={self.name}, description={self.description}, parameters={self.parameters})"

    @classmethod
    def load_slash_command_definition(cls, slash_command_name):
        """Loads slash command configurations from YAML file."""
        # Ensure the config directory exists

        # Use filename as fallback
        # return error if no fallback

        # Open and read YML file using pyyaml
