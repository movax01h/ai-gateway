from pathlib import Path
from typing import Any, Dict

import structlog
import yaml
from pydantic import BaseModel, Field

from duo_workflow_service.slash_commands.error_handler import SlashCommandConfigError

# Constants
SLASH_COMMANDS_CONFIG_DIR = Path(__file__).parents[1] / "config" / "slash_commands"
LOGGER = structlog.stdlib.get_logger("slash_commands")


class SlashCommandDefinition(BaseModel):
    """Defines the structure and configuration of a slash command.

    This class represents the configuration loaded from YAML files and provides
    a structured way to access slash command properties.

    Attributes:
        name: The name of the slash command
        description: A brief description of what the slash command does
        system_prompt: The prompt provided to the AI system
        goal: The intended outcome of the slash command
        parameters: Optional parameters that can be passed to the slash command
    """

    name: str = ""
    description: str = ""
    system_prompt: str = ""
    goal: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return f"SlashCommandDefinition(name={self.name}, description={self.description}, parameters={self.parameters})"

    @classmethod
    def load_slash_command_definition(
        cls, slash_command_name: str
    ) -> "SlashCommandDefinition":
        """Loads slash command configurations from YAML file.

        Args:
            slash_command_name: The name of the slash command to load

        Returns:
            SlashCommandDefinition: The loaded slash command definition

        Raises:
            SlashCommandConfigError: If the configuration file does not exist or has invalid YAML
        """
        config_file_path = _get_config_file_path(slash_command_name)

        config_data = _load_yaml_file(config_file_path)
        return cls(**config_data)


def _get_config_file_path(slash_command_name: str) -> Path:
    """Construct the path to the YAML configuration file and validate it exists.

    Args:
        slash_command_name: The name of the slash command

    Returns:
        Path: The validated path to the configuration file

    Raises:
        SlashCommandConfigError: If no configuration file exists
    """
    # Check both possible extensions
    yaml_path = SLASH_COMMANDS_CONFIG_DIR / f"{slash_command_name}.yaml"
    yml_path = SLASH_COMMANDS_CONFIG_DIR / f"{slash_command_name}.yml"

    for path in (yaml_path, yml_path):
        if path.exists():
            return path

    error_message = (
        f"Slash command configuration file for '{slash_command_name}' not found"
    )
    LOGGER.error(error_message)
    raise SlashCommandConfigError(error_message)


def _load_yaml_file(config_file_path: Path) -> Dict[str, Any]:
    """Load and parse YAML file.

    Args:
        config_file_path: Path to the YAML configuration file

    Returns:
        Dict[str, Any]: The parsed YAML content as a dictionary

    Raises:
        SlashCommandConfigError: If the file cannot be read, parsed, or has invalid format
    """
    try:
        with open(config_file_path, "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file)

        if not config_data or not isinstance(config_data, dict):
            error_message = f"Invalid configuration format in '{config_file_path}'"
            LOGGER.error(error_message)
            raise SlashCommandConfigError(error_message)

        return config_data

    except yaml.YAMLError as yaml_error:
        error_message = (
            f"Failed to parse YAML in '{config_file_path}': {str(yaml_error)}"
        )
        LOGGER.error(error_message)
        raise SlashCommandConfigError(error_message)

    except Exception as e:
        error_message = f"Error loading slash command configuration from '{config_file_path}': {str(e)}"
        LOGGER.error(error_message)
        raise SlashCommandConfigError(error_message)
