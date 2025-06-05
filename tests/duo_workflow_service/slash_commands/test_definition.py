# pylint: disable=unused-import

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

from duo_workflow_service.slash_commands.definition import SlashCommandDefinition
from duo_workflow_service.slash_commands.error_handler import SlashCommandConfigError


@pytest.fixture
def mock_dir():
    mock = MagicMock(spec=Path)
    return mock


class TestSlashCommandDefinition:

    def test_initialization_and_validation(self):
        data = {
            "name": "explain",
            "description": "Explain code",
            "system_prompt": "You are a code explainer",
            "goal": "Explain the following: {text}",
            "parameters": {"max_tokens": 1024},
        }

        # Test basic initialization
        command = SlashCommandDefinition.model_validate(data)
        assert command.name == "explain"
        assert command.description == "Explain code"
        assert command.system_prompt == "You are a code explainer"
        assert command.goal == "Explain the following: {text}"
        assert command.parameters == {"max_tokens": 1024}

        # Test model_dump method
        model_dict = command.model_dump()
        assert model_dict["name"] == "explain"
        assert model_dict["description"] == "Explain code"
        assert model_dict["system_prompt"] == "You are a code explainer"
        assert model_dict["goal"] == "Explain the following: {text}"
        assert model_dict["parameters"] == {"max_tokens": 1024}

        # Test serialization and deserialization
        json_data = command.model_dump_json()
        recreated_command = SlashCommandDefinition.model_validate_json(json_data)
        assert recreated_command.name == "explain"
        assert recreated_command.parameters == {"max_tokens": 1024}

    def test_init_with_missing_fields(self):
        data = {"name": "explain"}

        command = SlashCommandDefinition.model_validate(data)

        assert command.name == "explain"
        assert command.description == ""
        assert command.system_prompt == ""
        assert command.goal == ""
        assert not command.parameters

    def test_repr(self):
        data = {
            "name": "explain",
            "description": "Explain code",
            "parameters": {"max_tokens": 1024},
        }

        command = SlashCommandDefinition.model_validate(data)

        assert (
            repr(command)
            == "SlashCommandDefinition(name=explain, description=Explain code, parameters={'max_tokens': 1024})"
        )

    @patch("pathlib.Path.exists")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=(
            "name: explain\n"
            "description: Explains code or concepts\n"
            "system_prompt: You are a helpful assistant who explains code.\n"
            "goal: Explain the following code {code}\n"
            "parameters:\n"
            "  max_tokens: 1024\n"
        ),
    )
    def test_load_slash_command_definition_success(self, mock_file, mock_exists):
        mock_exists.return_value = True

        command = SlashCommandDefinition.load_slash_command_definition(
            slash_command_name="explain"
        )

        assert command.name == "explain"
        assert command.description == "Explains code or concepts"
        assert command.system_prompt == "You are a helpful assistant who explains code."
        assert command.goal == "Explain the following code {code}"
        assert command.parameters == {"max_tokens": 1024}

        mock_exists.assert_called_once()
        mock_file.assert_called_once()

    @patch("pathlib.Path.exists")
    def test_load_slash_command_definition_file_not_found(self, mock_exists):
        mock_exists.return_value = False

        with pytest.raises(SlashCommandConfigError) as excinfo:
            SlashCommandDefinition.load_slash_command_definition(
                slash_command_name="non_existent"
            )

        assert "not found" in str(excinfo.value)

    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="invalid: yaml: :")
    @patch("yaml.safe_load", side_effect=yaml.YAMLError("YAML parsing error"))
    def test_load_slash_command_definition_invalid_yaml(
        self, _mock_yaml, _mock_file, mock_exists
    ):
        mock_exists.return_value = True

        with pytest.raises(SlashCommandConfigError) as excinfo:
            SlashCommandDefinition.load_slash_command_definition(
                slash_command_name="explain"
            )

        assert "Failed to parse YAML" in str(excinfo.value)
        assert "YAML parsing error" in str(excinfo.value)

    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="")
    @patch("yaml.safe_load", return_value=None)
    def test_load_slash_command_definition_empty_yaml(
        self, _mock_yaml, _mock_file, mock_exists
    ):
        mock_exists.return_value = True

        with pytest.raises(SlashCommandConfigError) as excinfo:
            SlashCommandDefinition.load_slash_command_definition(
                slash_command_name="explain"
            )

        assert "Invalid configuration format" in str(excinfo.value)

    @patch("pathlib.Path.exists")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="- just a list\n- not a dictionary",
    )
    @patch("yaml.safe_load", return_value=["just a list", "not a dictionary"])
    def test_load_slash_command_definition_non_dict_yaml(
        self, _mock_yaml, _mock_file, mock_exists
    ):
        mock_exists.return_value = True

        with pytest.raises(SlashCommandConfigError) as excinfo:
            SlashCommandDefinition.load_slash_command_definition(
                slash_command_name="explain"
            )

        assert "Invalid configuration format" in str(excinfo.value)

    @patch("pathlib.Path.exists")
    @patch("builtins.open", side_effect=IOError("File read error"))
    def test_load_slash_command_definition_io_error(self, _mock_file, mock_exists):
        mock_exists.return_value = True

        with pytest.raises(SlashCommandConfigError) as excinfo:
            SlashCommandDefinition.load_slash_command_definition(
                slash_command_name="explain"
            )

        assert "Error loading slash command configuration" in str(excinfo.value)
        assert "File read error" in str(excinfo.value)
