"""Tests for the slash_commands.definition module."""

# pylint: disable=file-naming-for-tests,unused-import

import os
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml
from pydantic import ValidationError

from duo_workflow_service.slash_commands.definition import SlashCommandDefinition


class TestSlashCommandDefinition:
    """Tests for the SlashCommandDefinition class."""

    def test_initialization_and_validation(self):
        """Test initialization with valid data."""
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
        """Test initialization with missing fields."""
        data = {"name": "explain"}

        command = SlashCommandDefinition.model_validate(data)

        assert command.name == "explain"
        assert command.description == ""
        assert command.system_prompt == ""
        assert command.goal == ""
        assert not command.parameters

    def test_repr(self):
        """Test string representation."""
        data = {
            "name": "explain",
            "description": "Explain code",
            "parameters": {"max_tokens": 1024},
        }

        command = SlashCommandDefinition.model_validate(data)

        # Test the __repr__ method returns the expected string
        assert (
            repr(command)
            == "SlashCommandDefinition(name=explain, description=Explain code, parameters={'max_tokens': 1024})"
        )

    def test_pydantic_model_validation(self):
        """Test that Pydantic validation works as expected with correct data types."""
        valid_data = {
            "name": "explain",
            "description": "Explain code",
            "system_prompt": "You are a code explainer",
            "goal": "Explain the following: {text}",
            "parameters": {"text": "string"},
        }

        command = SlashCommandDefinition.model_validate(valid_data)
        assert command.name == "explain"
        assert command.parameters == {"text": "string"}

        json_data = command.model_dump_json()
        recreated_command = SlashCommandDefinition.model_validate_json(json_data)
        assert recreated_command.name == "explain"
        assert recreated_command.parameters == {"text": "string"}

    def test_pydantic_model_with_extra_fields(self):
        """Test that extra fields are ignored when strict=False (default)."""
        data_with_extra = {
            "name": "explain",
            "description": "Explain code",
            "extra_field": "This should be ignored",
            "parameters": {"text": "string"},
        }

        command = SlashCommandDefinition.model_validate(data_with_extra)
        assert command.name == "explain"
        assert not hasattr(command, "extra_field")

        command = SlashCommandDefinition.model_validate(data_with_extra)
        assert command.name == "explain"
        assert not hasattr(command, "extra_field")

    def test_pydantic_model_with_invalid_types(self):
        """Test that Pydantic validation catches invalid data types."""
        invalid_data = {
            "name": "explain",
            "description": "Explain code",
            "parameters": "this should be a dict not a string",
        }

        with pytest.raises(ValidationError) as excinfo:
            SlashCommandDefinition.model_validate(invalid_data)

        error_msg = str(excinfo.value)
        assert "parameters" in error_msg
        assert "dict" in error_msg.lower()

    def test_model_with_nested_validation(self):
        """Test validation with complex nested parameter structure."""
        valid_nested = {
            "name": "complex",
            "parameters": {
                "max_tokens": 1024,
                "nested": {"sub1": "value1", "sub2": 42},
            },
        }

        command = SlashCommandDefinition.model_validate(valid_nested)

        params = command.model_dump()["parameters"]
        assert params["nested"]["sub2"] == 42
        assert params["max_tokens"] == 1024

        json_data = command.model_dump_json()
        recreated = SlashCommandDefinition.model_validate_json(json_data)
        assert recreated.parameters["nested"]["sub1"] == "value1"

    @patch("pathlib.Path.exists")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=(
            "name: explain\n"
            "description: Explains code or concepts\n"
            "system_prompt: You are a helpful assistant who explains code.\n"
            "goal: Explain the following code: {code}\n"
            "parameters:\n"
            "  max_tokens: 1024\n"
        ),
    )
    def test_load_slash_command_definition_success(self, mock_file, mock_exists):
        """Test loading a slash command definition from YAML."""
        # TODO: Implement this test once the load_slash_command_definition method is fully implemented
        # Mock that the file exists
        mock_exists.return_value = True

        # This would test the actual implementation
        # command = SlashCommandDefinition.load_slash_command_definition("explain")
        # assert command.name == "explain"
        # assert command.description == "Explains code or concepts"
        # ...

        # Placeholder until implementation is complete
        pytest.skip()

    @patch("pathlib.Path.exists")
    def test_load_slash_command_definition_file_not_found(self, mock_exists):
        """Test error handling when a command definition file doesn't exist."""
        # TODO: Implement this test once the load_slash_command_definition method is fully implemented
        # Mock that the file doesn't exist
        mock_exists.return_value = False

        # This would test the actual implementation
        # with pytest.raises(Exception) as excinfo:
        #     SlashCommandDefinition.load_slash_command_definition("non_existent")
        # assert "not found" in str(excinfo.value)

        # Placeholder until implementation is complete
        pytest.skip()

    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="invalid: yaml: content")
    def test_load_slash_command_definition_invalid_yaml(self, mock_file, mock_exists):
        """Test error handling when YAML is invalid."""
        # TODO: Implement this test once the load_slash_command_definition method is fully implemented
        # Mock that the file exists
        mock_exists.return_value = True

        # Mock that yaml.safe_load raises an exception
        # with patch('yaml.safe_load', side_effect=yaml.YAMLError):
        #     with pytest.raises(Exception) as excinfo:
        #         SlashCommandDefinition.load_slash_command_definition("explain")
        #     assert "YAML" in str(excinfo.value)

        # Placeholder until implementation is complete
        pytest.skip()
