# pylint: disable=unused-import

from unittest.mock import MagicMock, patch

import pytest

from duo_workflow_service.slash_commands.error_handler import SlashCommandError
from duo_workflow_service.slash_commands.goal_parser import parse
from duo_workflow_service.slash_commands.processor import SlashCommandsProcessor
from lib.result import Error, Ok

# Constants for patch paths
MODULE_PATH = "duo_workflow_service.slash_commands"
GOAL_PARSER_PATH = f"{MODULE_PATH}.goal_parser.parse"
DEFINITION_PATH = f"{MODULE_PATH}.definition.SlashCommandDefinition"
LOAD_DEFINITION_PATH = f"{DEFINITION_PATH}.load_slash_command_definition"
PROMPT_EXPANDER_PARSER_PATH = f"{MODULE_PATH}.processor.parse"
LOG_COMMAND_ERROR_PATH = f"{MODULE_PATH}.processor.log_command_error"
LOG_PATH = f"{MODULE_PATH}.processor.log"


class TestSlashCommandsGoalExpander:
    @pytest.fixture
    def processor(self):
        return SlashCommandsProcessor()

    @pytest.fixture
    def mock_parser_instance(self):
        return MagicMock()

    @patch(GOAL_PARSER_PATH)
    def test_not_slash_command(self, mock_parser, processor):

        mock_parser.parse().return_value = (None, "/")

        result = processor.process("/")
        error_message = "The message does not contain a command after the slash."

        assert isinstance(result, Error)
        assert result.value is None
        assert str(result.error) == error_message

    @patch(GOAL_PARSER_PATH)
    @patch(LOAD_DEFINITION_PATH)
    def test_success(
        self,
        mock_definition_load,
        mock_parser,
        processor,
    ):
        mock_command_def = MagicMock(
            name="explain",
            system_prompt="You are an expert at explaining code",
            goal="Explain this code: {code}",
            parameters={"max_tokens": 1024},
        )
        mock_definition_load.return_value = mock_command_def

        mock_parser.parse().return_value = ("explain", "def add(a, b): return a + b")

        result = processor.process("/explain def add(a, b): return a + b")

        assert isinstance(result, Ok)
        assert result.value["success"] is True
        assert result.value["command_name"] == "explain"
        assert result.value["goal"] == "Explain this code: {code}"
        assert result.value["message_context"] == "def add(a, b): return a + b"
        mock_definition_load.assert_called_once_with("explain")

    @patch(GOAL_PARSER_PATH)
    def test_command_not_found_error(self, mock_parser, processor):
        mock_parser.parse().return_value = ("nonexistent", "some text")

        error_message = "Slash command configuration file for 'nonexistent' not found"

        result = processor.process("/nonexistent some text")

        assert isinstance(result, Error)
        assert str(result.error) == error_message

    @patch(GOAL_PARSER_PATH)
    @patch(DEFINITION_PATH)
    def test_missing_required_parameters(
        self, mock_definition, mock_parser, processor, mock_parser_instance
    ):
        mock_parser.return_value = mock_parser_instance
        parsed_command = ("test", "")
        mock_parser_instance.parse.return_value = parsed_command

        mock_definition_instance = MagicMock(
            name="test",
            goal="Test goal with required params",
            parameters={"required_param": {"type": "string", "required": True}},
        )
        mock_definition.load_slash_command_definition.return_value = (
            mock_definition_instance
        )

        result = processor.process("/test")

        assert isinstance(result, Error)

    @patch(LOG_COMMAND_ERROR_PATH)
    @patch(PROMPT_EXPANDER_PARSER_PATH)
    def test_slash_command_error_handling(
        self, mock_parse, mock_log_command_error, processor
    ):
        mock_parse.parse().return_value = ("test_command", "remaining text")
        test_error = SlashCommandError("Test slash command error")

        mock_parse.side_effect = test_error

        result = processor.process("/test_command remaining text")

        assert isinstance(result, Error)
        assert result.error == test_error
        mock_log_command_error.assert_called_once_with(
            command_name=None, error=test_error
        )

    @patch(LOG_PATH)
    @patch(PROMPT_EXPANDER_PARSER_PATH)
    def test_general_exception_handling(self, mock_parse, mock_log, processor):
        mock_parse.parse().return_value = ("test_command", "remaining text")
        test_error = ValueError("Test general exception")
        mock_parse.side_effect = test_error

        result = processor.process("/test_command remaining text")

        assert isinstance(result, Error)
        assert result.error == test_error
        mock_log.error.assert_called_once_with(
            f"Error processing slash command: {str(test_error)}"
        )
