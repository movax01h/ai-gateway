from unittest.mock import AsyncMock, MagicMock

import pytest

from contract import contract_pb2
from duo_workflow_service.tools.search_system import Grep, GrepInput


class TestGrep:
    valid_test_cases = [
        # Basic recursive search
        pytest.param(
            {
                "pattern": "test",
                "search_directory": ".",
                "case_insensitive": False,
            },
            "test.py:10:test line",
            "-r test",
            id="basic_recursive_grep",
        ),
        # Test with directory
        pytest.param(
            {
                "pattern": "test",
                "search_directory": "src",
                "case_insensitive": False,
            },
            "src/test.py:10:test line",
            "test -- src",
            id="with_directory",
        ),
        # Test with case_insensitive
        pytest.param(
            {"pattern": "test", "search_directory": ".", "case_insensitive": True},
            "test.py:10:TEST line",
            "-i test",
            id="ignore_case",
        ),
    ]

    no_matches_test_cases = [
        # Test empty string response
        pytest.param(
            {
                "pattern": "nonexistent",
                "search_directory": ".",
                "case_insensitive": False,
            },
            "",
            "No matches found for pattern 'nonexistent' in '.'.",
            id="empty_string_response",
        ),
        # Test "No such file or directory" response
        pytest.param(
            {
                "pattern": "test",
                "search_directory": "nonexistent_dir",
                "case_insensitive": False,
            },
            "No such file or directory",
            "No matches found for pattern 'test' in 'nonexistent_dir'.",
            id="no_such_file_or_directory",
        ),
        # Test "exit status 1" response
        pytest.param(
            {
                "pattern": "test",
                "search_directory": ".",
                "case_insensitive": False,
            },
            "exit status 1",
            "No matches found for pattern 'test' in '.'.",
            id="exit_status_1",
        ),
    ]

    def _setup_grep_tool_with_mocks(self, mock_response: str) -> Grep:
        """Helper method to set up Grep tool with mocked outbox and inbox.

        Args:
            mock_response: The response string to return from the mocked inbox

        Returns:
            Configured Grep tool instance with mocked dependencies
        """
        mock_outbox = MagicMock()
        mock_outbox.put = AsyncMock()

        mock_inbox = MagicMock()
        mock_inbox.get = AsyncMock(
            return_value=contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(response=mock_response)
            )
        )

        metadata = {"outbox": mock_outbox, "inbox": mock_inbox}
        grep_tool = Grep()
        grep_tool.metadata = metadata

        return grep_tool

    @pytest.mark.asyncio
    @pytest.mark.parametrize("params,expected_output,expected_args", valid_test_cases)
    async def test_grep_arun(self, params, expected_output, expected_args):
        grep_tool = self._setup_grep_tool_with_mocks(expected_output)

        result = await grep_tool._arun(
            pattern=params["pattern"],
            search_directory=params["search_directory"],
            case_insensitive=params["case_insensitive"],
        )

        assert result == expected_output

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "params,mock_response,expected_output", no_matches_test_cases
    )
    async def test_grep_no_matches_handling(
        self, params, mock_response, expected_output
    ):
        """Test that various 'no matches' responses are properly formatted."""
        grep_tool = self._setup_grep_tool_with_mocks(mock_response)

        result = await grep_tool._arun(
            pattern=params["pattern"],
            search_directory=params["search_directory"],
            case_insensitive=params["case_insensitive"],
        )

        assert result == expected_output

    @pytest.mark.asyncio
    async def test_grep_security_check(self):
        grep_tool = Grep()
        result = await grep_tool._arun(
            pattern="test",
            search_directory="../parent",
        )

        assert result == "Searching above the current directory is not allowed"


def test_grep_format_display_message():
    tool = Grep(description="Grep description")

    # Basic test with directory
    input_data = GrepInput(pattern="TODO", search_directory="./src")
    message = tool.format_display_message(input_data)
    expected_message = "Search for 'TODO' in files in './src'"
    assert message == expected_message

    # Test with options
    input_data = GrepInput(
        pattern="TODO", search_directory="./src", case_insensitive=True
    )
    message = tool.format_display_message(input_data)
    expected_message = "Search for 'TODO' in files in './src'"
    assert message == expected_message

    # Test with all options
    input_data = GrepInput(
        pattern="TODO",
        search_directory="./src",
        case_insensitive=True,
    )
    message = tool.format_display_message(input_data)
    expected_message = "Search for 'TODO' in files in './src'"
    assert message == expected_message


def test_grep_format_display_message_no_directory():
    tool = Grep(description="Grep description")

    # Basic test with no directory
    input_data = GrepInput(pattern="TODO", search_directory=None)
    message = tool.format_display_message(input_data)
    expected_message = "Search for 'TODO' in files in 'directory'"
    assert message == expected_message

    # Test with options and no directory
    input_data = GrepInput(pattern="TODO", search_directory=None, case_insensitive=True)
    message = tool.format_display_message(input_data)
    expected_message = "Search for 'TODO' in files in 'directory'"
    assert message == expected_message
