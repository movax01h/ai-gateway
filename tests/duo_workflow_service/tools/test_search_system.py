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

    @pytest.mark.asyncio
    @pytest.mark.parametrize("params,expected_output,expected_args", valid_test_cases)
    async def test_grep_arun(self, params, expected_output, expected_args):
        # Set up the mock outbox and inbox
        mock_outbox = MagicMock()
        mock_outbox.put = AsyncMock()

        # Create a mock inbox that returns the expected_output for each test case
        mock_inbox = MagicMock()
        mock_inbox.get = AsyncMock(
            return_value=contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(response=expected_output)
            )
        )

        metadata = {"outbox": mock_outbox, "inbox": mock_inbox}
        grep_tool = Grep()
        grep_tool.metadata = metadata

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
