import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from contract import contract_pb2
from duo_workflow_service.tools.search_system import (
    ExtractLinesFromText,
    ExtractLinesFromTextInput,
    Grep,
    GrepInput,
)
from tests.duo_workflow_service.tools.conftest import (
    create_mock_client_event_with_response,
)


class TestGrep:
    valid_test_cases = [
        # Basic recursive search
        pytest.param(
            {
                "keywords": "test",
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
                "keywords": "test",
                "search_directory": "src",
                "case_insensitive": False,
            },
            "src/test.py:10:test line",
            "test -- src",
            id="with_directory",
        ),
        # Test with case_insensitive
        pytest.param(
            {"keywords": "test", "search_directory": ".", "case_insensitive": True},
            "test.py:10:TEST line",
            "-i test",
            id="ignore_case",
        ),
    ]

    no_matches_test_cases = [
        # Test empty string response
        pytest.param(
            {
                "keywords": "nonexistent",
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
                "keywords": "test",
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
                "keywords": "test",
                "search_directory": ".",
                "case_insensitive": False,
            },
            "exit status 1",
            "No matches found for pattern 'test' in '.'.",
            id="exit_status_1",
        ),
    ]

    def _setup_grep_tool_with_mocks(self, mock_response: str) -> Grep:
        """Helper method to set up Grep tool with mocked outbox.

        Args:
            mock_response: The response string to return from the mocked outbox

        Returns:
            Configured Grep tool instance with mocked dependencies
        """
        mock_outbox = MagicMock()
        mock_outbox.put_action_and_wait_for_response = AsyncMock(
            return_value=create_mock_client_event_with_response(mock_response)
        )

        metadata = {"outbox": mock_outbox}
        grep_tool = Grep()
        grep_tool.metadata = metadata

        return grep_tool

    @pytest.mark.asyncio
    @pytest.mark.parametrize("params,expected_output,expected_args", valid_test_cases)
    async def test_grep_arun(self, params, expected_output, expected_args):
        grep_tool = self._setup_grep_tool_with_mocks(expected_output)

        result = await grep_tool._arun(
            keywords=params["keywords"],
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
            keywords=params["keywords"],
            search_directory=params["search_directory"],
            case_insensitive=params["case_insensitive"],
        )

        assert result == expected_output

    @pytest.mark.asyncio
    async def test_grep_security_check(self):
        grep_tool = Grep()
        result = await grep_tool._arun(
            keywords="test",
            search_directory="../parent",
        )

        assert result == "Searching above the current directory is not allowed"


def test_grep_format_display_message():
    tool = Grep(description="Grep description")

    # Basic test with directory
    input_data = GrepInput(keywords="TODO", search_directory="./src")
    message = tool.format_display_message(input_data)
    expected_message = "Search for 'TODO' in files in './src'"
    assert message == expected_message

    # Test with options
    input_data = GrepInput(
        keywords="TODO", search_directory="./src", case_insensitive=True
    )
    message = tool.format_display_message(input_data)
    expected_message = "Search for 'TODO' in files in './src'"
    assert message == expected_message

    # Test with all options
    input_data = GrepInput(
        keywords="TODO",
        search_directory="./src",
        case_insensitive=True,
    )
    message = tool.format_display_message(input_data)
    expected_message = "Search for 'TODO' in files in './src'"
    assert message == expected_message


def test_grep_format_display_message_no_directory():
    tool = Grep(description="Grep description")

    # Basic test with no directory
    input_data = GrepInput(keywords="TODO", search_directory=None)
    message = tool.format_display_message(input_data)
    expected_message = "Search for 'TODO' in files in 'directory'"
    assert message == expected_message

    # Test with options and no directory
    input_data = GrepInput(
        keywords="TODO", search_directory=None, case_insensitive=True
    )
    message = tool.format_display_message(input_data)
    expected_message = "Search for 'TODO' in files in 'directory'"
    assert message == expected_message


class TestExtractLinesFromText:
    valid_test_cases = [
        # Single line extraction
        pytest.param(
            {
                "content": "line1\nline2\nline3\nline4\nline5",
                "start_line": 3,
                "end_line": None,
            },
            {
                "lines": "line3",
                "start_line": 3,
                "end_line": 3,
                "total_lines_extracted": 1,
            },
            id="single_line_extraction",
        ),
        # Range extraction
        pytest.param(
            {
                "content": "line1\nline2\nline3\nline4\nline5",
                "start_line": 2,
                "end_line": 4,
            },
            {
                "lines": "line2\nline3\nline4",
                "start_line": 2,
                "end_line": 4,
                "total_lines_extracted": 3,
            },
            id="range_extraction",
        ),
        # Handle content with leading/trailing whitespace
        pytest.param(
            {
                "content": "  line1  \n  line2  \n  line3  ",
                "start_line": 1,
                "end_line": 2,
            },
            {
                "lines": "  line1\n  line2",
                "start_line": 1,
                "end_line": 2,
                "total_lines_extracted": 2,
            },
            id="whitespace_handling",
        ),
    ]

    error_test_cases = [
        # start_line out of range
        pytest.param(
            {
                "content": "line1\nline2\nline3",
                "start_line": 5,
                "end_line": None,
            },
            "start_line 5 is out of range. Content has 3 lines.",
            id="start_line_out_of_range",
        ),
        # end_line out of range
        pytest.param(
            {
                "content": "line1\nline2\nline3",
                "start_line": 1,
                "end_line": 5,
            },
            "end_line 5 is out of range. Content has 3 lines.",
            id="end_line_out_of_range",
        ),
        # end_line less than start_line
        pytest.param(
            {
                "content": "line1\nline2\nline3",
                "start_line": 3,
                "end_line": 1,
            },
            "end_line 1 cannot be less than start_line 3.",
            id="end_line_less_than_start_line",
        ),
        # start_line zero
        pytest.param(
            {
                "content": "line1\nline2\nline3",
                "start_line": 0,
                "end_line": None,
            },
            "start_line 0 is out of range. Content has 3 lines.",
            id="start_line_zero",
        ),
        pytest.param(
            {
                "content": "",
                "start_line": 0,
                "end_line": None,
            },
            "start_line 0 is out of range. Content has 1 lines.",
            id="empty_content",
        ),
    ]

    def _setup_extract_lines_from_text_tool(self) -> ExtractLinesFromText:
        """Helper method to set up ExtractLinesFromText tool.

        Returns:
            Configured ExtractLinesFromText tool instance
        """
        extract_lines_from_text_tool = ExtractLinesFromText()
        extract_lines_from_text_tool.metadata = {}
        return extract_lines_from_text_tool

    @pytest.mark.asyncio
    @pytest.mark.parametrize("params,expected_output", valid_test_cases)
    async def test_extract_lines_from_text_arun_valid(self, params, expected_output):
        """Test valid line extraction scenarios."""
        extract_lines_from_text_tool = self._setup_extract_lines_from_text_tool()

        result = await extract_lines_from_text_tool._arun(**params)

        result_dict = json.loads(result)

        assert result_dict == expected_output

    @pytest.mark.asyncio
    @pytest.mark.parametrize("params,expected_error", error_test_cases)
    async def test_extract_lines_from_text_error_handling(self, params, expected_error):
        """Test error handling for invalid inputs."""
        extract_lines_from_text_tool = self._setup_extract_lines_from_text_tool()

        result = await extract_lines_from_text_tool._arun(**params)

        result_dict = json.loads(result)
        assert "error" in result_dict
        assert result_dict["error"] == expected_error


def test_extract_lines_from_text_format_display_message():
    """Test the display message formatting."""
    tool = ExtractLinesFromText()

    # Test single line extraction message
    input_data = ExtractLinesFromTextInput(content="test content", start_line=5)
    message = tool.format_display_message(input_data)
    expected_message = "Extract line 5 from content"
    assert message == expected_message

    # Test range extraction message
    input_data = ExtractLinesFromTextInput(
        content="test content", start_line=3, end_line=7
    )
    message = tool.format_display_message(input_data)
    expected_message = "Extract lines 3-7 from content"
    assert message == expected_message
