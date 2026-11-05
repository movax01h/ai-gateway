import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.tools.search_system import (
    ExtractLinesFromText,
    ExtractLinesFromTextInput,
    Grep,
    GrepInput,
    GrepLiteral,
    GrepLiteralInput,
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

    def _setup_grep_tool_with_mocks(
        self, mock_response: str, tool_class: type[Grep] = Grep
    ) -> Grep:
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
        grep_tool = tool_class()
        grep_tool.metadata = metadata

        return grep_tool

    @pytest.mark.asyncio
    @pytest.mark.parametrize("params,expected_output,expected_args", valid_test_cases)
    async def test_grep_arun(
        self,
        params,
        expected_output,
        expected_args,  # pylint: disable=unused-argument  # parametrize value
    ):
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

    def _setup_grep_tool_with_error(self, error: str) -> Grep:
        event = create_mock_client_event_with_response("")
        event.actionResponse.plainTextResponse.error = error
        mock_outbox = MagicMock()
        mock_outbox.put_action_and_wait_for_response = AsyncMock(return_value=event)
        grep_tool = Grep()
        grep_tool.metadata = {"outbox": mock_outbox}
        return grep_tool

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            "ripgrep failed: RangeError [ERR_CHILD_PROCESS_STDIO_MAXBUFFER]:"
            " stdout maxBuffer length exceeded",
            "ripgrep failed: stdout maxBuffer length exceeded",
        ],
    )
    async def test_grep_output_overflow_raises_guidance(self, error):
        grep_tool = self._setup_grep_tool_with_error(error)

        with pytest.raises(ToolException) as exc_info:
            await grep_tool._arun(keywords="end")

        assert str(exc_info.value) == (
            "Too many matches to return. Retry with a more distinctive search term,"
            " or set `search_directory` to a narrower directory."
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            "ripgrep failed: spawn rg ENOENT",
            "ripgrep failed: RangeError [ERR_CHILD_PROCESS_STDIO_MAXBUFFER]:"
            " stderr maxBuffer length exceeded",
        ],
    )
    async def test_grep_other_action_errors_still_raise(self, error):
        grep_tool = self._setup_grep_tool_with_error(error)

        with pytest.raises(ToolException, match="ripgrep failed"):
            await grep_tool._arun(keywords="end")

    @staticmethod
    def _project(exclusion_rules):
        return Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=exclusion_rules,
        )

    def _setup_grep_tool_with_project(self, mock_response, exclusion_rules):
        grep_tool = self._setup_grep_tool_with_mocks(mock_response)
        grep_tool.metadata["project"] = self._project(exclusion_rules)
        return grep_tool

    # Shaped like the node executor's formatter (gitlab-lsp workflow_executor grep.ts).
    snippet_output = (
        "Found 4 matches across 4 files\n"
        "src/app.py:3_4\n"
        "token = load()\n"
        "ratio = 1:16_9\n"
        "secrets/key.pem:1_1\n"
        "token: abc:12_30\n"
        "docs/internal/notes.md:2_2\n"
        "token here\n"
        "lib/util.py:7_7\n"
        "return token\n"
    )
    hidden_note = (
        "Some matches are hidden by the project's file exclusion settings;"
        " searching again will not return them.\n"
    )
    unfilterable = (
        "Grep results from this client cannot be checked against the project's file exclusion"
        " settings. Use find_files and read_file instead."
    )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool_class", [Grep, GrepLiteral])
    async def test_grep_drops_snippets_of_excluded_files(self, tool_class):
        grep_tool = self._setup_grep_tool_with_mocks(self.snippet_output, tool_class)
        grep_tool.metadata["project"] = self._project(["secrets/", "docs/internal/"])

        result = await grep_tool._arun(keywords="token")

        assert result == (
            "Found 4 matches across 4 files\n"
            "src/app.py:3_4\n"
            "token = load()\n"
            "ratio = 1:16_9\n"
            "lib/util.py:7_7\n"
            "return token\n" + self.hidden_note
        )

    @pytest.mark.asyncio
    async def test_grep_drops_default_exclusions_without_project_rules(self):
        output = (
            "Found 2 matches across 2 files\n"
            ".gitlab/duo/flows.yml:1_1\n"
            "token\n"
            ".gitlab/ci.yml:2_2\n"
            "token\n"
        )
        grep_tool = self._setup_grep_tool_with_project(output, None)

        result = await grep_tool._arun(keywords="token", search_directory=".gitlab")

        assert result == (
            "Found 2 matches across 2 files\n.gitlab/ci.yml:2_2\ntoken\n"
            + self.hidden_note
        )

    @pytest.mark.asyncio
    async def test_grep_ends_a_snippet_only_where_the_executor_trimmed(self):
        long_line = "x" * 5000 + "... (trimmed)"
        output = (
            "Found 3 matches across 3 files (showing first 2)\n"
            f"secrets/a.pem:1_3\n{long_line}\n"
            "src/b.py:4_5\n"
            "log(... (trimmed)\n"
            "ok\n"
            "(Results truncated: showing 2 of 3 matches (1 hidden).)\n"
        )
        grep_tool = self._setup_grep_tool_with_project(output, ["secrets/"])

        result = await grep_tool._arun(keywords="token")

        assert result == (
            "Found 3 matches across 3 files (showing first 2)\n"
            "src/b.py:4_5\n"
            "log(... (trimmed)\n"
            "ok\n"
            "(Results truncated: showing 2 of 3 matches (1 hidden).)\n"
            + self.hidden_note
        )

    @pytest.mark.asyncio
    async def test_grep_keeps_lines_ending_like_the_trim_note_as_content(self):
        output = (
            "Found 2 matches across 2 files\n"
            "secrets/a.pem:1_3\n"
            "log(... (trimmed)\n"
            "SECRET=abc\n"
            "more\n"
            "src/b.py:4_4\n"
            "ok\n"
        )
        grep_tool = self._setup_grep_tool_with_project(output, ["secrets/"])

        result = await grep_tool._arun(keywords="token")

        assert result == (
            "Found 2 matches across 2 files\nsrc/b.py:4_4\nok\n" + self.hidden_note
        )

    @pytest.mark.asyncio
    async def test_grep_handles_output_that_ends_inside_a_snippet(self):
        output = (
            "Found 2 matches across 2 files\nsrc/b.py:4_4\nok\nsecrets/a.pem:1_3\nkey"
        )
        grep_tool = self._setup_grep_tool_with_project(output, ["secrets/"])

        result = await grep_tool._arun(keywords="token")

        assert result == (
            "Found 2 matches across 2 files\nsrc/b.py:4_4\nok\n" + self.hidden_note
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "output",
        [
            pytest.param("src/a.py:token\nconfig/x.secret:token", id="path_preview"),
            pytest.param(
                "Snippet: config/x.secret?start_line=1&end_line=1\ntoken",
                id="go_executor",
            ),
        ],
    )
    async def test_grep_refuses_other_formats_when_the_project_has_rules(self, output):
        grep_tool = self._setup_grep_tool_with_project(output, ["*.secret"])

        result = await grep_tool._arun(keywords="token")

        assert result == self.unfilterable

    @pytest.mark.asyncio
    async def test_grep_refuses_other_formats_that_name_a_default_exclusion(self):
        grep_tool = self._setup_grep_tool_with_project(
            "src/a.py:token\n.ssh/id_rsa:token", None
        )

        result = await grep_tool._arun(keywords="token")

        assert result == self.unfilterable

    @pytest.mark.asyncio
    async def test_grep_passes_no_matches_through_when_the_project_has_rules(self):
        grep_tool = self._setup_grep_tool_with_project(
            "No matches found.\n", ["secrets/"]
        )

        result = await grep_tool._arun(keywords="token")

        assert result == "No matches found.\n"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exclusion_rules", [None, []])
    async def test_grep_without_exclusion_rules_keeps_non_sensitive_output(
        self, exclusion_rules
    ):
        output = "Found 1 match across 1 file\nsrc/app.py:3_3\ntoken = load()\n"
        grep_tool = self._setup_grep_tool_with_project(output, exclusion_rules)

        result = await grep_tool._arun(keywords="token")

        assert result == output

    @pytest.mark.asyncio
    async def test_grep_refuses_an_absolute_search_directory(self):
        grep_tool = self._setup_grep_tool_with_project("unused", None)

        with pytest.raises(ToolException):
            await grep_tool._arun(keywords="token", search_directory="/etc")

        grep_tool.metadata[
            "outbox"
        ].put_action_and_wait_for_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_grep_searches_the_root_when_no_directory_is_given(self):
        grep_tool = self._setup_grep_tool_with_project("No matches found.\n", None)

        result = await grep_tool._arun(keywords="token", search_directory=None)

        assert result == "No matches found.\n"

    @pytest.mark.asyncio
    async def test_grep_refuses_an_excluded_search_directory(self):
        grep_tool = self._setup_grep_tool_with_project("unused", ["secrets/"])

        result = await grep_tool._arun(keywords="token", search_directory="secrets")

        assert result == (
            "Files excluded due to policy, continue without files:\nsecrets"
        )
        grep_tool.metadata[
            "outbox"
        ].put_action_and_wait_for_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_grep_refuses_a_sensitive_search_directory(self):
        grep_tool = self._setup_grep_tool_with_project("unused", None)

        with pytest.raises(ToolException, match="Duo Context Exclusion"):
            await grep_tool._arun(keywords="token", search_directory=".ssh")

    @pytest.mark.asyncio
    async def test_grep_security_check(self):
        grep_tool = Grep()
        result = await grep_tool._arun(
            keywords="test",
            search_directory="../parent",
        )

        assert result == "Searching above the current directory is not allowed"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool_class", [Grep, GrepLiteral])
    @pytest.mark.parametrize(
        "keywords,expected_pattern",
        [
            pytest.param("foo,", "foo", id="trailing_comma"),
            pytest.param("a,,b", "a,b", id="double_comma"),
            pytest.param("a, b", "a,b", id="space_after_comma"),
            pytest.param(" foo bar , baz ", "foo bar,baz", id="internal_space_kept"),
            pytest.param("foo  bar,baz", "foo  bar,baz", id="inner_double_space_kept"),
        ],
    )
    async def test_grep_normalizes_search_terms(
        self, tool_class, keywords, expected_pattern
    ):
        grep_tool = self._setup_grep_tool_with_mocks("test.py:1:foo", tool_class)

        await grep_tool._arun(keywords=keywords)

        outbox = grep_tool.metadata["outbox"]
        action = outbox.put_action_and_wait_for_response.call_args[0][0]
        assert action.grep.pattern == expected_pattern

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool_class", [Grep, GrepLiteral])
    @pytest.mark.parametrize("keywords", ["", ",", " , ,"])
    async def test_grep_rejects_empty_search_terms(self, tool_class, keywords):
        grep_tool = self._setup_grep_tool_with_mocks("test.py:1:foo", tool_class)

        result = await grep_tool._arun(keywords=keywords)

        assert result == "No search terms given"
        outbox = grep_tool.metadata["outbox"]
        outbox.put_action_and_wait_for_response.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool_class", [Grep, GrepLiteral])
    async def test_grep_no_matches_message_shows_cleaned_terms(self, tool_class):
        grep_tool = self._setup_grep_tool_with_mocks("", tool_class)

        result = await grep_tool._arun(keywords="a, b")

        assert result == "No matches found for pattern 'a,b' in '.'."


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


class TestGrepLiteral:
    def test_supersedes_grep(self):
        assert GrepLiteral.supersedes is Grep

    def test_required_capability(self):
        assert GrepLiteral.required_capability == frozenset({"grep_fixed_strings"})

    def test_shares_tool_name_with_grep(self):
        assert GrepLiteral.model_fields["name"].default == "grep"

    def test_only_literal_variant_says_punctuation_is_fine(self):
        literal = GrepLiteralInput.model_fields["keywords"].description
        base = GrepInput.model_fields["keywords"].description

        assert "punctuation is fine" in literal
        assert "Do NOT provide regex expressions" in base
        assert "punctuation" not in base


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

        with pytest.raises(ToolException) as exc_info:
            await extract_lines_from_text_tool._arun(**params)

        assert str(exc_info.value) == expected_error


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
