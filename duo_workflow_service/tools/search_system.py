import json
from typing import Any, ClassVar, Optional, Type

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from contract import contract_pb2
from duo_workflow_service.executor.action import _execute_action
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.filesystem import _format_no_matches_message


class GrepInput(BaseModel):
    search_directory: Optional[str] = Field(
        default=".",
        description="The relative path of directory in which to search. Scope this to a specific subdirectory "
        "(e.g. 'src/api', 'pkg/controller') whenever possible instead of '.' to reduce irrelevant results and token usage."
        " A directory that does not exist is reported as an error, but a directory that exists and does not"
        " hold the code returns no matches, which looks the same as the code not existing.",
    )
    keywords: str = Field(
        description="A comma-separated list of keywords for searching relevant snippets."
        " Do NOT provide regex expressions."
        " Every keyword should be either camel-case or snake-case."
        " Examples: 'authentication,login,user_session' or 'LoginComponent,LogoutComponent,Dashboard'"
    )
    case_insensitive: bool = Field(
        default=True,
        description="Whether to ignore letter case. Set false when case distinguishes the thing you"
        " are looking for, such as a constant from a local of the same name.",
    )


class Grep(DuoBaseTool):
    name: str = "grep"
    description: str = """Search code and text content within files across the codebase.

    This tool searches, recursively, through all files in the given directory, respecting .gitignore rules.

    **Primary use cases:**
    Use this search tool for finding:
    - Function definitions, class names, variable usage
    - Code patterns, imports, API calls
    - Error messages, comments, configuration values

    **How to use:**
    - Always scope `search_directory` to a specific subdirectory rather than the repository root '.' whenever known.
    - Provide 3-5 specific keywords per search to maximize precision and minimize irrelevant results.
    - Avoid long iterative trial-and-error grep chains; use `find_files` to locate relevant files or directories first.
    - Terms in one call are pooled and ranked together, so group terms that answer the same question.
    - Call this tool in parallel when you have several independent questions; do not spend a turn on
        each one in sequence.

    **Output structure:**
    - Matches are ranked for relevance across the whole result set and the top 100 snippets are returned,
        so a rare term outweighs a common one and results are not grouped by file
    - At most 5 matches per file survive to ranking, so a file that uses a term heavily is represented
        by a sample of its matches rather than all of them; read the file when the snippets do not settle it
    - Snippets include start and end line numbers for each match, where the first line of a file is line 1

    **Don't use this for:**
    - Finding files by name patterns (use find_files instead)
    - Listing directory contents (use list_dir instead)
    """
    args_schema: Type[BaseModel] = GrepInput

    async def _execute(
        self,
        keywords: str,
        search_directory: str = ".",
        case_insensitive: bool = True,
    ) -> str:
        """Execute the standard grep command with the specified parameters."""
        if search_directory and ".." in search_directory:
            return "Searching above the current directory is not allowed"

        result = await _execute_action(
            self.metadata,  # type: ignore
            contract_pb2.Action(
                grep=contract_pb2.Grep(
                    pattern=keywords,
                    search_directory=search_directory,
                    case_insensitive=case_insensitive,
                )
            ),
        )

        if (
            "No such file or directory" in result
            or "exit status 1" in result
            or result == ""
        ):
            return _format_no_matches_message(keywords, search_directory)

        return result

    def format_display_message(
        self, args: GrepInput, _tool_response: Any = None
    ) -> str:
        if not (search_dir := args.search_directory):
            search_dir = "directory"
        message = f"Search for '{args.keywords}' in files in '{search_dir}'"
        return message


class GrepLiteralInput(GrepInput):
    keywords: str = Field(
        description="A comma-separated list of literal strings to search for. Each one is matched"
        " literally, so punctuation is fine and no escaping is needed, but a comma cannot appear"
        " inside a term because it separates them. Results pool every term together, so use one call"
        " for terms you want ranked against each other and separate parallel calls for unrelated"
        " questions. Examples: 'authentication,login,user_session' or 'LoginComponent,Dashboard'"
    )


class GrepLiteral(Grep):
    """Grep for clients whose executor matches terms literally.

    Older executors pass each term to ripgrep as a regex, so telling the model punctuation is safe would make a term
    like `foo(` fail the whole call on them.
    """

    args_schema: Type[BaseModel] = GrepLiteralInput
    supersedes: ClassVar[Optional[Type[DuoBaseTool]]] = Grep
    required_capability: ClassVar[frozenset[str]] = frozenset({"grep_fixed_strings"})


class ExtractLinesFromTextInput(BaseModel):
    content: str = Field(description="The content string separated by '\\n' characters")
    start_line: int = Field(description="The starting line number (1-indexed)")
    end_line: Optional[int] = Field(
        default=None,
        description="The ending line number (1-indexed). If None, only returns the start_line",
    )


class ExtractLinesFromText(DuoBaseTool):
    name: str = "extract_lines_from_text"
    description: str = """Extract specific lines from a text content.

    The tool extracts lines from a large string content that is separated by '\\n' characters.
    It returns the exact block of lines starting from start_line and ending at end_line.

    Line numbers are 1-indexed (first line is line 1).

    For example:
    - Get a single line (line 5):
        extract_lines_from_text(
            content="line1\\nline2\\nline3\\nline4\\nline5\\nline6",
            start_line=5
        )

    - Get a range of lines (lines 3 to 5):
        extract_lines_from_text(
            content="line1\\nline2\\nline3\\nline4\\nline5\\nline6",
            start_line=3,
            end_line=5
        )
    """
    args_schema: Type[BaseModel] = ExtractLinesFromTextInput

    async def _execute(self, **kwargs: Any) -> str:
        content = kwargs.pop("content")
        start_line = kwargs.pop("start_line")
        end_line = kwargs.pop("end_line", None)

        lines = content.split("\n")
        total_lines = len(lines)

        if start_line < 1 or start_line > total_lines:
            raise ToolException(
                f"start_line {start_line} is out of range. Content has {total_lines} lines."
            )

        # If end_line is not provided, just return the start_line
        if end_line is None:
            result_lines = [lines[start_line - 1]]
        else:
            if end_line < 1 or end_line > total_lines:
                raise ToolException(
                    f"end_line {end_line} is out of range. Content has {total_lines} lines."
                )

            if end_line < start_line:
                raise ToolException(
                    f"end_line {end_line} cannot be less than start_line {start_line}."
                )

            result_lines = lines[start_line - 1 : end_line]

        result_lines = [line.rstrip() for line in result_lines]

        extracted_snippet = "\n".join(result_lines)

        return json.dumps(
            {
                "lines": extracted_snippet,
                "start_line": start_line,
                "end_line": end_line if end_line else start_line,
                "total_lines_extracted": len(result_lines),
            }
        )

    def format_display_message(
        self, args: ExtractLinesFromTextInput, _tool_response: Any = None
    ) -> str:
        if args.end_line:
            return f"Extract lines {args.start_line}-{args.end_line} from content"
        return f"Extract line {args.start_line} from content"
