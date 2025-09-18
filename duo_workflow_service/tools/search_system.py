import json
from typing import Any, Optional, Type

from pydantic import BaseModel, Field

from contract import contract_pb2
from duo_workflow_service.executor.action import _execute_action
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.filesystem import _format_no_matches_message


class GrepInput(BaseModel):
    search_directory: Optional[str] = Field(
        default=".",
        description="The relative path of directory in which to search. Defaults to current directory.",
    )
    pattern: str = Field(description="The PATTERN to search for")
    case_insensitive: bool = Field(
        default=False,
        description="Ignore case distinctions (equivalent to -i flag)",
    )


class Grep(DuoBaseTool):
    name: str = "grep"
    description: str = """Search code and text content within files across the entire codebase.

    This tool uses searches, recursively, through all files in the given directory, respecting .gitignore rules.

    **Primary use cases:**
    Fastest local search: Use this as your PRIMARY search tool for finding:
    - Function definitions, class names, variable usage
    - Code patterns, imports, API calls
    - Error messages, comments, configuration values

    **Examples:**
    - Search for "TODO" in all files: grep(pattern="TODO")
    - Case-insensitive search: grep(pattern="error", case_insensitive=True)
    - Search in specific directory: grep(pattern="bug", search_directory="src/")

    **Don't use this for:**
    - Finding files by name patterns (use find_files instead)
    - Listing directory contents (use list_dir instead)
    """
    args_schema: Type[BaseModel] = GrepInput  # type: ignore

    async def _arun(
        self,
        pattern: str,
        search_directory: str = ".",
        case_insensitive: bool = False,
    ) -> str:
        """Execute the standard grep command with the specified parameters."""
        if search_directory and ".." in search_directory:
            return "Searching above the current directory is not allowed"

        result = await _execute_action(
            self.metadata,  # type: ignore
            contract_pb2.Action(
                grep=contract_pb2.Grep(
                    pattern=pattern,
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
            return _format_no_matches_message(pattern, search_directory)

        return result

    def format_display_message(
        self, args: GrepInput, _tool_response: Any = None
    ) -> str:
        if not (search_dir := args.search_directory):
            search_dir = "directory"
        message = f"Search for '{args.pattern}' in files in '{search_dir}'"
        return message


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

    async def _arun(self, **kwargs: Any) -> str:
        content = kwargs.pop("content")
        start_line = kwargs.pop("start_line")
        end_line = kwargs.pop("end_line", None)

        try:
            lines = content.split("\n")
            total_lines = len(lines)

            if start_line < 1 or start_line > total_lines:
                return json.dumps(
                    {
                        "error": f"start_line {start_line} is out of range. Content has {total_lines} lines."
                    }
                )

            # If end_line is not provided, just return the start_line
            if end_line is None:
                result_lines = [lines[start_line - 1]]
            else:
                if end_line < 1 or end_line > total_lines:
                    return json.dumps(
                        {
                            "error": f"end_line {end_line} is out of range. Content has {total_lines} lines."
                        }
                    )

                if end_line < start_line:
                    return json.dumps(
                        {
                            "error": f"end_line {end_line} cannot be less than start_line {start_line}."
                        }
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

        except Exception as e:
            return json.dumps({"error": f"Failed to extract lines: {str(e)}"})

    def format_display_message(
        self, args: ExtractLinesFromTextInput, _tool_response: Any = None
    ) -> str:
        if args.end_line:
            return f"Extract lines {args.start_line}-{args.end_line} from content"
        return f"Extract line {args.start_line} from content"
