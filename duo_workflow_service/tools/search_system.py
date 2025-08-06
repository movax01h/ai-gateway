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
