from typing import Any, Type

from pydantic import BaseModel, Field

from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.command import RunCommand
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool


class RunTestsInput(BaseModel):
    command: str = Field(
        description="The complete test command to execute (e.g., 'pytest -v', 'npm test', 'go test ./...')"
    )


class RunTests(DuoBaseTool):
    name: str = "run_tests"
    description: str = """Execute test commands for any language or framework.

    The agent should determine the appropriate test command based on:
    - Project files (package.json, go.mod, Cargo.toml, etc.)
    - Test frameworks detected (pytest, jest, rspec, etc.)
    - Existing test scripts or Makefiles

    Examples:
    - Python: run_tests(command="pytest")
    - JavaScript: run_tests(command="npm test")
    - Go: run_tests(command="go test ./...")
    - Ruby: run_tests(command="bundle exec rspec")
    - Custom: run_tests(command="make test")
    """
    args_schema: Type[BaseModel] = RunTestsInput
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL

    async def _execute(self, command: str, **kwargs: Any) -> str:
        parts = command.split(maxsplit=1)
        program = parts[0]
        args = parts[1] if len(parts) > 1 else ""

        run_command = RunCommand()
        run_command.metadata = self.metadata
        return await run_command._arun(program=program, args=args)

    def format_display_message(
        self, args: RunTestsInput, _tool_response: Any = None
    ) -> str:
        return f"Running tests: {args.command}"
