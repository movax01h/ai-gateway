from typing import Any, Optional, Type

from pydantic import BaseModel, Field

from contract import contract_pb2
from duo_workflow_service.executor.action import _execute_action
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

_DISALLOWED_COMMANDS = ["git"]
_DISALLOWED_OPERATORS = ["&&", "||", "|"]


class RunCommandInput(BaseModel):
    program: str = Field(description="The name of bash program to execute eg: 'cp'")
    args: Optional[str] = Field(
        description="All arguments and flags for the bash program as a single string. "
        "eg: '-v -p source.txt destination.txt'",
        default=None,
    )


class RunCommand(DuoBaseTool):
    name: str = "run_command"
    description: str = (
        "Run a bash command in the current working directory. "
        "This tool should be reserved for cases where specialized tools cannot accomplish the task. "
        f"Following bash commands are not supported: {', '.join(_DISALLOWED_COMMANDS)} "
        "and will result in error. "
        "Pay extra attention to correctly escape special characters like '`'"
    )
    args_schema: Type[BaseModel] = RunCommandInput  # type: ignore

    async def _arun(
        self,
        program: str,
        args: Optional[str] = None,
    ) -> str:
        args = args or ""

        for disallowed_operator in _DISALLOWED_OPERATORS:
            if disallowed_operator in program:
                # pylint: disable=line-too-long
                return f"""'{disallowed_operator}' operators are not supported with {self.name} tool.
Instead of '{disallowed_operator}' please use {self.name} multiple times consecutively to emulate '{disallowed_operator}' behaviour
"""
        for disallowed_command in _DISALLOWED_COMMANDS:
            if program.startswith(disallowed_command):
                return f"{disallowed_command} commands are not supported with {self.name} tool."

        return await _execute_action(
            self.metadata,  # type: ignore
            contract_pb2.Action(
                runCommand=contract_pb2.RunCommandAction(
                    program=program,
                    arguments=args.split(),
                    flags=[],
                )
            ),
        )

    def format_display_message(
        self, args: RunCommandInput, _tool_response: Any = None
    ) -> str:
        command = f"{args.program} {args.args}".strip()
        return f"Run command: {command}"
