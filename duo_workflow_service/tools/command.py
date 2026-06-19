import shlex
import textwrap
from typing import Any, ClassVar, Optional, Type

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from contract import contract_pb2
from duo_workflow_service.client_capabilities import is_client_capable
from duo_workflow_service.executor.action import _execute_action
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

_DEFAULT_COMMAND_TIMEOUT_SECONDS = 120


class RunCommandInput(BaseModel):
    program: str = Field(description="The name of bash program to execute eg: 'cp'")
    args: Optional[str] = Field(
        description="All arguments and flags for the bash program as a single string. "
        "eg: '-v -p source.txt destination.txt'",
        default=None,
    )
    timeout: Optional[int] = Field(
        description="Timeout in seconds. "
        "Use a higher value for long-running commands like 'npm install' or 'docker build'.",
        default=None,
    )


class RunCommand(DuoBaseTool):
    name: str = "run_command"
    description: str = (
        "Run a bash command in the current working directory. "
        "This tool should be reserved for cases where specialized tools cannot accomplish the task. "
        "Pay extra attention to correctly escape special characters like '`'"
    )
    args_schema: Type[BaseModel] = RunCommandInput

    async def _execute(
        self,
        program: str,
        args: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> str:
        args = args or ""

        # Command validation is handled by Rails via GraphQL approval query.
        # RunCommand sends structured program + arguments (not a shell string),
        # so shell operators are not interpreted.

        try:
            arguments = shlex.split(args)
        except ValueError as e:
            raise ToolException(
                f"Invalid argument syntax: {e}. Check for unclosed quotes."
            )

        run_command_kwargs: dict = {
            "program": program,
            "arguments": arguments,
            "flags": [],
        }

        if timeout is None:
            if is_client_capable("command_timeout"):
                timeout = _DEFAULT_COMMAND_TIMEOUT_SECONDS

        if timeout is not None:
            if not is_client_capable("command_timeout"):
                raise ToolException(
                    "timeout parameter is not supported by this client version."
                )
            run_command_kwargs["timeout"] = timeout

        return await _execute_action(
            self.metadata,  # type: ignore
            contract_pb2.Action(
                runCommand=contract_pb2.RunCommandAction(**run_command_kwargs)
            ),
        )

    def format_display_message(
        self,
        args: RunCommandInput,
        _tool_response: Any = None,
        max_len: int = 100,
    ) -> str:
        command = f"{args.program} {args.args}".strip()
        message = f"Run command: {command}"
        if _tool_response is not None:
            return f"{message} {textwrap.shorten(_tool_response, max_len)}"
        return message


class ShellCommandInput(BaseModel):
    command: str = Field(
        description="The shell script to execute in the user's default shell"
    )
    timeout: Optional[int] = Field(
        description="Timeout in seconds. "
        "Use a higher value for long-running commands like 'npm install' or 'docker build'.",
        default=None,
    )


class ShellCommand(DuoBaseTool):
    name: str = "run_command"
    description: str = (
        "Execute a shell command in the current project directory. "
        "Do not prefix command with 'cd' as the working directory is already set."
    )
    args_schema: Type[BaseModel] = ShellCommandInput
    supersedes: ClassVar[Optional[Type[DuoBaseTool]]] = RunCommand
    required_capability: ClassVar[frozenset[str]] = frozenset({"shell_command"})

    async def _execute(
        self,
        command: str,
        timeout: Optional[int] = None,
    ) -> str:
        run_shell_kwargs: dict = {"command": command}

        if timeout is None:
            if is_client_capable("command_timeout"):
                timeout = _DEFAULT_COMMAND_TIMEOUT_SECONDS

        if timeout is not None:
            if not is_client_capable("command_timeout"):
                raise ToolException(
                    "timeout parameter is not supported by this client version."
                )
            run_shell_kwargs["timeout"] = timeout

        return await _execute_action(
            self.metadata,  # type: ignore
            contract_pb2.Action(
                runShellCommand=contract_pb2.RunShellCommand(**run_shell_kwargs)
            ),
        )

    def format_display_message(
        self,
        args: ShellCommandInput,
        _tool_response: Any = None,
        max_len: int = 100,
    ) -> str:
        message = f"Run shell command: {args.command}"
        if _tool_response is not None:
            return f"{message} {textwrap.shorten(_tool_response, max_len)}"
        return message
