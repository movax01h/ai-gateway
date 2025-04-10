from typing import Optional, Type

from pydantic import BaseModel, Field

from contract import contract_pb2
from duo_workflow_service.executor.action import _execute_action
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

_DISALLOWED_COMMANDS = ["git"]
_DISALLOWED_OPERATORS = ["&&", "||", "|"]


class RunCommandInput(BaseModel):
    program: str = Field(description="The name of bash program to execute eg: 'echo'")
    arguments: list[str] = Field(
        description="The argv to pass into the bash program eg: ['/home']"
    )
    flags: list[str] = Field(
        description="The flags to pass into the bash program eg: ['-l']"
    )


class RunCommand(DuoBaseTool):
    name: str = "run_command"
    description: str = (
        "Run a bash command in the current working directory. "
        f"Following bash commands are not supported: {', '.join(_DISALLOWED_COMMANDS)} "
        "and will result in error."
        "Pay extra attention to correctly escape special characters like '`'"
    )
    args_schema: Type[BaseModel] = RunCommandInput  # type: ignore

    async def _arun(
        self,
        program: str,
        arguments: Optional[list[str]] = None,
        flags: Optional[list[str]] = None,
    ) -> str:
        # handle mutable default arguments https://docs.python-guide.org/writing/gotchas/#mutable-default-arguments
        if arguments is None:
            arguments = []
        if flags is None:
            flags = []

        for disallowed_operator in _DISALLOWED_OPERATORS:
            if disallowed_operator in program:
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
                    arguments=arguments,
                    flags=flags,
                )
            ),
        )

    def format_display_message(self, args: RunCommandInput) -> str:
        args_str = " ".join(args.arguments) if args.arguments else ""
        flags_str = " ".join(args.flags) if args.flags else ""

        command = f"{args.program} {flags_str} {args_str}".strip()
        return f"Run command: {command}"
