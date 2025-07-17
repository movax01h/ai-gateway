import json
from functools import partialmethod
from typing import Any

from langchain.tools import BaseTool

from contract import contract_pb2
from duo_workflow_service.executor.action import _execute_action


class McpTool(BaseTool):
    """A tool that executes MCP (Model Control Protocol) operations asynchronously."""

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("This tool can only be run asynchronously")

    async def _arun(self, **arguments):
        return await _execute_action(
            self.metadata,
            contract_pb2.Action(
                runMCPTool=contract_pb2.RunMCPTool(
                    name=self.name, args=json.dumps(arguments)
                )
            ),
        )

    def format_display_message(self, arguments) -> str:
        return f"Run MCP tool {self.name}: {arguments}"


def convert_mcp_tools_to_langchain_tool_classes(
    mcp_tools: list[contract_pb2.McpTool],
) -> list[type[BaseTool]]:
    """Converts a list of MCP tools into LangChain tool classes.

    This function dynamically creates tool classes that inherit from McpTool
    for each MCP tool in the provided list. Each generated class is configured
    with the name, description, and input schema of the corresponding MCP tool.

    Args:
        mcp_tools: A list of MCP tools defined using the contract_pb2.McpTool protocol buffer.

    Returns:
        A list of dynamically created tool classes that inherit from BaseTool.
    """

    result: list[type[BaseTool]] = []

    for tool in mcp_tools:
        try:
            args_schema = json.loads(tool.inputSchema)
        except json.JSONDecodeError:
            args_schema = {}

        tool_cls = type(
            f"McpTool_{tool.name}",
            (McpTool,),
            {
                "__init__": partialmethod(
                    McpTool.__init__,
                    name=tool.name,
                    description=tool.description,
                    args_schema=args_schema,
                )
            },
        )
        setattr(tool_cls, "name", tool.name)

        result.append(tool_cls)

    return result
