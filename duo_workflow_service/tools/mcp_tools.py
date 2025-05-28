import json
from typing import Any

from langchain.tools import BaseTool

from contract import contract_pb2
from duo_workflow_service.executor.action import _execute_action


class McpTool(BaseTool):
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


def convert_mcp_tools_to_langchain_tools(
    metadata: dict[str, Any], mcp_tools: list[contract_pb2.McpTool]
) -> list[BaseTool]:
    result: list[BaseTool] = []

    for tool in mcp_tools:
        try:
            args_schema = json.loads(tool.inputSchema)
        except json.JSONDecodeError:
            args_schema = {}

        result.append(
            McpTool(
                name=tool.name,
                description=tool.description,
                metadata=metadata,
                args_schema=args_schema,
            )
        )

    return result
