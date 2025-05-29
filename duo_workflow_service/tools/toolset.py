import collections.abc
from typing import Type, Union

from langchain.tools import BaseTool
from langchain_core.messages import ToolCall
from pydantic import BaseModel, ValidationError

ToolType = Union[BaseTool, Type[BaseModel]]


class UnknownToolError(Exception):
    """Exception raised when trying to access an unknown tool."""


class MalformedToolCallError(Exception):
    tool_call: ToolCall

    def __init__(self, msg: str, tool_call: ToolCall):
        super().__init__(msg)
        self.tool_call = tool_call


class Toolset(collections.abc.Mapping):
    _all_tools: dict[str, ToolType]
    _pre_approved: set[str]
    _executable_tools: dict[str, BaseTool]

    def __init__(self, pre_approved: set[str], all_tools: dict[str, ToolType]):
        """Initialize a Toolset with pre-approved tools and all available tools.

        Args:
            pre_approved: A list of tool names that are pre-approved for use.
            all_tools: A lists of all enabled tools.
        """
        self._all_tools = all_tools
        self._pre_approved = pre_approved

        self._executable_tools: dict[str, BaseTool] = {
            tool.name: tool
            for tool in self._all_tools.values()
            if isinstance(tool, BaseTool)
        }

    @property
    def bindable(self) -> list[ToolType]:
        return list(self._all_tools.values())

    def __getitem__(self, tool_name: str) -> BaseTool:
        """Get an executable tool by name.

        Args:
            tool_name: The name of the tool to get.

        Returns:
            The requested tool.

        Raises:
            KeyError: If the tool is not found.
        """
        if tool_name not in self._executable_tools:
            raise KeyError(f"Tool '{tool_name}' does not exist in executable tools")

        return self._executable_tools[tool_name]

    def __iter__(self):
        """Return an iterator over executable tool names and classes."""
        return iter(self._executable_tools)

    def __len__(self) -> int:
        """Return the number of executable tools."""
        return len(self._executable_tools)

    def approved(self, tool_name: str) -> bool:
        """Check if a tool is pre-approved for use.

        Args:
            tool_name: The name of the tool to check.

        Returns:
            True if the tool is pre-approved, False otherwise.

        Raises:
            UnknownToolError: If the tool is not found in all_tools.
        """
        if tool_name not in self._all_tools:
            raise UnknownToolError(f"Tool '{tool_name}' does not exist")

        return tool_name in self._pre_approved

    def validate_tool_call(self, call: ToolCall) -> ToolCall:
        tool_name = call.get("name")

        if tool_name not in self._all_tools:
            raise MalformedToolCallError(
                f"Tool: '{call['name']}' not found. Please provide a valid tool name",
                tool_call=call,
            )

        try:
            tool = self._all_tools[tool_name]

            if isinstance(tool, BaseTool):
                tool_input_schema_cls = tool.get_input_schema()
            else:
                tool_input_schema_cls = tool

            tool_input_schema_cls.model_validate(call["args"])

            return call
        except ValidationError:
            raise MalformedToolCallError(
                (
                    f"Invalid arguments {call['args']} were passed to the tool: '{tool_name}'."
                    f"Please adhere to the tool schema {tool_input_schema_cls.model_json_schema()}.'"
                ),
                tool_call=call,
            )
