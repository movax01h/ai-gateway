import collections.abc
from typing import Any, Optional, Type, Union

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
    _tool_options: dict[str, dict[str, Any]]

    def __init__(
        self,
        pre_approved: set[str],
        all_tools: dict[str, ToolType],
        tool_options: Optional[dict[str, dict[str, Any]]] = None,
    ):
        """Initialize a Toolset with pre-approved tools and all available tools.

        Args:
            pre_approved: A list of tool names that are pre-approved for use.
            all_tools: A lists of all enabled tools.
            tool_options: Optional dict mapping tool names to their option overrides.
        """
        self._all_tools = all_tools
        self._pre_approved = pre_approved
        self._tool_options = tool_options or {}

        # Validate and apply tool options to each tool instance.
        # Tools with options should already be cloned by ToolsRegistry.toolset()
        # so this mutation is safe and won't affect other Toolsets.
        for tool_name, options in self._tool_options.items():
            tool = self._all_tools.get(tool_name)
            if tool is None:
                continue
            if isinstance(tool, BaseTool):
                self._validate_tool_options(tool_name, tool, options)

        for tool in self._all_tools.values():
            if isinstance(tool, BaseTool):
                tool._tool_options = self._tool_options  # type: ignore[attr-defined]

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

    @staticmethod
    def _validate_tool_options(
        tool_name: str, tool: BaseTool, options: dict[str, Any]
    ) -> None:
        """Validate that each option key is a valid parameter on the tool's input schema and that option values do not
        contain deeply nested structures.

        Args:
            tool_name: The name of the tool.
            tool: The BaseTool instance.
            options: The option overrides to validate.

        Raises:
            ValueError: If an option key is not a valid parameter for the tool,
                or if an option value contains nested structures.
        """
        schema_cls = tool.get_input_schema()
        valid_fields = set(schema_cls.model_fields.keys())
        for key in options:
            if key not in valid_fields:
                raise ValueError(
                    f"Invalid tool option '{key}' for tool '{tool_name}'. "
                    f"Valid parameters are: {sorted(valid_fields)}"
                )
        Toolset._validate_tool_option_values(tool_name, options)

    @staticmethod
    def _validate_tool_option_values(tool_name: str, options: dict[str, Any]) -> None:
        """Validate that tool option values are not arbitrarily nested.

        Values can be primitives or flat lists/dicts (1 level deep), but not
        nested structures like {"a": {"b": {"c": ...}}}.

        Args:
            tool_name: The name of the tool (for error messages).
            options: The option overrides to validate.

        Raises:
            ValueError: If an option value contains nested structures.
        """
        for key, value in options.items():
            if isinstance(value, dict):
                if any(isinstance(v, (dict, list)) for v in value.values()):
                    raise ValueError(
                        f"Tool option '{key}' for tool '{tool_name}' contains "
                        "nested structures. Only flat key-value mappings are supported."
                    )
            elif isinstance(value, list):
                if any(isinstance(v, (dict, list)) for v in value):
                    raise ValueError(
                        f"Tool option '{key}' for tool '{tool_name}' contains "
                        "nested structures. Only flat lists are supported."
                    )

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
