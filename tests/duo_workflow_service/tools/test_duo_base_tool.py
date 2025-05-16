from typing import Type
from unittest.mock import MagicMock

import pytest
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import (
    DuoBaseTool,
    format_tool_display_message,
)


class DummyTool(DuoBaseTool):
    name: str = "dummy_tool"
    description: str = "A dummy tool for testing"
    args_schema: Type[BaseModel] = BaseModel  # type: ignore

    async def _arun(self, *args, **kwargs):
        return "dummy result"


class DummyToolWithArgs(DuoBaseTool):
    name: str = "dummy_tool_with_args"
    description: str = "A dummy tool with args for testing"

    class ArgsSchema(BaseModel):
        param1: str = Field(description="First parameter")
        param2: int = Field(description="Second parameter")
        optional_param: str = Field(default="default", description="Optional parameter")

    args_schema: Type[BaseModel] = ArgsSchema  # type: ignore

    async def _arun(self, param1, param2, optional_param="default"):
        return f"{param1} {param2} {optional_param}"


def test_gitlab_client():
    tool = DummyTool(metadata={})
    with pytest.raises(RuntimeError):
        tool.gitlab_client

    client = MagicMock()
    tool = DummyTool(metadata={"gitlab_client": client})
    assert tool.gitlab_client == client


def test_format_display_message_with_dict_args():
    tool = DummyTool(metadata={})
    args = {"key1": "value1", "key2": "value2"}

    result = tool.format_display_message(args)

    assert result == "Using dummy_tool: key1=value1, key2=value2"


def test_format_display_message_with_pydantic_args():
    tool = DummyToolWithArgs(metadata={})

    args = DummyToolWithArgs.ArgsSchema(param1="test", param2=123)

    result = tool.format_display_message(args)

    assert (
        result
        == "Using dummy_tool_with_args: param1=test, param2=123, optional_param=default"
    )


def test_format_display_message_inheritance():

    class CustomTool(DummyTool):
        name: str = "custom_tool"

        def format_display_message(self, args):
            return f"Overridden in child: {args}"

    tool = CustomTool(metadata={})
    args = {"test": "value"}

    result = tool.format_display_message(args)

    assert result == "Overridden in child: {'test': 'value'}"


def test_gitlab_host_property_when_set():
    metadata = {"gitlab_host": "gitlab.example.com"}
    tool = DummyTool(metadata=metadata)

    assert tool.gitlab_host == "gitlab.example.com"


def test_gitlab_host_property_when_not_set():
    tool = DummyTool(metadata={})

    with pytest.raises(RuntimeError, match="gitlab_host is not set"):
        _ = tool.gitlab_host


def test_format_tool_display_message_non_duo_base_tool_child():
    mock_tool = MagicMock(spec=BaseTool)
    args = {"test": "value"}

    assert format_tool_display_message(mock_tool, args) == None


def test_format_tool_display_message_for_tool_without_args_schema():
    mock_tool = MagicMock(spec=DuoBaseTool)
    mock_tool.format_display_message.return_value = "Tool msg"
    args = {"test": "value"}

    assert format_tool_display_message(mock_tool, args) == "Tool msg"
    mock_tool.format_display_message.assert_called_once_with(args)


class DummyArgsModel(BaseModel):
    """Minimal Pydantic model that mirrors the 'args' dict in the tests."""

    test: str


class ErrorArgsModel(BaseModel):
    """Model that raises on instantiation so we can hit the except-branch."""

    test: str

    def __init__(self, **data):
        raise Exception("Something went wrong")


def test_format_tool_display_message_for_tool_with_pydantic_args_schema():
    mock_tool = MagicMock(spec=DuoBaseTool)
    mock_tool.args_schema = DummyArgsModel
    mock_tool.format_display_message.return_value = "Tool msg"
    args = {"test": "value"}

    result = format_tool_display_message(mock_tool, args)

    assert result == "Tool msg"
    mock_tool.format_display_message.assert_called_once()
    passed_instance = mock_tool.format_display_message.call_args.args[0]
    assert isinstance(passed_instance, DummyArgsModel)
    assert passed_instance.test == "value"


def test_format_tool_display_message_for_tool_with_args_schema_when_error():
    mock_tool = MagicMock(spec=DuoBaseTool)
    mock_tool.args_schema = ErrorArgsModel
    mock_tool.format_display_message.return_value = "Tool msg"
    args = {"test": "value"}

    result = format_tool_display_message(mock_tool, args)

    assert result == "Tool msg"
    mock_tool.format_display_message.assert_called_once_with(args)
