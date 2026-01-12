import json
from typing import Any, Type
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.duo_base_tool import (
    DuoBaseTool,
    format_tool_display_message,
)


class DummyTool(DuoBaseTool):
    name: str = "dummy_tool"
    description: str = "A dummy tool for testing"
    args_schema: Type[BaseModel] = BaseModel

    async def _execute(self, *args, **kwargs):
        return "dummy result"


class DummyToolWithArgs(DuoBaseTool):
    name: str = "dummy_tool_with_args"
    description: str = "A dummy tool with args for testing"

    class ArgsSchema(BaseModel):
        param1: str = Field(description="First parameter")
        param2: int = Field(description="Second parameter")
        optional_param: str = Field(default="default", description="Optional parameter")

    args_schema: Type[BaseModel] = ArgsSchema

    async def _execute(self, param1, param2, optional_param="default"):
        return f"{param1} {param2} {optional_param}"


class DummyToolWithResponseHandling(DuoBaseTool):
    name: str = "dummy_tool_with_response"
    description: str = "A dummy tool that uses tool_response in display message"

    class ArgsSchema(BaseModel):
        action: str = Field(description="Action to perform")

    args_schema: Type[BaseModel] = ArgsSchema

    def format_display_message(self, args: ArgsSchema, tool_response: str = "") -> str:
        base_msg = f"Performing {args.action}"
        if tool_response:
            return f"{base_msg} - Result: {tool_response[:50]}..."
        return base_msg

    async def _execute(self, action):
        return f"Completed {action}"


@pytest.fixture(name="gitlab_client_mock")
def gitlab_client_mock_fixture():
    mock = Mock()
    mock.aget = AsyncMock()
    return mock


@pytest.fixture(name="tool_with_client")
def tool_with_client_fixture(gitlab_client_mock):
    return DummyTool(
        metadata={
            "gitlab_client": gitlab_client_mock,
        }
    )


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

        def format_display_message(self, args, _tool_response: Any = None):
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
    mock_tool.format_display_message.assert_called_once_with(args, None)


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


def test_format_tool_display_message_for_tool_with_pydantic_args_schema_and_tool_response():
    mock_tool = MagicMock(spec=DuoBaseTool)
    mock_tool.args_schema = DummyArgsModel
    mock_tool.format_display_message.return_value = "Tool msg with response"
    args = {"test": "value"}
    tool_response = "Success: Data retrieved"

    result = format_tool_display_message(mock_tool, args, tool_response)

    assert result == "Tool msg with response"
    mock_tool.format_display_message.assert_called_once()
    call_args = mock_tool.format_display_message.call_args
    passed_instance = call_args.args[0]
    passed_tool_response = call_args.args[1]
    assert isinstance(passed_instance, DummyArgsModel)
    assert passed_instance.test == "value"
    assert passed_tool_response == tool_response


def test_format_tool_display_message_for_tool_with_args_schema_when_error():
    mock_tool = MagicMock(spec=DuoBaseTool)
    mock_tool.args_schema = ErrorArgsModel
    args = {"test": "value"}

    with patch.object(
        DuoBaseTool,
        "format_display_message",
        return_value="Using MagicMock: test=value",
    ) as mock_parent_method:

        result = format_tool_display_message(mock_tool, args)
        assert result == "Using MagicMock: test=value"

        mock_parent_method.assert_called_once_with(mock_tool, args, None)

        mock_tool.format_display_message.assert_not_called()


def test_format_tool_display_message_with_tool_that_uses_response():
    """Test that tools can use tool_response parameter in their display messages."""
    tool = DummyToolWithResponseHandling(metadata={})
    args = {"action": "data_processing"}

    # Test without tool_response
    result = format_tool_display_message(tool, args)
    assert result == "Performing data_processing"

    # Test with tool_response
    tool_response = "Successfully processed 100 records in 2.5 seconds"
    result = format_tool_display_message(tool, args, tool_response)
    assert (
        result
        == "Performing data_processing - Result: Successfully processed 100 records in 2.5 seconds..."
    )


@pytest.mark.parametrize(
    "response,expected_result,should_raise",
    [
        ("string_response", "string_response", False),
        ({"key": "value"}, {"key": "value"}, False),
        (42, 42, False),
        (None, None, False),
        (GitLabHttpResponse(200, {"data": "success"}), {"data": "success"}, False),
        (GitLabHttpResponse(201, "created"), "created", False),
        (GitLabHttpResponse(399, [1, 2, 3]), [1, 2, 3], False),
        (GitLabHttpResponse(400, {"error": "bad request"}), None, True),
        (GitLabHttpResponse(404, "not found"), None, True),
        (GitLabHttpResponse(500, {"error": "internal server error"}), None, True),
    ],
)
def test_process_http_response(response, expected_result, should_raise):
    tool = DummyTool()

    if should_raise:
        with pytest.raises(
            ToolException, match=r"Request failed \(test_identifier\): HTTP \d+"
        ):
            tool._process_http_response("test_identifier", response)
    else:
        result = tool._process_http_response("test_identifier", response)
        assert result == expected_result


def test_process_http_response_error_message_truncation():
    tool = DummyTool()

    # Create a long error message (over 300 characters)
    long_error_body = "A" * 400
    response = GitLabHttpResponse(500, long_error_body)

    with pytest.raises(ToolException) as exc_info:
        tool._process_http_response("test_identifier", response)

    error_message = str(exc_info.value)
    # Verify the message contains the expected prefix and truncated body
    expected_prefix = "Request failed (test_identifier): HTTP 500: "
    truncated_body = "A" * 300  # Should be truncated to exactly 300 chars
    expected_message = expected_prefix + truncated_body

    assert error_message == expected_message
    assert len(error_message) == len(expected_prefix) + 300


@pytest.mark.parametrize(
    "project_id,resource_type,resource_iid,note_id,discussions,expected_result",
    [
        (
            "456",
            "merge_requests",
            99,
            2001,
            [
                {
                    "id": "mr-discussion-123",
                    "notes": [
                        {"id": 2001, "body": "MR comment"},
                    ],
                }
            ],
            {"discussionId": "mr-discussion-123"},
        ),
        # Success: multiple discussions
        (
            "123",
            "issues",
            42,
            202,
            [
                {
                    "id": "discussion1",
                    "notes": [
                        {"id": 101, "body": "Comment 1"},
                        {"id": 102, "body": "Comment 2"},
                    ],
                },
                {
                    "id": "discussion2",
                    "notes": [
                        {"id": 201, "body": "Comment 3"},
                        {"id": 202, "body": "Comment 4"},
                    ],
                },
            ],
            {"discussionId": "discussion2"},
        ),
    ],
)
@pytest.mark.asyncio
async def test_get_discussion_id_from_note_rest_success(
    tool_with_client,
    gitlab_client_mock,
    project_id,
    resource_type,
    resource_iid,
    note_id,
    discussions,
    expected_result,
):
    """Test successfully finding discussion ID."""
    mock_response = Mock()
    mock_response.is_success.return_value = True
    mock_response.body = json.dumps(discussions)

    gitlab_client_mock.aget.return_value = mock_response

    result = await tool_with_client._get_discussion_id_from_note_rest(
        project_id=project_id,
        resource_type=resource_type,
        resource_iid=resource_iid,
        note_id=note_id,
    )

    assert result == expected_result
    gitlab_client_mock.aget.assert_called_once_with(
        path=f"/api/v4/projects/{project_id}/{resource_type}/{resource_iid}/discussions",
        parse_json=False,
    )


@pytest.mark.asyncio
async def test_get_discussion_id_from_note_rest_not_found(
    tool_with_client, gitlab_client_mock
):
    """Test when note is not found in discussions."""
    discussions = [
        {
            "id": "discussion1",
            "notes": [
                {"id": 101, "body": "Comment 1"},
                {"id": 102, "body": "Comment 2"},
            ],
        },
    ]

    mock_response = Mock()
    mock_response.is_success.return_value = True
    mock_response.body = json.dumps(discussions)

    gitlab_client_mock.aget.return_value = mock_response

    result = await tool_with_client._get_discussion_id_from_note_rest(
        project_id="123",
        resource_type="issues",
        resource_iid=42,
        note_id=999,
    )

    assert isinstance(result, dict)
    assert "error" in result
    assert "Note 999 not found" in result["error"]


@pytest.mark.asyncio
async def test_get_discussion_id_from_note_rest_api_failure(
    tool_with_client, gitlab_client_mock
):
    """Test when API call fails."""
    mock_response = Mock()
    mock_response.is_success.return_value = False

    gitlab_client_mock.aget.return_value = mock_response

    result = await tool_with_client._get_discussion_id_from_note_rest(
        project_id="123",
        resource_type="issues",
        resource_iid=42,
        note_id=1,
    )

    assert isinstance(result, dict)
    assert "error" in result
    assert "Failed to fetch issues discussions" in result["error"]


@pytest.mark.parametrize(
    "exception_msg",
    [
        "Connection timeout",
        "Network error",
        "Invalid response",
    ],
)
@pytest.mark.asyncio
async def test_get_discussion_id_from_note_rest_api_exception(
    tool_with_client, gitlab_client_mock, exception_msg
):
    """Test when API call raises an exception."""
    gitlab_client_mock.aget.side_effect = Exception(exception_msg)

    result = await tool_with_client._get_discussion_id_from_note_rest(
        project_id="123",
        resource_type="issues",
        resource_iid=42,
        note_id=1,
    )

    assert isinstance(result, dict)
    assert "error" in result
    assert exception_msg in result["error"]
