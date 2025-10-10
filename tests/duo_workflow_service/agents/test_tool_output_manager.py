import json
from typing import Any
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from duo_workflow_service.agents.tool_output_manager import (
    _add_truncation_instruction,
    truncate_tool_response,
)


def test_add_truncation_instruction():
    notice = _add_truncation_instruction(
        truncated_text="random ted unique",
        original_token_size=77,
        truncated_token_size=54,
    )
    assert "70.1%" in notice
    assert notice.endswith("\n</instructions>\n</truncation_notice>\n")
    assert "random ted unique" in notice


@pytest.mark.parametrize(
    ("response", "should_truncated"),
    [
        ("A response under limit", False),
        (None, False),
        (1.0, False),
        ("", False),
        (
            ToolMessage(
                content="A response under limit",
                tool_call_id="call_id",
            ),
            False,
        ),
        ("This is a response that exceed the byte limit", True),
        (
            ToolMessage(
                content="This is a response that exceed the byte limit",
                tool_call_id="call_id",
            ),
            True,
        ),
        (
            ToolMessage(
                content=[{"data": "B" * 30}, {"more_data": list(range(20))}],
                tool_call_id="call_id",
            ),
            True,
        ),
        (
            {"key": "This is a response that exceed the byte limit", "data": [1, 2, 3]},
            True,
        ),
        ({"data": "B" * 30, "more_data": list(range(20))}, True),
    ],
)
@patch(
    "duo_workflow_service.agents.tool_output_manager.TOOL_RESPONSE_MAX_BYTES",
    30,
)
@patch(
    "duo_workflow_service.agents.tool_output_manager.TOOL_RESPONSE_TRUNCATED_SIZE",
    10,
)
@patch("duo_workflow_service.agents.tool_output_manager.token_counter")
@patch("duo_workflow_service.agents.tool_output_manager.logger")
def test_truncate_tool_response(
    mock_logger: Mock,
    mock_token_counter: Mock,
    response: Any,
    should_truncated: bool,
):
    if isinstance(response, ToolMessage):
        expected_json_str = (
            json.dumps(response.content)
            if not isinstance(response.content, str)
            else response.content
        )
    else:
        expected_json_str = (
            json.dumps(response) if not isinstance(response, str) else response
        )

    mock_token_counter.count_string_content.return_value = 1

    result = truncate_tool_response(response, tool_name="test_tool")

    if should_truncated:
        mock_logger.info.assert_called_once_with(
            "Tool response exceeds max size and will be truncated",
            tool_name="test_tool",
            original_token_size=1,
            truncated_token_size=1,
        )
        result = result.content if isinstance(result, ToolMessage) else result
        assert expected_json_str[:10] in result
        if isinstance(result, str):
            assert result.startswith("\n<truncation_notice>")
            assert result.endswith("</instructions>\n</truncation_notice>\n")
    else:
        assert result == response


@patch("duo_workflow_service.agents.tool_output_manager.logger")
def test_truncate_tool_response_exception(
    mock_logger: Mock,
):
    tool_response: Command = Command(
        update={
            "tool_response": ToolMessage(
                content="",
                tool_call_id="call_id",
            )
        }
    )
    result = truncate_tool_response(tool_response, tool_name="test_tool")
    mock_logger.error.assert_called_once_with(
        "Abort tool response truncation due to unexpected error: Object of type Command is not JSON serializable"
    )
    assert result == tool_response
