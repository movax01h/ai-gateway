import json
from typing import Any
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from duo_workflow_service.tools.duo_base_tool import TruncationConfig
from duo_workflow_service.tools.tool_output_manager import (
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


def test_truncate_string_reverse():
    """Test reverse truncation keeps the end of the content."""
    from duo_workflow_service.tools.tool_output_manager import (
        TruncationDirection,
        truncate_string,
    )

    # Create content where we can identify beginning vs end
    content = "START" + ("x" * 200 * 1024) + "END"

    reverse_config = TruncationConfig(
        max_bytes=200 * 1024,
        truncated_size=100 * 1024,
        direction=TruncationDirection.FROM_END,
    )

    result = truncate_string(content, "test_tool", reverse_config)

    # Should contain END but not START
    assert "END" in result
    assert "START" not in result
    assert "<truncation_notice>" in result


def test_truncate_string_default_direction():
    """Test that default truncation direction is FROM_START."""
    from duo_workflow_service.tools.tool_output_manager import (
        TruncationDirection,
        truncate_string,
    )

    # Create content where we can identify beginning vs end
    content = "START" + ("x" * 200 * 1024) + "END"

    default_config = TruncationConfig(
        max_bytes=200 * 1024,
        truncated_size=100 * 1024,
    )

    # Verify default is FROM_START
    assert default_config.direction == TruncationDirection.FROM_START

    result = truncate_string(content, "test_tool", default_config)

    # Should contain START but not END (same as FROM_START behavior)
    assert "START" in result
    assert "END" not in result
    assert "<truncation_notice>" in result


def test_truncate_string_forward():
    """Test forward truncation keeps the beginning of the content."""
    from duo_workflow_service.tools.tool_output_manager import (
        TruncationDirection,
        truncate_string,
    )

    # Create content where we can identify beginning vs end
    content = "START" + ("x" * 200 * 1024) + "END"

    forward_config = TruncationConfig(
        max_bytes=200 * 1024,
        truncated_size=100 * 1024,
        direction=TruncationDirection.FROM_START,
    )

    result = truncate_string(content, "test_tool", forward_config)

    # Should contain START but not END
    assert "START" in result
    assert "END" not in result
    assert "<truncation_notice>" in result


def test_truncate_string_reverse_no_truncation_needed():
    """Test reverse truncation when content is under limit."""
    from duo_workflow_service.tools.tool_output_manager import (
        TruncationDirection,
        truncate_string,
    )

    content = "Small content"

    reverse_config = TruncationConfig(
        max_bytes=200 * 1024,
        truncated_size=100 * 1024,
        direction=TruncationDirection.FROM_END,
    )

    result = truncate_string(content, "test_tool", reverse_config)

    # Should return unchanged
    assert result == content
    assert "<truncation_notice>" not in result


def test_truncate_tool_response_with_custom_config():
    """Test truncation with custom config (1MB/800KB)."""
    custom_config = TruncationConfig(
        max_bytes=1 * 1024 * 1024, truncated_size=800 * 1024  # 1 MiB  # 800 KiB
    )

    # Response that would be truncated with default config but not with custom
    medium_response = "x" * (200 * 1024)  # 200KB
    result = truncate_tool_response(
        medium_response, "build_review_merge_request_context", custom_config
    )
    assert result == medium_response  # Should NOT be truncated

    # Response that exceeds even the custom limit
    huge_response = "x" * (1 * 1024 * 1024 + 1000)  # Exceeds 1MB
    result = truncate_tool_response(
        huge_response, "build_review_merge_request_context", custom_config
    )
    assert len(result) < len(huge_response)
    assert "<truncation_notice>" in result


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
@patch("duo_workflow_service.tools.tool_output_manager.token_counter")
@patch("duo_workflow_service.tools.tool_output_manager.logger")
def test_truncate_tool_response(
    mock_logger: Mock,
    mock_token_counter: Mock,
    response: Any,
    should_truncated: bool,
):
    test_config = TruncationConfig(max_bytes=30, truncated_size=10)

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

    result = truncate_tool_response(
        response, tool_name="test_tool", truncation_config=test_config
    )

    if should_truncated:
        mock_logger.info.assert_called_once_with(
            "Tool response exceeds max size and will be truncated",
            tool_name="test_tool",
            original_token_size=1,
            truncated_token_size=1,
            max_bytes=30,
            truncated_size=10,
            direction="from_start",
        )
        result = result.content if isinstance(result, ToolMessage) else result
        assert expected_json_str[:10] in result
        if isinstance(result, str):
            assert result.startswith("\n<truncation_notice>")
            assert result.endswith("</instructions>\n</truncation_notice>\n")
    else:
        assert result == response


@patch("duo_workflow_service.tools.tool_output_manager.logger")
def test_truncate_tool_response_exception(
    mock_logger: Mock,
):
    default_config = TruncationConfig()

    tool_response: Command = Command(
        update={
            "tool_response": ToolMessage(
                content="",
                tool_call_id="call_id",
            )
        }
    )
    result = truncate_tool_response(
        tool_response, tool_name="test_tool", truncation_config=default_config
    )
    mock_logger.info.assert_called_once_with(
        "Skip truncation for Command tool response"
    )
    assert result == tool_response
