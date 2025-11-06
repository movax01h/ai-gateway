import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.tools import ToolException

from contract import contract_pb2
from duo_workflow_service.executor.action import (
    _execute_action,
    _execute_action_and_get_action_response,
)
from duo_workflow_service.executor.outbox import Outbox


@pytest.fixture
def metadata():
    outbox = Outbox()
    return {"outbox": outbox}


@pytest.mark.asyncio
async def test_execute_action_success_http_response(metadata):
    action = contract_pb2.Action()
    client_event = contract_pb2.ClientEvent()

    # Set up HTTP response without error
    client_event.actionResponse.requestID = "test-request-123"
    client_event.actionResponse.httpResponse.statusCode = 200
    client_event.actionResponse.httpResponse.body = '{"result": "success"}'
    client_event.actionResponse.httpResponse.error = ""

    # Set up action type for logging
    action.runHTTPRequest.path = "https://example.com"

    metadata["outbox"].put_action_and_wait_for_response = AsyncMock(
        return_value=client_event
    )

    with patch(
        "duo_workflow_service.executor.action.structlog.stdlib.get_logger"
    ) as mock_get_logger:
        mock_logger = mock_get_logger.return_value

        response = await _execute_action(metadata, action)

        assert response == '{"result": "success"}'

        # Verify the log entry was created for the _execute_action call
        mock_logger.warning.assert_any_call(
            "HTTP response when plain text response expected, returning body",
            requestID="test-request-123",
            action_class="runHTTPRequest",
        )


@pytest.mark.asyncio
async def test_execute_action_success_plaintext_response(metadata):
    action = contract_pb2.Action()
    client_event = contract_pb2.ClientEvent()

    # Set up plaintext response without error
    client_event.actionResponse.plainTextResponse.response = "plaintext success"
    client_event.actionResponse.plainTextResponse.error = ""

    metadata["outbox"].put_action_and_wait_for_response = AsyncMock(
        return_value=client_event
    )

    with patch(
        "duo_workflow_service.executor.action.structlog.stdlib.get_logger"
    ) as mock_get_logger:
        mock_logger = mock_get_logger.return_value

        response = await _execute_action(metadata, action)

        assert response == "plaintext success"

        # Verify no log entry for the specific HTTP response message
        # Check that the HTTP-specific log message was not called
        http_log_calls = [
            call
            for call in mock_logger.warning.call_args_list
            if len(call[0]) > 0
            and "HTTP response when plain text response expected" in call[0][0]
        ]
        assert len(http_log_calls) == 0


@pytest.mark.asyncio
async def test__execute_action_and_get_action_response_http_error_raises_tool_exception(
    metadata,
):
    """Test that HTTP errors in _execute_action_and_get_action_response raise ToolException."""
    action = contract_pb2.Action()
    client_event = contract_pb2.ClientEvent()

    # Set up HTTP response with error
    client_event.actionResponse.httpResponse.error = "Connection timeout"

    metadata["outbox"].put_action_and_wait_for_response = AsyncMock(
        return_value=client_event
    )

    with pytest.raises(ToolException, match="HTTP action error: Connection timeout"):
        await _execute_action_and_get_action_response(metadata, action)


@pytest.mark.asyncio
async def test__execute_action_and_get_action_response_plaintext_error_raises_tool_exception(
    metadata,
):
    """Test that plaintext errors in _execute_action_and_get_action_response raise ToolException."""
    action = contract_pb2.Action()
    client_event = contract_pb2.ClientEvent()

    # Set up plaintext response with error
    client_event.actionResponse.plainTextResponse.error = "File not found"

    metadata["outbox"].put_action_and_wait_for_response = AsyncMock(
        return_value=client_event
    )

    with pytest.raises(ToolException, match="Action error: File not found"):
        await _execute_action_and_get_action_response(metadata, action)


@pytest.mark.asyncio
async def test_execute_action_empty_inbox(metadata):
    action = contract_pb2.Action()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(_execute_action(metadata, action), timeout=1.0)


@pytest.mark.asyncio
async def test__execute_action_and_get_action_response_http_success(
    metadata,
):
    action = contract_pb2.Action()
    client_event = contract_pb2.ClientEvent()

    client_event.actionResponse.httpResponse.statusCode = 200
    client_event.actionResponse.httpResponse.body = '{"result": "ok"}'
    client_event.actionResponse.httpResponse.error = ""

    metadata["outbox"].put_action_and_wait_for_response = AsyncMock(
        return_value=client_event
    )

    response = await _execute_action_and_get_action_response(metadata, action)

    assert response.httpResponse.statusCode == 200
    assert response.httpResponse.body == '{"result": "ok"}'


@pytest.mark.asyncio
async def test__execute_action_and_get_action_response_http_not_found(
    metadata,
):
    action = contract_pb2.Action()
    client_event = contract_pb2.ClientEvent()

    client_event.actionResponse.httpResponse.statusCode = 404
    client_event.actionResponse.httpResponse.body = ""
    client_event.actionResponse.httpResponse.error = ""

    metadata["outbox"].put_action_and_wait_for_response = AsyncMock(
        return_value=client_event
    )

    response = await _execute_action_and_get_action_response(metadata, action)

    assert response.httpResponse.statusCode == 404
    assert response.httpResponse.body == ""


@pytest.mark.asyncio
async def test__execute_action_and_get_action_response_http_error(
    metadata,
):
    action = contract_pb2.Action()
    client_event = contract_pb2.ClientEvent()

    client_event.actionResponse.httpResponse.statusCode = 0
    client_event.actionResponse.httpResponse.body = ""
    client_event.actionResponse.httpResponse.error = "Some HTTP error"

    metadata["outbox"].put_action_and_wait_for_response = AsyncMock(
        return_value=client_event
    )

    # This should now raise ToolException instead of returning a response
    with pytest.raises(ToolException, match="HTTP action error: Some HTTP error"):
        await _execute_action_and_get_action_response(metadata, action)


@pytest.mark.asyncio
async def test__execute_action_and_get_action_response_plaintext_success(
    metadata,
):
    action = contract_pb2.Action()
    client_event = contract_pb2.ClientEvent()

    client_event.actionResponse.plainTextResponse.response = "Response"
    client_event.actionResponse.plainTextResponse.error = ""

    metadata["outbox"].put_action_and_wait_for_response = AsyncMock(
        return_value=client_event
    )

    response = await _execute_action_and_get_action_response(metadata, action)

    assert response.plainTextResponse.response == "Response"


@pytest.mark.asyncio
async def test__execute_action_and_get_action_response_plaintext_error(
    metadata,
):
    action = contract_pb2.Action()
    client_event = contract_pb2.ClientEvent()

    client_event.actionResponse.plainTextResponse.response = ""
    client_event.actionResponse.plainTextResponse.error = "file not found"

    metadata["outbox"].put_action_and_wait_for_response = AsyncMock(
        return_value=client_event
    )

    # This should now raise ToolException instead of returning a response
    with pytest.raises(ToolException, match="Action error: file not found"):
        await _execute_action_and_get_action_response(metadata, action)


@pytest.mark.asyncio
async def test_execute_action_fallback_to_legacy_response(metadata):
    """Test that _execute_action falls back to legacy response when response_type is not set."""
    action = contract_pb2.Action()
    client_event = contract_pb2.ClientEvent()

    # Set up an ActionResponse with no response_type set (neither plainTextResponse nor httpResponse)
    client_event.actionResponse.response = "some legacy response"
    # Don't set either plainTextResponse or httpResponse, so WhichOneof("response_type") returns None

    metadata["outbox"].put_action_and_wait_for_response = AsyncMock(
        return_value=client_event
    )

    with patch(
        "duo_workflow_service.executor.action.structlog.stdlib.get_logger"
    ) as mock_get_logger:
        mock_logger = mock_get_logger.return_value

        response = await _execute_action(metadata, action)

        assert response == "some legacy response"

        # Verify the warning was logged
        mock_logger.warning.assert_called_once_with(
            "Executor doesn't return expected response fields, falling back to legacy response"
        )


@pytest.mark.asyncio
async def test__execute_action_and_get_action_response_runcommand_error(
    metadata,
):
    action = contract_pb2.Action()
    client_event = contract_pb2.ClientEvent()

    client_event.actionResponse.plainTextResponse.error = "Command failed"
    client_event.actionResponse.plainTextResponse.response = (
        "Additional error details from command output"
    )

    metadata["outbox"].put_action_and_wait_for_response = AsyncMock(
        return_value=client_event
    )

    with pytest.raises(ToolException) as exc_info:
        await _execute_action_and_get_action_response(metadata, action)

    expected_text = "Action error: Command failed"
    assert expected_text in str(exc_info.value)
