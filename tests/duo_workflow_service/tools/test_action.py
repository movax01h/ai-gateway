import asyncio

import pytest

from contract import contract_pb2
from duo_workflow_service.executor.action import (
    HTTPConnectionError,
    _execute_action,
    _execute_action_and_get_http_response,
)


@pytest.fixture
def metadata():
    outbox = asyncio.Queue()
    inbox = asyncio.Queue()
    return {"outbox": outbox, "inbox": inbox}


@pytest.mark.asyncio
async def test_execute_action_success(metadata):
    action = contract_pb2.Action()
    expected_response = "expected_response"
    client_event = contract_pb2.ClientEvent()
    client_event.actionResponse.response = expected_response

    await metadata["inbox"].put(client_event)

    response = await _execute_action(metadata, action)

    put_action = await metadata["outbox"].get()
    metadata["outbox"].task_done()
    assert put_action == action
    assert response == expected_response
    assert metadata["inbox"].empty()


@pytest.mark.asyncio
async def test_execute_action_empty_inbox(metadata):
    action = contract_pb2.Action()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(_execute_action(metadata, action), timeout=1.0)


@pytest.mark.asyncio
async def test_execute_action_and_get_http_response_success(metadata):
    action = contract_pb2.Action()
    client_event = contract_pb2.ClientEvent()
    client_event.actionResponse.response = "success"

    # Create httpResponse with no error
    client_event.actionResponse.httpResponse.statusCode = 200
    client_event.actionResponse.httpResponse.body = '{"result": "ok"}'
    client_event.actionResponse.httpResponse.error = ""

    await metadata["inbox"].put(client_event)

    response = await _execute_action_and_get_http_response(metadata, action)

    put_action = await metadata["outbox"].get()
    metadata["outbox"].task_done()
    assert put_action == action
    assert response.response == "success"
    assert response.httpResponse.statusCode == 200
    assert metadata["inbox"].empty()


@pytest.mark.asyncio
async def test_execute_action_and_get_http_response_connection_error(metadata):
    action = contract_pb2.Action()
    client_event = contract_pb2.ClientEvent()
    client_event.actionResponse.response = "failed"

    # Create httpResponse with error
    client_event.actionResponse.httpResponse.error = "Connection refused"

    await metadata["inbox"].put(client_event)

    with pytest.raises(
        HTTPConnectionError, match="HTTP connection failed: Connection refused"
    ):
        await _execute_action_and_get_http_response(metadata, action)

    put_action = await metadata["outbox"].get()
    metadata["outbox"].task_done()
    assert put_action == action
    assert metadata["inbox"].empty()
