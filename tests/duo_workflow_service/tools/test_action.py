import asyncio

import pytest

from contract import contract_pb2
from duo_workflow_service.executor.action import _execute_action


@pytest.mark.asyncio
async def test_execute_action_success():
    outbox: asyncio.Queue = asyncio.Queue()
    inbox: asyncio.Queue = asyncio.Queue()
    metadata = {"outbox": outbox, "inbox": inbox}

    action = contract_pb2.Action()
    expected_response = "expected_response"
    client_event = contract_pb2.ClientEvent()
    client_event.actionResponse.response = expected_response

    await inbox.put(client_event)

    response = await _execute_action(metadata, action)

    put_action = await outbox.get()
    outbox.task_done()
    assert put_action == action
    assert response == expected_response
    assert inbox.empty()


@pytest.mark.asyncio
async def test_execute_action_empty_inbox():
    outbox: asyncio.Queue = asyncio.Queue()
    inbox: asyncio.Queue = asyncio.Queue()
    metadata = {"outbox": outbox, "inbox": inbox}

    action = contract_pb2.Action()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(_execute_action(metadata, action), timeout=1.0)
