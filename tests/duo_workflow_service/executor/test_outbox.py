import asyncio
from unittest.mock import MagicMock, Mock, patch
from uuid import UUID

import pytest
from structlog.testing import capture_logs

from contract import contract_pb2
from duo_workflow_service.executor.outbox import (
    MAX_MESSAGE_LENGTH,
    Outbox,
    OutboxSignal,
)


class TestOutbox:
    @pytest.fixture
    def outbox(self) -> Outbox:
        return Outbox()

    @pytest.mark.parametrize(
        ("action", "future", "expected_to_receive_response"),
        [
            (contract_pb2.Action(), None, True),
            (contract_pb2.Action(), asyncio.Future(), True),
        ],
    )
    def test_put_action(
        self, outbox: Outbox, action, future, expected_to_receive_response
    ):
        assert action.requestID == ""

        request_id = outbox.put_action(action, result=future)

        assert action.requestID == request_id
        assert UUID(request_id)
        assert outbox._queue.qsize() == 1

        if not expected_to_receive_response:
            assert len(outbox._action_response) == 0
            assert len(outbox._legacy_action_response) == 0
            return

        assert len(outbox._action_response) == 1
        assert len(outbox._legacy_action_response) == 1
        assert request_id in outbox._action_response

        if future:
            assert outbox._action_response[request_id] is future
        else:
            assert outbox._action_response[request_id] is None

    @pytest.mark.asyncio
    async def test_put_action_and_wait_for_response(self, outbox: Outbox):
        action = contract_pb2.Action()
        client_response: contract_pb2.ClientEvent | None = None

        async def set_result():
            nonlocal client_response

            item = await outbox.get()

            client_response = contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(
                    requestID=item.requestID,
                ),
            )

            outbox.set_action_response(client_response)

        asyncio.create_task(set_result())

        response = await outbox.put_action_and_wait_for_response(action)

        assert response is client_response
        assert action.requestID == client_response.actionResponse.requestID

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("result"),
        [
            (None),
            (asyncio.Future()),
        ],
    )
    async def test_set_action_response(
        self,
        outbox: Outbox,
        result,
    ):
        action = contract_pb2.Action()

        outbox.put_action(action, result=result)

        assert action.requestID in outbox._action_response

        response = contract_pb2.ClientEvent(
            actionResponse=contract_pb2.ActionResponse(requestID=action.requestID)
        )

        outbox.set_action_response(response)

        if result:
            assert result.result() is response

        assert action.requestID not in outbox._action_response

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("action", "response", "result"),
        [
            (
                contract_pb2.Action(runHTTPRequest=contract_pb2.RunHTTPRequest()),
                contract_pb2.ClientEvent(
                    actionResponse=contract_pb2.ActionResponse(
                        httpResponse=contract_pb2.HttpResponse()
                    )
                ),
                None,
            ),
            (
                contract_pb2.Action(runCommand=contract_pb2.RunCommandAction()),
                contract_pb2.ClientEvent(
                    actionResponse=contract_pb2.ActionResponse(
                        plainTextResponse=contract_pb2.PlainTextResponse()
                    )
                ),
                asyncio.Future(),
            ),
        ],
    )
    async def test_legacy_set_action_response(
        self,
        action: contract_pb2.Action,
        response: contract_pb2.ClientEvent,
        outbox: Outbox,
        result,
    ):
        outbox.put_action(action, result=result)

        with capture_logs() as cap_logs:
            outbox.set_action_response(response)

            assert len(cap_logs) == 3
            assert cap_logs[0]["event"] == "Request ID not found."

        if result:
            assert result.result() is response

        assert action.requestID not in outbox._action_response

    @pytest.mark.asyncio
    async def test_legacy_new_checkpoint_response(
        self,
        outbox: Outbox,
    ):
        """This test ensures that the outbox works even if the legacy clients still return a response to NewCheckpoint
        action.

        For the reference, this is the case in GoLang Duo Workflow Executor ver 0.0.51.
        https://gitlab.com/gitlab-org/duo-workflow/duo-workflow-executor/-/blob/main/internal/services/runner/runner.go#L313
        """
        result: asyncio.Future = asyncio.Future()
        new_checkpoint_action = contract_pb2.Action(
            newCheckpoint=contract_pb2.NewCheckpoint()
        )
        outbox.put_action(
            contract_pb2.Action(runCommand=contract_pb2.RunCommandAction()),
            result=result,
        )
        outbox.put_action(new_checkpoint_action)

        response = contract_pb2.ClientEvent(
            actionResponse=contract_pb2.ActionResponse(
                plainTextResponse=contract_pb2.PlainTextResponse(),
                requestID=new_checkpoint_action.requestID,
            )
        )
        outbox.set_action_response(response)

        assert result.done() is False

    @pytest.mark.asyncio
    async def test_close(self, outbox: Outbox):
        assert outbox._queue.empty()

        outbox.close()

        item = await outbox.get()

        assert item == OutboxSignal.NO_MORE_OUTBOUND_REQUESTS

    @pytest.mark.asyncio
    async def test_check_empty(self, outbox: Outbox):
        outbox.put_action(contract_pb2.Action())

        assert not outbox._queue.empty()

        with capture_logs() as cap_logs:
            outbox.check_empty()

        assert outbox._queue.empty()
        assert len(cap_logs) == 1
        assert cap_logs[0]["event"] == "Found unsent items in outbox"
