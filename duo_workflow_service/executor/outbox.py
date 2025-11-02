import asyncio
from enum import StrEnum
from uuid import uuid4

import structlog

from contract import contract_pb2
from duo_workflow_service.tracking import log_exception

log = structlog.stdlib.get_logger("outbox")

type ActionRequestID = str
type ActionType = str

MAX_MESSAGE_LENGTH = 200


class OutboxSignal(StrEnum):
    NO_MORE_OUTBOUND_REQUESTS = "no_more_outbound_requests"


class Outbox:
    """Class to manage outbound requests to clients."""

    def __init__(self):
        self._queue: asyncio.Queue[contract_pb2.Action | OutboxSignal] = asyncio.Queue()
        self._action_response: dict[
            ActionRequestID, asyncio.Future[contract_pb2.ClientEvent] | None
        ] = {}
        self._legacy_action_response: dict[ActionRequestID, ActionType] = {}

    def put_action(
        self,
        action: contract_pb2.Action,
        result: asyncio.Future[contract_pb2.ClientEvent] | None = None,
    ) -> ActionRequestID:
        """Put an item into the outbox queue."""

        action.requestID = str(uuid4())
        self._action_response[action.requestID] = result
        self._legacy_action_response[action.requestID] = action.WhichOneof("action")
        self._queue.put_nowait(action)

        return action.requestID

    async def put_action_and_wait_for_response(
        self, action: contract_pb2.Action
    ) -> contract_pb2.ClientEvent:
        """Put an action request into the queue and wait for the client response."""

        result = asyncio.get_event_loop().create_future()
        self.put_action(action, result=result)
        return await result

    async def get(self) -> contract_pb2.Action | OutboxSignal:
        """Get an item from the outbox."""

        return await self._queue.get()

    def fail_action(self, request_id: str, message: str):
        """Fail an action by setting an exception on its future."""
        log.error("Failing action", requestID=request_id, failure_reason=message)

        future = self._action_response.get(request_id)
        if future and not future.cancelled():
            try:
                future.set_exception(Exception(message))
            except asyncio.InvalidStateError:
                log.warning(
                    "Future for request already in final state when failing",
                    request_id=request_id,
                )

        if request_id in self._action_response:
            del self._action_response[request_id]
        if request_id in self._legacy_action_response:
            del self._legacy_action_response[request_id]

    def set_action_response(self, event: contract_pb2.ClientEvent):
        """Set action response to the future object which is awaited by the caller."""

        if event.actionResponse.requestID in self._action_response:
            self._set_action_response_for_request_id(
                event.actionResponse.requestID, event
            )
        else:
            log.error(
                "Request ID not found.",
                responseType=event.WhichOneof("response"),
                request_id=event.actionResponse.requestID,
                awaiting_request_ids=self.awaiting_request_ids(),
            )
            self.legacy_set_action_response(event)

    def awaiting_request_ids(self) -> str:
        return ",".join(list(self._action_response.keys()))

    def legacy_set_action_response(self, event: contract_pb2.ClientEvent) -> None:
        """Set action response best-effort basis for legacy LSP clients that do not return request ID in the response.

        The 8.20+ of the Language Server and 6.49.7+ of the VSCode extension return request ID properly by the fix
        https://gitlab.com/gitlab-org/editor-extensions/gitlab-lsp/-/merge_requests/2332.
        """
        if not self._legacy_action_response:
            log.warn("No legacy action responses are registered")
            return

        response_type = event.actionResponse.WhichOneof("response_type")

        request_id_expecting_http_response: ActionRequestID | None = None
        request_id_expecting_plain_response: ActionRequestID | None = None

        for request_id, action_type in self._legacy_action_response.items():
            log.info(
                "legacy_set_action_response",
                request_id=request_id,
                action_type=action_type,
            )

            if (
                request_id_expecting_http_response
                and request_id_expecting_plain_response
            ):
                break

            if action_type in ["newCheckpoint", "runHTTPRequest"]:
                if not request_id_expecting_http_response:
                    request_id_expecting_http_response = request_id
            else:
                if not request_id_expecting_plain_response:
                    request_id_expecting_plain_response = request_id

        if response_type == "httpResponse" and request_id_expecting_http_response:
            self._set_action_response_for_request_id(
                request_id_expecting_http_response, event
            )
        elif (
            response_type == "plainTextResponse" and request_id_expecting_plain_response
        ):
            self._set_action_response_for_request_id(
                request_id_expecting_plain_response, event
            )
        else:
            log.error("Failed to legacy set action response")

    def close(self) -> None:
        """Close the outbox for exiting send events loop."""

        self._queue.put_nowait(OutboxSignal.NO_MORE_OUTBOUND_REQUESTS)

    def check_empty(self) -> None:
        try:
            while True:
                try:
                    item: contract_pb2.Action | OutboxSignal = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    # Queue is empty, exit loop
                    break

                content = str(item)

                if len(content) > MAX_MESSAGE_LENGTH:
                    content = f"{content[:MAX_MESSAGE_LENGTH]}..."

                log.error(
                    "Found unsent items in outbox",
                    content=content,
                )
        except Exception as e:
            log_exception(
                e,
                extra={
                    "context": "Error draining outbox queue",
                },
            )
            raise

    def _set_action_response_for_request_id(
        self, request_id: ActionRequestID, event: contract_pb2.ClientEvent
    ):
        log.info(
            "Setting action response for request ID.",
            request_id=request_id,
            responseType=event.WhichOneof("response"),
        )

        future = self._action_response[request_id]

        if future and not future.cancelled():
            try:
                future.set_result(event)
            except asyncio.InvalidStateError:
                log.warning(
                    "Future for request already in final state", request_id=request_id
                )

        del self._action_response[request_id]
        del self._legacy_action_response[request_id]
