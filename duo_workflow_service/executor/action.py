import asyncio
import time
from typing import Any, Dict

import structlog
from prometheus_client import Histogram

from contract import contract_pb2
from duo_workflow_service.tracking import log_exception

ACTION_LATENCY = Histogram(
    name="executor_actions_duration_seconds",
    documentation="Latency for all actions that go to the Executor.",
    labelnames=["action_class"],
)

TIMEOUT_OUTBOX_PUT = 60
TIMEOUT_INBOX_GET = 60


def record_metrics(action_class: str, duration: float):
    """Record Prometheus metrics for an action execution."""
    ACTION_LATENCY.labels(action_class=action_class).observe(duration)


async def _execute_action_and_get_action_response(
    metadata: Dict[str, Any], action: contract_pb2.Action
) -> contract_pb2.ActionResponse:
    outbox: asyncio.Queue = metadata["outbox"]
    inbox: asyncio.Queue = metadata["inbox"]
    log = structlog.stdlib.get_logger("workflow")

    action_class = action.WhichOneof("action")
    log.info(
        "Attempting action from the egress queue",
        requestID=action.requestID,
        action_class=action_class,
    )

    start_time = time.time()
    try:
        log.debug("Waiting for putting action to outbox...", extra={"source": __name__})
        # `outbox.put` could hung indefinitely if the queue is full and no tasks get an item from it.
        # There are several cases when this could happen:
        # - Connection between client and server is terminated during the workflow is running
        # - Deadlock between workflow and client-server message loop
        # TODO: Handle the `asyncio.TimeoutError` properly.
        await asyncio.wait_for(outbox.put(action), TIMEOUT_OUTBOX_PUT)
    except asyncio.TimeoutError as ex:
        log_exception(ex)
        raise

    try:
        log.debug(
            "Waiting for getting action from inbox...", extra={"source": __name__}
        )
        # `inbox.get` could hung indefinitely if the queue is full and no tasks get an item from it.
        # There are several cases when this could happen:
        # - Connection between client and server is terminated during the workflow is running
        # - Deadlock between workflow and client-server message loop
        # TODO: Handle the `asyncio.TimeoutError` properly.
        event: contract_pb2.ClientEvent = await asyncio.wait_for(
            inbox.get(), TIMEOUT_INBOX_GET
        )
    except asyncio.TimeoutError as ex:
        log_exception(ex)
        raise

    if event.actionResponse:
        duration = time.time() - start_time
        log.info(
            "Read ClientEvent into the ingres queue",
            requestID=event.actionResponse.requestID,
            action_class=action_class,
            duration_s=duration,
        )

        # Record all metrics in the separate function
        record_metrics(action_class, duration)

    inbox.task_done()
    return event.actionResponse


async def _execute_action(metadata: Dict[str, Any], action: contract_pb2.Action) -> str:
    actionResponse = await _execute_action_and_get_action_response(metadata, action)

    return actionResponse.response
