import asyncio
import time
from typing import Any, Dict

import structlog
from prometheus_client import Histogram

from contract import contract_pb2

ACTION_LATENCY = Histogram(
    name="executor_actions_duration_seconds",
    documentation="Latency for all actions that go to the Executor.",
    labelnames=["action_class"],
)


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
    await outbox.put(action)
    event: contract_pb2.ClientEvent = await inbox.get()

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


class HTTPConnectionError(Exception):
    """Exception raised when HTTP client connection fails."""

    pass


async def _execute_action_and_get_http_response(
    metadata: Dict[str, Any], action: contract_pb2.Action
) -> contract_pb2.ActionResponse:
    """Execute action and return HTTP response, checking for connection errors.

    This method checks if actionResponse.httpResponse.error has content (length > 0)
    and raises HTTPConnectionError if an error exists. This handles connection-level
    failures that would otherwise be silently ignored.

    Args:
        metadata: Dictionary containing outbox and inbox queues
        action: The action to execute

    Returns:
        ActionResponse with httpResponse if no error

    Raises:
        HTTPConnectionError: If actionResponse.httpResponse.error has content
    """
    actionResponse = await _execute_action_and_get_action_response(metadata, action)

    # Check for HTTP connection errors
    if actionResponse.httpResponse.error:
        raise HTTPConnectionError(
            f"HTTP connection failed: {actionResponse.httpResponse.error}"
        )

    return actionResponse


async def _execute_action(metadata: Dict[str, Any], action: contract_pb2.Action) -> str:
    actionResponse = await _execute_action_and_get_action_response(metadata, action)

    return actionResponse.response
