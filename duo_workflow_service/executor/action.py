import time
from typing import Any, Dict, Optional

import structlog
from langchain_core.tools import ToolException
from prometheus_client import Histogram

from contract import contract_pb2
from duo_workflow_service.executor.outbox import Outbox
from duo_workflow_service.tools.tool_output_manager import (
    TruncationConfig,
    truncate_string,
)

ACTION_LATENCY = Histogram(
    name="executor_actions_duration_seconds",
    documentation="Latency for all actions that go to the Executor.",
    labelnames=["action_class"],
)


class ToolExceptionWithResponse(ToolException):
    def __init__(self, error, response: Optional[str] = None):
        super().__init__(error)
        self.response = response


def record_metrics(action_class: str, duration: float):
    """Record Prometheus metrics for an action execution."""
    ACTION_LATENCY.labels(action_class=action_class).observe(duration)


async def _execute_action_and_get_action_response(
    metadata: Dict[str, Any], action: contract_pb2.Action
) -> contract_pb2.ActionResponse:
    outbox: Outbox = metadata["outbox"]
    log = structlog.stdlib.get_logger("workflow")

    action_class = action.WhichOneof("action")
    log.info(
        "Attempting action from the egress queue",
        request_id=action.requestID,
        action_class=action_class,
    )

    start_time = time.time()

    event: contract_pb2.ClientEvent = await outbox.put_action_and_wait_for_response(
        action
    )

    if event.actionResponse:
        duration = time.time() - start_time
        log.info(
            "Read ClientEvent into the ingres queue",
            request_id=event.actionResponse.requestID,
            action_class=action_class,
            duration_s=duration,
        )

        if event.actionResponse.httpResponse.error:
            log.error(
                "Http response error",
                request_id=event.actionResponse.requestID,
                action_class=action_class,
            )
            raise ToolException(
                f"HTTP action error: {event.actionResponse.httpResponse.error}"
            )

        if event.actionResponse.plainTextResponse.error:
            log.error(
                "Plaintext response error",
                request_id=event.actionResponse.requestID,
                action_class=action_class,
            )
            error = f"Action error: {event.actionResponse.plainTextResponse.error}"
            # Some actions (e.g. RunCommand) can have important error information in the `response` field.
            if event.actionResponse.plainTextResponse.response:
                response = truncate_string(
                    text=event.actionResponse.plainTextResponse.response,
                    tool_name=action_class,
                    truncation_config=TruncationConfig(),
                )
                raise ToolExceptionWithResponse(error, response)
            raise ToolException(error)

        # Record all metrics in the separate function
        record_metrics(action_class, duration)

    return event.actionResponse


async def _execute_action(metadata: Dict[str, Any], action: contract_pb2.Action) -> str:
    log = structlog.stdlib.get_logger("workflow")
    actionResponse = await _execute_action_and_get_action_response(metadata, action)

    # Return the appropriate response type based on action type
    response_type = actionResponse.WhichOneof("response_type")
    if response_type == "httpResponse":
        log.warning(
            "HTTP response when plain text response expected, returning body",
            request_id=actionResponse.requestID,
            action_class=action.WhichOneof("action"),
        )
        return actionResponse.httpResponse.body
    elif response_type == "plainTextResponse":
        return actionResponse.plainTextResponse.response
    else:
        # Backward compatibility: This fallback to legacy response field is required for GitLab versions < 18.5
        # that doesn't have the following fix in Go Executor. Do not delete until 18.8
        # https://gitlab.com/gitlab-org/duo-workflow/duo-workflow-executor/-/merge_requests/243
        log.warning(
            "Executor doesn't return expected response fields, falling back to legacy response"
        )
        return actionResponse.response
