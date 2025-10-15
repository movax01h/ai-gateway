import time
from typing import Any, Dict

import structlog
from langchain_core.tools import ToolException
from prometheus_client import Histogram

from contract import contract_pb2
from contract.contract_pb2 import HttpResponse, PlainTextResponse
from duo_workflow_service.executor.outbox import Outbox

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
    outbox: Outbox = metadata["outbox"]
    log = structlog.stdlib.get_logger("workflow")

    action_class = action.WhichOneof("action")
    log.info(
        "Attempting action from the egress queue",
        requestID=action.requestID,
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
            requestID=event.actionResponse.requestID,
            action_class=action_class,
            duration_s=duration,
        )

        if event.actionResponse.httpResponse.error:
            log.error(
                "Http response error",
                requestID=event.actionResponse.requestID,
                action_class=action_class,
            )
            raise ToolException(
                f"HTTP action error: {event.actionResponse.httpResponse.error}"
            )

        if event.actionResponse.plainTextResponse.error:
            log.error(
                "Plaintext response error",
                requestID=event.actionResponse.requestID,
                action_class=action_class,
            )
            raise ToolException(
                f"Action error: {event.actionResponse.plainTextResponse.error}"
            )

        if not event.actionResponse.response:
            response_type = event.actionResponse.WhichOneof("response_type")
            if response_type == "plainTextResponse":
                log.info(
                    "Legacy response empty, setting it from plaintext response",
                    requestID=event.actionResponse.requestID,
                    action_class=action_class,
                )
                event.actionResponse.response = _get_action_response_from_plaintext(
                    event.actionResponse.plainTextResponse
                )
            elif response_type == "httpResponse":
                log.info(
                    "Legacy response empty, setting it from http response",
                    requestID=event.actionResponse.requestID,
                    action_class=action_class,
                )
                event.actionResponse.response = _get_action_response_from_http(
                    event.actionResponse.httpResponse
                )

        # Record all metrics in the separate function
        record_metrics(action_class, duration)

    return event.actionResponse


def _get_action_response_from_plaintext(plaintext_response: PlainTextResponse):
    if plaintext_response.error:
        return f"Error running tool: {plaintext_response.error}"

    return plaintext_response.response


def _get_action_response_from_http(http_response: HttpResponse):
    if http_response.error:
        return f"Error: {http_response.error}"

    if http_response.statusCode < 200 or http_response.statusCode >= 300:
        return f"Error: unexpected status code: {http_response.statusCode}"

    return http_response.body


async def _execute_action(metadata: Dict[str, Any], action: contract_pb2.Action) -> str:
    log = structlog.stdlib.get_logger("workflow")
    actionResponse = await _execute_action_and_get_action_response(metadata, action)

    # Return the appropriate response type based on action type
    response_type = actionResponse.WhichOneof("response_type")
    if response_type == "httpResponse":
        log.info(
            "HTTP response with use_http_response=False, returning body instead",
            requestID=actionResponse.requestID,
            action_class=action.WhichOneof("action"),
        )
        return actionResponse.httpResponse.body
    elif response_type == "plainTextResponse":
        return actionResponse.plainTextResponse.response
    else:
        log.warning(
            "Executor doesn't return expected response fields, falling back to legacy response"
        )
        return actionResponse.response
