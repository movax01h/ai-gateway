import re
import time
from typing import Any, Awaitable, Callable, Dict, Optional

import structlog
from langchain_core.tools import ToolException
from prometheus_client import Histogram

from contract import contract_pb2
from duo_workflow_service.executor.outbox import Outbox, OutgoingMessageTooLargeError
from duo_workflow_service.tools.tool_output_manager import (
    TruncationConfig,
    truncate_string,
)
from duo_workflow_service.workflows.type_definitions import (
    MAX_MESSAGE_SIZE,
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

    try:
        actionResponse = await _execute_action_and_get_action_response(metadata, action)
    except OutgoingMessageTooLargeError as e:
        action_class = action.WhichOneof("action")
        log.error(
            "Action rejected before send: payload exceeds transport limit",
            request_id=action.requestID,
            action_class=action_class,
        )
        raise ToolException(
            f"The {action_class} action was not sent to the executor because its "
            f"payload exceeds the {MAX_MESSAGE_SIZE // (1024 * 1024)} MiB transport limit. "
            "Reduce the size of the command or request (e.g. write large content to a "
            "file in smaller chunks, or split the request) and try again."
        ) from e

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
        log.error(
            "Response error, missing plain text or http response",
            request_id=actionResponse.requestID,
        )
        raise ToolException("Executor doesn't return expected response fields")


# The Node `duo` CLI executor pages `runReadFile` responses at 50 KiB or 2,000
# lines, cut on a whole-line boundary, and ends a truncated page with a footer:
#   (Showing lines 1-46 of 154 total. Use offset=46 to continue reading.)
# Resume from the offset the footer advertises rather than computing one, so
# the executor's own indexing is always respected.
_READ_PAGE_FOOTER_RE = re.compile(
    r"^\s*[\[(]Showing lines \d+-\d+ of \d+ total\."
    r"(?: Use offset=(\d+) to continue reading\.)?[\])]\s*$"
)
_READ_PAST_EOF_RE = re.compile(r"^\s*[\[(]\s*Offset\s+\d+\s+is beyond", re.IGNORECASE)


def _split_read_page(page: str) -> tuple[str, Optional[int], Optional[str]]:
    """Split one ``runReadFile`` page into (content, next_offset, footer).

    ``next_offset`` is ``None`` when there is nothing to resume. ``footer`` is
    the raw footer text when one was present, so callers can log it.
    """
    lines = page.split("\n")
    idx = len(lines) - 1
    while idx >= 0 and not lines[idx].strip():
        idx -= 1
    if idx < 0:
        return page, None, None
    last = lines[idx]
    if _READ_PAST_EOF_RE.match(last):
        return "", None, last
    match = _READ_PAGE_FOOTER_RE.match(last)
    if not match:
        return page, None, None
    # Drop only the blank separator line; any further blank lines are content.
    content = "\n".join(lines[:idx]).removesuffix("\n")
    return content, (int(match.group(1)) if match.group(1) else None), last


async def _read_file_fully(
    metadata: Dict[str, Any],
    filepath: str,
    execute: Optional[
        Callable[[Dict[str, Any], contract_pb2.Action], Awaitable[str]]
    ] = None,
    max_pages: int = 256,
) -> str:
    """Read a file from the executor in full, following ``runReadFile`` pagination.

    A paginated read succeeds with only the first page plus a footer, so a
    single read silently returns a prefix. This follows each footer's resume
    offset until the file is complete, and raises ``ToolException`` rather
    than return a partial file (no resume offset, a stalled offset, a
    non-string response, or ``max_pages`` reached).

    With no footer this issues exactly one read, so it is a no-op against an
    executor that does not paginate.
    """
    log = structlog.stdlib.get_logger("workflow")
    run = execute or _execute_action
    pages: list[str] = []
    offset: Optional[int] = None
    lines_read = 0

    def _partial(reason: str) -> ToolException:
        return ToolException(
            f"Could not read {filepath} in full: {reason} "
            f"({lines_read} lines read). A partial file was not returned."
        )

    for page_num in range(max_pages):
        read = contract_pb2.ReadFile(filepath=filepath)
        if offset is not None:
            read.offset = offset
        page = await run(metadata, contract_pb2.Action(runReadFile=read))
        if not isinstance(page, str):
            raise _partial(f"runReadFile returned a non-string {type(page).__name__}")
        content, next_offset, footer = _split_read_page(page)
        if footer and not page_num:
            log.info(
                "runReadFile returned a pagination footer",
                filepath=filepath,
                footer=footer[:200],
                page_bytes=len(page),
            )
        if next_offset is not None and offset is not None and next_offset <= offset:
            raise _partial(f"pagination stalled at offset {offset}")
        if next_offset is not None and not content:
            # The executor skips a single line longer than its page size with
            # an empty page that still advances the offset.
            raise _partial(f"empty page at offset {offset or 0}")
        if content:
            pages.append(content)
            lines_read += content.count("\n") + 1
        if next_offset is None:
            if page_num:
                log.info(
                    "runReadFile completed across multiple pages",
                    filepath=filepath,
                    pages=page_num + 1,
                    lines=lines_read,
                )
            elif footer and not _READ_PAST_EOF_RE.match(footer):
                # Truncated on the first page with no way to continue.
                raise _partial(f"truncated with no resume offset ({footer[:200]})")
            return "\n".join(pages)
        offset = next_offset

    raise _partial(f"hit the {max_pages}-page limit")
