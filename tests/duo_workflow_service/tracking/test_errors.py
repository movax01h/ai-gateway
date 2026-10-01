import asyncio

import pytest
from structlog.testing import capture_logs

from duo_workflow_service.tracking.errors import log_exception, log_workflow_failure


def test_log_exception():
    with capture_logs() as cap_logs:
        log_exception(ValueError("boom"), extra={"workflow_id": "123"})

    assert len(cap_logs) == 1
    assert cap_logs[0]["event"] == "boom"
    assert cap_logs[0]["log_level"] == "error"
    assert cap_logs[0]["exception_class"] == "ValueError"
    assert cap_logs[0]["additional_details"] == {"workflow_id": "123"}


@pytest.mark.parametrize(
    "error,expected",
    [
        (
            asyncio.CancelledError("Client-side streaming has been closed."),
            {
                "event": "Workflow run cancelled",
                "log_level": "info",
                "reason": "Client-side streaming has been closed.",
                "exception_class": "CancelledError",
            },
        ),
        (
            ValueError("boom"),
            {
                "event": "boom",
                "log_level": "error",
                "exception_class": "ValueError",
            },
        ),
    ],
    ids=["cancelled", "other_error"],
)
def test_log_workflow_failure(error, expected):
    with capture_logs() as cap_logs:
        log_workflow_failure(error, extra={"workflow_id": "123"})

    assert len(cap_logs) == 1
    assert cap_logs[0] == cap_logs[0] | expected
    assert cap_logs[0]["additional_details"] == {"workflow_id": "123"}


def test_log_workflow_failure_without_extra():
    with capture_logs() as cap_logs:
        log_workflow_failure(asyncio.CancelledError())

    assert cap_logs[0]["additional_details"] == {}
