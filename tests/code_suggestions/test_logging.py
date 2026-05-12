from unittest import mock

import pytest
import structlog
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from starlette.testclient import TestClient
from starlette_context.middleware import RawContextMiddleware
from structlog.testing import capture_logs

from ai_gateway.api.middleware import AccessLogMiddleware


def broken_page(request):
    raise RuntimeError("Something broke!")


def raise_exception_group(request):
    raise ExceptionGroup(
        "ExceptionGroup", [ValueError("value error in an ExceptionGroup")]
    )


def log_and_respond(_request: Request) -> Response:
    """Endpoint that emits a log entry so we can verify context vars are propagated."""
    logger = structlog.stdlib.get_logger("test.endpoint")
    logger.info("inside request handler")
    return Response("ok")


app = Starlette(
    middleware=[
        Middleware(RawContextMiddleware),
        Middleware(AccessLogMiddleware, skip_endpoints=[]),
    ],
    routes=[
        Route("/", endpoint=broken_page, methods=["POST"]),
        Route("/exception_group", endpoint=raise_exception_group, methods=["POST"]),
        Route("/log", endpoint=log_and_respond, methods=["POST"]),
    ],
)
client = TestClient(app)


@mock.patch("ai_gateway.api.middleware.access_log.log_exception")
def test_x_gitlab_headers_logged_when_set(mock_log_exception):
    with (
        capture_logs(processors=[structlog.contextvars.merge_contextvars]) as cap_logs,
        pytest.raises(RuntimeError),
    ):
        client.post(
            "/",
            headers={
                "X-Gitlab-Instance-Id": "ABC",
                "X-Gitlab-Global-User-Id": "DEF",
                "X-Gitlab-Host-Name": "awesome-org.com",
                "X-Gitlab-Feature-Enabled-By-Namespace-Ids": "1,2",
                "X-Gitlab-Feature-Enablement-Type": "duo_pro",
                "X-Gitlab-Realm": "saas",
                "x-gitlab-root-namespace-id": "123",
            },
            data={"foo": "bar"},
        )

        mock_log_exception.assert_called_once()

    assert cap_logs[0]["gitlab_instance_id"] == "ABC"
    assert cap_logs[0]["gitlab_global_user_id"] == "DEF"
    assert cap_logs[0]["gitlab_host_name"] == "awesome-org.com"
    assert cap_logs[0]["gitlab_feature_enabled_by_namespace_ids"] == "1,2"
    assert cap_logs[0]["gitlab_feature_enablement_type"] == "duo_pro"
    assert cap_logs[0]["gitlab_realm"] == "saas"
    assert cap_logs[0]["gitlab_root_namespace_id"] == "123"


@mock.patch("ai_gateway.api.middleware.access_log.log_exception")
def test_x_gitlab_headers_not_logged_when_not_set(mock_log_exception):
    with (
        capture_logs(processors=[structlog.contextvars.merge_contextvars]) as cap_logs,
        pytest.raises(RuntimeError),
    ):
        client.post("/", headers={}, data={"foo": "bar"})

        mock_log_exception.assert_called_once()

    assert cap_logs[0]["gitlab_instance_id"] is None
    assert cap_logs[0]["gitlab_global_user_id"] is None
    assert cap_logs[0]["gitlab_host_name"] is None
    assert cap_logs[0]["gitlab_feature_enabled_by_namespace_ids"] is None
    assert cap_logs[0]["gitlab_feature_enablement_type"] is None
    assert cap_logs[0]["gitlab_realm"] is None
    assert cap_logs[0]["gitlab_root_namespace_id"] is None


def test_x_gitlab_headers_propagated_to_all_log_entries():
    """Verify that GitLab headers are bound to context vars and appear in all log entries."""
    with capture_logs(processors=[structlog.contextvars.merge_contextvars]) as cap_logs:
        client.post(
            "/log",
            headers={
                "X-Gitlab-Instance-Id": "ABC",
                "X-Gitlab-Global-User-Id": "DEF",
                "X-Gitlab-Host-Name": "awesome-org.com",
                "X-Gitlab-Feature-Enabled-By-Namespace-Ids": "1,2",
                "X-Gitlab-Feature-Enablement-Type": "duo_pro",
                "X-Gitlab-Realm": "saas",
                "x-gitlab-root-namespace-id": "123",
                "X-Gitlab-Version": "17.0.0",
                "X-Gitlab-Language-Server-Version": "1.2.3",
                "x-gitlab-organization-id": "456",
            },
            data={"foo": "bar"},
        )

    # Find the log entry emitted inside the request handler (not the access log)
    handler_log = next(
        (entry for entry in cap_logs if entry.get("event") == "inside request handler"),
        None,
    )
    assert handler_log is not None, "Expected log entry from request handler not found"

    # All GitLab headers should be present in the handler log entry via context vars
    assert handler_log["gitlab_instance_id"] == "ABC"
    assert handler_log["gitlab_global_user_id"] == "DEF"
    assert handler_log["gitlab_host_name"] == "awesome-org.com"
    assert handler_log["gitlab_feature_enabled_by_namespace_ids"] == "1,2"
    assert handler_log["gitlab_feature_enablement_type"] == "duo_pro"
    assert handler_log["gitlab_realm"] == "saas"
    assert handler_log["gitlab_root_namespace_id"] == "123"
    assert handler_log["gitlab_version"] == "17.0.0"
    assert handler_log["gitlab_language_server_version"] == "1.2.3"
    assert handler_log["gitlab_organization_id"] == "456"


@mock.patch("ai_gateway.api.middleware.access_log.log_exception")
def test_exception_capture(mock_log_exception):
    with capture_logs() as cap_logs, pytest.raises(RuntimeError):
        response = client.post("/", headers={}, data={"foo": "bar"})

        mock_log_exception.assert_called_once()

        assert response.status_code == 500

    assert cap_logs[0]["exception_message"] == "Something broke!"
    assert cap_logs[0]["exception_backtrace"].startswith("Traceback")


@mock.patch("ai_gateway.api.middleware.access_log.log_exception")
def test_exception_group_capture(mock_log_exception):
    with capture_logs() as cap_logs, pytest.raises(ValueError):
        response = client.post("/exception_group", headers={}, data={"foo": "bar"})

        mock_log_exception.assert_called_once()

        assert response.status_code == 500

    assert cap_logs[0]["exception_message"] == "value error in an ExceptionGroup"
    assert "Traceback" in cap_logs[0]["exception_backtrace"]
