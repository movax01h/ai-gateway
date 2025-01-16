import asyncio
import os
import socket
from typing import Iterator, cast
from unittest import mock
from unittest.mock import MagicMock, patch

import litellm
import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from ai_gateway.api import create_fast_api_server, server
from ai_gateway.api.server import (
    custom_http_exception_handler,
    model_api_exception_handler,
    setup_custom_exception_handlers,
    setup_gcp_service_account,
    validation_exception_handler,
)
from ai_gateway.config import (
    Config,
    ConfigAuth,
    ConfigGoogleCloudPlatform,
    ConfigInternalEvent,
)
from ai_gateway.container import ContainerApplication
from ai_gateway.models import ModelAPIError
from ai_gateway.structured_logging import setup_logging

_ROUTES_V1 = [
    ("/v1/chat/{chat_invokable}", ["POST"]),
    ("/v1/x-ray/libraries", ["POST"]),
]

_ROUTES_V2 = [
    ("/v2/code/completions", ["POST"]),
    ("/v2/completions", ["POST"]),  # legacy path
    ("/v2/code/generations", ["POST"]),
]

_ROUTES_V3 = [
    ("/v3/code/completions", ["POST"]),
]

_ROUTES_V4 = [
    ("/v4/code/suggestions", ["POST"]),
]


@pytest.fixture(scope="module")
def unused_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return port


@pytest.fixture
def vertex_project():
    yield "vertex-project"


@pytest.fixture
def config(vertex_project: str):
    yield Config(
        google_cloud_platform=ConfigGoogleCloudPlatform(project=vertex_project)
    )


@pytest.fixture
def app():
    return FastAPI()


@pytest.fixture(scope="session")
def auth_enabled():
    # pylint: disable=direct-environment-variable-reference
    return os.environ.get("AIGW_AUTH__BYPASS_EXTERNAL", "False") == "False"
    # pylint: enable=direct-environment-variable-reference


@pytest.fixture(scope="session")
def fastapi_server_app(auth_enabled) -> Iterator[FastAPI]:
    config = Config(_env_file=None, auth=ConfigAuth(bypass_external=not auth_enabled))
    fast_api_container = ContainerApplication()
    fast_api_container.config.from_dict(config.model_dump())
    setup_logging(config.logging)
    yield create_fast_api_server(config)


@pytest.mark.parametrize(
    "routes_expected", [_ROUTES_V1, _ROUTES_V2, _ROUTES_V3, _ROUTES_V4]
)
class TestServerRoutes:
    def test_routes_available(
        self,
        fastapi_server_app: FastAPI,
        routes_expected: list,
    ):
        routes_expected = [
            (path, method) for path, methods in routes_expected for method in methods
        ]

        routes_actual = [
            (cast(APIRoute, route).path, method)
            for route in fastapi_server_app.routes
            for method in cast(APIRoute, route).methods
        ]

        assert set(routes_expected).issubset(routes_actual)

    def test_routes_reachable(
        self,
        fastapi_server_app: FastAPI,
        auth_enabled: bool,
        routes_expected: list,
    ):
        client = TestClient(fastapi_server_app)

        routes_expected = [
            (path, method) for path, methods in routes_expected for method in methods
        ]

        for path, method in routes_expected:
            res = client.request(method, path)
            if auth_enabled:
                assert res.status_code == 401
            else:
                if method == "POST":
                    # We're checking the route availability only
                    assert res.status_code == 422
                else:
                    assert False


def test_setup_router():
    app = FastAPI()
    server.setup_router(app)

    assert any(route.path == "/v1/chat/{chat_invokable}" for route in app.routes)
    assert any(route.path == "/v2/code/completions" for route in app.routes)
    assert any(route.path == "/v3/code/completions" for route in app.routes)
    assert any(route.path == "/v4/code/suggestions" for route in app.routes)
    assert any(route.path == "/monitoring/healthz" for route in app.routes)


def test_setup_prometheus_fastapi_instrumentator():
    app = FastAPI()
    server.setup_prometheus_fastapi_instrumentator(app)

    assert any(
        "Prometheus" in middleware.cls.__name__ for middleware in app.user_middleware
    )


@pytest.mark.asyncio
async def test_lifespan(config, app, unused_port, monkeypatch, vertex_project):
    mock_credentials = MagicMock()
    mock_credentials.client_id = "mocked_client_id"

    def mock_default(*args, **kwargs):
        return (mock_credentials, "mocked_project_id")

    monkeypatch.setattr("google.auth.default", mock_default)

    mock_container_app = MagicMock(spec=ContainerApplication)
    monkeypatch.setattr(
        "ai_gateway.api.server.ContainerApplication", mock_container_app
    )
    monkeypatch.setattr(asyncio, "get_running_loop", MagicMock())

    config.fastapi.metrics_port = unused_port

    app.extra = {"extra": {"config": config}}

    async with server.lifespan(app):
        mock_container_app.assert_called_once()
        mock_container_app.return_value.config.from_dict.assert_called_once_with(
            config.model_dump()
        )

        if config.instrumentator.thread_monitoring_enabled:
            asyncio.get_running_loop.assert_called_once()

        assert litellm.vertex_project == vertex_project


def test_middleware_authentication(fastapi_server_app: FastAPI, auth_enabled: bool):
    client = TestClient(fastapi_server_app)

    response = client.post("/v1/chat/agent")
    if auth_enabled:
        assert response.status_code == 401
    else:
        assert response.status_code == 422

    response = client.get("/monitoring/healthz")
    assert response.status_code == 200


def test_middleware_log_request(fastapi_server_app: FastAPI, caplog):
    client = TestClient(fastapi_server_app)

    with caplog.at_level("INFO"):
        client.post("/v1/chat/agent")
        log_messages = [record.message for record in caplog.records]
        assert any("correlation_id" in msg for msg in log_messages)

    caplog.clear()


@pytest.mark.parametrize(
    "test_path,expected", [("/v1/chat/agent", True), ("/monitoring/healthz", False)]
)
def test_middleware_internal_event(fastapi_server_app: FastAPI, test_path, expected):
    config = Config(
        _env_file=None,
        auth=ConfigAuth(bypass_external=True),
        internal_event=ConfigInternalEvent(enabled=True),
    )
    server = create_fast_api_server(config)
    client = TestClient(server)

    with patch("ai_gateway.api.middleware.current_event_context") as mock_event_context:
        client.post(test_path)
        if expected:
            mock_event_context.set.assert_called_once()
        else:
            mock_event_context.set.assert_not_called()


@pytest.mark.parametrize(
    "test_path,expected", [("/v1/chat/agent", True), ("/monitoring/healthz", False)]
)
def test_middleware_distributed_trace(fastapi_server_app: FastAPI, test_path, expected):
    config = Config(
        _env_file=None,
        auth=ConfigAuth(bypass_external=True),
        environment="development",
    )
    server = create_fast_api_server(config)
    client = TestClient(server)

    with patch("ai_gateway.api.middleware.tracing_context") as mock_tracing_context:
        client.post(
            test_path,
            headers={
                "langsmith-trace": "20240808T090953171943Z18dfa1db-1dfc-4a48-aaf8-a139960955ce"
            },
        )
        if expected:
            mock_tracing_context.assert_called_once()
        else:
            mock_tracing_context.assert_not_called()


def test_middleware_feature_flag(fastapi_server_app: FastAPI):
    config = Config(
        _env_file=None,
        auth=ConfigAuth(bypass_external=True),
    )
    server = create_fast_api_server(config)
    client = TestClient(server)

    with patch(
        "ai_gateway.api.middleware.current_feature_flag_context"
    ) as mock_feature_flag_context:
        client.post(
            "/v1/chat/agent",
            headers={"x-gitlab-enabled-feature-flags": "feature_a,feature_b"},
        )
        mock_feature_flag_context.set.assert_called_once_with(
            {"feature_a", "feature_b"}
        )


def test_setup_custom_exception_handlers(app, monkeypatch):
    mock_add_exception_handler = MagicMock()
    monkeypatch.setattr(app, "add_exception_handler", mock_add_exception_handler)

    setup_custom_exception_handlers(app)

    assert mock_add_exception_handler.mock_calls == [
        mock.call(StarletteHTTPException, custom_http_exception_handler),
        mock.call(ModelAPIError, model_api_exception_handler),
        mock.call(RequestValidationError, validation_exception_handler),
    ]


def test_custom_http_exception_handler(app):
    @app.get("/test")
    def test_route():
        raise StarletteHTTPException(status_code=400, detail="Test Exception")

    setup_custom_exception_handlers(app)

    client = TestClient(app)

    with patch("ai_gateway.api.server.context") as mock_context:
        response = client.get("/test")

        mock_context.__setitem__.assert_called_once_with(
            "http_exception_details", "400: Test Exception"
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "Test Exception"}


def test_model_exception_handler(app):
    @app.get("/test")
    def test_route():
        raise ModelAPIError("model call failed")

    setup_custom_exception_handlers(app)

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 503
    assert response.json() == {"detail": "Inference failed"}


@pytest.mark.parametrize(
    ("service_account_json_key", "should_create_cred_file"),
    [
        (
            "",
            False,
        ),
        (
            '{ "type": "service_account" }',
            True,
        ),
    ],
)
def test_setup_gcp_service_account(service_account_json_key, should_create_cred_file):
    config = MagicMock(Config)
    google_cloud_platform = ConfigGoogleCloudPlatform
    config.google_cloud_platform = google_cloud_platform
    google_cloud_platform.service_account_json_key = service_account_json_key
    setup_gcp_service_account(config=config)

    if should_create_cred_file:
        # pylint: disable=direct-environment-variable-reference
        with open("/tmp/gcp-service-account.json", "r") as f:
            assert f.read() == service_account_json_key
        assert (
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
            == "/tmp/gcp-service-account.json"
        )
        # Cleanup
        os.remove("/tmp/gcp-service-account.json")
        del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        # pylint: enable=direct-environment-variable-reference
    else:
        assert not os.path.exists("/tmp/gcp-service-account.json")


@patch("ai_gateway.api.server.context")
@patch("ai_gateway.api.server.is_feature_enabled")
def test_validation_exception_handler_without_expanded_logging_ff(
    mock_is_feature_enabled, mock_context, app
):
    @app.post("/test")
    def test_route(required_field: str):
        return {"message": "success"}

    setup_custom_exception_handlers(app)
    mock_is_feature_enabled.return_value = False
    client = TestClient(app)

    response = client.post("/test", json={})
    assert response.status_code == 422
    assert response.json() == {"detail": "Validation error"}
    assert not mock_context.__setitem__.called


@patch("ai_gateway.api.server.context")
@patch("ai_gateway.api.server.is_feature_enabled")
def test_validation_exception_handler_with_expanded_logging_ff(
    mock_is_feature_enabled, mock_context, app
):
    @app.post("/test")
    def test_route(required_field: str):
        return {"message": "success"}

    setup_custom_exception_handlers(app)
    mock_is_feature_enabled.return_value = True
    client = TestClient(app)

    response = client.post("/test", json={})
    assert response.status_code == 422
    assert "required_field" in str(response.json()["detail"])
    mock_context.__setitem__.assert_called_once()
    error_message = mock_context.__setitem__.call_args[0][1]
    assert "required_field" in error_message
