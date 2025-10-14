import asyncio
import os
import socket
from typing import cast
from unittest import mock
from unittest.mock import MagicMock, patch

import litellm
import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException
from structlog.testing import capture_logs

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
from ai_gateway.models.base import ModelAPICallError
from ai_gateway.structured_logging import setup_logging

_ROUTES_V1 = [
    ("/v1/chat/{chat_invokable}", ["POST"]),  # legacy path
    ("/v1/x-ray/libraries", ["POST"]),
]

_ROUTES_V2 = [
    ("/v2/code/completions", ["POST"]),
    ("/v2/completions", ["POST"]),  # legacy path
    ("/v2/code/generations", ["POST"]),
    ("/v2/chat/agent", ["POST"]),
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


@pytest.fixture(name="config")
def config_fixture(vertex_project: str):
    return Config(
        google_cloud_platform=ConfigGoogleCloudPlatform(project=vertex_project)
    )


@pytest.fixture(name="app")
def app_fixture():
    return FastAPI()


@pytest.fixture(name="auth_enabled", scope="session")
def auth_enabled_fixture():
    # pylint: disable=direct-environment-variable-reference
    return os.environ.get("AIGW_AUTH__BYPASS_EXTERNAL", "False") == "False"
    # pylint: enable=direct-environment-variable-reference


@pytest.fixture(name="fastapi_server_app", scope="session")
def fastapi_server_app_fixture(auth_enabled) -> FastAPI:
    config = Config(_env_file=None, auth=ConfigAuth(bypass_external=not auth_enabled))
    fast_api_container = ContainerApplication()
    fast_api_container.wire(
        modules=[
            "ai_gateway.api.v1.x_ray.libraries",
            "ai_gateway.api.v1.chat.agent",
            "ai_gateway.api.v1.search.docs",
            "ai_gateway.api.v2.code.completions",
            "ai_gateway.api.v3.code.completions",
            "ai_gateway.api.v4.code.suggestions",
            "ai_gateway.api.server",
            "ai_gateway.api.monitoring",
            "ai_gateway.async_dependency_resolver",
        ],
    )

    fast_api_container.config.from_dict(config.model_dump())
    setup_logging(
        config.logging, custom_models_enabled=False, cache_logger_on_first_use=False
    )
    return create_fast_api_server(config)


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
    assert any(route.path == "/v1/models/definitions" for route in app.routes)
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

    def mock_default(*_args, **_kwargs):
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


def test_cloud_connector_auth_provider_in_app_state():
    config = Config(_env_file=None, auth=ConfigAuth(bypass_external=True))

    with patch("gitlab_cloud_connector.cloud_connector_ready", return_value=True):
        app = create_fast_api_server(config)

    assert hasattr(app.state, "cloud_connector_auth_provider")
    assert app.state.cloud_connector_auth_provider is not None


def test_middleware_authentication(fastapi_server_app: FastAPI, auth_enabled: bool):
    client = TestClient(fastapi_server_app)

    response = client.post("/v2/chat/agent")
    if auth_enabled:
        assert response.status_code == 401
    else:
        assert response.status_code == 422

    response = client.get("/monitoring/healthz")
    assert response.status_code == 200

    response = client.get("/v1/models/definitions")
    assert response.status_code == 200


def test_middleware_log_request(fastapi_server_app: FastAPI):
    client = TestClient(fastapi_server_app)

    with capture_logs() as cap_logs:
        client.post("/v2/chat/agent")
        correlation_ids = [log.get("correlation_id") for log in cap_logs]
        assert len(correlation_ids) > 0


@pytest.mark.usefixtures("fastapi_server_app")
@pytest.mark.parametrize(
    "test_path,expected", [("/v2/chat/agent", True), ("/monitoring/healthz", False)]
)
def test_middleware_internal_event(test_path, expected):
    config = Config(
        _env_file=None,
        auth=ConfigAuth(bypass_external=True),
        internal_event=ConfigInternalEvent(enabled=True),
    )
    app = create_fast_api_server(config)
    client = TestClient(app)

    with patch(
        "ai_gateway.api.middleware.internal_event.current_event_context"
    ) as mock_event_context:
        client.post(test_path)
        if expected:
            mock_event_context.set.assert_called_once()
        else:
            mock_event_context.set.assert_not_called()


@pytest.mark.usefixtures("fastapi_server_app")
@pytest.mark.parametrize(
    "test_path,expected", [("/v2/chat/agent", True), ("/monitoring/healthz", False)]
)
def test_middleware_distributed_trace_in_development(test_path, expected):
    """Test that distributed tracing works in development environment with langsmith-trace header."""
    config = Config(
        _env_file=None,
        auth=ConfigAuth(bypass_external=True),
        environment="development",
    )
    # pylint: disable=direct-environment-variable-reference
    with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"}):
        app = create_fast_api_server(config)
        client = TestClient(app)
        with patch(
            "ai_gateway.api.middleware.distributed_trace.tracing_context"
        ) as mock_tracing_context:
            client.post(
                test_path,
                headers={
                    "langsmith-trace": "20240808T090953171943Z18dfa1db-1dfc-4a48-aaf8-a139960955ce"
                },
            )
            # pylint: enable=direct-environment-variable-reference
            if expected:
                # Should be called with enabled=True in development
                mock_tracing_context.assert_called_once()
                call_args = mock_tracing_context.call_args
                assert call_args.kwargs["enabled"] is True
            else:
                mock_tracing_context.assert_not_called()


@pytest.mark.usefixtures("fastapi_server_app")
@pytest.mark.parametrize("test_path", ["/v2/chat/agent", "/monitoring/healthz"])
def test_middleware_distributed_trace_disabled_in_non_development(test_path):
    """Test that distributed tracing is disabled in non-development environments."""
    config = Config(
        _env_file=None,
        auth=ConfigAuth(bypass_external=True),
        environment="production",
    )
    # pylint: disable=direct-environment-variable-reference
    with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"}):
        app = create_fast_api_server(config)
        client = TestClient(app)
        with patch(
            "ai_gateway.api.middleware.distributed_trace.tracing_context"
        ) as mock_tracing_context:
            client.post(
                test_path,
                headers={
                    "langsmith-trace": "20240808T090953171943Z18dfa1db-1dfc-4a48-aaf8-a139960955ce"
                },
            )
            # pylint: enable=direct-environment-variable-reference
            if test_path == "/v2/chat/agent":
                # tracing_context should be called but with enabled=False in non-development environments
                mock_tracing_context.assert_called_once()
                call_args = mock_tracing_context.call_args
                assert call_args.kwargs["enabled"] is False
            else:
                # Health endpoint should skip tracing entirely
                mock_tracing_context.assert_not_called()


@pytest.mark.usefixtures("fastapi_server_app")
@pytest.mark.parametrize("test_path", ["/v2/chat/agent", "/monitoring/healthz"])
def test_middleware_distributed_trace_disabled_without_header(test_path):
    """Test that distributed tracing is disabled without langsmith-trace header."""
    config = Config(
        _env_file=None,
        auth=ConfigAuth(bypass_external=True),
        environment="development",
    )
    # pylint: disable=direct-environment-variable-reference
    with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"}):
        app = create_fast_api_server(config)
        client = TestClient(app)
        with patch(
            "ai_gateway.api.middleware.distributed_trace.tracing_context"
        ) as mock_tracing_context:
            client.post(test_path)  # No langsmith-trace header
            # pylint: enable=direct-environment-variable-reference
            if test_path == "/v2/chat/agent":
                # tracing_context should be called but with enabled=True and parent=None in development
                mock_tracing_context.assert_called_once_with(parent=None, enabled=True)
            else:
                # Health endpoint should skip tracing entirely
                mock_tracing_context.assert_not_called()


@pytest.mark.usefixtures("fastapi_server_app")
def test_middleware_feature_flag():
    config = Config(
        _env_file=None,
        auth=ConfigAuth(bypass_external=True),
    )
    app = create_fast_api_server(config)
    client = TestClient(app)

    with patch(
        "ai_gateway.api.middleware.feature_flag.current_feature_flag_context"
    ) as mock_feature_flag_context:
        client.post(
            "/v2/chat/agent",
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


def test_model_exception_handler_with_429_error(app):
    @app.get("/test")
    def test_route():
        class TestTooManyRequestsError(ModelAPICallError):
            code = 429

        error = TestTooManyRequestsError("Too many requests")
        raise error

    setup_custom_exception_handlers(app)

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 429
    assert response.json() == {"detail": "Too many requests. Please try again later."}


def test_model_exception_handler_propagates_retry_after_header(app):
    @app.get("/test")
    def test_route():
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "30"}

        original_error = MagicMock()
        original_error.response = mock_response

        class TestTooManyRequestsError(ModelAPICallError):
            code = 429

        too_many_requests_error = TestTooManyRequestsError(
            "Too many requests", errors=(original_error,)
        )
        raise too_many_requests_error

    setup_custom_exception_handlers(app)

    client = TestClient(app)
    response = client.get("/test")

    assert "Retry-After" in response.headers
    assert response.headers["Retry-After"] == "30"


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
@patch("ai_gateway.api.server.can_log_request_data")
def test_validation_exception_handler_without_log_request_data(
    mock_can_log_request_data, mock_context, app
):
    @app.post("/test")
    def test_route(_required_field: str):
        return {"message": "success"}

    setup_custom_exception_handlers(app)
    mock_can_log_request_data.return_value = False
    client = TestClient(app)

    response = client.post("/test", json={})
    assert response.status_code == 422
    assert response.json() == {"detail": "Validation error"}
    assert not mock_context.__setitem__.called


@patch("ai_gateway.api.server.context")
@patch("ai_gateway.api.server.can_log_request_data")
def test_validation_exception_handler_with_log_request_data(
    mock_can_log_request_data, mock_context, app
):
    @app.post("/test")
    def test_route(_required_field: str):
        return {"message": "success"}

    setup_custom_exception_handlers(app)
    mock_can_log_request_data.return_value = True
    client = TestClient(app)

    response = client.post("/test", json={})
    assert response.status_code == 422
    assert "required_field" in str(response.json()["detail"])
    mock_context.__setitem__.assert_called_once()
    error_message = mock_context.__setitem__.call_args[0][1]
    assert "required_field" in error_message
