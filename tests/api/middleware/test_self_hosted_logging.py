# create tests for EnabledInstanceVerboseAiLogsHeaderPlugin

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from starlette_context import context
from starlette_context.middleware import ContextMiddleware

from ai_gateway.api.middleware.self_hosted_logging import (
    HEADER_KEY,
    EnabledInstanceVerboseAiLogsHeaderPlugin,
)


class TestEnabledInstanceVerboseAiLogsHeaderPlugin:

    @pytest.fixture(scope="class")
    def plugin(self):
        return EnabledInstanceVerboseAiLogsHeaderPlugin()

    @pytest.fixture
    def headers(self):
        return []

    @pytest.fixture
    def mock_request(self, headers):
        return Request(
            {
                "type": "http",
                "headers": headers,
                "method": "GET",
                "path": "/",
            }
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("headers", [[(HEADER_KEY.encode(), b"true")]])
    async def test_process_request_with_header(self, mock_request, plugin):
        result = await plugin.process_request(mock_request)
        assert result

    @pytest.mark.asyncio
    async def test_process_request_without_header(self, mock_request, plugin):
        result = await plugin.process_request(mock_request)
        assert not result

    def test_plugin_key(self, plugin):
        # Verify the plugin key is set correctly
        assert plugin.key == "enabled-instance-verbose-ai-logs"


class TestEnabledInstanceVerboseAiLogsHeader:
    @pytest.fixture
    def client(self):
        async def test_endpoint(_request):
            return JSONResponse(context.data)

        app = Starlette(
            routes=[Route("/test", test_endpoint)],
            middleware=[
                Middleware(
                    ContextMiddleware,
                    plugins=(EnabledInstanceVerboseAiLogsHeaderPlugin(),),
                )
            ],
        )

        return TestClient(app)

    def test_enabled_instance_verbose_ai_logs_set(self, client):
        response = client.get("/test", headers={HEADER_KEY: "true"})

        assert response.json()["enabled-instance-verbose-ai-logs"]

    def test_enabled_instance_verbose_ai_logs_not_set(self, client):
        response = client.get("/test")

        assert not response.json()["enabled-instance-verbose-ai-logs"]
