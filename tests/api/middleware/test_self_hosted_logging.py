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
    EnabledInstanceVerboseAiLogsHeaderPlugin,
)
from lib.verbose_ai_logs import VERBOSE_AI_LOGS_HEADER


class TestEnabledInstanceVerboseAiLogsHeaderPlugin:

    @pytest.fixture(name="plugin", scope="class")
    def plugin_fixture(self):
        return EnabledInstanceVerboseAiLogsHeaderPlugin()

    @pytest.fixture(name="headers")
    def headers_fixture(self):
        return []

    @pytest.fixture(name="mock_request")
    def mock_request_fixture(self, headers):
        return Request(
            {
                "type": "http",
                "headers": headers,
                "method": "GET",
                "path": "/",
            }
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("headers", [[(VERBOSE_AI_LOGS_HEADER.encode(), b"true")]])
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
    @pytest.fixture(name="client")
    def client_fixture(self):
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
        response = client.get("/test", headers={VERBOSE_AI_LOGS_HEADER: "true"})

        assert response.json()["enabled-instance-verbose-ai-logs"]

    def test_enabled_instance_verbose_ai_logs_not_set(self, client):
        response = client.get("/test")

        assert not response.json()["enabled-instance-verbose-ai-logs"]
