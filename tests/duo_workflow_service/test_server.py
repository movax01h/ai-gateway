# pylint: disable=direct-environment-variable-reference

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest
from langchain.globals import get_llm_cache
from langchain_community.cache import SQLiteCache

from duo_workflow_service.internal_events.client import DuoWorkflowInternalEvent
from duo_workflow_service.server import configure_cache, run, start_servers


@pytest.fixture
def mock_env_vars():
    original_env = dict(os.environ)
    os.environ.clear()
    os.environ.update(
        {"PORT": "50052", "WEBSOCKET_SERVER": "false", "LLM_CACHE": "false"}
    )
    yield
    os.environ.clear()
    os.environ.update(original_env)


def test_configure_cache_disabled():
    with patch.dict(os.environ, {"LLM_CACHE": "false"}):
        configure_cache()
        assert get_llm_cache() is None


def test_configure_cache_enabled():
    with patch.dict(os.environ, {"LLM_CACHE": "true"}):
        configure_cache()
        cache = get_llm_cache()
        assert isinstance(cache, SQLiteCache)
        assert cache is not None


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_env_vars")
async def test_start_servers_grpc_only():
    with patch("duo_workflow_service.server.grpc_serve") as mock_grpc_serve, patch(
        "duo_workflow_service.server.websocket_serve"
    ) as mock_websocket_serve:

        mock_grpc_serve.return_value = None

        await start_servers()

        mock_grpc_serve.assert_called_once_with(50052)
        mock_websocket_serve.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_env_vars")
async def test_start_servers_with_websocket():
    with patch.dict(
        os.environ, {"WEBSOCKET_SERVER": "true", "WEBSOCKET_PORT": "8080"}
    ), patch("duo_workflow_service.server.grpc_serve") as mock_grpc_serve, patch(
        "duo_workflow_service.server.websocket_serve"
    ) as mock_websocket_serve:

        mock_grpc_serve.return_value = None
        mock_websocket_serve.return_value = None

        await start_servers()

        mock_grpc_serve.assert_called_once_with(50052)
        mock_websocket_serve.assert_called_once_with(8080)


@pytest.mark.asyncio
async def test_start_servers_with_custom_ports():
    with patch.dict(
        os.environ,
        {
            "PORT": "5000",
            "WEBSOCKET_SERVER": "TRUE",  # Test case insensitive
            "WEBSOCKET_PORT": "9000",
        },
    ), patch("duo_workflow_service.server.grpc_serve") as mock_grpc_serve, patch(
        "duo_workflow_service.server.websocket_serve"
    ) as mock_websocket_serve:

        mock_grpc_serve.return_value = None
        mock_websocket_serve.return_value = None

        await start_servers()

        mock_grpc_serve.assert_called_once_with(5000)
        mock_websocket_serve.assert_called_once_with(9000)


@pytest.mark.usefixtures("mock_env_vars")
def test_run():
    with patch(
        "duo_workflow_service.server.setup_profiling"
    ) as mock_setup_profiling, patch(
        "duo_workflow_service.server.setup_error_tracking"
    ) as mock_setup_error_tracking, patch(
        "duo_workflow_service.server.setup_monitoring"
    ) as mock_setup_monitoring, patch(
        "duo_workflow_service.server.setup_logging"
    ) as mock_setup_logging, patch(
        "duo_workflow_service.server.validate_llm_access"
    ) as mock_validate_llm_access, patch.object(
        DuoWorkflowInternalEvent, "setup"
    ) as mock_internal_event_setup, patch(
        "duo_workflow_service.server.start_servers", autospec=True
    ) as mock_start_servers, patch(
        "asyncio.get_event_loop"
    ) as mock_get_loop:

        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop

        run()

        mock_setup_profiling.assert_called_once()
        mock_setup_error_tracking.assert_called_once()
        mock_setup_monitoring.assert_called_once()
        mock_setup_logging.assert_called_once_with(json_format=True, to_file=None)
        mock_validate_llm_access.assert_called_once()
        mock_internal_event_setup.assert_called_once()

        mock_start_servers.assert_called_once()

        assert mock_loop.run_until_complete.call_count == 1
        actual_arg = mock_loop.run_until_complete.call_args[0][0]
        assert asyncio.iscoroutine(actual_arg)
