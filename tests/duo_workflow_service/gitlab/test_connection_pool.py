"""Tests for the connection pool manager."""

import ssl
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
import pytest_asyncio

from duo_workflow_service.gitlab.connection_pool import (
    ConnectionPoolManager,
    connection_pool,
)


@pytest_asyncio.fixture
async def pool_manager():
    """Create a fresh connection pool manager for each test."""
    # Reset the singleton instance
    ConnectionPoolManager._instance = None
    ConnectionPoolManager._session = None

    # Create a new instance
    manager = ConnectionPoolManager()
    yield manager

    # Cleanup
    if manager._session:
        await manager._session.close()
        manager._session = None


@pytest.mark.asyncio
async def test_singleton_pattern():
    """Test that ConnectionPoolManager follows the singleton pattern."""
    manager1 = ConnectionPoolManager()
    manager2 = ConnectionPoolManager()
    assert manager1 is manager2


@pytest.mark.asyncio
async def test_session_not_initialized_error():
    """Test that accessing session before initialization raises an error."""
    manager = ConnectionPoolManager()
    with pytest.raises(
        RuntimeError, match="HTTP client connection pool is not initialized"
    ):
        _ = manager.session


@pytest.mark.asyncio
async def test_set_options():
    """Test setting pool options."""
    manager = ConnectionPoolManager()
    custom_timeout = aiohttp.ClientTimeout(total=60)
    manager.set_options(pool_size=200, timeout=custom_timeout)

    mock_session = MagicMock(spec=aiohttp.ClientSession)
    mock_connector = MagicMock(spec=aiohttp.TCPConnector)

    with (
        patch("aiohttp.ClientSession") as mock_session_cls,
        patch("aiohttp.TCPConnector") as mock_connector_cls,
        patch(
            "duo_workflow_service.gitlab.connection_pool.os.getenv", return_value=None
        ),
    ):
        mock_connector_cls.return_value = mock_connector
        mock_session_cls.return_value = mock_session
        async with manager:
            # Verify that ClientSession was created with correct parameters
            mock_connector_cls.assert_called_once_with(limit=200)
            mock_session_cls.assert_called_once_with(
                connector=mock_connector, timeout=custom_timeout
            )

            assert manager.session is mock_session


@pytest.mark.asyncio
async def test_multiple_context_entries():
    """Test that multiple context entries reuse the same session."""
    # Create a mock session with proper async close method
    mock_session = AsyncMock()
    mock_session.close = AsyncMock()
    mock_session.close.return_value = None  # Ensure close() returns None

    with patch(
        "aiohttp.ClientSession", return_value=mock_session
    ) as mock_session_class:
        connection_pool.set_options(
            pool_size=100, timeout=aiohttp.ClientTimeout(total=30)
        )
        async with connection_pool:
            session1 = connection_pool._session

            async with connection_pool:
                session2 = connection_pool._session

                # Should be the same session
                assert session1 is session2

                # Session creation should only happen once
                assert mock_session_class.call_count == 1

        # Session should be closed only once at the end
        mock_session.close.assert_awaited_once()
        assert connection_pool._session is None


@pytest.mark.parametrize(
    "env_setup,test_description",
    [
        (
            lambda mp: mp.delenv("DUO_WORKFLOW_GITLAB_SSL_CA_FILE", raising=False),
            "no CA file configured",
        ),
        (
            lambda mp: mp.setenv("DUO_WORKFLOW_GITLAB_SSL_CA_FILE", ""),
            "empty CA file variable",
        ),
        (
            lambda mp: mp.setenv(
                "DUO_WORKFLOW_GITLAB_SSL_CA_FILE", "/nonexistent/ca.crt"
            ),
            "nonexistent CA file path",
        ),
    ],
    ids=["no_ca_file", "empty_ca_file", "nonexistent_ca_file"],
)
@pytest.mark.asyncio
async def test_ssl_context_default_verification(
    pool_manager, monkeypatch, env_setup, test_description
):
    """Test SSL behavior when falling back to default verification."""
    env_setup(monkeypatch)

    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_connector = MagicMock(spec=aiohttp.TCPConnector)

    with (
        patch(
            "aiohttp.TCPConnector", return_value=mock_connector
        ) as mock_connector_cls,
        patch("aiohttp.ClientSession", return_value=mock_session) as mock_session_cls,
        patch("duo_workflow_service.gitlab.connection_pool.log") as mock_log,
    ):
        async with pool_manager:
            mock_connector_cls.assert_called_once_with(limit=100)
            mock_session_cls.assert_called_once_with(connector=mock_connector)
            mock_log.info.assert_any_call(
                "Using default SSL verification for HTTP connection pool"
            )


@pytest.mark.asyncio
async def test_ssl_context_via_environment_valid_ca_file(pool_manager, monkeypatch):
    """Test SSL behavior when valid CA file is configured."""
    ca_file_path = "/path/to/ca.crt"
    monkeypatch.setenv("DUO_WORKFLOW_GITLAB_SSL_CA_FILE", ca_file_path)

    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_connector = MagicMock(spec=aiohttp.TCPConnector)

    with (
        patch("os.path.exists", return_value=True),
        patch("ssl.create_default_context") as mock_ssl_create,
        patch(
            "aiohttp.TCPConnector", return_value=mock_connector
        ) as mock_connector_cls,
        patch("aiohttp.ClientSession", return_value=mock_session) as mock_session_cls,
        patch("duo_workflow_service.gitlab.connection_pool.log") as mock_log,
    ):
        mock_ssl_context = MagicMock(spec=ssl.SSLContext)
        mock_ssl_create.return_value = mock_ssl_context

        async with pool_manager:
            mock_ssl_create.assert_called_once()
            mock_ssl_context.load_verify_locations.assert_called_once_with(ca_file_path)

            mock_connector_cls.assert_called_once_with(limit=100, ssl=mock_ssl_context)
            mock_session_cls.assert_called_once_with(connector=mock_connector)

            mock_log.info.assert_any_call("Loaded custom CA file", ca_file=ca_file_path)
            mock_log.info.assert_any_call("SSL context created with custom CA")
            mock_log.info.assert_any_call("Using custom SSL context with custom CA")


@pytest.mark.asyncio
async def test_ssl_context_via_environment_ca_file_load_error(
    pool_manager, monkeypatch
):
    """Test SSL behavior when CA file loading fails."""
    ca_file_path = "/path/to/ca.crt"
    monkeypatch.setenv("DUO_WORKFLOW_GITLAB_SSL_CA_FILE", ca_file_path)

    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_connector = MagicMock(spec=aiohttp.TCPConnector)

    with (
        patch("os.path.exists", return_value=True),
        patch("ssl.create_default_context") as mock_ssl_create,
        patch(
            "aiohttp.TCPConnector", return_value=mock_connector
        ) as mock_connector_cls,
        patch("aiohttp.ClientSession", return_value=mock_session) as mock_session_cls,
        patch("duo_workflow_service.gitlab.connection_pool.log") as mock_log,
    ):
        mock_ssl_context = MagicMock(spec=ssl.SSLContext)
        mock_ssl_context.load_verify_locations.side_effect = Exception("SSL load error")
        mock_ssl_create.return_value = mock_ssl_context

        async with pool_manager:
            mock_ssl_create.assert_called_once()
            mock_ssl_context.load_verify_locations.assert_called_once_with(ca_file_path)

            mock_connector_cls.assert_called_once_with(limit=100)
            mock_session_cls.assert_called_once_with(connector=mock_connector)

            mock_log.error.assert_called_once_with(
                "Failed to create SSL context with custom CA",
                error="SSL load error",
                ca_file=ca_file_path,
            )
