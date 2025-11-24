from unittest.mock import AsyncMock, MagicMock

import pytest

from duo_workflow_service.interceptors.mcp_server_tools_interceptor import (
    McpServerToolsInterceptor,
)
from lib.mcp_server_tools.context import current_mcp_server_tools_context


@pytest.fixture(name="reset_context")
def reset_context_fixture():
    """Reset the context variable after each test."""
    token = current_mcp_server_tools_context.set(set())
    yield
    current_mcp_server_tools_context.reset(token)


@pytest.fixture(name="mock_handler_call_details")
def mock_handler_call_details_fixture():
    """Create a mock for the handler_call_details."""
    details = MagicMock()
    details.invocation_metadata = ()
    return details


@pytest.fixture(name="mock_continuation")
def mock_continuation_fixture():
    """Create a mock for the continuation function."""
    return AsyncMock()


@pytest.fixture(name="interceptor")
def interceptor_fixture():
    return McpServerToolsInterceptor()


class TestMcpServerToolsInterceptor:
    @pytest.mark.asyncio
    async def test_intercept_service_no_mcp_tools(
        self, reset_context, mock_handler_call_details, mock_continuation, interceptor
    ):
        """Test interceptor with no MCP server tools in metadata."""
        # Execute
        await interceptor.intercept_service(
            mock_continuation, mock_handler_call_details
        )

        # Assert
        mock_continuation.assert_called_once_with(mock_handler_call_details)
        assert current_mcp_server_tools_context.get() == set()

    @pytest.mark.asyncio
    async def test_intercept_service_with_mcp_tools(
        self, reset_context, mock_handler_call_details, mock_continuation, interceptor
    ):
        """Test interceptor with MCP server tools in metadata."""
        # Setup
        mock_handler_call_details.invocation_metadata = [
            ("x-gitlab-enabled-mcp-server-tools", "tool1,tool2,tool3"),
        ]

        # Execute
        await interceptor.intercept_service(
            mock_continuation, mock_handler_call_details
        )

        # Assert
        mock_continuation.assert_called_once_with(mock_handler_call_details)
        assert current_mcp_server_tools_context.get() == {"tool1", "tool2", "tool3"}

    @pytest.mark.asyncio
    async def test_intercept_service_with_empty_mcp_tools(
        self, reset_context, mock_handler_call_details, mock_continuation, interceptor
    ):
        """Test interceptor with empty MCP server tools string."""
        # Setup
        mock_handler_call_details.invocation_metadata = [
            ("x-gitlab-enabled-mcp-server-tools", ""),
        ]

        # Execute
        await interceptor.intercept_service(
            mock_continuation, mock_handler_call_details
        )

        # Assert
        mock_continuation.assert_called_once_with(mock_handler_call_details)
        assert current_mcp_server_tools_context.get() == set()

    @pytest.mark.asyncio
    async def test_intercept_service_with_whitespace(
        self, reset_context, mock_handler_call_details, mock_continuation, interceptor
    ):
        """Test interceptor with whitespace in MCP server tools."""
        # Setup
        mock_handler_call_details.invocation_metadata = [
            ("x-gitlab-enabled-mcp-server-tools", " tool1 , tool2 , tool3 "),
        ]

        # Execute
        await interceptor.intercept_service(
            mock_continuation, mock_handler_call_details
        )

        # Assert
        mock_continuation.assert_called_once_with(mock_handler_call_details)
        assert current_mcp_server_tools_context.get() == {"tool1", "tool2", "tool3"}

    @pytest.mark.asyncio
    async def test_intercept_service_with_empty_values(
        self, reset_context, mock_handler_call_details, mock_continuation, interceptor
    ):
        """Test interceptor filters out empty strings from split."""
        # Setup
        mock_handler_call_details.invocation_metadata = [
            ("x-gitlab-enabled-mcp-server-tools", "tool1,,tool2,,,tool3"),
        ]

        # Execute
        await interceptor.intercept_service(
            mock_continuation, mock_handler_call_details
        )

        # Assert
        mock_continuation.assert_called_once_with(mock_handler_call_details)
        assert current_mcp_server_tools_context.get() == {"tool1", "tool2", "tool3"}

    @pytest.mark.asyncio
    async def test_intercept_service_metadata_conversion(
        self, reset_context, mock_handler_call_details, mock_continuation, interceptor
    ):
        """Test that metadata tuples are correctly converted to a dict."""
        # Setup
        mock_handler_call_details.invocation_metadata = [
            ("x-gitlab-enabled-mcp-server-tools", "tool1,tool2"),
            ("x-gitlab-global-user-id", "user123"),
            ("other-header", "value"),
        ]

        # Execute
        await interceptor.intercept_service(
            mock_continuation, mock_handler_call_details
        )

        # Assert
        mock_continuation.assert_called_once_with(mock_handler_call_details)
        assert current_mcp_server_tools_context.get() == {"tool1", "tool2"}

    @pytest.mark.asyncio
    async def test_intercept_service_single_tool(
        self, reset_context, mock_handler_call_details, mock_continuation, interceptor
    ):
        """Test interceptor with a single MCP server tool."""
        # Setup
        mock_handler_call_details.invocation_metadata = [
            ("x-gitlab-enabled-mcp-server-tools", "single-tool"),
        ]

        # Execute
        await interceptor.intercept_service(
            mock_continuation, mock_handler_call_details
        )

        # Assert
        mock_continuation.assert_called_once_with(mock_handler_call_details)
        assert current_mcp_server_tools_context.get() == {"single-tool"}
