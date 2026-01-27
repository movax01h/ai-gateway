from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from duo_workflow_service.interceptors.metadata_context_interceptor import (
    MetadataContextInterceptor,
)
from lib.mcp_server_tools.context import current_mcp_server_tools_context
from lib.self_hosted_dap_billing_context import (
    X_GITLAB_SELF_HOSTED_DAP_BILLING_ENABLED,
    current_self_hosted_dap_billing_enabled,
)


@pytest.mark.asyncio
async def test_client_type_header():
    """Test that client type header is properly set."""
    interceptor = MetadataContextInterceptor()
    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        ("x-gitlab-client-type", "node-grpc"),
    ]

    continuation = AsyncMock()
    continuation.return_value = "mocked_response"

    with patch(
        "duo_workflow_service.interceptors.metadata_context_interceptor.client_type"
    ) as mock_client_type:
        result = await interceptor.intercept_service(continuation, handler_call_details)

        mock_client_type.set.assert_called_once_with("node-grpc")
        continuation.assert_called_once_with(handler_call_details)
        assert result == "mocked_response"


@pytest.mark.asyncio
async def test_gitlab_realm_header():
    """Test that GitLab realm header is properly set."""
    interceptor = MetadataContextInterceptor()
    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        ("x-gitlab-realm", "saas"),
    ]

    continuation = AsyncMock()
    continuation.return_value = "mocked_response"

    with patch(
        "duo_workflow_service.interceptors.metadata_context_interceptor.gitlab_realm"
    ) as mock_gitlab_realm:
        result = await interceptor.intercept_service(continuation, handler_call_details)

        mock_gitlab_realm.set.assert_called_once_with("saas")
        continuation.assert_called_once_with(handler_call_details)
        assert result == "mocked_response"


@pytest.mark.asyncio
async def test_gitlab_version_header():
    """Test that GitLab version header is properly set."""
    interceptor = MetadataContextInterceptor()
    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        ("x-gitlab-version", "16.5.0"),
    ]

    continuation = AsyncMock()
    continuation.return_value = "mocked_response"

    with patch(
        "duo_workflow_service.interceptors.metadata_context_interceptor.gitlab_version"
    ) as mock_gitlab_version:
        result = await interceptor.intercept_service(continuation, handler_call_details)

        mock_gitlab_version.set.assert_called_once_with("16.5.0")
        continuation.assert_called_once_with(handler_call_details)
        assert result == "mocked_response"


@pytest.mark.asyncio
async def test_language_server_version_header():
    """Test that language server version header is properly set with transformation."""
    interceptor = MetadataContextInterceptor()
    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        ("x-gitlab-language-server-version", "1.2.3"),
    ]

    continuation = AsyncMock()
    continuation.return_value = "mocked_response"

    with (
        patch(
            "duo_workflow_service.interceptors.metadata_context_interceptor.language_server_version_context"
        ) as mock_lsp_version,
        patch(
            "duo_workflow_service.interceptors.metadata_context_interceptor.LanguageServerVersion"
        ) as mock_lsp_class,
    ):
        mock_lsp_instance = MagicMock()
        mock_lsp_class.from_string.return_value = mock_lsp_instance

        result = await interceptor.intercept_service(continuation, handler_call_details)

        mock_lsp_class.from_string.assert_called_once_with("1.2.3")
        mock_lsp_version.set.assert_called_once_with(mock_lsp_instance)
        continuation.assert_called_once_with(handler_call_details)
        assert result == "mocked_response"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "header_value,expected_bool",
    [
        ("true", True),
        ("false", False),
        ("True", False),
        ("", False),
    ],
)
async def test_verbose_ai_logs_header(header_value, expected_bool):
    """Test that verbose AI logs header is properly converted to boolean."""
    interceptor = MetadataContextInterceptor()
    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        ("x-gitlab-enabled-instance-verbose-ai-logs", header_value),
    ]

    continuation = AsyncMock()
    continuation.return_value = "mocked_response"

    with patch(
        "duo_workflow_service.interceptors.metadata_context_interceptor.current_verbose_ai_logs_context"
    ) as mock_verbose_logs:
        result = await interceptor.intercept_service(continuation, handler_call_details)

        mock_verbose_logs.set.assert_called_once_with(expected_bool)
        continuation.assert_called_once_with(handler_call_details)
        assert result == "mocked_response"


@pytest.mark.asyncio
async def test_prompt_caching_header():
    """Test that prompt caching header calls the setter function."""
    interceptor = MetadataContextInterceptor()
    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        ("x-gitlab-model-prompt-cache-enabled", "true"),
    ]

    continuation = AsyncMock()
    continuation.return_value = "mocked_response"

    with patch(
        "duo_workflow_service.interceptors.metadata_context_interceptor.set_prompt_caching_enabled_to_current_request"
    ) as mock_setter:
        result = await interceptor.intercept_service(continuation, handler_call_details)

        mock_setter.assert_called_once_with("true")
        continuation.assert_called_once_with(handler_call_details)
        assert result == "mocked_response"


@pytest.mark.asyncio
async def test_all_headers_together():
    """Test that all headers are processed correctly when present together."""
    interceptor = MetadataContextInterceptor()
    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        ("x-gitlab-client-type", "node-grpc"),
        ("x-gitlab-realm", "saas"),
        ("x-gitlab-version", "16.5.0"),
        ("x-gitlab-language-server-version", "1.2.3"),
        ("x-gitlab-enabled-instance-verbose-ai-logs", "true"),
        ("x-gitlab-model-prompt-cache-enabled", "false"),
        ("x-gitlab-enabled-mcp-server-tools", "postgres,context7"),
    ]

    continuation = AsyncMock()
    continuation.return_value = "mocked_response"

    with (
        patch(
            "duo_workflow_service.interceptors.metadata_context_interceptor.client_type"
        ) as mock_client_type,
        patch(
            "duo_workflow_service.interceptors.metadata_context_interceptor.gitlab_realm"
        ) as mock_gitlab_realm,
        patch(
            "duo_workflow_service.interceptors.metadata_context_interceptor.gitlab_version"
        ) as mock_gitlab_version,
        patch(
            "duo_workflow_service.interceptors.metadata_context_interceptor.language_server_version_context"
        ) as mock_lsp_version,
        patch(
            "duo_workflow_service.interceptors.metadata_context_interceptor.LanguageServerVersion"
        ) as mock_lsp_class,
        patch(
            "duo_workflow_service.interceptors.metadata_context_interceptor.current_verbose_ai_logs_context"
        ) as mock_verbose_logs,
        patch(
            "duo_workflow_service.interceptors.metadata_context_interceptor.set_prompt_caching_enabled_to_current_request"
        ) as mock_prompt_caching,
    ):
        mock_lsp_instance = MagicMock()
        mock_lsp_class.from_string.return_value = mock_lsp_instance

        result = await interceptor.intercept_service(continuation, handler_call_details)

        mock_client_type.set.assert_called_once_with("node-grpc")
        mock_gitlab_realm.set.assert_called_once_with("saas")
        mock_gitlab_version.set.assert_called_once_with("16.5.0")
        mock_lsp_class.from_string.assert_called_once_with("1.2.3")
        mock_lsp_version.set.assert_called_once_with(mock_lsp_instance)
        mock_verbose_logs.set.assert_called_once_with(True)
        mock_prompt_caching.assert_called_once_with("false")
        assert current_mcp_server_tools_context.get() == {
            "postgres",
            "context7",
        }
        continuation.assert_called_once_with(handler_call_details)
        assert result == "mocked_response"


@pytest.mark.asyncio
async def test_missing_headers():
    """Test that missing headers don't cause errors."""
    interceptor = MetadataContextInterceptor()
    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        ("other-header", "other-value"),
    ]

    continuation = AsyncMock()
    continuation.return_value = "mocked_response"

    with (
        patch(
            "duo_workflow_service.interceptors.metadata_context_interceptor.client_type"
        ) as mock_client_type,
        patch(
            "duo_workflow_service.interceptors.metadata_context_interceptor.gitlab_realm"
        ) as mock_gitlab_realm,
        patch(
            "duo_workflow_service.interceptors.metadata_context_interceptor.gitlab_version"
        ) as mock_gitlab_version,
        patch(
            "duo_workflow_service.interceptors.metadata_context_interceptor.language_server_version_context"
        ) as mock_lsp_version,
        patch(
            "duo_workflow_service.interceptors.metadata_context_interceptor.current_verbose_ai_logs_context"
        ) as mock_verbose_logs,
        patch(
            "duo_workflow_service.interceptors.metadata_context_interceptor.set_prompt_caching_enabled_to_current_request"
        ) as mock_prompt_caching,
        patch(
            "duo_workflow_service.interceptors.metadata_context_interceptor.set_self_hosted_dap_billing_enabled"
        ) as mock_self_hosted_billing,
    ):
        result = await interceptor.intercept_service(continuation, handler_call_details)

        # None of the context setters should be called for missing headers
        mock_client_type.set.assert_not_called()
        mock_gitlab_realm.set.assert_not_called()
        mock_gitlab_version.set.assert_not_called()
        mock_lsp_version.set.assert_not_called()
        # Verbose logs is always set (defaults to False)
        mock_verbose_logs.set.assert_called_once_with(False)
        # Prompt caching is always called (with None)
        mock_prompt_caching.assert_called_once_with(None)
        # Self-hosted DAP billing is always called (with None)
        mock_self_hosted_billing.assert_called_once_with("")
        # MCP server tools should be empty set when header is missing
        assert current_mcp_server_tools_context.get() == set()
        continuation.assert_called_once_with(handler_call_details)
        assert result == "mocked_response"


@pytest.mark.asyncio
async def test_self_hosted_dap_billing_header():
    """Test that self-hosted DAP billing header is properly stored in context."""
    interceptor = MetadataContextInterceptor()
    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        (X_GITLAB_SELF_HOSTED_DAP_BILLING_ENABLED, "true"),
    ]

    continuation = AsyncMock()
    continuation.return_value = "mocked_response"

    result = await interceptor.intercept_service(continuation, handler_call_details)

    assert current_self_hosted_dap_billing_enabled.get() is True
    continuation.assert_called_once_with(handler_call_details)
    assert result == "mocked_response"


@pytest.mark.asyncio
async def test_empty_header_values():
    """Test that empty header values are handled correctly."""
    interceptor = MetadataContextInterceptor()
    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        ("x-gitlab-client-type", ""),
        ("x-gitlab-realm", ""),
        ("x-gitlab-version", ""),
        ("x-gitlab-enabled-mcp-server-tools", ""),
    ]

    continuation = AsyncMock()
    continuation.return_value = "mocked_response"

    with (
        patch(
            "duo_workflow_service.interceptors.metadata_context_interceptor.client_type"
        ) as mock_client_type,
        patch(
            "duo_workflow_service.interceptors.metadata_context_interceptor.gitlab_realm"
        ) as mock_gitlab_realm,
        patch(
            "duo_workflow_service.interceptors.metadata_context_interceptor.gitlab_version"
        ) as mock_gitlab_version,
    ):
        result = await interceptor.intercept_service(continuation, handler_call_details)

        # Empty strings should not trigger set calls (walrus operator filters them out)
        mock_client_type.set.assert_not_called()
        mock_gitlab_realm.set.assert_not_called()
        mock_gitlab_version.set.assert_not_called()
        # Empty MCP server tools string should result in empty set
        assert current_mcp_server_tools_context.get() == set()
        continuation.assert_called_once_with(handler_call_details)
        assert result == "mocked_response"


@pytest.mark.asyncio
async def test_mcp_server_tools_header():
    """Test that MCP server tools header is properly parsed."""
    interceptor = MetadataContextInterceptor()
    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        ("x-gitlab-enabled-mcp-server-tools", "tool1,tool2,tool3"),
    ]

    continuation = AsyncMock()
    continuation.return_value = "mocked_response"

    result = await interceptor.intercept_service(continuation, handler_call_details)

    assert current_mcp_server_tools_context.get() == {
        "tool1",
        "tool2",
        "tool3",
    }
    continuation.assert_called_once_with(handler_call_details)
    assert result == "mocked_response"
