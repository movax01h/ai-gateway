from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_gateway.config import Config
from duo_workflow_service.interceptors.metadata_context_interceptor import (
    MetadataContextInterceptor,
)
from lib.events.contextvar import (
    X_GITLAB_SELF_HOSTED_DAP_BILLING_ENABLED,
    self_hosted_dap_billing_enabled,
)
from lib.mcp_server_tools.context import current_mcp_server_tools_context


@pytest.fixture
def mock_config():
    """Create a properly configured mock Config for tests."""
    config = MagicMock(spec=Config)
    config.custom_models = MagicMock()
    config.custom_models.enabled = False
    return config


@pytest.mark.asyncio
async def test_client_type_header(mock_config):
    """Test that client type header is properly set."""
    interceptor = MetadataContextInterceptor(mock_config)
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
async def test_gitlab_realm_header(mock_config):
    """Test that GitLab realm header is properly set."""
    interceptor = MetadataContextInterceptor(mock_config)
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
async def test_gitlab_version_header(mock_config):
    """Test that GitLab version header is properly set."""
    interceptor = MetadataContextInterceptor(mock_config)
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
async def test_language_server_version_header(mock_config):
    """Test that language server version header is properly set with transformation."""
    interceptor = MetadataContextInterceptor(mock_config)
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
async def test_verbose_ai_logs_header(mock_config, header_value, expected_bool):
    """Test that verbose AI logs header is properly converted to boolean."""
    interceptor = MetadataContextInterceptor(mock_config)
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
async def test_prompt_caching_header(mock_config):
    """Test that prompt caching header calls the setter function."""
    interceptor = MetadataContextInterceptor(mock_config)
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
async def test_all_headers_together(mock_config):
    """Test that all headers are processed correctly when present together."""
    interceptor = MetadataContextInterceptor(mock_config)
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
async def test_missing_headers(mock_config):
    """Test that missing headers don't cause errors."""
    interceptor = MetadataContextInterceptor(mock_config)
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
        # Self-hosted DAP billing is only called when custom_models.enabled is True
        # Since mock_config has custom_models.enabled = False, it won't be called
        # MCP server tools should be empty set when header is missing
        assert current_mcp_server_tools_context.get() == set()
        continuation.assert_called_once_with(handler_call_details)
        assert result == "mocked_response"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("custom_models_enabled", "header_present", "header_value", "expected_result"),
    [
        # When custom_models.enabled is True and header is present with "true"
        (True, True, "true", True),
        # When custom_models.enabled is True and header is present with "false"
        (True, True, "false", False),
        # When custom_models.enabled is True and header is not present
        (True, False, None, False),
        # When custom_models.enabled is False and header is present (should not process)
        (False, True, "true", False),
        # When custom_models.enabled is False and header is not present
        (False, False, None, False),
    ],
)
async def test_self_hosted_dap_billing_header(
    mock_config, custom_models_enabled, header_present, header_value, expected_result
):
    """Test that self-hosted DAP billing header is properly handled based on config and header presence."""
    # Configure custom models setting
    mock_config.custom_models.enabled = custom_models_enabled

    interceptor = MetadataContextInterceptor(mock_config)
    handler_call_details = MagicMock()

    # Set up metadata based on whether header should be present
    if header_present:
        handler_call_details.invocation_metadata = [
            (X_GITLAB_SELF_HOSTED_DAP_BILLING_ENABLED, header_value),
        ]
    else:
        handler_call_details.invocation_metadata = []

    continuation = AsyncMock()
    continuation.return_value = "mocked_response"

    result = await interceptor.intercept_service(continuation, handler_call_details)

    # Verify the context value matches expected result
    assert self_hosted_dap_billing_enabled.get() is expected_result
    continuation.assert_called_once_with(handler_call_details)
    assert result == "mocked_response"


@pytest.mark.asyncio
async def test_empty_header_values(mock_config):
    """Test that empty header values are handled correctly."""
    interceptor = MetadataContextInterceptor(mock_config)
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
async def test_mcp_server_tools_header(mock_config):
    """Test that MCP server tools header is properly parsed."""
    interceptor = MetadataContextInterceptor(mock_config)
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
