import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims

from duo_workflow_service.interceptors.authentication_interceptor import current_user
from duo_workflow_service.interceptors.model_metadata_interceptor import (
    ModelMetadataInterceptor,
)


@pytest.fixture(name="mock_user")
def mock_user_fixture():
    return CloudConnectorUser(True, claims=UserClaims(gitlab_realm="test-realm"))


@pytest.mark.asyncio
async def test_model_metadata_interceptor_sets_metadata(mock_user):
    """Test that the interceptor processes model metadata correctly."""
    current_user.set(mock_user)
    interceptor = ModelMetadataInterceptor()

    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        (
            "x-gitlab-agent-platform-model-metadata",
            json.dumps(
                {
                    "model": "claude-3-5-sonnet-20240620",
                    "version": "1.0",
                    "provider": "anthropic",
                }
            ),
        ),
        ("other-header", "other-value"),
    ]

    continuation = AsyncMock(return_value="mocked_response")

    with (
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.create_model_metadata_by_tag"
        ) as mock_create,
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.current_model_metadata_context"
        ) as mock_context,
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.current_model_metadata_with_size_context"
        ) as mock_size_context,
    ):
        mock_metadata_by_size = MagicMock()
        mock_create.return_value = mock_metadata_by_size

        result = await interceptor.intercept_service(continuation, handler_call_details)

        mock_create.assert_called_once_with(
            {
                "model": "claude-3-5-sonnet-20240620",
                "version": "1.0",
                "provider": "anthropic",
            }
        )
        mock_context.set.assert_called_once_with(mock_metadata_by_size.default)
        mock_size_context.set.assert_called_once_with(mock_metadata_by_size)
        continuation.assert_called_once_with(handler_call_details)
        assert result == "mocked_response"


@pytest.mark.asyncio
async def test_model_metadata_interceptor_null_json_skips_context_setting(mock_user):
    """JSON 'null' parses to None, which create_model_metadata_by_tag rejects with ValueError.

    Both context vars must remain unset and the request must still continue.
    """
    current_user.set(mock_user)
    interceptor = ModelMetadataInterceptor()

    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        ("x-gitlab-agent-platform-model-metadata", "null"),
    ]

    continuation = AsyncMock(return_value="mocked_response")

    with (
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.current_model_metadata_context"
        ) as mock_context,
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.current_model_metadata_with_size_context"
        ) as mock_size_context,
    ):
        result = await interceptor.intercept_service(continuation, handler_call_details)

        mock_context.set.assert_not_called()
        mock_size_context.set.assert_not_called()
        continuation.assert_called_once_with(handler_call_details)
        assert result == "mocked_response"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invocation_metadata,_test_case,expected_header_present",
    [
        ([("other-header", "other-value")], "no_metadata_header", False),
        (
            [
                ("x-gitlab-agent-platform-model-metadata", ""),
                ("other-header", "other-value"),
            ],
            "empty_metadata",
            False,
        ),
        (
            [
                ("x-gitlab-agent-platform-model-metadata", "invalid-json{"),
                ("other-header", "other-value"),
            ],
            "invalid_json",
            True,
        ),
    ],
)
async def test_model_metadata_interceptor_no_processing_scenarios(
    invocation_metadata, _test_case, expected_header_present
):
    """Test that the interceptor handles cases where no model metadata processing occurs."""
    interceptor = ModelMetadataInterceptor()

    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = invocation_metadata

    continuation = AsyncMock(return_value="mocked_response")

    with (
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.create_model_metadata_by_tag"
        ) as mock_create,
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.current_model_metadata_context"
        ) as mock_context,
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.current_model_metadata_with_size_context"
        ) as mock_size_context,
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.log"
        ) as mock_log,
    ):
        result = await interceptor.intercept_service(continuation, handler_call_details)

        mock_create.assert_not_called()
        mock_context.set.assert_not_called()
        mock_size_context.set.assert_not_called()
        continuation.assert_called_once_with(handler_call_details)
        assert result == "mocked_response"

        mock_log.warning.assert_called_once()
        _, log_kwargs = mock_log.warning.call_args
        assert log_kwargs["header_present"] is expected_header_present
        # The raw header may carry provider API keys, so it must never be logged.
        assert "invalid-json{" not in str(log_kwargs)


@pytest.mark.asyncio
async def test_gitlab_provider_with_feature_setting_uses_build_default(mock_user):
    """Gitlab-provider payload with feature_setting routes through build_default_feature_setting_metadata."""
    current_user.set(mock_user)
    interceptor = ModelMetadataInterceptor()

    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        (
            "x-gitlab-agent-platform-model-metadata",
            json.dumps(
                {
                    "provider": "gitlab",
                    "feature_setting": "duo_agent_platform_agentic_chat",
                }
            ),
        ),
    ]

    continuation = AsyncMock(return_value="ok")
    fake_default = MagicMock()
    fake_by_tag = {"small": MagicMock()}

    fake_model_keys = MagicMock()
    fake_model_keys.model_dump.return_value = {"fireworks_provider_api_key": "fw"}
    fake_config = MagicMock()
    fake_config.model_keys = fake_model_keys
    fake_config.fireworks_api_base_url = "https://fw"

    with (
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.get_config",
            return_value=fake_config,
        ),
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.build_default_feature_setting_metadata",
            return_value=fake_default,
        ) as mock_build,
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.build_model_metadata_by_tag",
            return_value=fake_by_tag,
        ) as mock_build_by_tag,
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.ModelMetadataByTag"
        ) as mock_by_size_cls,
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.current_model_metadata_context"
        ) as mock_context,
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.current_model_metadata_with_size_context"
        ) as mock_size_context,
    ):
        fake_by_size = MagicMock()
        fake_by_size.default = fake_default
        mock_by_size_cls.return_value = fake_by_size

        result = await interceptor.intercept_service(continuation, handler_call_details)

        mock_build.assert_called_once_with(
            feature_setting="duo_agent_platform_agentic_chat",
            identifier=None,
            model_keys={"fireworks_provider_api_key": "fw"},
            fireworks_api_base_url="https://fw",
            user=mock_user,
        )
        mock_build_by_tag.assert_called_once_with("duo_agent_platform_agentic_chat")
        mock_by_size_cls.assert_called_once_with(
            default=fake_default, by_tag=fake_by_tag
        )
        mock_context.set.assert_called_once_with(fake_default)
        mock_size_context.set.assert_called_once_with(fake_by_size)
        continuation.assert_called_once_with(handler_call_details)
        assert result == "ok"


@pytest.mark.asyncio
async def test_gitlab_provider_without_identifier_or_feature_setting_falls_through(
    mock_user,
):
    """Gitlab-provider payload without identifier/feature_setting skips the new branch."""
    current_user.set(mock_user)
    interceptor = ModelMetadataInterceptor()

    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        ("x-gitlab-agent-platform-model-metadata", json.dumps({"provider": "gitlab"})),
    ]

    continuation = AsyncMock(return_value="ok")

    with (
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.build_default_feature_setting_metadata"
        ) as mock_build,
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.create_model_metadata_by_tag"
        ) as mock_create,
    ):
        mock_create.return_value = MagicMock()
        await interceptor.intercept_service(continuation, handler_call_details)

        mock_build.assert_not_called()
        mock_create.assert_called_once()
