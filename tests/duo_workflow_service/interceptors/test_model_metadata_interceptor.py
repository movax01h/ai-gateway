import json
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims
from structlog.testing import capture_logs

from ai_gateway.config import get_config
from ai_gateway.model_selection import ModelSelectionConfig
from ai_gateway.model_selection.model_selection_config import ChatLiteLLMDefinition
from duo_workflow_service.entities.state import (
    get_current_model_max_context_token_limit,
)
from duo_workflow_service.interceptors.authentication_interceptor import current_user
from duo_workflow_service.interceptors.model_metadata_interceptor import (
    ModelMetadataInterceptor,
)
from lib.context import (
    current_model_metadata_context,
    current_model_metadata_with_size_context,
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

        # provider_keys/fireworks_api_base_url are backfilled from server config since
        # this payload didn't supply its own (see the "provider stickiness" fix).
        mock_create.assert_called_once_with(
            {
                "model": "claude-3-5-sonnet-20240620",
                "version": "1.0",
                "provider": "anthropic",
                "provider_keys": {"fireworks_provider_api_key": "fw"},
                "fireworks_api_base_url": "https://fw",
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
        mock_build_by_tag.assert_called_once_with(
            "duo_agent_platform_agentic_chat",
            provider_keys={"fireworks_provider_api_key": "fw"},
            fireworks_api_base_url="https://fw",
        )
        mock_by_size_cls.assert_called_once_with(
            default=fake_default,
            by_tag=fake_by_tag,
            feature_setting="duo_agent_platform_agentic_chat",
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


@pytest.mark.asyncio
async def test_stickiness_replay_of_fireworks_model_still_gets_a_working_key(mock_user):
    """Regression test for provider-stickiness replay.

    GitLab Rails echoes back a previously checkpointed model_metadata blob verbatim on
    workflow resume. That blob has `provider: "fireworks_ai"` directly (not
    `provider: "gitlab"` + identifier) and, per the checkpoint fix, carries no `api_key`/
    `provider_keys` at all. Without the backfill, this would resolve to a
    FireworksModelMetadata with no API key and fail Fireworks auth with a 401 on every
    resumed turn. The interceptor must backfill provider_keys/fireworks_api_base_url from
    server config so the replayed model still authenticates.
    """
    current_user.set(mock_user)
    interceptor = ModelMetadataInterceptor()

    # Shape of a real checkpoint replay: no provider_keys, no api_key, no endpoint —
    # exactly what a FireworksModelMetadata serializes to after excluding api_key.
    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        (
            "x-gitlab-agent-platform-model-metadata",
            json.dumps(
                {
                    "provider": "fireworks_ai",
                    "name": "kimi_k2_6_fireworks",
                    "model_identifier": "accounts/gitlab/deployments/z6hbhxrt",
                }
            ),
        ),
    ]

    continuation = AsyncMock(return_value="ok")

    fireworks_def = ChatLiteLLMDefinition(
        gitlab_identifier="kimi_k2_6_fireworks",
        name="Kimi K2.6 - Fireworks",
        max_context_tokens=256000,
        family=["kimi"],
        params={
            "model": "accounts/gitlab/deployments/z6hbhxrt",
            "custom_llm_provider": "fireworks_ai",
        },
    )

    fake_model_keys = MagicMock()
    fake_model_keys.model_dump.return_value = {"fireworks_provider_api_key": "fw"}
    fake_config = MagicMock()
    fake_config.model_keys = fake_model_keys
    fake_config.fireworks_api_base_url = "https://api.fireworks.ai/inference/v1"

    with (
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.get_config",
            return_value=fake_config,
        ),
        patch(
            "ai_gateway.model_selection.ModelSelectionConfig.instance"
        ) as mock_instance,
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.current_model_metadata_context"
        ) as mock_context,
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.current_model_metadata_with_size_context"
        ),
    ):
        mock_instance.return_value.get_model.return_value = fireworks_def

        result = await interceptor.intercept_service(continuation, handler_call_details)

        resolved = mock_context.set.call_args[0][0]
        assert resolved.provider == "fireworks_ai"
        assert resolved.to_params()["api_key"] == "fw"
        assert (
            resolved.to_params()["api_base"] == "https://api.fireworks.ai/inference/v1"
        )
        continuation.assert_called_once_with(handler_call_details)
        assert result == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("identifier", "expected_limit"),
    [
        ("bedrock/eu.anthropic.claude-sonnet-4-6", 1_000_000),
        ("vertex_ai/claude-sonnet-4-6@default", 1_000_000),
        ("anthropic/claude-sonnet-4-6", 1_000_000),
        # No match: the `claude_3` family limit applies
        (
            "bedrock/converse/arn:aws:bedrock:eu-west-1:123:application-inference-profile/x",
            200_000,
        ),
        ("custom_openai/claude-sonnet-4-6", 200_000),
    ],
)
async def test_self_hosted_identifier_sets_matched_context_limit(
    mock_user, identifier, expected_limit
):
    """A self-hosted request gets the context window of the matched models.yml entry.

    Uses the real ModelSelectionConfig and context vars, from the header to the limit that DWS reads.
    """
    current_user.set(mock_user)
    interceptor = ModelMetadataInterceptor()

    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        (
            "x-gitlab-agent-platform-model-metadata",
            json.dumps(
                {
                    "provider": "openai",
                    "name": "claude_3",
                    "endpoint": "https://example.com",
                    "api_key": "key",
                    "identifier": identifier,
                }
            ),
        ),
    ]

    fake_config = MagicMock()
    fake_config.model_keys.model_dump.return_value = {}
    fake_config.fireworks_api_base_url = ""

    with patch(
        "duo_workflow_service.interceptors.model_metadata_interceptor.get_config",
        return_value=fake_config,
    ):
        await interceptor.intercept_service(AsyncMock(), handler_call_details)

    assert get_current_model_max_context_token_limit() == expected_limit


AGENT_PLATFORM = "duo_agent_platform"
METHOD = "/DuoWorkflow/ExecuteWorkflow"
NOT_ALLOWED_FOR_AGENT_PLATFORM = {
    "provider": "gitlab",
    "identifier": "codestral_2508_fireworks",
    "feature_setting": AGENT_PLATFORM,
}


def _agent_platform_default() -> str:
    return (
        ModelSelectionConfig.instance()
        .get_unit_primitive_config_map()[AGENT_PLATFORM]
        .default_model_identifiers[0]
    )


def _call_details(model_metadata: dict) -> MagicMock:
    details = MagicMock()
    details.method = METHOD
    details.invocation_metadata = [
        ("x-gitlab-agent-platform-model-metadata", json.dumps(model_metadata))
    ]
    return details


async def _intercept(model_metadata: dict):
    """Run the interceptor with the real config; return its result and the metadata the call sees."""
    seen = {}

    async def continuation(_):
        seen["default"] = current_model_metadata_context.get()
        seen["by_tag"] = current_model_metadata_with_size_context.get()
        return "continued"

    result = await ModelMetadataInterceptor().intercept_service(
        continuation, _call_details(model_metadata)
    )
    return result, seen


@pytest.fixture(name="cloud_gateway")
def cloud_gateway_fixture(monkeypatch, mock_user):
    """Run as GitLab's own gateway: a gateway that serves custom models skips the allowlist check."""
    monkeypatch.setattr(get_config().custom_models, "enabled", False)
    token = current_user.set(mock_user)
    yield
    current_user.reset(token)


@pytest.mark.asyncio
@pytest.mark.usefixtures("cloud_gateway", "feature_allowlist_mode")
class TestFeatureAllowlist:
    @pytest.mark.parametrize("feature_allowlist_mode", ["log"], indirect=True)
    async def test_log_mode_serves_the_model_and_logs_the_mismatch(self):
        assert (
            "codestral_2508_fireworks"
            not in ModelSelectionConfig.instance()
            .get_unit_primitive_config_map()[AGENT_PLATFORM]
            .allowed_model_identifiers
        )

        with capture_logs() as logs:
            result, seen = await _intercept(NOT_ALLOWED_FOR_AGENT_PLATFORM)

        assert result == "continued"
        assert seen["default"].llm_definition.gitlab_identifier == (
            "codestral_2508_fireworks"
        )
        assert [
            (entry["reason"], entry["surface"], entry["method"])
            for entry in logs
            if entry["event"] == "Model allowlist mismatch"
        ] == [("model_not_allowed", "dws", METHOD)]

    @pytest.mark.parametrize("feature_allowlist_mode", ["fallback"], indirect=True)
    async def test_fallback_mode_serves_the_feature_default(self):
        result, seen = await _intercept(NOT_ALLOWED_FOR_AGENT_PLATFORM)

        assert result == "continued"
        assert seen["default"].llm_definition.gitlab_identifier == (
            _agent_platform_default()
        )
        assert seen["by_tag"].default is seen["default"]

    @pytest.mark.parametrize("feature_allowlist_mode", ["fallback"], indirect=True)
    async def test_fallback_mode_leaves_an_unknown_feature_on_the_prompt_default(
        self,
    ):
        result, seen = await _intercept(
            {
                "provider": "gitlab",
                "identifier": "codestral_2508_fireworks",
                "feature_setting": "no_such_feature",
            }
        )

        assert result == "continued"
        assert seen == {"default": None, "by_tag": None}

    @pytest.mark.parametrize("feature_allowlist_mode", ["enforce"], indirect=True)
    async def test_enforce_mode_rejects_the_call(self):
        continuation = AsyncMock()
        handler = await ModelMetadataInterceptor().intercept_service(
            continuation, _call_details(NOT_ALLOWED_FOR_AGENT_PLATFORM)
        )
        context = AsyncMock()

        await handler.unary_unary(None, context)

        continuation.assert_not_called()
        context.abort.assert_awaited_once_with(
            grpc.StatusCode.PERMISSION_DENIED,
            "Model 'codestral_2508_fireworks' is not available for 'duo_agent_platform'",
        )

    @pytest.mark.parametrize("feature_allowlist_mode", ["enforce"], indirect=True)
    @pytest.mark.parametrize(
        "with_feature_setting",
        [
            pytest.param(True, id="allowed-model"),
            pytest.param(False, id="resume-replay-without-feature"),
        ],
    )
    async def test_enforce_mode_serves_an_allowed_model(self, with_feature_setting):
        model_metadata = (
            {
                "provider": "gitlab",
                "identifier": _agent_platform_default(),
                "feature_setting": AGENT_PLATFORM,
            }
            if with_feature_setting
            else {"provider": "gitlab", "name": _agent_platform_default()}
        )

        result, seen = await _intercept(model_metadata)

        assert result == "continued"
        assert seen["default"].llm_definition.gitlab_identifier == (
            _agent_platform_default()
        )

    @pytest.mark.parametrize("feature_allowlist_mode", ["enforce"], indirect=True)
    async def test_enforce_mode_keeps_an_identifier_without_a_feature_with_a_note(
        self,
    ):
        with capture_logs() as logs:
            result, seen = await _intercept(
                {"provider": "gitlab", "identifier": "codestral_2508_fireworks"}
            )

        assert result == "continued"
        assert seen["default"].llm_definition.gitlab_identifier == (
            "codestral_2508_fireworks"
        )
        assert seen["by_tag"].default is seen["default"]
        assert [
            (entry["reason"], entry["surface"], entry["method"])
            for entry in logs
            if entry["event"] == "Model allowlist note"
        ] == [("missing_feature_setting", "dws", METHOD)]

    @pytest.mark.parametrize("feature_allowlist_mode", ["enforce"], indirect=True)
    async def test_enforce_mode_rejects_a_provider_the_gateway_does_not_serve(self):
        continuation = AsyncMock()
        handler = await ModelMetadataInterceptor().intercept_service(
            continuation,
            _call_details(
                {
                    "provider": "anthropic",
                    "name": _agent_platform_default(),
                    "feature_setting": AGENT_PLATFORM,
                }
            ),
        )
        context = AsyncMock()

        await handler.unary_unary(None, context)

        continuation.assert_not_called()
        context.abort.assert_awaited_once_with(
            grpc.StatusCode.PERMISSION_DENIED,
            "This gateway does not serve models from provider 'anthropic'",
        )
