# pylint: disable=unused-argument,file-naming-for-tests
from typing import Optional
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from ai_gateway.api.v3 import api_router
from ai_gateway.model_metadata import AmazonQModelMetadata, TypeModelMetadata
from ai_gateway.model_selection.model_selection_config import ChatAmazonQDefinition
from ai_gateway.models.base import KindModelProvider
from ai_gateway.tracking import SnowplowEventContext
from lib.feature_flags.context import current_feature_flag_context

__all__ = [
    "TestCodeSuggestionTypeLogged",
]


@pytest.fixture(name="route")
def route_fixture():
    return "/code/completions"


@pytest.fixture(name="fast_api_router", scope="class")
def fast_api_router_fixture():
    return api_router


@pytest.fixture(name="scopes")
def scopes_fixture():
    return ["complete_code", "generate_code", "amazon_q_integration"]


@pytest.fixture(name="config_values")
def config_values_fixture(assets_dir):
    return {
        "custom_models": {
            "enabled": True,
        },
        "fireworks_api_base_url": "https://api.fireworks.ai/inference/v1",
        "model_keys": {
            "fireworks_provider_api_key": "mock_fireworks_key",
        },
        "self_signed_jwt": {
            "signing_key": open(assets_dir / "keys" / "signing_key.pem").read(),
        },
        "amazon_q": {
            "region": "us-west-2",
        },
    }


class TestEditorContentCompletionStream:
    def test_successful_stream_response(
        self,
        mock_client: TestClient,
        mock_completions_stream: Mock,
        mock_suggestions_output_text: str,
        route: str,
    ):
        payload = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
            "language_identifier": "python",
            "stream": True,
        }

        prompt_component = {
            "type": "code_editor_completion",
            "payload": payload,
        }

        data = {
            "prompt_components": [prompt_component],
        }

        response = mock_client.post(
            route,
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
                "X-Gitlab-Global-User-Id": "test-user-id",
            },
            json=data,
        )

        assert response.status_code == 200
        assert response.text == mock_suggestions_output_text
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        expected_snowplow_event = SnowplowEventContext(
            prefix_length=30,
            suffix_length=1,
            language="python",
            gitlab_realm="self-managed",
            is_direct_connection=False,
            gitlab_instance_id="1234",
            gitlab_global_user_id="test-user-id",
            gitlab_host_name="",
            gitlab_saas_duo_pro_namespace_ids=[],
            suggestion_source="network",
            region="us-central1",
        )
        mock_completions_stream.assert_called_with(
            prefix=payload["content_above_cursor"],
            suffix=payload["content_below_cursor"],
            file_name=payload["file_name"],
            editor_lang=payload["language_identifier"],
            stream=True,
            code_context=None,
            user=mock_completions_stream.call_args.kwargs["user"],
            snowplow_event_context=expected_snowplow_event,
        )


class TestEditorContentGenerationStream:
    def test_successful_stream_response(
        self,
        mock_client: TestClient,
        mock_generations_stream: Mock,
        mock_suggestions_output_text: str,
        route: str,
    ):
        payload = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
            "language_identifier": "python",
            "model_provider": "anthropic",
            "stream": True,
        }

        prompt_component = {
            "type": "code_editor_generation",
            "payload": payload,
        }

        data = {
            "prompt_components": [prompt_component],
        }

        response = mock_client.post(
            route,
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json=data,
        )

        assert response.status_code == 200
        assert response.text == mock_suggestions_output_text
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


class TestAmazonQIntegrationV3:
    @pytest.mark.parametrize(
        (
            "mock_suggestions_output_text",
            "mock_suggestions_model",
            "mock_suggestions_engine",
            "role_arn",
            "model_metadata",
            "expected_response",
        ),
        [
            # non-empty suggestions from model
            (
                "def search",
                "amazon_q",
                "amazon_q",
                "test:role",
                None,
                {
                    "choices": [
                        {
                            "text": "def search",
                            "index": 0,
                            "finish_reason": "length",
                        }
                    ],
                    "metadata": {
                        "model": {
                            "engine": "amazon_q",
                            "name": "amazon_q",
                            "lang": "python",
                        },
                        "enabled_feature_flags": ["flag_a", "flag_b"],
                    },
                },
            ),
            # empty suggestions from model
            (
                "",
                "amazon_q",
                "amazon_q",
                "test:role",
                None,
                {
                    "choices": [],
                    "metadata": {
                        "model": {
                            "engine": "amazon_q",
                            "name": "amazon_q",
                            "lang": "python",
                        },
                        "enabled_feature_flags": ["flag_a", "flag_b"],
                    },
                },
            ),
            # non-empty suggestions from model
            (
                "def search",
                "amazon_q",
                "amazon_q",
                "",
                AmazonQModelMetadata(
                    provider="amazon_q",
                    name="amazon_q",
                    role_arn="test:role",
                    llm_definition=ChatAmazonQDefinition(
                        gitlab_identifier="amazon_q",
                        name="Amazon Q",
                        max_context_tokens=100000,
                    ),
                ),
                {
                    "choices": [
                        {
                            "text": "def search",
                            "index": 0,
                            "finish_reason": "length",
                        }
                    ],
                    "metadata": {
                        "model": {
                            "engine": "amazon_q",
                            "name": "amazon_q",
                            "lang": "python",
                        },
                        "enabled_feature_flags": ["flag_a", "flag_b"],
                    },
                },
            ),
            # empty suggestions from model
            (
                "",
                "amazon_q",
                "amazon_q",
                "",
                AmazonQModelMetadata(
                    provider="amazon_q",
                    name="amazon_q",
                    role_arn="test:role",
                    llm_definition=ChatAmazonQDefinition(
                        gitlab_identifier="amazon_q",
                        name="Amazon Q",
                        max_context_tokens=100000,
                    ),
                ),
                {
                    "choices": [],
                    "metadata": {
                        "model": {
                            "engine": "amazon_q",
                            "name": "amazon_q",
                            "lang": "python",
                        },
                        "enabled_feature_flags": ["flag_a", "flag_b"],
                    },
                },
            ),
        ],
    )
    def test_code_generation_successful_response(
        self,
        mock_client: TestClient,
        mock_generations: Mock,
        mock_suggestions_output_text: str,
        role_arn: str,
        model_metadata: Optional[TypeModelMetadata],
        expected_response: dict,
        route: str,
    ):
        payload = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
            "language_identifier": "python",
            "model_provider": "amazon_q",
            "prompt_id": "code_suggestions/generations",
            "prompt_version": "^1.0.0",
            "role_arn": role_arn,
        }

        prompt_component = {
            "type": "code_editor_generation",
            "payload": payload,
        }

        data = {
            "prompt_components": [prompt_component],
            "model_metadata": model_metadata
            and model_metadata.model_dump(
                exclude={"llm_definition", "friendly_name"},
                mode="json",
            ),
        }

        current_feature_flag_context.set({"flag_a", "flag_b"})

        response = mock_client.post(
            route,
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
                "X-Gitlab-Global-User-Id": "test-user-id",
            },
            json=data,
        )

        assert response.status_code == 200

        body = response.json()

        assert body["choices"] == expected_response["choices"]
        assert body["metadata"]["model"] == expected_response["metadata"]["model"]
        assert body["metadata"]["timestamp"] > 0
        assert set(body["metadata"]["enabled_feature_flags"]) == set(
            expected_response["metadata"]["enabled_feature_flags"]
        )

        expected_snowplow_event = SnowplowEventContext(
            prefix_length=30,
            suffix_length=1,
            language="python",
            gitlab_realm="self-managed",
            is_direct_connection=False,
            gitlab_instance_id="1234",
            gitlab_global_user_id="test-user-id",
            gitlab_host_name="",
            gitlab_saas_duo_pro_namespace_ids=[],
            suggestion_source="network",
            region="us-central1",
        )
        mock_generations.assert_called_with(
            prefix=payload["content_above_cursor"],
            file_name=payload["file_name"],
            editor_lang=payload["language_identifier"],
            model_provider=KindModelProvider.AMAZON_Q,
            stream=False,
            user=mock_generations.call_args.kwargs["user"],
            snowplow_event_context=expected_snowplow_event,
            prompt_enhancer=None,
            suffix="\n",
        )

    @pytest.mark.parametrize(
        (
            "mock_suggestions_output_text",
            "mock_suggestions_model",
            "mock_suggestions_engine",
            "role_arn",
            "model_metadata",
            "expected_response",
        ),
        [
            # non-empty suggestions from model
            (
                "def search",
                "amazon_q",
                "amazon_q",
                "test:role",
                None,
                {
                    "choices": [
                        {
                            "text": "def search",
                            "index": 0,
                            "finish_reason": "length",
                        },
                    ],
                    "metadata": {
                        "model": {
                            "engine": "amazon_q",
                            "name": "amazon_q",
                            "lang": "python",
                        },
                        "enabled_feature_flags": ["flag_a", "flag_b"],
                    },
                },
            ),
            # empty suggestions from model
            (
                "",
                "amazon_q",
                "amazon_q",
                "test:role",
                None,
                {
                    "choices": [],
                    "metadata": {
                        "model": {
                            "engine": "amazon_q",
                            "name": "amazon_q",
                            "lang": "python",
                        },
                        "enabled_feature_flags": ["flag_a", "flag_b"],
                    },
                },
            ),
        ],
    )
    def test_code_completion_successful_response(
        self,
        mock_client: TestClient,
        mock_completions: Mock,
        mock_suggestions_output_text: str,
        role_arn: str,
        model_metadata: Optional[TypeModelMetadata],
        expected_response: dict,
        route: str,
    ):
        payload = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
            "language_identifier": "python",
            "choices_count": 3,
            "model_provider": "amazon_q",
            "role_arn": role_arn,
        }

        prompt_component = {
            "type": "code_editor_completion",
            "payload": payload,
        }

        data = {
            "prompt_components": [prompt_component],
            "model_metadata": model_metadata
            and model_metadata.model_dump(
                exclude={"llm_definition", "friendly_name"},
                mode="json",
            ),
        }

        current_feature_flag_context.set({"flag_a", "flag_b"})

        response = mock_client.post(
            route,
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
                "X-Gitlab-Global-User-Id": "test-user-id",
            },
            json=data,
        )

        assert response.status_code == 200

        body = response.json()

        assert body["choices"] == expected_response["choices"]
        assert body["metadata"]["model"] == expected_response["metadata"]["model"]
        assert body["metadata"]["timestamp"] > 0
        assert set(body["metadata"]["enabled_feature_flags"]) == set(
            expected_response["metadata"]["enabled_feature_flags"]
        )


class TestCustomModelsSSRF:
    @pytest.fixture(name="config_values")
    def config_values_fixture(self, assets_dir):
        return {
            "custom_models": {
                "enabled": False,
            },
            "self_signed_jwt": {
                "signing_key": open(assets_dir / "keys" / "signing_key.pem").read(),
            },
        }

    @pytest.fixture(name="scopes")
    def scopes_fixture(self):
        return ["complete_code", "generate_code"]

    @pytest.fixture(name="route")
    def route_fixture(self):
        return "/code/completions"

    def test_completions_ssrf_model_metadata_endpoint_rejected_when_custom_models_disabled(
        self,
        mock_client: TestClient,
        route: str,
    ):
        with pytest.raises(
            ValueError, match="specifying custom models endpoint is disabled"
        ):
            mock_client.post(
                route,
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-GitLab-Instance-Id": "1234",
                    "X-GitLab-Realm": "self-managed",
                    "X-Gitlab-Global-User-Id": "test-user-id",
                },
                json={
                    "prompt_components": [
                        {
                            "type": "code_editor_completion",
                            "payload": {
                                "file_name": "main.py",
                                "content_above_cursor": "# test",
                                "content_below_cursor": "\n",
                                "language_identifier": "python",
                            },
                        }
                    ],
                    "model_metadata": {
                        "endpoint": "http://internal-server.local/ssrf",
                        "api_key": "malicious-key",
                        "identifier": "provider/model",
                        "name": "mistral",
                        "provider": "litellm",
                    },
                },
            )


class TestCodeSuggestionTypeLogged:
    @pytest.mark.parametrize(
        "component_type",
        [
            "code_editor_completion",
            "code_editor_generation",
        ],
    )
    def test_code_suggestion_type_in_access_log(
        self,
        mock_client: TestClient,
        mock_completions: Mock,
        mock_generations: Mock,
        component_type: str,
        route: str,
    ):
        payload = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
            "language_identifier": "python",
        }

        prompt_component = {
            "type": component_type,
            "payload": payload,
        }

        data = {
            "prompt_components": [prompt_component],
        }

        with capture_logs() as cap_logs:
            response = mock_client.post(
                route,
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                    "X-GitLab-Instance-Id": "1234",
                    "X-GitLab-Realm": "self-managed",
                    "X-Gitlab-Global-User-Id": "test-user-id",
                },
                json=data,
            )

        assert response.status_code == 200

        access_logs = [log for log in cap_logs if "code_suggestion_type" in log]
        assert access_logs
        assert access_logs[0]["code_suggestion_type"] == component_type
