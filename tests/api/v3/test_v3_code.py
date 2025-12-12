from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest
from dependency_injector import containers
from fastapi.testclient import TestClient
from gitlab_cloud_connector import CloudConnectorUser, UserClaims
from langchain.tools import BaseTool
from pydantic import AnyUrl

from ai_gateway.api.v3 import api_router
from ai_gateway.model_selection import LLMDefinition
from ai_gateway.models.base import KindModelProvider
from ai_gateway.tracking import SnowplowEventContext
from lib.feature_flags.context import current_feature_flag_context

__all__ = [
    "auth_user",
    "TestEditorContentCompletion",
    "TestEditorContentGeneration",
    "TestUnauthorizedScopes",
    "TestIncomingRequest",
    "TestUnauthorizedIssuer",
]

from ai_gateway.model_metadata import (
    AmazonQModelMetadata,
    ModelMetadata,
    TypeModelMetadata,
)


@pytest.fixture(name="route")
def route_fixture():
    return "/code/completions"


@pytest.fixture(name="fast_api_router", scope="class")
def fast_api_router_fixture():
    return api_router


@pytest.fixture(name="unit_primitives")
def unit_primitives_fixture():
    return ["complete_code", "generate_code", "amazon_q_integration"]


@pytest.fixture(name="config_values")
def config_values_fixture(assets_dir):
    return {
        "custom_models": {
            "enabled": True,
        },
        "model_keys": {
            "fireworks_api_key": "mock_fireworks_key",
        },
        "self_signed_jwt": {
            "signing_key": open(assets_dir / "keys" / "signing_key.pem").read(),
        },
        "amazon_q": {
            "region": "us-west-2",
        },
        "model_endpoints": {
            "fireworks_regional_endpoints": {
                "us-central1": {
                    "qwen2p5-coder-7b": {
                        "endpoint": "https://fireworks.endpoint",
                        "identifier": "qwen2p5-coder-7b",
                    },
                },
            },
        },
    }


class TestEditorContentCompletion:
    @pytest.mark.parametrize(
        ("mock_suggestions_output_text", "expected_response"),
        [
            # non-empty suggestions from model
            (
                "def search",
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
                            "engine": "anthropic",
                            "name": "claude-3-haiku-20240307",
                            "lang": "python",
                        },
                        "enabled_feature_flags": ["flag_a", "flag_b"],
                    },
                },
            ),
            # empty suggestions from model
            (
                "",
                {
                    "choices": [],
                    "metadata": {
                        "model": {
                            "engine": "anthropic",
                            "name": "claude-3-haiku-20240307",
                            "lang": "python",
                        },
                        "enabled_feature_flags": ["flag_a", "flag_b"],
                    },
                },
            ),
        ],
    )
    def test_successful_response(
        self,
        mock_client: TestClient,
        mock_completions: Mock,
        mock_suggestions_output_text: str,
        expected_response: dict,
        route: str,
    ):
        payload = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
            "language_identifier": "python",
            "choices_count": 3,
        }

        prompt_component = {
            "type": "code_editor_completion",
            "payload": payload,
        }

        data = {
            "prompt_components": [prompt_component],
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
        mock_completions.assert_called_with(
            prefix=payload["content_above_cursor"],
            suffix=payload["content_below_cursor"],
            file_name=payload["file_name"],
            editor_lang=payload["language_identifier"],
            stream=False,
            code_context=None,
            user=mock_completions.call_args.kwargs["user"],
            snowplow_event_context=expected_snowplow_event,
        )

    @pytest.mark.parametrize(
        ("model_provider", "expected_code", "expected_response", "expected_model"),
        [
            (
                "vertex-ai",
                200,
                {
                    "choices": [
                        {
                            "text": "Test text completion response",
                            "index": 0,
                            "finish_reason": "length",
                        }
                    ]
                },
                {
                    "engine": "vertex-ai",
                    "name": "vertex_ai/codestral-2501",
                    "lang": "python",
                },
            ),
            (
                "anthropic",
                200,
                {
                    "choices": [
                        {
                            "text": "test completion",
                            "index": 0,
                            "finish_reason": "length",
                        }
                    ]
                },
                {
                    "engine": "anthropic",
                    "name": "claude-3-5-sonnet-20240620",
                    "lang": "python",
                },
            ),
            # default provider
            (
                "",
                200,
                {
                    "choices": [
                        {
                            "text": "Test text completion response",
                            "index": 0,
                            "finish_reason": "length",
                        }
                    ]
                },
                {
                    "engine": "vertex-ai",
                    "name": "vertex_ai/codestral-2501",
                    "lang": "python",
                },
            ),
            # unknown provider
            (
                "some-provider",
                422,
                "",
                {},
            ),
        ],
    )
    def test_model_provider(
        self,
        mock_litellm_acompletion: Mock,
        mock_client: TestClient,
        mock_anthropic_chat: Mock,
        model_provider: str,
        expected_code: int,
        expected_response: dict,
        expected_model: dict,
        route: str,
    ):
        payload = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
            "language_identifier": "python",
            "model_provider": model_provider or None,
            "model_name": "claude-3-5-sonnet-20240620",
            "prompt": [
                {
                    "role": "system",
                    "content": "You are a code completion tool that performs Fill-in-the-middle",
                },
                {
                    "role": "user",
                    "content": "<SUFFIX>\n// a function to find the max\n for \n</SUFFIX>\n<PREFIX>\n\n\treturn min\n}\n</PREFIX>",
                },
            ],
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
            },
            json=data,
        )

        assert response.status_code == expected_code

        if expected_code >= 400:
            # if we want 400+ status we don't need check the response
            return

        body = response.json()

        assert body["choices"] == expected_response["choices"]

        assert body["metadata"]["model"] == expected_model

        mock = (
            mock_anthropic_chat
            if model_provider == "anthropic"
            else mock_litellm_acompletion
        )

        mock.assert_called_once()


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
            "model_provider": "anthropic",
            "stream": True,
            "model_name": "claude-3-5-sonnet-20240620",
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
            raw_prompt=None,
        )


class TestEditorContentGeneration:
    @pytest.mark.parametrize(
        (
            "mock_suggestions_output_text",
            "mock_suggestions_model",
            "mock_suggestions_engine",
            "expected_response",
        ),
        [
            # non-empty suggestions from model
            (
                "def search",
                "code-gecko",
                "vertex-ai",
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
                            "engine": "vertex-ai",
                            "name": "code-gecko",
                            "lang": "python",
                        },
                        "enabled_feature_flags": ["flag_a", "flag_b"],
                    },
                },
            ),
            # empty suggestions from model
            (
                "",
                "code-gecko",
                "vertex-ai",
                {
                    "choices": [],
                    "metadata": {
                        "model": {
                            "engine": "vertex-ai",
                            "name": "code-gecko",
                            "lang": "python",
                        },
                        "enabled_feature_flags": ["flag_a", "flag_b"],
                    },
                },
            ),
        ],
    )
    def test_successful_response(
        self,
        mock_client: TestClient,
        mock_generations: Mock,
        mock_suggestions_output_text: str,
        expected_response: dict,
        route: str,
    ):
        payload = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
            "language_identifier": "python",
        }

        prompt_component = {
            "type": "code_editor_generation",
            "payload": payload,
        }

        data = {
            "prompt_components": [prompt_component],
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
            model_provider=None,
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
            "expected_response",
        ),
        [
            # non-empty suggestions from model
            (
                "def search",
                "Claude 3.7 Sonnet Code Generations Agent",
                "agent",
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
                            "engine": "agent",
                            "name": "Claude 3.7 Sonnet Code Generations Agent",
                            "lang": "python",
                        },
                        "enabled_feature_flags": ["flag_a", "flag_b"],
                    },
                },
            ),
            # empty suggestions from model
            (
                "",
                "Claude 3.7 Sonnet Code Generations Agent",
                "agent",
                {
                    "choices": [],
                    "metadata": {
                        "model": {
                            "engine": "agent",
                            "name": "Claude 3.7 Sonnet Code Generations Agent",
                            "lang": "python",
                        },
                        "enabled_feature_flags": ["flag_a", "flag_b"],
                    },
                },
            ),
        ],
    )
    def test_generation_agent_model_successful_response(
        self,
        mock_client: TestClient,
        mock_generations: Mock,
        mock_suggestions_output_text: str,
        expected_response: dict,
        route: str,
    ):
        prompt_enhancer = {
            "examples_array": [
                {
                    "example": "class Project:\n  def __init__(self, name, public):\n    self.name = name\n    self.visibility = 'PUBLIC' if public\n\n    # is this project public?\n{{cursor}}\n\n    # print name of this project",
                    "response": "<new_code>def is_public(self):\n  return self.visibility == 'PUBLIC'",
                    "trigger_type": "comment",
                },
                {
                    "example": "def get_user(session):\n  # get the current user's name from the session data\n{{cursor}}\n\n# is the current user an admin",
                    "response": "<new_code>username = None\nif 'username' in session:\n  username = session['username']\nreturn username",
                    "trigger_type": "comment",
                },
            ],
            "trimmed_prefix": "# Create a fast binary search\n",
            "trimmed_suffix": "",
            "related_files": [
                '<file_content file_name="main.go">\npackage main\n\nfunc main()\nfullName("John", "Doe")\n}\n\n</file_content>\n'
            ],
            "related_snippets": [
                '<snippet_content name="fullName">\nfunc fullName(first, last string) {\n  fmt.Println(first, last)\n}\n\n</snippet_content>\n'
            ],
            "libraries": ["some_library", "other_library"],
            "user_instruction": "Generate the best possible code based on instructions.",
        }

        data = {
            "prompt_components": [
                {
                    "type": "code_editor_generation",
                    "payload": {
                        "file_name": "main.py",
                        "content_above_cursor": "# Create a fast binary search\n",
                        "content_below_cursor": "\n",
                        "language_identifier": "python",
                        "prompt_id": "code_suggestions/generations",
                        "prompt_version": "^1.0.0",
                        "prompt_enhancer": prompt_enhancer,
                    },
                }
            ],
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
            prefix="# Create a fast binary search\n",
            file_name="main.py",
            editor_lang="python",
            model_provider="anthropic",
            stream=False,
            user=mock_generations.call_args.kwargs["user"],
            snowplow_event_context=expected_snowplow_event,
            prompt_enhancer=prompt_enhancer,
            suffix="\n",
        )

    class TestModelMetadataSelection:
        @dataclass
        class Case:
            model_metadata: Optional[dict]
            expected_arguments: tuple[
                str, str, Optional[ModelMetadata], Optional[BaseTool]
            ]

        CASE_WITH_MODEL_PARAMS = Case(
            model_metadata={
                "api_key": "token",
                "endpoint": "http://localhost:4000",
                "identifier": "provider/some-model",
                "name": "mistral",
                "provider": "litellm",
            },
            expected_arguments=(
                "code_suggestions/generations",
                "2.0.1",
                ModelMetadata(
                    name="mistral",
                    provider="litellm",
                    endpoint=AnyUrl("http://localhost:4000"),
                    api_key="token",
                    identifier="provider/some-model",
                    llm_definition=LLMDefinition(
                        name="Mistral",
                        gitlab_identifier="mistral",
                        max_context_tokens=128000,
                        family=["mistral", "litellm"],
                        params={
                            "model": "mistral",
                            "temperature": 0.0,
                            "max_tokens": 4096,
                        },
                    ),
                    family=["mistral", "litellm"],
                    friendly_name="Mistral",
                ),
                None,
            ),
        )

        CASE_WITHOUT_MODEL_PARAMS = Case(
            model_metadata=None,
            expected_arguments=("code_suggestions/generations", "2.0.1", None, None),
        )

        @pytest.mark.parametrize(
            "test_case", [CASE_WITH_MODEL_PARAMS, CASE_WITHOUT_MODEL_PARAMS]
        )
        def test_model_metadata_selection(
            self,
            mock_client: TestClient,
            mock_generations: Mock,
            route: str,
            test_case: Case,
        ):
            prompt_enhancer = {
                "examples_array": [],
                "trimmed_prefix": "# Create a fast binary search\n",
                "trimmed_suffix": "",
                "related_files": [],
                "related_snippets": [],
                "libraries": [],
                "user_instruction": "Generate the best possible code based on instructions.",
            }

            data = {
                "prompt_components": [
                    {
                        "type": "code_editor_generation",
                        "payload": {
                            "file_name": "main.py",
                            "content_above_cursor": "# Create a fast binary search\n",
                            "content_below_cursor": "\n",
                            "language_identifier": "python",
                            "prompt_id": "code_suggestions/generations",
                            "prompt_version": "2.0.1",
                            "prompt_enhancer": prompt_enhancer,
                        },
                    }
                ],
                "model_metadata": test_case.model_metadata,
            }

            with patch("ai_gateway.prompts.registry.LocalPromptRegistry.get") as mock:
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
                mock.assert_called_with(*test_case.expected_arguments)

    @pytest.mark.parametrize(
        ("prompt", "want_called"),
        [
            # All cases should not call with_prompt_prepared since we now use Prompt Registry
            (
                "",
                False,
            ),
            (
                "some prompt",
                False,
            ),
            (
                None,
                False,
            ),
        ],
    )
    def test_prompt(
        self,
        mock_client,
        mock_ai_gateway_container: containers.Container,
        mock_generations: Mock,
        mock_with_prompt_prepared: Mock,
        prompt: str,
        want_called: bool,
        route: str,
    ):
        payload = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
            "language_identifier": "python",
            "prompt": prompt,
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

        # with_prompt_prepared should never be called since we now use Prompt Registry
        assert mock_with_prompt_prepared.called == want_called
        if want_called:
            mock_with_prompt_prepared.assert_called_with(prompt)

    @pytest.mark.parametrize(
        (
            "model_provider",
            "expected_code",
            "expected_response",
            "expected_model",
        ),
        [
            (
                "vertex-ai",
                200,
                {
                    "choices": [
                        {
                            "text": "test completion",
                            "index": 0,
                            "finish_reason": "length",
                        }
                    ]
                },
                {
                    "engine": "agent",
                    "name": "Code Generations Agent",
                    "lang": "python",
                },
            ),
            (
                "anthropic",
                200,
                {
                    "choices": [
                        {
                            "text": "test completion",
                            "index": 0,
                            "finish_reason": "length",
                        }
                    ]
                },
                {
                    "engine": "agent",
                    "name": "Claude Sonnet 4 Code Generations Agent",
                    "lang": "python",
                },
            ),
            # default provider
            (
                "",
                200,
                {
                    "choices": [
                        {
                            "text": "test completion",
                            "index": 0,
                            "finish_reason": "length",
                        }
                    ]
                },
                {
                    "engine": "agent",
                    "name": "Code Generations Agent",
                    "lang": "python",
                },
            ),
            # unknown provider
            (
                "some-provider",
                422,
                "",
                {},
            ),
        ],
    )
    def test_model_provider(
        self,
        mock_client: TestClient,
        mock_agent_model: Mock,
        model_provider: str,
        expected_code: int,
        expected_response: dict,
        expected_model: dict,
        route: str,
    ):
        payload = {
            "file_name": "main.py",
            "content_above_cursor": "# Create a fast binary search\n",
            "content_below_cursor": "\n",
            "language_identifier": "python",
            "model_provider": model_provider or None,
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

        assert response.status_code == expected_code

        if expected_code >= 400:
            # if we want 400+ status we don't need check the response
            return

        body = response.json()

        assert body["choices"] == expected_response["choices"]
        assert body["metadata"]["model"] == expected_model

    @pytest.mark.parametrize(
        ("model_provider", "expected_model"),
        [
            ("anthropic", "claude_sonnet_4_5_20250929"),
            ("vertex-ai", "claude_sonnet_4_5_20250929_vertex"),
        ],
    )
    def test_legacy_generation_model_provider_override(
        self, mock_client, route, mock_generations, model_provider, expected_model
    ):
        """Test that legacy generation requests respect model_provider parameter."""
        with patch(
            "ai_gateway.prompts.registry.LocalPromptRegistry.get_on_behalf"
        ) as mock_registry_get:
            payload = {
                "file_name": "main.py",
                "content_above_cursor": "# Create a fast binary search\n",
                "content_below_cursor": "\n",
                "language_identifier": "python",
                "model_provider": model_provider,
            }

            data = {
                "prompt_components": [
                    {
                        "type": "code_editor_generation",
                        "payload": payload,
                    }
                ],
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

            # Verify prompt_registry.get_on_behalf was called with model_metadata
            assert mock_registry_get.called
            call_kwargs = mock_registry_get.call_args.kwargs
            assert call_kwargs["prompt_id"] == "code_suggestions/generations"

            # Verify model_metadata was passed and contains the expected model
            model_metadata = call_kwargs.get("model_metadata")
            assert model_metadata is not None
            assert model_metadata.provider == "gitlab"
            assert expected_model == model_metadata.name


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


class TestUnauthorizedScopes:
    @pytest.fixture(name="auth_user")
    def auth_user_fixture(self):
        return CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(
                scopes=["unauthorized_scope"],
                subject="1234",
                gitlab_realm="self-managed",
            ),
        )

    def test_failed_authorization_scope(self, mock_client, route: str):
        response = mock_client.post(
            route,
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json={
                "prompt_components": [
                    {
                        "type": "code_editor_completion",
                        "payload": {
                            "file_name": "test",
                            "content_above_cursor": "def hello_world():",
                            "content_below_cursor": "",
                            "model_provider": "vertex-ai",
                        },
                    }
                ]
            },
        )

        assert response.status_code == 403
        assert response.json() == {"detail": "Unauthorized to access code suggestions"}


class TestIncomingRequest:
    @pytest.mark.parametrize(
        ("request_body", "expected_code"),
        [
            # valid request
            (
                {
                    "prompt_components": [
                        {
                            "type": "code_editor_completion",
                            "payload": {
                                "file_name": "test",
                                "content_above_cursor": "def hello_world():",
                                "content_below_cursor": "",
                                # FIXME: Forcing anthropic as vertex-ai is not working
                                "model_provider": "anthropic",
                                "model_name": "claude-3-5-sonnet-20240620",
                            },
                        },
                    ],
                },
                200,
            ),
            # unknown component type
            (
                {
                    "prompt_components": [
                        {
                            "type": "some_type",
                            "payload": {
                                "file_name": "test",
                                "content_above_cursor": "def hello_world():",
                                "content_below_cursor": "",
                                "model_name": "claude-3-5-sonnet-20240620",
                            },
                        },
                    ],
                },
                422,
            ),
            # too many prompt_components
            (
                {
                    "prompt_components": [
                        {
                            "type": "code_editor_completion",
                            "payload": {
                                "file_name": "test",
                                "content_above_cursor": "def hello_world():",
                                "content_below_cursor": "",
                                "model_name": "claude-3-5-sonnet-20240620",
                            },
                        },
                    ]
                    * 101,
                },
                422,
            ),
            # missing required field
            (
                {
                    "prompt_components": [
                        {
                            "type": "code_editor_completion",
                            "payload": {
                                "content_above_cursor": "def hello_world():",
                                "content_below_cursor": "",
                                "model_name": "claude-3-5-sonnet-20240620",
                            },
                        },
                    ],
                },
                422,
            ),
        ],
    )
    def test_valid_request(
        self,
        mock_client: TestClient,
        mock_completions: Mock,
        request_body: dict,
        expected_code: int,
        route: str,
    ):
        response = mock_client.post(
            route,
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-GitLab-Instance-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json=request_body,
        )

        assert response.status_code == expected_code


class TestUnauthorizedIssuer:
    @pytest.fixture(name="auth_user")
    def auth_user_fixture(self):
        return CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(
                scopes=["complete_code"],
                subject="1234",
                gitlab_realm="self-managed",
                issuer="gitlab-ai-gateway",
            ),
        )

    def test_failed_authorization_scope(self, mock_client, route: str):
        response = mock_client.post(
            route,
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Global-User-Id": "1234",
                "X-GitLab-Realm": "self-managed",
            },
            json={
                "prompt_components": [
                    {
                        "type": "code_editor_completion",
                        "payload": {
                            "file_name": "test",
                            "content_above_cursor": "def hello_world():",
                            "content_below_cursor": "",
                            "model_provider": "vertex-ai",
                        },
                    }
                ]
            },
        )

        assert response.status_code == 403
        assert response.json() == {"detail": "Unauthorized to access code suggestions"}


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
                    llm_definition=LLMDefinition(
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
                    llm_definition=LLMDefinition(
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
                exclude={"llm_definition", "family", "friendly_name"},
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
                exclude={"llm_definition", "family", "friendly_name"},
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
