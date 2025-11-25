from typing import Any, List, Optional
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from gitlab_cloud_connector import CloudConnectorUser, UserClaims
from pydantic import AnyUrl

from ai_gateway.api.v2 import api_router
from ai_gateway.instrumentators.model_requests import TokenUsage
from ai_gateway.model_metadata import (
    AmazonQModelMetadata,
    ModelMetadata,
    TypeModelMetadata,
)
from ai_gateway.model_selection import LLMDefinition


@pytest.fixture(name="fast_api_router", scope="class")
def fast_api_router_fixture():
    return api_router


@pytest.fixture(name="mock_get_usage_metadata")
def mock_get_usage_metadata_fixture(token_usage: TokenUsage | None):
    with patch(
        "ai_gateway.api.v2.prompts.invoke.get_token_usage", return_value=token_usage
    ) as mock:
        yield mock


class TestPrompt:
    @pytest.mark.usefixtures("mock_get_usage_metadata", "frozen_datetime_now")
    @pytest.mark.parametrize(
        (
            "inputs",
            "prompt_version",
            "input_model_metadata",
            "token_usage",
            "expected_get_args",
            "expected_status",
            "expected_response",
            "compatible_versions",
        ),
        [
            (
                {"name": "John", "age": 20},
                None,
                None,
                None,
                ("test", "^1.0.0", None, None),
                200,
                {"content": "Hi John!"},
                ["1.0.0"],
            ),
            (
                {"name": "John", "age": 20},
                None,
                None,
                {"model": {"input_tokens": 10, "output_tokens": 20}},
                ("test", "^1.0.0", None, None),
                200,
                {
                    "content": "Hi John!",
                    "usage": {"model": {"input_tokens": 10, "output_tokens": 20}},
                },
                ["1.0.0"],
            ),
            (
                {"name": "John", "age": 20},
                "^2.0.0",
                None,
                None,
                ("test", "^2.0.0", None, None),
                200,
                {"content": "Hi John!"},
                ["2.0.0"],
            ),
            (
                {"name": "John", "age": 20},
                None,
                ModelMetadata(
                    name="mistral",
                    provider="litellm",
                    endpoint=AnyUrl("http://localhost:4000"),
                    api_key="token",
                    llm_definition=LLMDefinition(
                        gitlab_identifier="mistral", name="Mistral"
                    ),
                ),
                None,
                (
                    "test",
                    "^1.0.0",
                    ModelMetadata(
                        name="mistral",
                        provider="litellm",
                        endpoint=AnyUrl("http://localhost:4000"),
                        api_key="token",
                        llm_definition=LLMDefinition(
                            name="Mistral",
                            gitlab_identifier="mistral",
                            family=["mistral"],
                            params={
                                "model": "mistral",
                                "temperature": 0.0,
                                "max_tokens": 4096,
                            },
                        ),
                        family=["mistral"],
                        friendly_name="Mistral",
                    ),
                    None,
                ),
                200,
                {"content": "Hi John!"},
                ["1.0.0"],
            ),
            (
                {"name": "John", "age": 20},
                None,
                AmazonQModelMetadata(
                    name="amazon_q",
                    provider="amazon_q",
                    role_arn="role-arn",
                    llm_definition=LLMDefinition(
                        gitlab_identifier="amazon_q", name="Amazon Q"
                    ),
                ),
                None,
                (
                    "test",
                    "^1.0.0",
                    AmazonQModelMetadata(
                        name="amazon_q",
                        provider="amazon_q",
                        role_arn="role-arn",
                        llm_definition=LLMDefinition(
                            gitlab_identifier="amazon_q",
                            name="Amazon Q",
                            family=["amazon_q"],
                            params={
                                "model": "amazon_q",
                                "model_class_provider": "amazon_q",
                            },
                        ),
                        family=["amazon_q"],
                        friendly_name="Amazon Q",
                    ),
                    None,
                ),
                200,
                {"content": "Hi John!"},
                ["1.0.0"],
            ),
            (
                {"name": "John", "age": 20},
                "^2.0.0",
                None,
                None,
                ("test", "^2.0.0", None, None),
                400,
                {"detail": "No prompt version found matching the query"},
                [],
            ),
            (
                {"name": "John", "age": 20},
                None,
                None,
                None,
                ("test", "^1.0.0", None, None),
                404,
                {"detail": "Prompt 'test' not found"},
                None,
            ),
            (
                {"name": "John"},
                None,
                None,
                None,
                ("test", "^1.0.0", None, None),
                422,
                {
                    "detail": "\"Input to ChatPromptTemplate is missing variables {'age'}.  Expected: ['age', 'current_date', 'name'] Received: ['name', 'current_date']"
                },
                ["1.0.0"],
            ),
        ],
    )
    def test_request(
        self,
        mock_registry_get,
        mock_client,
        mock_track_internal_event,
        inputs: dict[str, str],
        prompt_version: Optional[str],
        input_model_metadata: Optional[TypeModelMetadata],
        token_usage: TokenUsage | None,
        expected_get_args: dict,
        expected_status: int,
        expected_response: Any,
        compatible_versions: Optional[List[str]],
    ):
        response = mock_client.post(
            "/prompts/test",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            json={
                "inputs": inputs,
                "prompt_version": prompt_version,
                "model_metadata": input_model_metadata
                and input_model_metadata.model_dump(
                    exclude={"llm_definition", "family", "friendly_name"},
                    mode="json",
                ),
            },
        )

        mock_registry_get.compatible_versions = compatible_versions
        mock_registry_get.assert_called_with(*expected_get_args)
        assert response.status_code == expected_status

        actual_response = response.json()

        if "detail" in expected_response:
            assert expected_response["detail"] in actual_response["detail"]
        else:
            assert actual_response == expected_response

        if compatible_versions is not None and len(compatible_versions) > 0:
            mock_track_internal_event.assert_called_once_with(
                "request_explain_vulnerability",
                category="ai_gateway.api.v2.prompts.invoke",
            )
        else:
            mock_track_internal_event.assert_not_called()

    @pytest.mark.usefixtures("mock_get_usage_metadata", "frozen_datetime_now")
    @pytest.mark.parametrize(
        ("token_usage", "expected_response"),
        [
            (None, {"content": "Hi John!"}),
            (
                {"model": {"input_tokens": 10, "output_tokens": 20}},
                {
                    "content": "Hi John!",
                    "usage": {"model": {"input_tokens": 10, "output_tokens": 20}},
                },
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_streaming_request(
        self,
        mock_client,
        mock_registry_get,
        expected_response,
    ):
        response = mock_client.post(
            "/prompts/test",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            json={
                "inputs": {"name": "John", "age": 20},
                "prompt_version": "^2.0.0",
                "stream": True,
            },
        )

        mock_registry_get.assert_called_with("test", "^2.0.0", None, None)
        assert response.status_code == 200
        assert response.json() == expected_response
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


class TestUnauthorizedScopes:
    @pytest.fixture(name="auth_user")
    def auth_user_fixture(self):
        return CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(scopes=["unauthorized_scope"]),
        )

    def test_failed_authorization_scope(
        self, mock_ai_gateway_container, mock_client, mock_registry_get
    ):
        response = mock_client.post(
            "/prompts/test",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            json={"inputs": {}},
        )

        assert response.status_code == 403
        assert response.json() == {"detail": "Unauthorized to access 'test'"}


class TestMisdirectedRequest:
    @pytest.fixture(name="mock_model_misdirection")
    def mock_model_misdirection_fixture(self):
        with patch("ai_gateway.prompts.base.Prompt.ainvoke") as mock:
            mock.side_effect = HTTPException(
                status_code=401, detail="Invalid credentials"
            )

            yield mock

    def test_misdirected_request(
        self, mock_registry_get, mock_model_misdirection, mock_client, model_metadata
    ):
        response = mock_client.post(
            "/prompts/test",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            json={
                "inputs": {"name": "John", "age": 20},
                "model_metadata": model_metadata
                and model_metadata.model_dump(
                    exclude={"llm_definition", "family", "friendly_name"},
                    mode="json",
                ),
            },
        )
        mock_registry_get.assert_called_with("test", "^1.0.0", model_metadata, None)
        assert response.status_code == 421
        assert response.json() == {"detail": "401: Unauthorized"}
