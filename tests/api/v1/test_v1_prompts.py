from typing import Any, List, Optional
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from gitlab_cloud_connector import CloudConnectorUser, UserClaims
from pydantic import AnyUrl

from ai_gateway.api.v1 import api_router
from ai_gateway.model_metadata import (
    AmazonQModelMetadata,
    ModelMetadata,
    TypeModelMetadata,
)
from ai_gateway.prompts import Prompt


@pytest.fixture(name="fast_api_router", scope="class")
def fast_api_router_fixture():
    return api_router


class TestPrompt:
    @pytest.mark.parametrize(
        (
            "prompt_class",
            "inputs",
            "prompt_version",
            "input_model_metadata",
            "expected_get_args",
            "expected_status",
            "expected_response",
            "compatible_versions",
        ),
        [
            (
                Prompt,
                {"name": "John", "age": 20},
                None,
                None,
                ("test", "^1.0.0", None, None),
                200,
                "Hi John!",
                ["1.0.0"],
            ),
            (
                Prompt,
                {"name": "John", "age": 20},
                "^2.0.0",
                None,
                ("test", "^2.0.0", None, None),
                200,
                "Hi John!",
                ["2.0.0"],
            ),
            (
                Prompt,
                {"name": "John", "age": 20},
                None,
                ModelMetadata(
                    name="mistral",
                    provider="litellm",
                    endpoint=AnyUrl("http://localhost:4000"),
                    api_key="token",
                ),
                (
                    "test",
                    "^1.0.0",
                    ModelMetadata(
                        name="mistral",
                        provider="litellm",
                        endpoint=AnyUrl("http://localhost:4000"),
                        api_key="token",
                        llm_definition_params={
                            "model": "mistral",
                            "temperature": 0.0,
                            "max_tokens": 4096,
                        },
                        family=["mistral"],
                    ),
                    None,
                ),
                200,
                "Hi John!",
                ["1.0.0"],
            ),
            (
                Prompt,
                {"name": "John", "age": 20},
                None,
                AmazonQModelMetadata(
                    name="amazon_q",
                    provider="amazon_q",
                    role_arn="role-arn",
                ),
                (
                    "test",
                    "^1.0.0",
                    AmazonQModelMetadata(
                        name="amazon_q",
                        provider="amazon_q",
                        role_arn="role-arn",
                        llm_definition_params={"model": "amazon_q"},
                        family=["amazon_q"],
                    ),
                    None,
                ),
                200,
                "Hi John!",
                ["1.0.0"],
            ),
            (
                Prompt,
                {"name": "John", "age": 20},
                "^2.0.0",
                None,
                ("test", "^2.0.0", None, None),
                400,
                {"detail": "No prompt version found matching the query"},
                [],
            ),
            (
                None,
                {"name": "John", "age": 20},
                None,
                None,
                ("test", "^1.0.0", None, None),
                404,
                {"detail": "Prompt 'test' not found"},
                None,
            ),
            (
                Prompt,
                {"name": "John"},
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
        prompt_class,
        mock_registry_get,
        frozen_datetime_now,
        mock_client,
        mock_track_internal_event,
        inputs: dict[str, str],
        prompt_version: Optional[str],
        input_model_metadata: Optional[TypeModelMetadata],
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
                    exclude={"llm_definition_params", "family"}, mode="json"
                ),
            },
        )

        mock_registry_get.compatible_versions = compatible_versions
        mock_registry_get.assert_called_with(*expected_get_args)
        assert response.status_code == expected_status

        actual_response = response.json()
        if isinstance(expected_response, str):
            assert actual_response == expected_response
        else:
            assert expected_response["detail"] in actual_response["detail"]

        if (
            prompt_class
            and compatible_versions is not None
            and len(compatible_versions) > 0
        ):
            mock_track_internal_event.assert_called_once_with(
                "request_explain_vulnerability",
                category="ai_gateway.api.v1.prompts.invoke",
            )
        else:
            mock_track_internal_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_streaming_request(
        self,
        mock_client,
        mock_registry_get,
        frozen_datetime_now,
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
        assert response.text == "Hi John!"
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
                    exclude={"llm_definition_params", "family"}, mode="json"
                ),
            },
        )
        mock_registry_get.assert_called_with("test", "^1.0.0", model_metadata, None)
        assert response.status_code == 421
        assert response.json() == {"detail": "401: Unauthorized"}
