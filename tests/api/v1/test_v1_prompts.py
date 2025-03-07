from typing import Any, List, Optional, Type
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from gitlab_cloud_connector import CloudConnectorUser, GitLabUnitPrimitive, UserClaims
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage
from pydantic import AnyUrl

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.api.v1 import api_router
from ai_gateway.prompts import Prompt
from ai_gateway.prompts.typing import (
    AmazonQModelMetadata,
    ModelMetadata,
    TypeModelMetadata,
)


class FakeModel(SimpleChatModel):
    expected_message: str
    response: str

    @property
    def _llm_type(self) -> str:
        return "fake-provider"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": "fake-model"}

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        assert self.expected_message == messages[0].content

        return self.response


@pytest.fixture
def model_factory():
    yield lambda model, **kwargs: FakeModel(
        expected_message="Hi, I'm John and I'm 20 years old",
        response="Hi John!",
    )


@pytest.fixture
def prompt_template():
    yield {"system": "Hi, I'm {{name}} and I'm {{age}} years old"}


@pytest.fixture
def compatible_versions():
    yield ["1.0.0"]


@pytest.fixture
def mock_registry_get(
    request,
    prompt_class: Optional[Type[Prompt]],
    compatible_versions: Optional[List[str]],
):
    with patch("ai_gateway.prompts.registry.LocalPromptRegistry.get") as mock:
        if prompt_class and len(compatible_versions) > 0:
            mock.return_value = request.getfixturevalue("prompt")
        elif prompt_class and not compatible_versions:
            mock.side_effect = ValueError("No prompt version found matching the query:")
        else:
            mock.side_effect = KeyError()

        yield mock


@pytest.fixture(scope="class")
def fast_api_router():
    return api_router


@pytest.fixture
def unit_primitives():
    yield ["explain_vulnerability"]


@pytest.fixture
def auth_user(unit_primitives: list[GitLabUnitPrimitive]):
    return CloudConnectorUser(
        authenticated=True, claims=UserClaims(scopes=unit_primitives)
    )


class TestPrompt:
    @pytest.mark.parametrize(
        (
            "prompt_class",
            "inputs",
            "prompt_version",
            "model_metadata",
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
                ("test", "^1.0.0", None),
                200,
                "Hi John!",
                ["1.0.0"],
            ),
            (
                Prompt,
                {"name": "John", "age": 20},
                "^2.0.0",
                None,
                ("test", "^2.0.0", None),
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
                    ),
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
                    ),
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
                ("test", "^2.0.0", None),
                400,
                {"detail": "No prompt version found matching the query"},
                [],
            ),
            (
                None,
                {"name": "John", "age": 20},
                None,
                None,
                ("test", "^1.0.0", None),
                404,
                {"detail": "Prompt 'test' not found"},
                None,
            ),
            (
                Prompt,
                {"name": "John"},
                None,
                None,
                ("test", "^1.0.0", None),
                422,
                {
                    "detail": "\"Input to ChatPromptTemplate is missing variables {'age'}.  Expected: ['age', 'name'] Received: ['name']"
                },
                ["1.0.0"],
            ),
        ],
    )
    def test_request(
        self,
        prompt_class,
        mock_registry_get,
        mock_client,
        mock_track_internal_event,
        inputs: dict[str, str],
        prompt_version: Optional[str],
        model_metadata: Optional[TypeModelMetadata],
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
                "model_metadata": model_metadata
                and model_metadata.model_dump(mode="json"),
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

        if prompt_class and len(compatible_versions) > 0:
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

        mock_registry_get.assert_called_with("test", "^2.0.0", None)
        assert response.status_code == 200
        assert response.text == "Hi John!"
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


class TestUnauthorizedScopes:
    @pytest.fixture
    def auth_user(self):
        return CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(scopes=["unauthorized_scope"]),
        )

    def test_failed_authorization_scope(
        self, mock_container, mock_client, mock_registry_get
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
    @pytest.fixture
    def mock_model_misdirection(self):
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
                and model_metadata.model_dump(mode="json"),
            },
        )
        mock_registry_get.assert_called_with("test", "^1.0.0", model_metadata)
        assert response.status_code == 421
        assert response.json() == {"detail": "401: Unauthorized"}
