# pylint: disable=line-too-long,file-naming-for-tests
import asyncio
from typing import Any, List, Optional
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException, Request
from gitlab_cloud_connector import CloudConnectorUser, GitLabUnitPrimitive, UserClaims
from langchain_core.messages.ai import AIMessage, AIMessageChunk
from pydantic import AnyUrl
from structlog.testing import capture_logs

from ai_gateway.api.client_disconnect import ClientDisconnectedError
from ai_gateway.api.v1 import api_router
from ai_gateway.model_metadata import (
    AmazonQModelMetadata,
    ModelMetadata,
    TypeModelMetadata,
)
from ai_gateway.model_selection.model_selection_config import (
    ChatAmazonQDefinition,
    ChatLiteLLMDefinition,
)
from ai_gateway.model_selection.models import ChatAmazonQParams, ChatLiteLLMParams
from tests.conftest import FakeModel


@pytest.fixture(name="fast_api_router", scope="class")
def fast_api_router_fixture():
    return api_router


class TestPrompt:
    @pytest.mark.parametrize(
        (
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
                {"name": "John", "age": 20},
                None,
                None,
                ("test", None, None, None),
                200,
                "Hi John!",
                ["1.0.0"],
            ),
            (
                {"name": "John", "age": 20},
                "^2.0.0",
                None,
                ("test", "^2.0.0", None, None),
                200,
                "Hi John!",
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
                    llm_definition=ChatLiteLLMDefinition(
                        gitlab_identifier="mistral",
                        name="Mistral",
                        max_context_tokens=128000,
                    ),
                ),
                (
                    "test",
                    None,
                    ModelMetadata(
                        name="mistral",
                        provider="litellm",
                        endpoint=AnyUrl("http://localhost:4000"),
                        api_key="token",
                        llm_definition=ChatLiteLLMDefinition(
                            name="Mistral",
                            gitlab_identifier="mistral",
                            max_context_tokens=128000,
                            family=["mistral"],
                            params=ChatLiteLLMParams(
                                model="mistral",
                                temperature=0.0,
                                max_tokens=4096,
                            ),
                        ),
                        friendly_name="Mistral",
                        is_custom_model=True,
                    ),
                    None,
                ),
                200,
                "Hi John!",
                ["1.0.0"],
            ),
            (
                {"name": "John", "age": 20},
                None,
                AmazonQModelMetadata(
                    name="amazon_q",
                    provider="amazon_q",
                    role_arn="role-arn",
                    llm_definition=ChatAmazonQDefinition(
                        gitlab_identifier="amazon_q",
                        name="Amazon Q",
                        max_context_tokens=200000,
                    ),
                ),
                (
                    "test",
                    None,
                    AmazonQModelMetadata(
                        name="amazon_q",
                        provider="amazon_q",
                        role_arn="role-arn",
                        llm_definition=ChatAmazonQDefinition(
                            gitlab_identifier="amazon_q",
                            name="Amazon Q",
                            max_context_tokens=200000,
                            family=["amazon_q"],
                            params=ChatAmazonQParams(
                                model="amazon_q",
                            ),
                        ),
                        friendly_name="Amazon Q",
                    ),
                    None,
                ),
                200,
                "Hi John!",
                ["1.0.0"],
            ),
            (
                {"name": "John", "age": 20},
                "^2.0.0",
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
                ("test", None, None, None),
                404,
                {"detail": "Prompt 'test' not found"},
                None,
            ),
            (
                {"name": "John"},
                None,
                None,
                ("test", None, None, None),
                422,
                {
                    "detail": "\"Input to ChatPromptTemplate is missing variables {'age'}.  Expected: ['age', 'current_date', 'name'] Received: ['name', 'current_date']"
                },
                ["1.0.0"],
            ),
        ],
    )
    @pytest.mark.usefixtures("frozen_datetime_now", "custom_models_gateway")
    def test_request(
        self,
        mock_registry_get,
        mock_client,
        mock_track_internal_event,
        unit_primitive: GitLabUnitPrimitive,
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
                    exclude={"llm_definition", "friendly_name"},
                    mode="json",
                ),
            },
        )

        mock_registry_get.compatible_versions = compatible_versions
        mock_registry_get.assert_called_with(*expected_get_args, is_graph_node=False)
        assert response.status_code == expected_status

        actual_response = response.json()
        if isinstance(expected_response, str):
            assert actual_response == expected_response
        else:
            assert expected_response["detail"] in actual_response["detail"]

        if compatible_versions is not None and len(compatible_versions) > 0:
            mock_track_internal_event.assert_called_once_with(
                f"request_{unit_primitive.value}",
                category="ai_gateway.api.v1.prompts.invoke",
            )
        else:
            mock_track_internal_event.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("frozen_datetime_now")
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

        mock_registry_get.assert_called_with(
            "test", "^2.0.0", None, None, is_graph_node=False
        )
        assert response.status_code == 200
        assert response.text == "Hi John!"
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


def _post_prompt(mock_client, stream: bool = False):
    return mock_client.post(
        "/prompts/test",
        headers={
            "Authorization": "Bearer 12345",
            "X-Gitlab-Authentication-Type": "oidc",
        },
        json={"inputs": {"name": "John", "age": 20}, "stream": stream},
    )


def _logged_feature_category(cap_logs) -> Optional[str]:
    [access_log] = [log for log in cap_logs if log.get("method") == "POST"]
    return access_log.get("meta.feature_category")


@pytest.mark.usefixtures("frozen_datetime_now", "mock_registry_get")
class TestPromptFeatureCategory:
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        ("unit_primitive", "expected_category"),
        [
            (GitLabUnitPrimitive.COMPLETE_CODE, "code_suggestions"),
            (GitLabUnitPrimitive.EXPLAIN_VULNERABILITY, "vulnerability_management"),
            # Not a GitLabFeatureCategory member, so it's recorded as the catalog's string
            (GitLabUnitPrimitive.DUO_AGENT_PLATFORM, "duo_workflow"),
        ],
    )
    def test_category_of_prompt_unit_primitive(
        self, mock_client, stream: bool, expected_category: str
    ):
        with capture_logs() as cap_logs:
            response = _post_prompt(mock_client, stream=stream)

        assert response.status_code == 200
        assert _logged_feature_category(cap_logs) == expected_category

    @pytest.mark.parametrize(
        ("compatible_versions", "expected_status"),
        [
            (None, 404),  # prompt not found
            ([], 400),  # no version matches the constraint
        ],
    )
    def test_default_category_when_prompt_is_not_resolved(
        self, mock_client, expected_status: int
    ):
        with capture_logs() as cap_logs:
            response = _post_prompt(mock_client)

        assert response.status_code == expected_status
        assert _logged_feature_category(cap_logs) == "ai_abstraction_layer"


class TestUnauthorizedScopes:
    @pytest.fixture(name="auth_user")
    def auth_user_fixture(self):
        return CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(scopes=["unauthorized_scope"]),
        )

    @pytest.mark.usefixtures("mock_ai_gateway_container", "mock_registry_get")
    def test_failed_authorization_scope(self, mock_client):
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

    @pytest.mark.usefixtures("mock_ai_gateway_container", "mock_registry_get")
    def test_failed_authorization_keeps_default_category(self, mock_client):
        with capture_logs() as cap_logs:
            response = _post_prompt(mock_client)

        assert response.status_code == 403
        assert _logged_feature_category(cap_logs) == "ai_abstraction_layer"


class TestMisdirectedRequest:
    @pytest.fixture(name="mock_model_misdirection")
    def mock_model_misdirection_fixture(self):
        with patch("ai_gateway.prompts.base.Prompt.ainvoke") as mock:
            mock.side_effect = HTTPException(
                status_code=401, detail="Invalid credentials"
            )

            yield mock

    @pytest.mark.usefixtures("mock_model_misdirection", "custom_models_gateway")
    def test_misdirected_request(self, mock_registry_get, mock_client, model_metadata):
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
                    exclude={"llm_definition", "friendly_name"},
                    mode="json",
                ),
            },
        )
        mock_registry_get.assert_called_with(
            "test", None, model_metadata, None, is_graph_node=False
        )
        assert response.status_code == 421
        assert response.json() == {"detail": "401: Unauthorized"}


class TestClientDisconnect:
    """The client leaves while the model call is in flight."""

    @pytest.fixture(name="model_factory")
    def model_factory_fixture(self):
        return lambda *args, **kwargs: FakeModel(
            model_engine="fake-engine",
            model_name="fake-model",
            responses=["Hi John!"],
        )

    @pytest.fixture(name="model_call")
    def model_call_fixture(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def hanging(*_args, **_kwargs):
            started.set()
            try:
                # Bounded, so the test fails instead of hanging if nothing cancels it
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        # The client leaves only once the model call is in flight, so a fast first poll
        # can't cancel the call before it reaches the model
        disconnected = AsyncMock(side_effect=lambda: started.is_set())

        with (
            patch.object(FakeModel, "_agenerate", side_effect=hanging) as mock,
            patch.object(Request, "is_disconnected", disconnected),
            patch(
                "ai_gateway.api.client_disconnect._DISCONNECT_POLL_INTERVAL_SECONDS",
                0.01,
            ),
        ):
            mock.cancelled = cancelled
            yield mock

    @pytest.mark.usefixtures("mock_registry_get")
    def test_cancels_the_model_call_without_retrying(self, mock_client, model_call):
        # Goes through the real `Prompt.ainvoke`, so the cancellation has to pass the
        # retry loop and the instrumentator to reach the model
        with pytest.raises(ClientDisconnectedError):
            _post_prompt(mock_client)

        assert model_call.cancelled.is_set()
        model_call.assert_awaited_once()

    @pytest.mark.parametrize(
        "config_values", [{"feature_flags": {"cancel_on_client_disconnect": False}}]
    )
    @pytest.mark.usefixtures("mock_registry_get")
    def test_waits_for_the_model_when_disabled(self, mock_client):
        agenerate = FakeModel._agenerate

        async def slower_than_the_poll(*args, **kwargs):
            await asyncio.sleep(0.05)
            return await agenerate(*args, **kwargs)

        disconnected = AsyncMock(return_value=True)

        # With the flag on, the poll would see the client gone and cancel this call
        with (
            patch.object(FakeModel, "_agenerate", slower_than_the_poll),
            patch.object(Request, "is_disconnected", disconnected),
            patch(
                "ai_gateway.api.client_disconnect._DISCONNECT_POLL_INTERVAL_SECONDS",
                0.01,
            ),
        ):
            response = _post_prompt(mock_client)

        assert response.status_code == 200
        assert response.json() == "Hi John!"
        disconnected.assert_not_awaited()


class TestContentBlockResponse:
    """Some providers (e.g. `google_genai`) return message content as a list of blocks rather than a plain string."""

    @pytest.fixture(name="model_factory")
    def model_factory_fixture(self):
        return lambda *args, **kwargs: FakeModel(
            model_engine="fake-engine",
            model_name="fake-model",
            responses=[AIMessage(content=[{"type": "text", "text": "Hi John!"}])],
        )

    def test_flattens_content_blocks_to_text(self, mock_client, mock_registry_get):
        response = mock_client.post(
            "/prompts/test",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            json={"inputs": {"name": "John", "age": 20}},
        )

        assert response.status_code == 200
        assert response.json() == "Hi John!"


class TestContentBlockResponseWithNonTextBlocks:
    """Non-text blocks (e.g. thinking/tool blocks) are skipped when flattening."""

    @pytest.fixture(name="model_factory")
    def model_factory_fixture(self):
        return lambda *args, **kwargs: FakeModel(
            model_engine="fake-engine",
            model_name="fake-model",
            responses=[
                AIMessage(
                    content=[
                        {"type": "thinking", "thinking": "Let me think..."},
                        {"type": "text", "text": "Hi "},
                        {"type": "text", "text": "John!"},
                    ]
                )
            ],
        )

    def test_skips_non_text_blocks(self, mock_client, mock_registry_get):
        response = mock_client.post(
            "/prompts/test",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            json={"inputs": {"name": "John", "age": 20}},
        )

        assert response.status_code == 200
        assert response.json() == "Hi John!"


class TestStreamingContentBlockResponse:
    """Block content is flattened on the streaming path too.

    `StreamingResponse` can only yield `str`/`bytes`, so before flattening a list-valued `chunk.content` raised
    rather than streaming. `FakeModel` streams by iterating the response string, so the chunks are produced by
    patching `Prompt.astream` directly.
    """

    @pytest.fixture(name="mock_astream_content_blocks")
    def mock_astream_content_blocks_fixture(self):
        async def _astream(*_args: Any, **_kwargs: Any):
            yield AIMessageChunk(content=[{"type": "text", "text": "Hi "}])
            yield AIMessageChunk(
                content=[{"type": "thinking", "thinking": "Let me think..."}]
            )
            yield AIMessageChunk(content=[{"type": "text", "text": "John!"}])

        with patch(
            "ai_gateway.prompts.base.Prompt.astream", side_effect=_astream
        ) as mock:
            yield mock

    @pytest.mark.usefixtures("mock_astream_content_blocks")
    def test_streams_flattened_text(self, mock_client, mock_registry_get):
        response = mock_client.post(
            "/prompts/test",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            json={"inputs": {"name": "John", "age": 20}, "stream": True},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert response.text == "Hi John!"
