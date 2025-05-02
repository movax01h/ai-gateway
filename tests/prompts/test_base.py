import os
from typing import Optional, Type
from unittest import mock
from unittest.mock import Mock, call

import pytest
from anthropic import APITimeoutError, AsyncAnthropic
from gitlab_cloud_connector import GitLabUnitPrimitive, WrongUnitPrimitives
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages.ai import UsageMetadata
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from litellm.exceptions import Timeout
from pydantic import AnyUrl

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.config import ConfigModelLimits
from ai_gateway.model_metadata import (
    AmazonQModelMetadata,
    ModelMetadata,
    TypeModelMetadata,
    current_model_metadata_context,
)
from ai_gateway.models.v2.anthropic_claude import ChatAnthropic
from ai_gateway.prompts import BasePromptRegistry, Prompt
from ai_gateway.prompts.config.base import PromptParams
from ai_gateway.prompts.typing import Model


@pytest.fixture
def mock_watcher():
    return mock.AsyncMock()


@pytest.fixture
def mock_watch(mock_watcher: Mock):
    with mock.patch(
        "ai_gateway.instrumentators.model_requests.ModelRequestInstrumentator.watch"
    ) as mock_watch:
        mock_watch.return_value.__enter__.return_value = mock_watcher

        yield mock_watch


class TestPrompt:
    @pytest.mark.parametrize(
        ("model_params", "expected_model_engine"),
        [
            ({"model_class_provider": "litellm"}, "litellm"),
            (
                {"model_class_provider": "litellm", "custom_llm_provider": "my_engine"},
                "my_engine",
            ),
        ],
    )
    def test_initialize(
        self,
        prompt: Prompt,
        unit_primitives: list[GitLabUnitPrimitive],
        model_params: dict,
        expected_model_engine: str,
    ):
        assert prompt.name == "test_prompt"
        assert prompt.unit_primitives == unit_primitives
        assert prompt.model_provider == model_params["model_class_provider"]
        assert prompt.model_engine == expected_model_engine
        assert isinstance(prompt.bound, Runnable)

    def test_build_prompt_template(self, prompt_template, model_config):
        prompt_template = Prompt._build_prompt_template(prompt_template, model_config)

        assert prompt_template == ChatPromptTemplate.from_messages(
            [("system", "Hi, I'm {{name}}"), ("user", "{{content}}")],
            template_format="jinja2",
        )

    def test_instrumentator(self, model_engine: str, model_name: str, prompt: Prompt):
        assert prompt.instrumentator.labels == {
            "model_engine": model_engine,
            "model_name": model_name,
        }

    @pytest.mark.asyncio
    @mock.patch("ai_gateway.prompts.base.get_request_logger")
    async def test_ainvoke(
        self,
        mock_get_logger: mock.Mock,
        mock_watch: mock.Mock,
        prompt: Prompt,
        model_response: str,
    ):
        mock_logger = mock.MagicMock()
        mock_get_logger.return_value = mock_logger

        response = await prompt.ainvoke({"name": "Duo", "content": "What's up?"})

        assert response.content == model_response

        mock_logger.info.assert_called_with(
            "Performing LLM request",
            prompt="System: Hi, I'm Duo\nHuman: What's up?",
        )

        mock_watch.assert_called_with(stream=False)

    @pytest.mark.asyncio
    @mock.patch("ai_gateway.prompts.base.get_request_logger")
    async def test_astream(
        self,
        mock_get_logger: mock.Mock,
        mock_watch: mock.Mock,
        mock_watcher: mock.Mock,
        prompt: Prompt,
        model_response: str,
    ):
        response = ""

        mock_logger = mock.MagicMock()
        mock_get_logger.return_value = mock_logger

        async for c in prompt.astream({"name": "Duo", "content": "What's up?"}):
            response += c.content

            mock_watcher.afinish.assert_not_awaited()  # Make sure we don't finish prematurely

        mock_logger.info.assert_called_with(
            "Performing LLM request",
            prompt="System: Hi, I'm Duo\nHuman: What's up?",
        )

        assert response == model_response

        mock_watch.assert_called_with(stream=True)

        mock_watcher.afinish.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("usage_metadata"),
        [UsageMetadata(input_tokens=1, output_tokens=2, total_tokens=3)],
    )
    async def test_ainvoke_handle_usage_metadata(
        self,
        mock_watch: mock.Mock,
        mock_watcher: mock.Mock,
        internal_event_client: mock.Mock,
        prompt: Prompt,
        usage_metadata: UsageMetadata,
    ):
        prompt.internal_event_client = internal_event_client
        await prompt.ainvoke({"name": "Duo", "content": "What's up?"})

        _assert_usage_metadata_handling(
            mock_watcher, internal_event_client, prompt, usage_metadata
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("usage_metadata"),
        [UsageMetadata(input_tokens=1, output_tokens=2, total_tokens=3)],
    )
    async def test_astream_handle_usage_metadata(
        self,
        mock_watch: mock.Mock,
        mock_watcher: mock.Mock,
        internal_event_client: mock.Mock,
        prompt: Prompt,
        usage_metadata: UsageMetadata,
    ):
        prompt.internal_event_client = internal_event_client

        # Consume stream
        async for _ in prompt.astream({"name": "Duo", "content": "What's up?"}):
            pass

        _assert_usage_metadata_handling(
            mock_watcher, internal_event_client, prompt, usage_metadata
        )


def _assert_usage_metadata_handling(
    mock_watcher: mock.Mock,
    internal_event_client: mock.Mock,
    prompt: Prompt,
    usage_metadata: UsageMetadata,
):
    mock_watcher.register_token_usage.assert_called_once_with(
        prompt.model_name, usage_metadata
    )
    for unit_primitive in prompt.unit_primitives:
        internal_event_client.track_event.assert_any_call(
            f"token_usage_{unit_primitive}",
            category="ai_gateway.prompts.base",
            input_tokens=usage_metadata["input_tokens"],
            output_tokens=usage_metadata["output_tokens"],
            total_tokens=usage_metadata["total_tokens"],
            model_engine=prompt.model_engine,
            model_name=prompt.model_name,
            model_provider=prompt.model_provider,
        )


@pytest.mark.skipif(
    # pylint: disable=direct-environment-variable-reference
    os.getenv("REAL_AI_REQUEST") is None,
    # pylint: enable=direct-environment-variable-reference
    reason="3rd party requests not enabled",
)
class TestPromptTimeout:
    @pytest.fixture
    def prompt_params(self):
        return PromptParams(timeout=0.1)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("model", "expected_exception"),
        [
            (
                ChatAnthropic(
                    async_client=AsyncAnthropic(), model="claude-3-sonnet-20240229"  # type: ignore[call-arg]
                ),
                APITimeoutError,
            ),
            (
                ChatLiteLLM(
                    model="claude-3-sonnet@20240229", custom_llm_provider="vertex_ai"  # type: ignore[call-arg]
                ),
                Timeout,
            ),
            (
                ChatLiteLLM(
                    model="claude-3-5-sonnet-v2@20241022", custom_llm_provider="vertex_ai"  # type: ignore[call-arg]
                ),
                Timeout,
            ),
        ],
    )
    async def test_timeout(
        self, prompt: Prompt, model: Model, expected_exception: Type
    ):
        with pytest.raises(expected_exception):
            await prompt.ainvoke(
                {"name": "Duo", "content": "Print pi with 400 decimals"}
            )


@pytest.fixture
def registry(
    internal_event_client: Mock, model_limits: ConfigModelLimits, prompt: Prompt
):
    class Registry(BasePromptRegistry):
        def __init__(self):
            self.internal_event_client = internal_event_client
            self.model_limits = model_limits

        def get(self, *args, **kwargs):
            return prompt

    return Registry()


class TestBaseRegistry:
    @pytest.mark.parametrize(
        (
            "unit_primitives",
            "scopes",
            "model_metadata",
            "success",
            "expected_internal_events",
        ),
        [
            (
                [GitLabUnitPrimitive.COMPLETE_CODE],
                ["complete_code"],
                None,
                True,
                [call("request_complete_code", category="ai_gateway.prompts.base")],
            ),
            (
                [GitLabUnitPrimitive.COMPLETE_CODE, GitLabUnitPrimitive.ASK_BUILD],
                ["complete_code", "ask_build"],
                None,
                True,
                [
                    call("request_complete_code", category="ai_gateway.prompts.base"),
                    call("request_ask_build", category="ai_gateway.prompts.base"),
                ],
            ),
            (
                [GitLabUnitPrimitive.COMPLETE_CODE],
                ["complete_code"],
                ModelMetadata(
                    name="mistral",
                    provider="litellm",
                    endpoint=AnyUrl("http://localhost:4000"),
                    api_key="token",
                ),
                True,
                [
                    call("request_complete_code", category="ai_gateway.prompts.base"),
                ],
            ),
            (
                [GitLabUnitPrimitive.AMAZON_Q_INTEGRATION],
                ["amazon_q_integration"],
                AmazonQModelMetadata(
                    name="amazon_q",
                    provider="amazon_q",
                    role_arn="role-arn",
                ),
                True,
                [
                    call(
                        "request_amazon_q_integration",
                        category="ai_gateway.prompts.base",
                    ),
                ],
            ),
            ([GitLabUnitPrimitive.COMPLETE_CODE], [], None, False, []),
            (
                [
                    GitLabUnitPrimitive.COMPLETE_CODE,
                    GitLabUnitPrimitive.ASK_BUILD,
                ],
                ["complete_code"],
                None,
                False,
                [],
            ),
        ],
    )
    def test_get_on_behalf(
        self,
        internal_event_client: Mock,
        registry: BasePromptRegistry,
        user: StarletteUser,
        prompt: Prompt,
        model_metadata: Optional[TypeModelMetadata],
        unit_primitives: list[GitLabUnitPrimitive],
        scopes: list[str],
        success: bool,
        expected_internal_events,
    ):
        if success:
            assert registry.get_on_behalf(user, "test", None, model_metadata) == prompt
            assert prompt.internal_event_client == internal_event_client

            if model_metadata:
                assert model_metadata._user == user

            internal_event_client.track_event.assert_has_calls(expected_internal_events)
        else:
            with pytest.raises(WrongUnitPrimitives):
                registry.get_on_behalf(user=user, prompt_id="test")

            internal_event_client.track_event.assert_not_called()

    @pytest.mark.parametrize(
        (
            "unit_primitives",
            "scopes",
            "internal_event_category",
            "expected_internal_events",
        ),
        [
            (
                [GitLabUnitPrimitive.COMPLETE_CODE, GitLabUnitPrimitive.ASK_BUILD],
                ["complete_code", "ask_build"],
                "my_category",
                [
                    call("request_complete_code", category="my_category"),
                    call("request_ask_build", category="my_category"),
                ],
            ),
        ],
    )
    def test_get_on_behalf_with_internal_event_category(
        self,
        internal_event_client: Mock,
        registry: BasePromptRegistry,
        user: StarletteUser,
        prompt: Prompt,
        unit_primitives: list[GitLabUnitPrimitive],
        scopes: list[str],
        internal_event_category: str,
        expected_internal_events,
    ):
        registry.get_on_behalf(
            user=user, prompt_id="test", internal_event_category=internal_event_category
        )

        internal_event_client.track_event.assert_has_calls(expected_internal_events)

    @pytest.mark.parametrize(
        ("model_metadata", "unit_primitives", "scopes"),
        [
            (
                ModelMetadata(
                    name="mistral",
                    provider="litellm",
                    endpoint=AnyUrl("http://localhost:4000"),
                    api_key="token",
                ),
                [GitLabUnitPrimitive.COMPLETE_CODE],
                ["complete_code"],
            )
        ],
    )
    def test_get_on_behalf_with_context_model_metadata(
        self,
        registry: BasePromptRegistry,
        user: StarletteUser,
        prompt: Prompt,
        unit_primitives: list[GitLabUnitPrimitive],
        model_metadata: TypeModelMetadata,
    ):
        current_model_metadata_context.set(model_metadata)

        assert registry.get_on_behalf(user, "test", None) == prompt
