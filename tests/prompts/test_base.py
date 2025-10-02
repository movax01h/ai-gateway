# pylint: disable=too-many-lines
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Iterator, List, Optional, Type
from unittest import mock
from unittest.mock import Mock, PropertyMock, call

import pytest
from anthropic import APITimeoutError, AsyncAnthropic
from gitlab_cloud_connector import GitLabUnitPrimitive, WrongUnitPrimitives
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.ai import InputTokenDetails, UsageMetadata
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from litellm.exceptions import Timeout
from pydantic import AnyUrl
from pyfakefs.fake_filesystem import FakeFilesystem

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.config import ConfigModelLimits
from ai_gateway.instrumentators.model_requests import ModelRequestInstrumentator
from ai_gateway.model_metadata import (
    AmazonQModelMetadata,
    ModelMetadata,
    TypeModelMetadata,
    current_model_metadata_context,
)
from ai_gateway.models.v2.anthropic_claude import ChatAnthropic
from ai_gateway.prompts import BasePromptRegistry, Prompt
from ai_gateway.prompts.config.base import PromptConfig, PromptParams
from ai_gateway.prompts.typing import TypeModelFactory
from duo_workflow_service.tracking.llm_usage_context import (
    clear_workflow_checkpointer,
    set_workflow_checkpointer,
)
from lib.internal_events.context import InternalEventAdditionalProperties
from tests.conftest import FakeModel


@pytest.fixture(name="mock_watch")
def mock_watch_fixture() -> Generator[mock.MagicMock, None, None]:
    with mock.patch(
        "ai_gateway.prompts.base.ModelRequestInstrumentator.watch"
    ) as mock_watch:
        mock_watcher = mock.AsyncMock(spec=ModelRequestInstrumentator.WatchContainer)
        mock_watch.return_value.__enter__.return_value = mock_watcher

        yield mock_watch


class TestPrompt:
    # editorconfig-checker-disable
    @pytest.fixture(autouse=True)
    def mock_fs(self, fs: FakeFilesystem):
        ai_gateway_dir = Path(__file__).parent.parent.parent / "ai_gateway"
        model_selection_dir = ai_gateway_dir / "model_selection"
        prompts_definitions_dir = ai_gateway_dir / "prompts" / "definitions"

        fs.create_file(
            model_selection_dir / "models.yml",
            contents="""---
models:
  - name: Mistral
    gitlab_identifier: mistral
    params:
        model: mistral
  - name: Amazon Q
    gitlab_identifier: amazon_q
    params:
        model: amazon_q
""",
        )
        fs.create_file(
            model_selection_dir / "unit_primitives.yml",
            contents="""---
configurable_unit_primitives:
  - feature_setting: "duo_chat"
    unit_primitives:
      - "duo_chat"
    default_model: "mistral"
    selectable_models:
      - "mistral"
""",
        )
        fs.create_file(
            prompts_definitions_dir / "system.jinja",
            contents="Hi, I'm {{name}}",
        )

    # editorconfig-checker-enable

    @pytest.fixture(name="prompt_template")
    def prompt_template_fixture(self, request: Any):
        # Test inclusion and direct content
        tpl = {"system": "{% include 'system.jinja' %}", "user": "{{content}}"}

        if getattr(request, "param", None) and request.param.get(
            "with_messages_placeholder", None
        ):
            tpl["placeholder"] = "messages"

        return tpl

    @contextmanager
    def _mock_usage_metadata(
        self, model_name: str, usage_metadata: UsageMetadata
    ) -> Iterator[None]:
        with mock.patch(
            "ai_gateway.prompts.base.get_usage_metadata_callback"
        ) as mock_get_usage_callback:
            mock_callback = mock.MagicMock(usage_metadata={model_name: usage_metadata})
            mock_get_usage_callback.return_value.__enter__.return_value = mock_callback

            yield

    @pytest.mark.parametrize(
        ("model_params", "expected_model_engine"),
        [
            ({"model_class_provider": "litellm"}, "litellm"),
            (
                {
                    "model_class_provider": "litellm",
                    "custom_llm_provider": "my_engine",
                },
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

    def test_build_prompt_template(self, prompt_config: PromptConfig):
        prompt_template: Runnable = Prompt._build_prompt_template(prompt_config)

        assert prompt_template == ChatPromptTemplate.from_messages(
            [
                ("system", "{% include 'system.jinja' %}"),
                ("user", "{{content}}"),
            ],
            template_format="jinja2",
        )

    @pytest.mark.parametrize(
        "prompt_template", [{"with_messages_placeholder": True}], indirect=True
    )
    def test_build_prompt_template_with_placeholder(self, prompt_config: PromptConfig):
        prompt_template: Runnable = Prompt._build_prompt_template(prompt_config)

        assert prompt_template == ChatPromptTemplate.from_messages(
            [
                ("system", "{% include 'system.jinja' %}"),
                ("user", "{{content}}"),
                MessagesPlaceholder("messages"),
            ],
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

        expected_call = mock.call(
            "Performing LLM request", prompt="System: Hi, I'm Duo\nHuman: What's up?"
        )
        assert mock_logger.info.mock_calls[0] == expected_call

        mock_watch.assert_called_with(
            stream=False, unit_primitives=prompt.unit_primitives
        )

    @pytest.mark.asyncio
    @mock.patch("ai_gateway.prompts.base.get_request_logger")
    @pytest.mark.parametrize(
        "prompt_template", [{"with_messages_placeholder": True}], indirect=True
    )
    async def test_ainvoke_with_messages_placeholder(
        self,
        mock_get_logger: mock.Mock,
        mock_watch: mock.Mock,
        prompt: Prompt,
        model_response: str,
    ):
        mock_logger = mock.MagicMock()
        mock_get_logger.return_value = mock_logger

        response = await prompt.ainvoke(
            {
                "name": "Duo",
                "content": "What's up?",
                "messages": [
                    AIMessage(content="Fine, you?"),
                    HumanMessage(content="Good."),
                ],
            }
        )

        assert response.content == model_response

        expected_call = mock.call(
            "Performing LLM request",
            prompt="System: Hi, I'm Duo\nHuman: What's up?\nAI: Fine, you?\nHuman: Good.",
        )
        assert mock_logger.info.mock_calls[0] == expected_call

        mock_watch.assert_called_with(
            stream=False, unit_primitives=prompt.unit_primitives
        )

    @pytest.mark.asyncio
    @mock.patch("ai_gateway.prompts.base.get_request_logger")
    async def test_astream(
        self,
        mock_get_logger: mock.Mock,
        mock_watch: mock.Mock,
        prompt: Prompt,
        model_response: str,
    ):
        response = ""

        mock_watcher = mock_watch.return_value.__enter__.return_value
        mock_logger = mock.MagicMock()
        mock_get_logger.return_value = mock_logger

        async for c in prompt.astream({"name": "Duo", "content": "What's up?"}):
            response += c.content

            # Make sure we don't finish prematurely
            mock_watcher.afinish.assert_not_awaited()

        expected_call = mock.call(
            "Performing LLM request",
            prompt="System: Hi, I'm Duo\nHuman: What's up?",
        )
        assert mock_logger.info.mock_calls[0] == expected_call

        assert response == model_response

        mock_watch.assert_called_with(
            stream=True, unit_primitives=prompt.unit_primitives
        )

        mock_watcher.afinish.assert_awaited_once()

    @pytest.mark.asyncio
    @mock.patch("ai_gateway.prompts.base.get_request_logger")
    @pytest.mark.parametrize(
        "prompt_template", [{"with_messages_placeholder": True}], indirect=True
    )
    async def test_astream_with_messages_placeholder(
        self,
        mock_get_logger: mock.Mock,
        mock_watch: mock.Mock,
        prompt: Prompt,
        model_response: str,
    ):
        response = ""

        mock_watcher = mock_watch.return_value.__enter__.return_value
        mock_logger = mock.MagicMock()
        mock_get_logger.return_value = mock_logger

        async for c in prompt.astream(
            {
                "name": "Duo",
                "content": "What's up?",
                "messages": [
                    AIMessage(content="Fine, you?"),
                    HumanMessage(content="Good."),
                ],
            }
        ):
            response += c.content

            # Make sure we don't finish prematurely
            mock_watcher.afinish.assert_not_awaited()

        expected_call = mock.call(
            "Performing LLM request",
            prompt="System: Hi, I'm Duo\nHuman: What's up?\nAI: Fine, you?\nHuman: Good.",
        )
        assert mock_logger.info.mock_calls[0] == expected_call

        assert response == model_response

        mock_watch.assert_called_with(
            stream=True, unit_primitives=prompt.unit_primitives
        )

        mock_watcher.afinish.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("usage_metadata", "internal_event_extra", "expected_additional_properties"),
        [
            (
                UsageMetadata(
                    input_tokens=1,
                    output_tokens=2,
                    total_tokens=3,
                    input_token_details=InputTokenDetails(
                        audio=0, cache_creation=0, cache_read=0
                    ),
                ),
                {},
                InternalEventAdditionalProperties(
                    label="cache_details",
                    property=None,
                    value=None,
                    cache_read=0,
                    cache_creation=0,
                    ephemeral_5m_input_tokens=0,
                    ephemeral_1h_input_tokens=0,
                ),
            ),
            (
                UsageMetadata(
                    input_tokens=1,
                    output_tokens=2,
                    total_tokens=3,
                    input_token_details=InputTokenDetails(
                        audio=0,
                        cache_creation=0,
                        cache_read=0,
                        ephemeral_5m_input_tokens=10,  # type: ignore[typeddict-unknown-key]
                    ),
                ),
                {"test_key": "test_value"},
                InternalEventAdditionalProperties(
                    label="cache_details",
                    property=None,
                    value=None,
                    cache_read=0,
                    cache_creation=0,
                    ephemeral_5m_input_tokens=10,
                    ephemeral_1h_input_tokens=0,
                    test_key="test_value",
                ),
            ),
            (
                UsageMetadata(
                    input_tokens=1,
                    output_tokens=2,
                    total_tokens=3,
                    input_token_details=InputTokenDetails(
                        audio=0,
                        cache_creation=0,
                        cache_read=0,
                        ephemeral_1h_input_tokens=25,  # type: ignore[typeddict-unknown-key]
                    ),
                ),
                {},
                InternalEventAdditionalProperties(
                    label="cache_details",
                    property=None,
                    value=None,
                    cache_read=0,
                    cache_creation=0,
                    ephemeral_5m_input_tokens=0,
                    ephemeral_1h_input_tokens=25,
                ),
            ),
            (
                UsageMetadata(
                    input_tokens=1,
                    output_tokens=2,
                    total_tokens=3,
                    input_token_details=InputTokenDetails(
                        audio=0,
                        cache_creation=5,
                        cache_read=15,
                        ephemeral_5m_input_tokens=10,
                        ephemeral_1h_input_tokens=25,  # type: ignore[typeddict-unknown-key]
                    ),
                ),
                {},
                InternalEventAdditionalProperties(
                    label="cache_details",
                    property=None,
                    value=None,
                    cache_read=15,
                    cache_creation=5,
                    ephemeral_5m_input_tokens=10,
                    ephemeral_1h_input_tokens=25,
                ),
            ),
        ],
    )
    async def test_ainvoke_handle_usage_metadata(
        self,
        mock_watch: mock.Mock,
        internal_event_client: mock.Mock,
        prompt: Prompt,
        usage_metadata: UsageMetadata,
        internal_event_extra: dict[str, Any],
        expected_additional_properties: InternalEventAdditionalProperties,
    ):
        mock_watcher = mock_watch.return_value.__enter__.return_value

        prompt.internal_event_client = internal_event_client

        with (
            self._mock_usage_metadata(prompt.model_name, usage_metadata),
            mock.patch.object(
                Prompt,
                "internal_event_extra",
                new_callable=PropertyMock,
                return_value=internal_event_extra,
            ),
        ):
            await prompt.ainvoke({"name": "Duo", "content": "What's up?"})

        _assert_usage_metadata_handling(
            mock_watcher,
            internal_event_client,
            prompt,
            usage_metadata,
            expected_additional_properties,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("usage_metadata", "expected_additional_properties"),
        [
            (
                UsageMetadata(
                    input_tokens=1,
                    output_tokens=2,
                    total_tokens=3,
                    input_token_details=InputTokenDetails(
                        audio=0, cache_creation=0, cache_read=0
                    ),
                ),
                InternalEventAdditionalProperties(
                    label="cache_details",
                    property=None,
                    value=None,
                    cache_read=0,
                    cache_creation=0,
                    ephemeral_5m_input_tokens=0,
                    ephemeral_1h_input_tokens=0,
                ),
            ),
            (
                UsageMetadata(
                    input_tokens=1,
                    output_tokens=2,
                    total_tokens=3,
                    input_token_details=InputTokenDetails(
                        audio=0,
                        cache_creation=0,
                        cache_read=0,
                        ephemeral_5m_input_tokens=8,  # type: ignore[typeddict-unknown-key]
                    ),
                ),
                InternalEventAdditionalProperties(
                    label="cache_details",
                    property=None,
                    value=None,
                    cache_read=0,
                    cache_creation=0,
                    ephemeral_5m_input_tokens=8,
                    ephemeral_1h_input_tokens=0,
                ),
            ),
        ],
    )
    async def test_astream_handle_usage_metadata(
        self,
        mock_watch: mock.Mock,
        internal_event_client: mock.Mock,
        prompt: Prompt,
        usage_metadata: UsageMetadata,
        expected_additional_properties: InternalEventAdditionalProperties,
    ):
        mock_watcher = mock_watch.return_value.__enter__.return_value

        prompt.internal_event_client = internal_event_client

        with self._mock_usage_metadata(prompt.model_name, usage_metadata):
            # Consume stream
            async for _ in prompt.astream({"name": "Duo", "content": "What's up?"}):
                pass

        _assert_usage_metadata_handling(
            mock_watcher,
            internal_event_client,
            prompt,
            usage_metadata,
            expected_additional_properties,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("usage_metadata", "expected_additional_properties"),
        [
            (
                UsageMetadata(
                    input_tokens=1,
                    output_tokens=2,
                    total_tokens=3,
                    input_token_details=InputTokenDetails(
                        audio=0, cache_creation=4, cache_read=0
                    ),
                ),
                InternalEventAdditionalProperties(
                    label="cache_details",
                    property=None,
                    value=None,
                    cache_read=0,
                    cache_creation=4,
                    ephemeral_5m_input_tokens=0,
                    ephemeral_1h_input_tokens=0,
                ),
            ),
            (
                UsageMetadata(
                    input_tokens=4,
                    output_tokens=6,
                    total_tokens=3,
                    input_token_details=InputTokenDetails(
                        audio=0,
                        cache_creation=5,
                        cache_read=0,
                        ephemeral_1h_input_tokens=3,  # type: ignore[typeddict-unknown-key]
                    ),
                ),
                InternalEventAdditionalProperties(
                    label="cache_details",
                    property=None,
                    value=None,
                    cache_read=0,
                    cache_creation=5,
                    ephemeral_5m_input_tokens=0,
                    ephemeral_1h_input_tokens=3,
                ),
            ),
        ],
    )
    async def test_astream_handle_usage_metadata_with_cache_control(
        self,
        mock_watch: mock.Mock,
        internal_event_client: mock.Mock,
        prompt: Prompt,
        usage_metadata: UsageMetadata,
        expected_additional_properties: InternalEventAdditionalProperties,
    ):
        mock_watcher = mock_watch.return_value.__enter__.return_value

        prompt.internal_event_client = internal_event_client

        with self._mock_usage_metadata(prompt.model_name, usage_metadata):
            # Consume stream with cache control for Anthropic
            async for _ in prompt.astream(
                {"name": "Duo", "content": "What's up?"},
                cache_control={"type": "ephemeral"},
            ):
                pass

        _assert_usage_metadata_handling(
            mock_watcher,
            internal_event_client,
            prompt,
            usage_metadata,
            expected_additional_properties,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("model_response", "usage_metadata"),
        [
            # 2 characters ensures 2 yield before iteration stops
            ("ab", UsageMetadata(input_tokens=1, output_tokens=1, total_tokens=2)),
        ],
    )
    async def test_astream_usage_metadata_before_stream_end(
        self, mock_watch: mock.Mock, internal_event_client: mock.Mock, prompt: Prompt
    ):
        mock_watcher = mock_watch.return_value.__enter__.return_value
        prompt.internal_event_client = internal_event_client

        iterator = prompt.astream({"name": "Duo", "content": "What's up?"})

        # While the stream is being consumed, token usage is not registered yet
        await anext(iterator)
        mock_watcher.register_token_usage.assert_not_called()

        # When the last message is yield, but before iteration stops, token usage is registered
        await anext(iterator)
        mock_watcher.register_token_usage.assert_called_once()

        # Iteration ends
        with pytest.raises(StopAsyncIteration):
            await anext(iterator)

    @pytest.mark.asyncio
    async def test_ainvoke_model_input(self, prompt: Prompt):
        with mock.patch.object(FakeModel, "ainvoke") as mock_ainvoke:
            await prompt.ainvoke({"name": "Duo", "content": "What's up?"})

        mock_ainvoke.assert_called_once_with(
            ChatPromptValue(
                messages=[
                    SystemMessage(content="Hi, I'm Duo"),
                    HumanMessage(content="What's up?"),
                ]
            ),
            mock.ANY,
        )

    @pytest.mark.asyncio
    async def test_ainvoke_missing_inputs(self, prompt: Prompt):
        with pytest.raises(
            KeyError,
            match="Input to ChatPromptTemplate is missing variables {'name'}",
        ):
            await prompt.ainvoke({"content": "What's up?"})

    @pytest.mark.asyncio
    async def test_astream_model_input(self, prompt: Prompt):
        response = mock.AsyncMock()
        response.__aiter__.return_value = iter(["response"])

        with mock.patch.object(
            FakeModel, "astream", return_value=response
        ) as mock_astream:
            await anext(prompt.astream({"name": "Duo", "content": "What's up?"}))

        mock_astream.assert_called_once_with(
            ChatPromptValue(
                messages=[
                    SystemMessage(content="Hi, I'm Duo"),
                    HumanMessage(content="What's up?"),
                ]
            ),
            mock.ANY,
        )

    @pytest.mark.asyncio
    async def test_astream_missing_inputs(self, prompt: Prompt):
        with pytest.raises(
            KeyError,
            match="Input to ChatPromptTemplate is missing variables {'name'}",
        ):
            await anext(prompt.astream({"content": "What's up?"}))

    @pytest.mark.parametrize(
        "tool_choice",
        [
            "auto",
            None,
            "any",
        ],
    )
    def test_bind_tools_with_tool_choice(
        self,
        prompt_config: PromptConfig,
        model_metadata: TypeModelMetadata,
        model_factory: TypeModelFactory,
        model: FakeModel,
        tool_choice: str | None,
    ):
        """Test that tool_choice parameter is correctly passed to bind_tools method."""

        with mock.patch.object(FakeModel, "bind_tools") as mock_bind_tool:
            mock_bind_tool.return_value = model
            Prompt(
                model_factory=model_factory,
                config=prompt_config,
                model_metadata=model_metadata,
                tools=[mock.Mock(spec=BaseTool)],
                tool_choice=tool_choice,
            )

        kwargs = mock_bind_tool.call_args.kwargs

        assert kwargs["tool_choice"] == tool_choice


def _assert_usage_metadata_handling(
    mock_watcher: mock.Mock,
    internal_event_client: mock.Mock,
    prompt: Prompt,
    usage_metadata: UsageMetadata,
    additional_properties: Optional[InternalEventAdditionalProperties] = None,
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
            additional_properties=additional_properties,
        )


@pytest.mark.skipif(
    # pylint: disable=direct-environment-variable-reference
    os.getenv("REAL_AI_REQUEST") is None,
    # pylint: enable=direct-environment-variable-reference
    reason="3rd party requests not enabled",
)
class TestPromptTimeout:
    @pytest.fixture(name="prompt_params")
    def prompt_params_fixture(self):
        return PromptParams(timeout=0.1)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("model", "expected_exception"),
        [
            (
                ChatAnthropic(
                    async_client=AsyncAnthropic(),
                    model="claude-3-sonnet-20240229",  # type: ignore[call-arg]
                ),
                APITimeoutError,
            ),
            (
                ChatLiteLLM(
                    model="claude-3-sonnet@20240229",
                    custom_llm_provider="vertex_ai",  # type: ignore[call-arg]
                ),
                Timeout,
            ),
            (
                ChatLiteLLM(
                    model="claude-3-5-sonnet-v2@20241022",
                    custom_llm_provider="vertex_ai",  # type: ignore[call-arg]
                ),
                Timeout,
            ),
        ],
    )
    async def test_timeout(self, prompt: Prompt, expected_exception: Type):
        with pytest.raises(expected_exception):
            await prompt.ainvoke(
                {"name": "Duo", "content": "Print pi with 400 decimals"}
            )


@pytest.fixture(name="registry")
def registry_fixture(
    internal_event_client: Mock, model_limits: ConfigModelLimits, prompt: Prompt
):
    class Registry(BasePromptRegistry):
        def __init__(self):
            self.internal_event_client = internal_event_client
            self.model_limits = model_limits

        def get(self, *_args, **_kwargs):
            return prompt

    return Registry()


class TestBaseRegistry:
    @pytest.mark.parametrize(
        (
            "unit_primitives",
            "scopes",
            "input_model_metadata",
            "tools",
            "success",
            "expected_internal_events",
        ),
        [
            (
                [GitLabUnitPrimitive.COMPLETE_CODE],
                ["complete_code"],
                None,
                None,
                True,
                [call("request_complete_code", category="ai_gateway.prompts.base")],
            ),
            (
                [GitLabUnitPrimitive.COMPLETE_CODE, GitLabUnitPrimitive.ASK_BUILD],
                ["complete_code", "ask_build"],
                None,
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
                None,
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
                None,
                True,
                [
                    call(
                        "request_amazon_q_integration",
                        category="ai_gateway.prompts.base",
                    ),
                ],
            ),
            ([GitLabUnitPrimitive.COMPLETE_CODE], [], None, None, False, []),
            (
                [
                    GitLabUnitPrimitive.COMPLETE_CODE,
                    GitLabUnitPrimitive.ASK_BUILD,
                ],
                ["complete_code"],
                None,
                None,
                False,
                [],
            ),
            (
                [GitLabUnitPrimitive.DUO_CHAT],
                ["duo_chat"],
                None,
                [Mock(spec=BaseTool)],
                True,
                [
                    call(
                        "request_duo_chat",
                        category="ai_gateway.prompts.base",
                    ),
                ],
            ),
        ],
    )
    def test_get_on_behalf(
        self,
        internal_event_client: Mock,
        registry: BasePromptRegistry,
        user: StarletteUser,
        prompt: Prompt,
        input_model_metadata: Optional[TypeModelMetadata],
        tools: Optional[List[BaseTool]],
        success: bool,
        expected_internal_events,
    ):
        if success:
            assert (
                registry.get_on_behalf(
                    user,
                    "test",
                    None,
                    input_model_metadata,
                    "ai_gateway.prompts.base",
                    tools,
                )
                == prompt
            )
            assert prompt.internal_event_client == internal_event_client

            if input_model_metadata:
                assert input_model_metadata._user == user

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
        model_metadata: TypeModelMetadata,
    ):
        current_model_metadata_context.set(model_metadata)

        assert registry.get_on_behalf(user, "test", None) == prompt

    @pytest.mark.parametrize(
        ("unit_primitives", "scopes", "prompt_version"),
        [
            (
                [GitLabUnitPrimitive.COMPLETE_CODE],
                ["complete_code"],
                "^1.0.0",
            ),
            (
                [GitLabUnitPrimitive.COMPLETE_CODE],
                ["complete_code"],
                None,
            ),
        ],
    )
    def test_get_on_behalf_with_prompt_version(
        self,
        registry: BasePromptRegistry,
        user: StarletteUser,
        prompt: Prompt,
        prompt_version: str | None,
    ):
        result = registry.get_on_behalf(
            user,
            "test",
            prompt_version=prompt_version,
        )

        assert result == prompt


class TestPromptCheckpointerIntegration:
    """Test integration between Prompt and workflow checkpointer for LLM tracking."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("has_internal_event_client", [True, False])
    async def test_prompt_calls_checkpointer_track_llm_operation_integration(
        self, has_internal_event_client: bool
    ):
        """Test that Prompt.handle_usage_metadata calls checkpointer.track_llm_operation when available."""

        clear_workflow_checkpointer()

        try:
            mock_checkpointer = Mock()
            mock_checkpointer.track_llm_operation = Mock()

            set_workflow_checkpointer(mock_checkpointer)

            prompt = Mock(spec=Prompt)
            prompt.model_engine = "test_engine"
            prompt.model_provider = "test_provider"
            prompt.unit_primitives = ["complete_code"]
            prompt.internal_event_client = Mock() if has_internal_event_client else None
            prompt.internal_event_extra = {}

            mock_watcher = Mock(spec=ModelRequestInstrumentator.WatchContainer)

            usage_metadata = {
                "claude-3-sonnet": UsageMetadata(
                    input_tokens=80,
                    output_tokens=20,
                    total_tokens=100,
                )
            }

            Prompt.handle_usage_metadata(prompt, mock_watcher, usage_metadata)

            mock_checkpointer.track_llm_operation.assert_called_once_with(
                token_count=100,
                model_id="claude-3-sonnet",
                model_engine="test_engine",
                model_provider="test_provider",
                prompt_tokens=80,
                completion_tokens=20,
            )

            mock_watcher.register_token_usage.assert_called_once_with(
                "claude-3-sonnet", usage_metadata["claude-3-sonnet"]
            )

        finally:
            clear_workflow_checkpointer()

    @pytest.mark.asyncio
    async def test_prompt_handle_usage_metadata_no_checkpointer_no_internal_client(
        self,
    ):
        """Test early return when neither checkpointer nor internal event client available."""

        clear_workflow_checkpointer()

        try:
            prompt = Mock(spec=Prompt)
            prompt.internal_event_client = None

            mock_watcher = Mock(spec=ModelRequestInstrumentator.WatchContainer)

            usage_metadata = {
                "claude-3-sonnet": UsageMetadata(
                    input_tokens=80,
                    output_tokens=20,
                    total_tokens=100,
                )
            }

            Prompt.handle_usage_metadata(prompt, mock_watcher, usage_metadata)

            mock_watcher.register_token_usage.assert_not_called()

        finally:
            clear_workflow_checkpointer()

    @pytest.mark.asyncio
    async def test_prompt_handle_usage_metadata_multiple_models(self):
        """Test that checkpointer tracks multiple models correctly."""

        clear_workflow_checkpointer()

        try:
            mock_checkpointer = Mock()
            mock_checkpointer.track_llm_operation = Mock()
            set_workflow_checkpointer(mock_checkpointer)

            prompt = Mock(spec=Prompt)
            prompt.model_engine = "test_engine"
            prompt.model_provider = "test_provider"
            prompt.unit_primitives = ["complete_code"]
            prompt.internal_event_client = None
            prompt.internal_event_extra = {}

            mock_watcher = Mock(spec=ModelRequestInstrumentator.WatchContainer)

            usage_metadata = {
                "claude-3-sonnet": UsageMetadata(
                    input_tokens=80, output_tokens=20, total_tokens=100
                ),
                "gpt-4": UsageMetadata(
                    input_tokens=120, output_tokens=30, total_tokens=150
                ),
            }

            Prompt.handle_usage_metadata(prompt, mock_watcher, usage_metadata)

            expected_calls = [
                call(
                    token_count=100,
                    model_id="claude-3-sonnet",
                    model_engine="test_engine",
                    model_provider="test_provider",
                    prompt_tokens=80,
                    completion_tokens=20,
                ),
                call(
                    token_count=150,
                    model_id="gpt-4",
                    model_engine="test_engine",
                    model_provider="test_provider",
                    prompt_tokens=120,
                    completion_tokens=30,
                ),
            ]

            mock_checkpointer.track_llm_operation.assert_has_calls(
                expected_calls, any_order=True
            )
            assert mock_checkpointer.track_llm_operation.call_count == 2

        finally:
            clear_workflow_checkpointer()
