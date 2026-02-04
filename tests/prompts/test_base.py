# pylint: disable=too-many-lines
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Iterator, List, Optional, Type
from unittest import mock
from unittest.mock import Mock, call

import pytest
from anthropic import APITimeoutError, AsyncAnthropic
from gitlab_cloud_connector import GitLabUnitPrimitive, WrongUnitPrimitives
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.ai import InputTokenDetails, UsageMetadata
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from litellm.exceptions import Timeout
from pydantic import AnyUrl
from pyfakefs.fake_filesystem import FakeFilesystem
from structlog.testing import capture_logs

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.config import ConfigModelLimits
from ai_gateway.instrumentators.model_requests import ModelRequestInstrumentator
from ai_gateway.model_metadata import (
    AmazonQModelMetadata,
    ModelMetadata,
    TypeModelMetadata,
    current_model_metadata_context,
)
from ai_gateway.model_selection import LLMDefinition, PromptParams
from ai_gateway.models.v2.anthropic_claude import ChatAnthropic
from ai_gateway.prompts import BasePromptCallbackHandler, BasePromptRegistry, Prompt
from ai_gateway.prompts.config.base import PromptConfig
from ai_gateway.prompts.config.models import (
    ChatAnthropicParams,
    ChatLiteLLMParams,
    ModelClassProvider,
)
from ai_gateway.prompts.typing import TypeModelFactory, TypePromptTemplateFactory
from ai_gateway.vendor.langchain_litellm.litellm import ChatLiteLLM
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
        ("model_params", "expected_llm_provider"),
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
        expected_llm_provider: str,
    ):
        assert prompt.name == "test_prompt"
        assert prompt.unit_primitives == unit_primitives
        assert prompt.model_provider == model_params["model_class_provider"]
        assert prompt.llm_provider == expected_llm_provider
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
    @pytest.mark.parametrize("runnable_config", [None, RunnableConfig(callbacks=[])])
    async def test_ainvoke(
        self,
        mock_get_logger: mock.Mock,
        runnable_config: Optional[RunnableConfig],
        mock_watch: mock.Mock,
        prompt: Prompt,
        model_response: str,
    ):
        mock_logger = mock.MagicMock()
        mock_get_logger.return_value = mock_logger

        response = await prompt.ainvoke(
            {"name": "Duo", "content": "What's up?"}, runnable_config
        )

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
    @pytest.mark.parametrize("runnable_config", [None, RunnableConfig(callbacks=[])])
    async def test_astream(
        self,
        mock_get_logger: mock.Mock,
        runnable_config: Optional[RunnableConfig],
        mock_watch: mock.Mock,
        prompt: Prompt,
        model_response: str,
    ):
        response = ""

        mock_watcher = mock_watch.return_value.__enter__.return_value
        mock_logger = mock.MagicMock()
        mock_get_logger.return_value = mock_logger

        async for c in prompt.astream(
            {"name": "Duo", "content": "What's up?"}, runnable_config
        ):
            response += str(c.content)

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
            response += str(c.content)

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
    async def test_ainvoke_model_instrumentator_callbacks(
        self,
        mock_watch: mock.Mock,
        internal_event_client: mock.Mock,
        prompt: Prompt,
        usage_metadata: UsageMetadata,
        expected_additional_properties: InternalEventAdditionalProperties,
    ):
        mock_watcher = mock_watch.return_value.__enter__.return_value

        prompt.internal_event_client = internal_event_client

        with (
            self._mock_usage_metadata(prompt.model_name, usage_metadata),
            capture_logs() as cap_logs,
        ):
            result = await prompt.ainvoke({"name": "Duo", "content": "What's up?"})

        mock_watcher.register_message.assert_called_once_with(result)

        _assert_usage_metadata_handling(
            mock_watcher,
            internal_event_client,
            prompt,
            usage_metadata,
            cap_logs,
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
    async def test_astream_model_instrumentator_callbacks(
        self,
        mock_watch: mock.Mock,
        internal_event_client: mock.Mock,
        prompt: Prompt,
        usage_metadata: UsageMetadata,
        expected_additional_properties: InternalEventAdditionalProperties,
    ):
        mock_watcher = mock_watch.return_value.__enter__.return_value

        prompt.internal_event_client = internal_event_client

        with (
            self._mock_usage_metadata(prompt.model_name, usage_metadata),
            capture_logs() as cap_logs,
        ):
            async for message in prompt.astream(
                {"name": "Duo", "content": "What's up?"}
            ):
                mock_watcher.register_message.assert_any_call(message)

        _assert_usage_metadata_handling(
            mock_watcher,
            internal_event_client,
            prompt,
            usage_metadata,
            cap_logs,
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
    async def test_atransform_model_instrumentator_callbacks(
        self,
        mock_watch: mock.Mock,
        internal_event_client: mock.Mock,
        prompt: Prompt,
        usage_metadata: UsageMetadata,
        expected_additional_properties: InternalEventAdditionalProperties,
    ):
        mock_watcher = mock_watch.return_value.__enter__.return_value

        prompt.internal_event_client = internal_event_client

        async def input():
            yield {"name": "Duo", "content": "What's up?"}

        with (
            self._mock_usage_metadata(prompt.model_name, usage_metadata),
            capture_logs() as cap_logs,
        ):
            async for message in prompt.atransform(input()):
                mock_watcher.register_message.assert_any_call(message)

        _assert_usage_metadata_handling(
            mock_watcher,
            internal_event_client,
            prompt,
            usage_metadata,
            cap_logs,
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

        with (
            self._mock_usage_metadata(prompt.model_name, usage_metadata),
            capture_logs() as cap_logs,
        ):
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
            cap_logs,
            expected_additional_properties,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("model_response", "usage_metadata"),
        [
            # first yield for the single character, second yield empty content end of stream
            ("a", UsageMetadata(input_tokens=1, output_tokens=1, total_tokens=2)),
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
    async def test_astream_model_input(self, prompt: Prompt, end_message: AIMessage):
        response = mock.AsyncMock()
        response.__aiter__.return_value = iter([end_message])

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
            # Create a proper mock with name attribute for cache signature computation
            mock_tool = mock.Mock(spec=BaseTool)
            mock_tool.name = "test_tool"
            mock_tool.description = "Test tool description"

            Prompt(
                model_factory=model_factory,
                config=prompt_config,
                model_metadata=model_metadata,
                tools=[mock_tool],
                tool_choice=tool_choice,
            )

        kwargs = mock_bind_tool.call_args.kwargs

        assert kwargs["tool_choice"] == tool_choice


def _assert_usage_metadata_handling(
    mock_watcher: mock.Mock,
    internal_event_client: mock.Mock,
    prompt: Prompt,
    usage_metadata: UsageMetadata,
    cap_logs,
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
            model_engine=prompt.llm_provider,
            model_name=prompt.model_name,
            model_provider=prompt.model_provider,
            additional_properties=additional_properties,
        )

    assert cap_logs[0]["model_engine"] == prompt.llm_provider
    assert cap_logs[0]["model_name"] == prompt.model_name
    assert cap_logs[0]["model_provider"] == prompt.model_provider
    assert cap_logs[0]["input_tokens"] == usage_metadata["input_tokens"]
    assert cap_logs[0]["output_tokens"] == usage_metadata["output_tokens"]
    assert cap_logs[0]["total_tokens"] == usage_metadata["total_tokens"]
    assert (
        cap_logs[0]["cache_read"] == usage_metadata["input_token_details"]["cache_read"]
    )
    assert (
        cap_logs[0]["cache_creation"]
        == usage_metadata["input_token_details"]["cache_creation"]
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
                    custom_llm_provider="vertex_ai",
                ),
                Timeout,
            ),
            (
                ChatLiteLLM(
                    model="claude-3-5-sonnet-v2@20241022",
                    custom_llm_provider="vertex_ai",
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


class TestPromptCaching:
    @pytest.mark.asyncio
    @mock.patch("ai_gateway.prompts.base.filter_cache_control_injection_points")
    @mock.patch("ai_gateway.prompts.base.CacheControlInjectionPointsConverter")
    @pytest.mark.parametrize(
        ("prompt_params", "model_params", "expected_to_use_converter"),
        [
            (
                PromptParams(
                    cache_control_injection_points=[{"location": "message", "index": 0}]
                ),
                ChatAnthropicParams(model_class_provider=ModelClassProvider.ANTHROPIC),
                True,
            ),
            (
                PromptParams(
                    cache_control_injection_points=[{"location": "message", "index": 0}]
                ),
                ChatLiteLLMParams(
                    model="test_model", model_class_provider=ModelClassProvider.LITE_LLM
                ),
                False,
            ),
            (
                PromptParams(),
                ChatAnthropicParams(model_class_provider=ModelClassProvider.ANTHROPIC),
                False,
            ),
        ],
    )
    async def test_prompt_caching(
        self,
        mock_cache_control_injection_points_converter,
        mock_filter_cache_control_injection_points,
        model_factory: TypeModelFactory,
        prompt_config: PromptConfig,
        model_metadata: TypeModelMetadata | None,
        prompt_template_factory: TypePromptTemplateFactory | None,
        expected_to_use_converter: bool,
    ):
        Prompt(model_factory, prompt_config, model_metadata, prompt_template_factory)

        mock_filter_cache_control_injection_points.assert_called_once()

        if expected_to_use_converter:
            mock_cache_control_injection_points_converter.assert_called_once()
        else:
            mock_cache_control_injection_points_converter.assert_not_called()


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
                    llm_definition=LLMDefinition(
                        gitlab_identifier="mistral",
                        name="Mistral",
                        max_context_tokens=128000,
                    ),
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
                    llm_definition=LLMDefinition(
                        gitlab_identifier="amazon_q",
                        name="Amazon Q",
                        max_context_tokens=100000,
                    ),
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
                    llm_definition=LLMDefinition(
                        gitlab_identifier="mistral",
                        name="Mistral",
                        max_context_tokens=128000,
                    ),
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


class TestPromptCallbacks:
    """Test callback execution in Prompt class."""

    @pytest.mark.asyncio
    async def test_callbacks_executed_on_ainvoke(self, prompt: Prompt):
        """Test that internal callbacks are executed during ainvoke."""
        call_log = []

        class TestCallback(BasePromptCallbackHandler):
            async def on_before_llm_call(self):
                call_log.append("ainvoke_callback")

        # Set internal_callbacks directly on the prompt instance
        prompt.internal_callbacks = [TestCallback()]

        await prompt.ainvoke({"name": "Duo", "content": "What's up?"})

        # Callback should have been executed
        assert "ainvoke_callback" in call_log

    @pytest.mark.asyncio
    async def test_callbacks_executed_on_astream(self, prompt: Prompt):
        """Test that internal callbacks are executed during astream."""
        call_log = []

        class TestCallback(BasePromptCallbackHandler):
            async def on_before_llm_call(self):
                call_log.append("astream_callback")

        prompt.internal_callbacks = [TestCallback()]

        async for _ in prompt.astream({"name": "Duo", "content": "What's up?"}):
            pass

        # Callback should have been executed
        assert "astream_callback" in call_log

    @pytest.mark.asyncio
    async def test_multiple_callbacks_execution_order(self, prompt: Prompt):
        """Test that multiple callbacks are executed in registration order."""
        call_log = []

        class Callback1(BasePromptCallbackHandler):
            async def on_before_llm_call(self):
                call_log.append("first")

        class Callback2(BasePromptCallbackHandler):
            async def on_before_llm_call(self):
                call_log.append("second")

        class Callback3(BasePromptCallbackHandler):
            async def on_before_llm_call(self):
                call_log.append("third")

        prompt.internal_callbacks = [Callback1(), Callback2(), Callback3()]

        await prompt.ainvoke({"name": "Duo", "content": "What's up?"})

        # Callbacks should execute in order
        assert call_log == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_callback_exception_propagates(self, prompt: Prompt):
        """Test that exceptions in callbacks are propagated and stop execution."""

        class FailingCallback(BasePromptCallbackHandler):
            async def on_before_llm_call(self):
                raise RuntimeError("Callback error")

        prompt.internal_callbacks = [FailingCallback()]

        with pytest.raises(RuntimeError, match="Callback error"):
            await prompt.ainvoke({"name": "Duo", "content": "What's up?"})

    @pytest.mark.asyncio
    async def test_callbacks_with_runnable_config(self, prompt: Prompt):
        """Test that callbacks work correctly when RunnableConfig is provided."""
        call_log = []

        class TestCallback(BasePromptCallbackHandler):
            async def on_before_llm_call(self):
                call_log.append("config_callback")

        prompt.internal_callbacks = [TestCallback()]

        config = RunnableConfig(callbacks=[])
        await prompt.ainvoke({"name": "Duo", "content": "What's up?"}, config)

        # Callback should still execute
        assert "config_callback" in call_log

    @pytest.mark.asyncio
    async def test_empty_callback_list(self, prompt: Prompt):
        """Test that empty callback list doesn't cause errors."""
        prompt.internal_callbacks = []

        # Should execute without errors
        result = await prompt.ainvoke({"name": "Duo", "content": "What's up?"})
        assert result is not None

    @pytest.mark.asyncio
    async def test_callback_with_state(self, prompt: Prompt):
        """Test that callbacks can maintain state across calls."""

        class StatefulCallback(BasePromptCallbackHandler):
            def __init__(self):
                self.call_count = 0

            async def on_before_llm_call(self):
                self.call_count += 1

        callback = StatefulCallback()
        prompt.internal_callbacks = [callback]

        await prompt.ainvoke({"name": "Duo", "content": "First call"})
        assert callback.call_count == 1

        await prompt.ainvoke({"name": "Duo", "content": "Second call"})
        assert callback.call_count == 2

    @pytest.mark.asyncio
    async def test_callback_receives_context(self, prompt: Prompt):
        """Test that callbacks can access prompt context."""
        captured_data = {}

        class ContextCallback(BasePromptCallbackHandler):
            def __init__(self, prompt_instance: Prompt):
                self.prompt = prompt_instance

            async def on_before_llm_call(self):
                captured_data["model_name"] = self.prompt.model_name
                captured_data["llm_provider"] = self.prompt.llm_provider

        prompt.internal_callbacks = [ContextCallback(prompt)]

        await prompt.ainvoke({"name": "Duo", "content": "What's up?"})

        assert "model_name" in captured_data
        assert "llm_provider" in captured_data
        assert captured_data["model_name"] == prompt.model_name
        assert captured_data["llm_provider"] == prompt.llm_provider

    @pytest.mark.asyncio
    async def test_callbacks_execute_before_llm_call(self, prompt: Prompt):
        """Test that callbacks execute before the actual LLM call."""
        execution_order = []

        class OrderTrackingCallback(BasePromptCallbackHandler):
            async def on_before_llm_call(self):
                execution_order.append("callback")

        prompt.internal_callbacks = [OrderTrackingCallback()]

        with mock.patch.object(FakeModel, "ainvoke") as mock_ainvoke:
            mock_ainvoke.side_effect = lambda *args, **kwargs: (
                execution_order.append("llm_call"),  # type: ignore[func-returns-value]
                AIMessage(content="response"),
            )[1]

            await prompt.ainvoke({"name": "Duo", "content": "What's up?"})

        # Callback should execute before LLM call
        assert execution_order == ["callback", "llm_call"]

    @pytest.mark.asyncio
    async def test_multiple_callbacks_all_execute_on_exception(self, prompt: Prompt):
        """Test that all callbacks execute even if one raises an exception."""
        call_log = []

        class FailingCallback(BasePromptCallbackHandler):
            async def on_before_llm_call(self):
                call_log.append("failing")
                raise RuntimeError("Callback error")

        class SuccessCallback(BasePromptCallbackHandler):
            async def on_before_llm_call(self):
                call_log.append("success")

        # asyncio.gather will execute all callbacks concurrently
        prompt.internal_callbacks = [SuccessCallback(), FailingCallback()]

        with pytest.raises(RuntimeError, match="Callback error"):
            await prompt.ainvoke({"name": "Duo", "content": "What's up?"})

        # Both callbacks should have started execution
        # (asyncio.gather runs them concurrently)
        assert "success" in call_log
        assert "failing" in call_log


class TestValidateDefaultModels:
    @pytest.fixture
    def prompt_template(self):
        return {"system": "Just say hi"}

    @pytest.fixture(autouse=True)
    def mock_fs(self, fs: FakeFilesystem):
        ai_gateway_dir = Path(__file__).parent.parent.parent / "ai_gateway"
        model_selection_dir = ai_gateway_dir / "model_selection"
        prompts_definitions_dir = ai_gateway_dir / "prompts" / "definitions"

        # editorconfig-checker-disable
        fs.create_file(
            model_selection_dir / "models.yml",
            contents="""---
models:
  - name: Model A
    gitlab_identifier: model_a
    max_context_tokens: 1000
  - name: Model B
    gitlab_identifier: model_b
    max_context_tokens: 1000
  - name: Model C
    gitlab_identifier: model_c
    max_context_tokens: 1000
""",
        )
        fs.create_file(
            model_selection_dir / "unit_primitives.yml",
            contents="""---
configurable_unit_primitives:
  - feature_setting: "setting_a"
    unit_primitives:
      - "duo_chat"
    default_model: "model_a"
    selectable_models:
      - "model_a"
      - "model_c"
  - feature_setting: "b"
    unit_primitives:
      - "code_suggestions"
    default_model: "model_b"
    selectable_models:
      - "model_b"
""",
        )
        # editorconfig-checker-enable
        fs.create_file(
            prompts_definitions_dir / "system.jinja",
            contents="Hi, I'm {{name}}",
        )

    @pytest.mark.asyncio
    async def test_success(
        self,
        registry: BasePromptRegistry,
    ):
        """Test successful validation of all default models."""
        with capture_logs() as cap_logs:
            with mock.patch.object(FakeModel, "ainvoke") as mock_ainvoke:
                result = await registry.validate_default_models()

        assert result is True
        # Should validate both unique default models
        assert mock_ainvoke.call_count == 2

        # Verify logging - should have 2 log entries for the 2 models
        log_messages = [
            log for log in cap_logs if log.get("event") == "Validating default model"
        ]
        assert len(log_messages) == 2
        logged_models = {log["model"] for log in log_messages}
        assert logged_models == {"model_a", "model_b"}

    @pytest.mark.asyncio
    async def test_with_unit_primitive_filter_matching(
        self,
        registry: BasePromptRegistry,
    ):
        with mock.patch.object(FakeModel, "ainvoke") as mock_ainvoke:
            result = await registry.validate_default_models(
                unit_primitive=GitLabUnitPrimitive("duo_chat")
            )

        assert result is True
        mock_ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_with_non_matching_unit_primitive(
        self,
        registry: BasePromptRegistry,
    ):
        """Test validation skips models when unit primitive doesn't match.

        Using a unit primitive that doesn't exist in the config should skip all validations.
        """
        with mock.patch.object(FakeModel, "ainvoke") as mock_ainvoke:
            result = await registry.validate_default_models(
                unit_primitive=GitLabUnitPrimitive("duo_agent_platform")
            )

        assert result is True
        # Should not invoke any prompts since the unit primitive is not in the config
        mock_ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_caches_validations(
        self,
        registry: BasePromptRegistry,
    ):
        """Test that validations are cached per model.

        Validating model_a first, then all models should only validate model_b on second call.
        """
        with mock.patch.object(FakeModel, "ainvoke") as mock_ainvoke:
            # First call - validate only model_a
            result1 = await registry.validate_default_models(
                unit_primitive=GitLabUnitPrimitive("duo_chat")
            )
            assert result1 is True
            assert mock_ainvoke.call_count == 1

            # Second call - validate all models, but model_a is already cached
            result2 = await registry.validate_default_models()
            assert result2 is True
            # Should be 2 total (1 from first call + 1 new for model_b)
            assert mock_ainvoke.call_count == 2

            # Third call - validate all models, but all models are cached
            result2 = await registry.validate_default_models()
            assert result2 is True
            # Call count shouldn't have changed
            assert mock_ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_calls_get_with_correct_params(
        self,
        prompt: Prompt,
        registry: BasePromptRegistry,
    ):
        """Test that validate_default_models calls get() with correct parameters.

        Should call get() twice (once for model_a, once for model_b) with proper metadata.
        """
        with mock.patch.object(registry, "get", return_value=prompt) as mock_get:
            await registry.validate_default_models()

            # Verify get was called twice (once per unique default model)
            mock_get.assert_has_calls(
                [
                    mock.call(
                        "model_configuration/check",
                        registry._DEFAULT_VERSION,
                        model_metadata=ModelMetadata(
                            provider="gitlab",
                            name="model_a",
                            friendly_name="Model A",
                            llm_definition=LLMDefinition(
                                name="Model A",
                                gitlab_identifier="model_a",
                                max_context_tokens=1000,
                            ),
                        ),
                    ),
                    mock.call(
                        "model_configuration/check",
                        registry._DEFAULT_VERSION,
                        model_metadata=ModelMetadata(
                            provider="gitlab",
                            name="model_b",
                            friendly_name="Model B",
                            llm_definition=LLMDefinition(
                                name="Model B",
                                gitlab_identifier="model_b",
                                max_context_tokens=1000,
                            ),
                        ),
                    ),
                ]
            )

    @pytest.mark.asyncio
    async def test_validation_failed(
        self,
        registry: BasePromptRegistry,
    ):
        """Test that exceptions from prompt invocation are propagated."""
        with mock.patch.object(FakeModel, "ainvoke") as mock_ainvoke:
            mock_ainvoke.side_effect = Exception("Model validation failed")

            with pytest.raises(Exception, match="Model validation failed"):
                await registry.validate_default_models()

            # Tried to validate both models independently, both failed
            assert mock_ainvoke.call_count == 2

            with pytest.raises(Exception, match="Model validation failed"):
                await registry.validate_default_models()

            # Doesn't cache the failed results, tries to validate both models again
            assert mock_ainvoke.call_count == 4
