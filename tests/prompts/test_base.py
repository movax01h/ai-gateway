# pylint: disable=too-many-lines
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Iterator, List, Optional, Type
from unittest import mock
from unittest.mock import Mock, call

import httpx
import pytest
from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncAnthropic,
    BadRequestError,
)
from anthropic import InternalServerError as AnthropicInternalServerError
from anthropic._exceptions import OverloadedError as AnthropicOverloadedError
from gitlab_cloud_connector import GitLabUnitPrimitive, WrongUnitPrimitives
from jinja2.exceptions import SecurityError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from litellm.exceptions import InternalServerError as LiteLLMInternalServerError
from litellm.exceptions import MidStreamFallbackError, Timeout
from pydantic import AnyUrl
from pyfakefs.fake_filesystem import FakeFilesystem
from structlog.testing import capture_logs

from ai_gateway.config import ConfigModelLimits
from ai_gateway.instrumentators.model_requests import ModelRequestInstrumentator
from ai_gateway.model_metadata import (
    AmazonQModelMetadata,
    ModelMetadata,
    TypeModelMetadata,
)
from ai_gateway.model_selection import PromptParams
from ai_gateway.model_selection.model_selection_config import (
    ChatAmazonQDefinition,
    ChatAnthropicDefinition,
    ChatLiteLLMDefinition,
)
from ai_gateway.model_selection.models import (
    ChatAnthropicParams,
    ChatLiteLLMParams,
    ModelClassProvider,
)
from ai_gateway.models.v2.anthropic_claude import ChatAnthropic
from ai_gateway.prompts import (
    BasePromptCallbackHandler,
    BasePromptRegistry,
    Prompt,
    jinja2_formatter,
)
from ai_gateway.prompts.base import TemplateNotFoundError
from ai_gateway.prompts.config.base import ModelConfig, PromptConfig
from ai_gateway.prompts.typing import TypeModelFactory, TypePromptTemplateFactory
from ai_gateway.vendor.langchain_litellm.litellm import ChatLiteLLM
from lib.context import StarletteUser, current_model_metadata_context
from lib.prompts.utilities import render_security_block
from tests.conftest import FakeModel


@pytest.fixture(autouse=True)
def mock_security_suffix():
    with mock.patch("lib.prompts.utilities._security_suffix", return_value="test"):
        yield


@pytest.fixture(name="mock_watch")
def mock_watch_fixture() -> Generator[mock.MagicMock, None, None]:
    with mock.patch(
        "ai_gateway.prompts.base.ModelRequestInstrumentator.watch"
    ) as mock_watch:
        mock_watcher = mock.AsyncMock(spec=ModelRequestInstrumentator.WatchContainer)
        mock_watch.return_value.__enter__.return_value = mock_watcher

        yield mock_watch


class TestPrompt:  # pylint: disable=too-many-public-methods
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
    default_models:
      - "mistral"
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

    def test_initialize(
        self,
        prompt: Prompt,
        unit_primitive: GitLabUnitPrimitive,
    ):
        assert prompt.name == "test_prompt"
        assert prompt.unit_primitive == unit_primitive
        assert isinstance(prompt.bound, Runnable)

    def test_build_prompt_template(self, prompt_config: PromptConfig):
        prompt_template: Runnable = Prompt._build_prompt_template(prompt_config)

        assert prompt_template == ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    render_security_block() + "{% include 'system.jinja' %}",
                ),
                ("user", "{{content}}"),
                MessagesPlaceholder("history", optional=True),
            ],
            template_format="jinja2",
        )

    @pytest.mark.parametrize(
        ("template", "kwargs", "should_raise"),
        [
            ("{{ user.name }}", {"user": object()}, False),
            ("{{ user.method() }}", {"user": object()}, True),
            ("{{ array.append(1) }}", {"array": []}, True),
            ("{{ value * 2 }}", {"value": 2}, True),
            ("{{ value ** 2 }}", {"value": 2}, True),
            ("{{ value + 2 }}", {"value": 2}, False),
            ("{{ value / 2 }}", {"value": 2}, False),
            ("{{ text|split }}", {"text": "a b"}, False),
            ('{{ text|split(":", 1) }}', {"text": "a:b"}, False),
            ('{{ text|split("", 1) }}', {"text": "oops"}, True),
        ],
    )
    def test_jinja2_formatter_security_constraints(
        self, template: str, kwargs: dict[str, Any], should_raise: bool
    ):
        class Dummy:
            name = "secure"

            def method(self):
                return "unsafe"

        if "user" in kwargs:
            kwargs["user"] = Dummy()

        if should_raise:
            with pytest.raises(SecurityError):
                jinja2_formatter(template, **kwargs)
        else:
            jinja2_formatter(template, **kwargs)

    @pytest.mark.parametrize(
        "prompt_template", [{"with_messages_placeholder": True}], indirect=True
    )
    def test_build_prompt_template_with_placeholder(self, prompt_config: PromptConfig):
        prompt_template: Runnable = Prompt._build_prompt_template(prompt_config)

        assert prompt_template == ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    render_security_block() + "{% include 'system.jinja' %}",
                ),
                ("user", "{{content}}"),
                MessagesPlaceholder("messages"),
                MessagesPlaceholder("history", optional=True),
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
            "Performing LLM request",
            prompt=f"System: {render_security_block()}Hi, I'm Duo\nHuman: What's up?",
        )
        assert mock_logger.info.mock_calls[0] == expected_call

        mock_watch.assert_called_with(
            stream=False,
            unit_primitive=prompt.unit_primitive,
            internal_event_client=prompt.internal_event_client,
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
            prompt=(
                f"System: {render_security_block()}Hi, I'm Duo\n"
                "Human: What's up?\nAI: Fine, you?\nHuman: Good."
            ),
        )
        assert mock_logger.info.mock_calls[0] == expected_call

        mock_watch.assert_called_with(
            stream=False,
            unit_primitive=prompt.unit_primitive,
            internal_event_client=prompt.internal_event_client,
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
            prompt=f"System: {render_security_block()}Hi, I'm Duo\nHuman: What's up?",
        )
        assert mock_logger.info.mock_calls[0] == expected_call

        assert response == model_response

        mock_watch.assert_called_with(
            stream=True,
            unit_primitive=prompt.unit_primitive,
            internal_event_client=prompt.internal_event_client,
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
            prompt=(
                f"System: {render_security_block()}Hi, I'm Duo\n"
                "Human: What's up?\nAI: Fine, you?\nHuman: Good."
            ),
        )
        assert mock_logger.info.mock_calls[0] == expected_call

        assert response == model_response

        mock_watch.assert_called_with(
            stream=True,
            unit_primitive=prompt.unit_primitive,
            internal_event_client=prompt.internal_event_client,
        )

        mock_watcher.afinish.assert_awaited_once()

    @pytest.mark.asyncio
    @mock.patch("ai_gateway.prompts.base.get_request_logger")
    @pytest.mark.parametrize("runnable_config", [None, RunnableConfig(callbacks=None)])
    async def test_atransform(
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

        async def input():
            yield {"name": "Duo", "content": "What's up?"}

        async for c in prompt.atransform(input(), runnable_config):
            response += str(c.content)

            # Make sure we don't finish prematurely
            mock_watcher.afinish.assert_not_awaited()

        expected_call = mock.call(
            "Performing LLM request",
            prompt=f"System: {render_security_block()}Hi, I'm Duo\nHuman: What's up?",
        )
        assert mock_logger.info.mock_calls[0] == expected_call

        assert response == model_response

        mock_watch.assert_called_with(
            stream=True,
            unit_primitive=prompt.unit_primitive,
            internal_event_client=prompt.internal_event_client,
        )

        mock_watcher.afinish.assert_awaited_once()

    @pytest.mark.asyncio
    @mock.patch("ai_gateway.prompts.base.get_request_logger")
    @pytest.mark.parametrize(
        "prompt_template", [{"with_messages_placeholder": True}], indirect=True
    )
    async def test_atransform_with_messages_placeholder(
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

        async def input():
            yield {
                "name": "Duo",
                "content": "What's up?",
                "messages": [
                    AIMessage(content="Fine, you?"),
                    HumanMessage(content="Good."),
                ],
            }

        async for c in prompt.atransform(input()):
            response += str(c.content)

            # Make sure we don't finish prematurely
            mock_watcher.afinish.assert_not_awaited()

        expected_call = mock.call(
            "Performing LLM request",
            prompt=(
                f"System: {render_security_block()}Hi, I'm Duo\n"
                "Human: What's up?\nAI: Fine, you?\nHuman: Good."
            ),
        )
        assert mock_logger.info.mock_calls[0] == expected_call

        assert response == model_response

        mock_watch.assert_called_with(
            stream=True,
            unit_primitive=prompt.unit_primitive,
            internal_event_client=prompt.internal_event_client,
        )

        mock_watcher.afinish.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ainvoke_model_instrumentator_callbacks(
        self,
        mock_watch: mock.Mock,
        prompt: Prompt,
        usage_metadata: UsageMetadata,
    ):
        mock_watcher = mock_watch.return_value.__enter__.return_value

        with self._mock_usage_metadata(prompt.model_name, usage_metadata):
            result = await prompt.ainvoke({"name": "Duo", "content": "What's up?"})

        mock_watcher.register_message.assert_called_once_with(result)

        mock_watcher.register_token_usage.assert_called_once_with(
            prompt.model_name, usage_metadata, prompt.internal_event_extra
        )

    @pytest.mark.asyncio
    async def test_astream_model_instrumentator_callbacks(
        self,
        mock_watch: mock.Mock,
        prompt: Prompt,
        usage_metadata: UsageMetadata,
    ):
        mock_watcher = mock_watch.return_value.__enter__.return_value

        with self._mock_usage_metadata(prompt.model_name, usage_metadata):
            async for message in prompt.astream(
                {"name": "Duo", "content": "What's up?"}
            ):
                mock_watcher.register_message.assert_any_call(message)

        mock_watcher.register_token_usage.assert_called_once_with(
            prompt.model_name, usage_metadata, prompt.internal_event_extra
        )

    @pytest.mark.asyncio
    async def test_atransform_model_instrumentator_callbacks(
        self,
        mock_watch: mock.Mock,
        internal_event_client: mock.Mock,
        prompt: Prompt,
        usage_metadata: UsageMetadata,
    ):
        mock_watcher = mock_watch.return_value.__enter__.return_value

        prompt.internal_event_client = internal_event_client

        async def input():
            yield {"name": "Duo", "content": "What's up?"}

        with self._mock_usage_metadata(prompt.model_name, usage_metadata):
            async for message in prompt.atransform(input()):
                mock_watcher.register_message.assert_any_call(message)

        mock_watcher.register_token_usage.assert_called_once_with(
            prompt.model_name, usage_metadata, prompt.internal_event_extra
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
    @pytest.mark.parametrize(
        ("model_response", "usage_metadata"),
        [
            # first yield for the single character, second yield empty content end of stream
            ("a", UsageMetadata(input_tokens=1, output_tokens=1, total_tokens=2)),
        ],
    )
    async def test_atransform_usage_metadata_before_stream_end(
        self, mock_watch: mock.Mock, internal_event_client: mock.Mock, prompt: Prompt
    ):
        mock_watcher = mock_watch.return_value.__enter__.return_value
        prompt.internal_event_client = internal_event_client

        async def input():
            yield {"name": "Duo", "content": "What's up?"}

        iterator = prompt.atransform(input())

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

        mock_ainvoke.assert_called_once()
        assert mock_ainvoke.call_args.args[0] == ChatPromptValue(
            messages=[
                SystemMessage(content=render_security_block() + "Hi, I'm Duo"),
                HumanMessage(content="What's up?"),
            ]
        )
        assert len(mock_ainvoke.call_args.args) >= 2

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

        mock_astream.assert_called_once()
        assert mock_astream.call_args.args[0] == ChatPromptValue(
            messages=[
                SystemMessage(content=render_security_block() + "Hi, I'm Duo"),
                HumanMessage(content="What's up?"),
            ]
        )
        assert len(mock_astream.call_args.args) >= 2

    @pytest.mark.asyncio
    async def test_astream_missing_inputs(self, prompt: Prompt):
        with pytest.raises(
            KeyError,
            match="Input to ChatPromptTemplate is missing variables {'name'}",
        ):
            await anext(prompt.astream({"content": "What's up?"}))

    @pytest.mark.asyncio
    async def test_ainvoke_retries_on_read_error(self, prompt: Prompt):
        """Test that ainvoke retries on httpx.ReadError and succeeds on third attempt."""
        success_response = AIMessage(content="Hello!")
        call_count = 0

        async def flaky_ainvoke(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ReadError("Connection reset by peer")
            return success_response

        with mock.patch.object(FakeModel, "ainvoke", side_effect=flaky_ainvoke):
            with mock.patch("asyncio.sleep"):
                result = await prompt.ainvoke({"name": "Duo", "content": "What's up?"})

        assert result == success_response
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_ainvoke_raises_after_exhausting_retries_on_read_error(
        self, prompt: Prompt
    ):
        """Test that ainvoke raises httpx.ReadError after all retry attempts fail."""
        with mock.patch.object(
            FakeModel,
            "ainvoke",
            side_effect=httpx.ReadError("Connection reset by peer"),
        ):
            with mock.patch("asyncio.sleep"):
                with pytest.raises(httpx.ReadError):
                    await prompt.ainvoke({"name": "Duo", "content": "What's up?"})

    @pytest.mark.asyncio
    async def test_ainvoke_retries_on_timeout(self, prompt: Prompt):
        """Test that ainvoke retries on httpx.TimeoutException (covers ReadTimeout, ConnectTimeout, etc.)."""
        success_response = AIMessage(content="Hello!")
        call_count = 0
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

        async def flaky_ainvoke(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ReadTimeout(
                    "Timeout on reading data from socket", request=request
                )
            return success_response

        with mock.patch.object(FakeModel, "ainvoke", side_effect=flaky_ainvoke):
            with mock.patch("asyncio.sleep"):
                result = await prompt.ainvoke({"name": "Duo", "content": "What's up?"})

        assert result == success_response
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_ainvoke_retries_on_mid_stream_fallback_error(self, prompt: Prompt):
        """Test that ainvoke retries on litellm.MidStreamFallbackError (ServiceUnavailable subclass)."""
        success_response = AIMessage(content="Hello!")
        call_count = 0

        async def flaky_ainvoke(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise MidStreamFallbackError(
                    "Overloaded", model="claude-3", llm_provider="anthropic"
                )
            return success_response

        with mock.patch.object(FakeModel, "ainvoke", side_effect=flaky_ainvoke):
            with mock.patch("asyncio.sleep"):
                result = await prompt.ainvoke({"name": "Duo", "content": "What's up?"})

        assert result == success_response
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_ainvoke_retries_on_api_connection_error(self, prompt: Prompt):
        """Test that ainvoke retries on anthropic.APIConnectionError."""
        success_response = AIMessage(content="Hello!")
        call_count = 0
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

        async def flaky_ainvoke(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise APIConnectionError(request=request)
            return success_response

        with mock.patch.object(FakeModel, "ainvoke", side_effect=flaky_ainvoke):
            with mock.patch("asyncio.sleep"):
                result = await prompt.ainvoke({"name": "Duo", "content": "What's up?"})

        assert result == success_response
        assert call_count == 3

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(
                AnthropicInternalServerError(
                    "Internal server error",
                    response=httpx.Response(
                        500, request=httpx.Request("POST", "https://api.anthropic.com")
                    ),
                    body=None,
                ),
                id="anthropic_internal_server_error",
            ),
            pytest.param(
                # Streaming path: HTTP 200 with error embedded in the body.
                # _make_status_error raises the base APIStatusError (not InternalServerError)
                # because status_code comes from the body, not the HTTP response.
                APIStatusError(
                    "Internal server error",
                    response=httpx.Response(
                        200, request=httpx.Request("POST", "https://api.anthropic.com")
                    ),
                    body={
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": "Internal server error",
                        },
                    },
                ),
                id="anthropic_internal_server_error_streaming_path",
            ),
            pytest.param(
                LiteLLMInternalServerError(
                    "Internal error encountered.",
                    model="claude-haiku-4-5",
                    llm_provider="vertex_ai",
                ),
                id="litellm_internal_server_error",
            ),
        ],
    )
    async def test_ainvoke_retries_on_internal_server_error(
        self, prompt: Prompt, error: Exception
    ):
        """Test that ainvoke retries on InternalServerError from Anthropic and LiteLLM."""
        success_response = AIMessage(content="Hello!")
        call_count = 0

        async def flaky_ainvoke(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise error
            return success_response

        with mock.patch.object(FakeModel, "ainvoke", side_effect=flaky_ainvoke):
            with mock.patch("asyncio.sleep"):
                result = await prompt.ainvoke({"name": "Duo", "content": "What's up?"})

        assert result == success_response
        assert call_count == 3

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(
                AnthropicOverloadedError(
                    "Overloaded",
                    response=httpx.Response(
                        529,
                        request=httpx.Request(
                            "POST", "https://api.anthropic.com/v1/messages"
                        ),
                    ),
                    body=None,
                ),
                id="anthropic_overloaded_error",
            ),
            pytest.param(
                # Streaming path: HTTP 200 with overloaded_error in body.
                APIStatusError(
                    "Overloaded",
                    response=httpx.Response(
                        200,
                        request=httpx.Request(
                            "POST", "https://api.anthropic.com/v1/messages"
                        ),
                    ),
                    body={
                        "type": "error",
                        "error": {"type": "overloaded_error", "message": "Overloaded"},
                    },
                ),
                id="anthropic_overloaded_error_streaming_path",
            ),
        ],
    )
    async def test_ainvoke_retries_on_anthropic_overloaded_error(
        self, prompt: Prompt, error: Exception
    ):
        """Test that ainvoke retries on OverloadedError from Anthropic, including streaming path."""
        success_response = AIMessage(content="Hello!")
        call_count = 0

        async def flaky_ainvoke(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise error
            return success_response

        with mock.patch.object(FakeModel, "ainvoke", side_effect=flaky_ainvoke):
            with mock.patch("asyncio.sleep"):
                result = await prompt.ainvoke({"name": "Duo", "content": "What's up?"})

        assert result == success_response
        assert call_count == 3

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(
                BadRequestError(
                    "Bad request",
                    response=httpx.Response(
                        400,
                        request=httpx.Request(
                            "POST", "https://api.anthropic.com/v1/messages"
                        ),
                    ),
                    body=None,
                ),
                id="bad_request_error",
            ),
            pytest.param(
                # Streaming path: HTTP 200 with a non-retryable error type in body.
                APIStatusError(
                    "Invalid request",
                    response=httpx.Response(
                        200,
                        request=httpx.Request(
                            "POST", "https://api.anthropic.com/v1/messages"
                        ),
                    ),
                    body={
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": "Invalid request",
                        },
                    },
                ),
                id="invalid_request_error_streaming_path",
            ),
        ],
    )
    async def test_ainvoke_does_not_retry_on_anthropic_non_retryable_status(
        self, prompt: Prompt, error: Exception
    ):
        """Test that ainvoke does not retry on non-retryable Anthropic errors."""
        with mock.patch.object(FakeModel, "ainvoke", side_effect=error):
            with pytest.raises(type(error)):
                await prompt.ainvoke({"name": "Duo", "content": "What's up?"})

    @pytest.mark.asyncio
    async def test_ainvoke_uses_exponential_backoff(self, prompt: Prompt):
        """Test that retry waits follow exponential backoff: ~3s, ~9s, ~27s."""
        sleep_calls = []

        async def fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        with mock.patch.object(
            FakeModel,
            "ainvoke",
            side_effect=httpx.ReadError("Connection reset by peer"),
        ):
            with mock.patch("asyncio.sleep", side_effect=fake_sleep):
                with pytest.raises(httpx.ReadError):
                    await prompt.ainvoke({"name": "Duo", "content": "What's up?"})

        assert len(sleep_calls) == 3
        assert sleep_calls[0] == pytest.approx(3.0)
        assert sleep_calls[1] == pytest.approx(9.0)
        assert sleep_calls[2] == pytest.approx(27.0)

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
        model_provider: ModelClassProvider,
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
                model_provider=model_provider,
                model_factory=model_factory,
                config=prompt_config,
                model_metadata=model_metadata,
                tools=[mock_tool],
                tool_choice=tool_choice,
            )

        kwargs = mock_bind_tool.call_args.kwargs

        assert kwargs["tool_choice"] == tool_choice

    @pytest.mark.parametrize(
        (
            "model_provider",
            "model_params",
            "prompt_params",
            "expect_context_management",
        ),
        [
            (
                ModelClassProvider.LITE_LLM,
                ChatLiteLLMParams(
                    model="claude-sonnet-4-20250514",
                    custom_llm_provider="anthropic",
                ),
                PromptParams(
                    context_management={
                        "edits": [
                            {
                                "type": "clear_tool_uses_20250919",
                                "trigger": {"type": "input_tokens", "value": 1000},
                                "keep": {"type": "tool_uses", "value": 1},
                            }
                        ]
                    }
                ),
                True,
            ),
            (
                ModelClassProvider.ANTHROPIC,
                ChatAnthropicParams(),
                PromptParams(
                    context_management={
                        "edits": [
                            {
                                "type": "clear_tool_uses_20250919",
                                "trigger": {"type": "input_tokens", "value": 1000},
                                "keep": {"type": "tool_uses", "value": 1},
                            }
                        ]
                    }
                ),
                True,
            ),
            (
                ModelClassProvider.LITE_LLM,
                ChatLiteLLMParams(
                    model="gpt-4",
                    custom_llm_provider="openai",
                ),
                PromptParams(
                    context_management={
                        "edits": [
                            {
                                "type": "clear_tool_uses_20250919",
                                "trigger": {"type": "input_tokens", "value": 1000},
                                "keep": {"type": "tool_uses", "value": 1},
                            }
                        ]
                    }
                ),
                False,
            ),
            (
                ModelClassProvider.LITE_LLM,
                ChatLiteLLMParams(
                    model="some-model",
                ),
                PromptParams(
                    context_management={
                        "edits": [
                            {
                                "type": "clear_tool_uses_20250919",
                                "trigger": {"type": "input_tokens", "value": 1000},
                                "keep": {"type": "tool_uses", "value": 1},
                            }
                        ]
                    }
                ),
                False,
            ),
        ],
    )
    def test_context_management_filtered_by_provider(
        self,
        model_provider: ModelClassProvider,
        prompt_config: PromptConfig,
        model_metadata: TypeModelMetadata,
        model_factory: TypeModelFactory,
        model: FakeModel,
        expect_context_management: bool,
    ):
        with mock.patch.object(FakeModel, "bind") as mock_bind:
            mock_bind.return_value = model

            Prompt(
                model_provider=model_provider,
                model_factory=model_factory,
                config=prompt_config,
                model_metadata=model_metadata,
            )

        bind_kwargs = mock_bind.call_args.kwargs
        if expect_context_management:
            assert "context_management" in bind_kwargs
        else:
            assert "context_management" not in bind_kwargs


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
        ("prompt_params", "model_provider", "expected_to_use_converter"),
        [
            (
                PromptParams(
                    cache_control_injection_points=[{"location": "message", "index": 0}]
                ),
                ModelClassProvider.ANTHROPIC,
                True,
            ),
            (
                PromptParams(
                    cache_control_injection_points=[{"location": "message", "index": 0}]
                ),
                ModelClassProvider.LITE_LLM,
                False,
            ),
            (
                PromptParams(),
                ModelClassProvider.ANTHROPIC,
                True,
            ),
        ],
    )
    async def test_prompt_caching(
        self,
        mock_cache_control_injection_points_converter,
        mock_filter_cache_control_injection_points,
        model_provider: ModelClassProvider,
        model_factory: TypeModelFactory,
        prompt_config: PromptConfig,
        model_metadata: TypeModelMetadata | None,
        prompt_template_factory: TypePromptTemplateFactory | None,
        expected_to_use_converter: bool,
    ):
        Prompt(
            model_provider,
            model_factory,
            prompt_config,
            model_metadata,
            prompt_template_factory,
        )

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
            super().__init__(internal_event_client, model_limits)

        def get(self, *_args, **_kwargs):
            return prompt

        def get_required_variables(self, *_args, **_kwargs) -> set[str]:
            raise TemplateNotFoundError("not available in test registry")

    return Registry()


class TestBaseRegistry:
    @pytest.mark.parametrize(
        (
            "unit_primitive",
            "scopes",
            "input_model_metadata",
            "tools",
            "success",
            "expected_internal_events",
        ),
        [
            (
                GitLabUnitPrimitive.COMPLETE_CODE,
                ["complete_code"],
                None,
                None,
                True,
                [call("request_complete_code", category="ai_gateway.prompts.base")],
            ),
            (
                GitLabUnitPrimitive.COMPLETE_CODE,
                ["complete_code"],
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
                None,
                True,
                [
                    call("request_complete_code", category="ai_gateway.prompts.base"),
                ],
            ),
            (
                GitLabUnitPrimitive.AMAZON_Q_INTEGRATION,
                ["amazon_q_integration"],
                AmazonQModelMetadata(
                    name="amazon_q",
                    provider="amazon_q",
                    role_arn="role-arn",
                    llm_definition=ChatAmazonQDefinition(
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
            (GitLabUnitPrimitive.COMPLETE_CODE, [], None, None, False, []),
            (
                GitLabUnitPrimitive.DUO_CHAT,
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

            if input_model_metadata:
                assert input_model_metadata._user == user

            internal_event_client.track_event.assert_has_calls(expected_internal_events)
        else:
            with pytest.raises(WrongUnitPrimitives):
                registry.get_on_behalf(user=user, prompt_id="test")

            internal_event_client.track_event.assert_not_called()

    @pytest.mark.parametrize(
        (
            "unit_primitive",
            "scopes",
            "internal_event_category",
            "expected_internal_events",
        ),
        [
            (
                GitLabUnitPrimitive.COMPLETE_CODE,
                ["complete_code"],
                "my_category",
                [
                    call("request_complete_code", category="my_category"),
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
        ("model_metadata", "unit_primitive", "scopes"),
        [
            (
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
                GitLabUnitPrimitive.COMPLETE_CODE,
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
        ("unit_primitive", "scopes", "prompt_version"),
        [
            (
                GitLabUnitPrimitive.COMPLETE_CODE,
                ["complete_code"],
                "^1.0.0",
            ),
            (
                GitLabUnitPrimitive.COMPLETE_CODE,
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

        prompt.internal_callbacks = [ContextCallback(prompt)]

        await prompt.ainvoke({"name": "Duo", "content": "What's up?"})

        assert "model_name" in captured_data
        assert captured_data["model_name"] == prompt.model_name

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
    model_class_provider: anthropic
    max_context_tokens: 1000
  - name: Model B
    gitlab_identifier: model_b
    model_class_provider: litellm
    max_context_tokens: 1000
  - name: Model C
    gitlab_identifier: model_c
    model_class_provider: amazon_q
    max_context_tokens: 1000
  - name: "text-embedding-005 - Vertex"
    provider: "Vertex"
    gitlab_identifier: text_embedding_005_vertex
    description: "Natural language processing technique that converts textual data into numerical vectors."
    cost_indicator: "$"
    max_context_tokens: 20000
    model_class_provider: litellm_embedding
    family:
      - vertex
    params:
      model: "text-embedding-005"
      custom_llm_provider: vertex_ai
    prompt_params:
      vertex_location: global
""",
        )
        fs.create_file(
            model_selection_dir / "unit_primitives.yml",
            contents="""---
configurable_unit_primitives:
  - feature_setting: "setting_a"
    unit_primitives:
      - "duo_chat"
    default_models:
      - "model_a"
    selectable_models:
      - "model_a"
      - "model_c"
  - feature_setting: "b"
    unit_primitives:
      - "code_suggestions"
    default_models:
      - "model_b"
    selectable_models:
      - "model_b"
  - feature_setting: "c"
    unit_primitives:
      - "complete_code"
    default_models:
      - "model_b"
    selectable_models:
      - "model_b"
  - feature_setting: "embeddings_code"
    unit_primitives:
      - "generate_embeddings_codebase"
    default_models:
      - "text_embedding_005_vertex"
    selectable_models:
      - "text_embedding_005_vertex"
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
    async def test_skips_embedding_model_in_validation(
        self,
        registry: BasePromptRegistry,
    ):
        with mock.patch.object(registry, "get", wraps=registry.get) as mock_get:
            await registry.validate_default_models()

        validated_models = {
            call.kwargs["model_metadata"].name
            for call in mock_get.mock_calls
            if call.args and call.args[0] == "model_configuration/check"
        }
        assert validated_models == {"model_a", "model_b"}
        assert registry.validations is not None
        assert "text_embedding_005_vertex" in registry.validations

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
                            llm_definition=ChatAnthropicDefinition(
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
                            llm_definition=ChatLiteLLMDefinition(
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


@pytest.mark.skipif(
    # pylint: disable=direct-environment-variable-reference
    os.getenv("REAL_AI_REQUEST") is None,
    # pylint: enable=direct-environment-variable-reference
    reason="3rd party requests not enabled",
)
class TestContextManagementTokenReduction:
    """Integration test demonstrating that Anthropic context management reduces input tokens.

    Uses aggressive thresholds to trigger tool result clearing quickly.
    The test builds a conversation with simulated tool use/result pairs to
    generate enough token volume, then verifies that context_management
    clears old tool results and reduces input tokens.

    Run with: REAL_AI_REQUEST=1 pytest tests/prompts/test_base.py::TestContextManagementTokenReduction -v
    """

    AGGRESSIVE_CONTEXT_MANAGEMENT = {
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": 1000},
                "keep": {"type": "tool_uses", "value": 1},
                "clear_at_least": {"type": "input_tokens", "value": 500},
                "exclude_tools": [],
            }
        ]
    }

    BULKY_TOOL_RESULT = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris "
        "nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in "
        "reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla "
        "pariatur. Excepteur sint occaecat cupidatat non proident, sunt in "
        "culpa qui officia deserunt mollit anim id est laborum. "
    ) * 3

    NUM_TOOL_TURNS = 5

    @pytest.fixture(name="model_params")
    def model_params_fixture(self):
        return ChatLiteLLMParams(
            custom_llm_provider="anthropic",
            model="claude-sonnet-4-20250514",
            max_tokens=100,
        )

    @pytest.fixture(name="prompt_template")
    def prompt_template_fixture(self):
        return {
            "system": "You are a helpful assistant. Keep responses very short.",
            "placeholder": "messages",
            "user": "{{content}}",
        }

    @pytest.fixture(name="tools")
    def tools_fixture(self):
        def read_file(_path: str) -> str:
            return ""

        return [
            StructuredTool.from_function(
                func=read_file,
                name="read_file",
                description="Read a file from disk.",
            )
        ]

    @pytest.fixture(name="tool_history")
    def tool_history_fixture(self):
        messages = []
        for i in range(self.NUM_TOOL_TURNS):
            tool_use_id = f"toolu_{i:04d}"
            messages.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": tool_use_id,
                            "name": "read_file",
                            "args": {"path": f"/tmp/file_{i}.txt"},
                        }
                    ],
                )
            )
            messages.append(
                ToolMessage(
                    content=f"File {i} contents: {self.BULKY_TOOL_RESULT}",
                    tool_call_id=tool_use_id,
                )
            )
        return messages

    @staticmethod
    def _build_prompt(
        model_params: ChatLiteLLMParams,
        prompt_template: dict,
        tools: list[BaseTool],
        prompt_params: PromptParams | None = None,
    ) -> Prompt:
        config = PromptConfig(
            name="context_management_test",
            model=ModelConfig(params=model_params),
            unit_primitive="duo_chat",
            prompt_template=prompt_template,
            params=prompt_params,
        )

        def model_factory(**kwargs):
            return ChatLiteLLM(**kwargs)

        return Prompt(
            model_provider=ModelClassProvider.LITE_LLM,
            model_factory=model_factory,
            config=config,
            tools=tools,
        )

    @pytest.mark.asyncio
    async def test_context_management_reduces_input_tokens(
        self,
        model_params: ChatLiteLLMParams,
        prompt_template: dict,
        tools: list[BaseTool],
        tool_history: list,
    ):
        prompt_with_cm = self._build_prompt(
            model_params,
            prompt_template,
            tools,
            prompt_params=PromptParams(
                context_management=self.AGGRESSIVE_CONTEXT_MANAGEMENT,
            ),
        )
        prompt_without_cm = self._build_prompt(model_params, prompt_template, tools)

        response_with_cm = await prompt_with_cm.ainvoke(
            {"content": "Summarize all files you read.", "messages": tool_history}
        )
        response_without_cm = await prompt_without_cm.ainvoke(
            {"content": "Summarize all files you read.", "messages": tool_history}
        )

        assert isinstance(response_with_cm, AIMessage)
        assert isinstance(response_without_cm, AIMessage)
        assert response_with_cm.usage_metadata is not None
        assert response_without_cm.usage_metadata is not None
        tokens_with_cm = response_with_cm.usage_metadata.get("input_tokens", 0)
        tokens_without_cm = response_without_cm.usage_metadata.get("input_tokens", 0)

        assert tokens_without_cm > 0, "Baseline should have recorded input tokens"
        assert (
            tokens_with_cm > 0
        ), "Context management run should have recorded input tokens"
        assert tokens_with_cm < tokens_without_cm, (
            f"Context management should reduce input tokens: "
            f"with_cm={tokens_with_cm}, without_cm={tokens_without_cm}"
        )


class TestBuildModelExtraHeaders:
    """Tests for merging AIGW_CUSTOM_MODELS__EXTRA_HEADERS with YAML extra_headers."""

    @pytest.fixture(name="capturing_model_factory")
    def capturing_model_factory_fixture(self, model: FakeModel):
        class CapturingFactory:
            def __init__(self):
                self.captured_kwargs: dict[str, Any] = {}

            def __call__(self, **kwargs: Any) -> FakeModel:
                self.captured_kwargs.update(kwargs)
                return model

        return CapturingFactory()

    def test_custom_models_extra_headers_none(
        self,
        capturing_model_factory,
        model_config,
    ):
        """When custom_models_extra_headers is None, no extra_headers should be set beyond what the model config
        provides."""
        Prompt._build_model(
            Prompt,
            model_factory=capturing_model_factory,
            config=model_config,
            model_metadata=None,
            disable_streaming=False,
            custom_models_extra_headers=None,
        )

        assert "extra_headers" not in capturing_model_factory.captured_kwargs

    def test_custom_models_extra_headers_empty_dict(
        self,
        capturing_model_factory,
        model_config,
    ):
        """When custom_models_extra_headers is an empty dict, it should be treated as falsy and no merging should
        occur."""
        Prompt._build_model(
            Prompt,
            model_factory=capturing_model_factory,
            config=model_config,
            model_metadata=None,
            disable_streaming=False,
            custom_models_extra_headers={},
        )

        assert "extra_headers" not in capturing_model_factory.captured_kwargs

    def test_extra_headers_merged_env_and_yaml(
        self,
        capturing_model_factory,
    ):
        """When both env and YAML extra_headers are set, they should be merged with YAML values taking precedence over
        env values."""
        env_headers = {
            "x-env-header": "env-value",
            "x-shared-header": "env-shared-value",
        }
        config = ModelConfig(
            params=ChatLiteLLMParams(
                model="test_model",
                extra_headers={
                    "x-yaml-header": "yaml-value",
                    "x-shared-header": "yaml-shared-value",
                },
            )
        )

        Prompt._build_model(
            Prompt,
            model_factory=capturing_model_factory,
            config=config,
            model_metadata=None,
            disable_streaming=False,
            custom_models_extra_headers=env_headers,
        )

        result_headers = capturing_model_factory.captured_kwargs["extra_headers"]
        # Env-only header is preserved
        assert result_headers["x-env-header"] == "env-value"
        # YAML-only header is preserved
        assert result_headers["x-yaml-header"] == "yaml-value"
        # Shared key: YAML wins over env
        assert result_headers["x-shared-header"] == "yaml-shared-value"

    def test_extra_headers_env_only(
        self,
        capturing_model_factory,
        model_config,
    ):
        """When only env extra_headers are set (no YAML), only env headers should appear."""
        env_headers = {"x-env-header": "env-value"}

        Prompt._build_model(
            Prompt,
            model_factory=capturing_model_factory,
            config=model_config,
            model_metadata=None,
            disable_streaming=False,
            custom_models_extra_headers=env_headers,
        )

        result_headers = capturing_model_factory.captured_kwargs["extra_headers"]
        assert result_headers == {"x-env-header": "env-value"}

    def test_warning_logged_when_yaml_overrides_env_headers(
        self,
        capturing_model_factory,
    ):
        """A warning should be logged when YAML extra_headers override keys from AIGW_CUSTOM_MODELS__EXTRA_HEADERS."""
        env_headers = {
            "x-env-only": "env-value",
            "x-shared": "env-shared-value",
        }
        config = ModelConfig(
            params=ChatLiteLLMParams(
                model="test_model",
                extra_headers={
                    "x-yaml-only": "yaml-value",
                    "x-shared": "yaml-shared-value",
                },
            )
        )

        with capture_logs() as cap_logs:
            Prompt._build_model(
                Prompt,
                model_factory=capturing_model_factory,
                config=config,
                model_metadata=None,
                disable_streaming=False,
                custom_models_extra_headers=env_headers,
            )

        warning_logs = [l for l in cap_logs if l.get("log_level") == "warning"]
        assert len(warning_logs) == 1
        assert "overriding" in warning_logs[0]["event"].lower()
        assert warning_logs[0]["overridden_keys"] == ["x-shared"]
