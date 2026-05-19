from contextlib import contextmanager
from datetime import datetime
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest
from gitlab_cloud_connector import GitLabUnitPrimitive
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage

from ai_gateway.code_suggestions.base import CodeSuggestionsChunk, CodeSuggestionsOutput
from ai_gateway.code_suggestions.processing.typing import LanguageId
from ai_gateway.models.base import ModelMetadata as LegacyModelMetadata
from ai_gateway.models.base_text import TextGenModelChunk, TextGenModelOutput
from ai_gateway.safety_attributes import SafetyAttributes


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
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        assert self.expected_message in messages[0].content

        return self.response


@pytest.fixture(name="model_factory")
def model_factory_fixture():
    return lambda model, **kwargs: FakeModel(
        expected_message="Hi, I'm John and I'm 20 years old. It's now July 12, 2025.",
        response="Hi John!",
    )


@pytest.fixture(name="prompt_template")
def prompt_template_fixture():
    return {
        "system": "Hi, I'm {{name}} and I'm {{age}} years old. It's now {{current_date}}."
    }


@pytest.fixture(name="compatible_versions")
def compatible_versions_fixture():
    return ["1.0.0"]


@pytest.fixture(name="mock_registry_get")
def mock_registry_get_fixture(
    request,
    compatible_versions: list[str],
):
    with patch("ai_gateway.prompts.registry.LocalPromptRegistry.get") as mock:
        if compatible_versions is None:
            mock.side_effect = KeyError()
        elif not compatible_versions:
            mock.side_effect = ValueError("No prompt version found matching the query:")
        else:
            mock.return_value = request.getfixturevalue("prompt")

        yield mock


@pytest.fixture(name="frozen_datetime_now")
def frozen_datetime_now_fixture():
    frozen = datetime(2025, 7, 12, 12, 0, 0)
    with patch("ai_gateway.api.v1.prompts.invoke.datetime") as mock_datetime:
        mock_datetime.now.return_value = frozen
        yield mock_datetime


@pytest.fixture(name="scopes")
def scopes_fixture(unit_primitive: GitLabUnitPrimitive):
    return [unit_primitive.value]


@pytest.fixture(name="mock_track_internal_event")
def mock_track_internal_event_fixture():
    with patch("lib.internal_events.InternalEventsClient.track_event") as mock:
        yield mock


@pytest.fixture(name="mock_output_text")
def mock_output_text_fixture():
    return "test completion"


@pytest.fixture(name="mock_output")
def mock_output_fixture(mock_output_text: str):
    return TextGenModelOutput(
        text=mock_output_text,
        score=10_000,
        safety_attributes=SafetyAttributes(),
    )


@contextmanager
def _mock_generate(klass: str, mock_output: TextGenModelOutput):
    with patch(f"{klass}.generate", return_value=mock_output) as mock:
        yield mock


@contextmanager
def _mock_async_generate(klass: str, mock_output: TextGenModelOutput):
    async def _stream(*_args: Any, **_kwargs: Any) -> AsyncIterator[TextGenModelChunk]:
        for c in list(mock_output.text):
            yield TextGenModelChunk(text=c)

    with patch(f"{klass}.generate", side_effect=_stream) as mock:
        yield mock


@pytest.fixture(name="mock_code_bison")
def mock_code_bison_fixture(mock_output: TextGenModelOutput):
    with _mock_generate(
        "ai_gateway.models.vertex_text.PalmCodeBisonModel", mock_output
    ) as mock:
        yield mock


@pytest.fixture(name="mock_anthropic")
def mock_anthropic_fixture(mock_output: TextGenModelOutput):
    with _mock_generate(
        "ai_gateway.models.anthropic.AnthropicModel", mock_output
    ) as mock:
        yield mock


@pytest.fixture(name="mock_anthropic_chat")
def mock_anthropic_chat_fixture(mock_output: TextGenModelOutput):
    with _mock_generate(
        "ai_gateway.models.anthropic.AnthropicChatModel", mock_output
    ) as mock:
        yield mock


@pytest.fixture(name="mock_llm_chat")
def mock_llm_chat_fixture(mock_output: TextGenModelOutput):
    with _mock_generate(
        "ai_gateway.models.litellm.LiteLlmChatModel", mock_output
    ) as mock:
        yield mock


@pytest.fixture(name="mock_llm_text")
def mock_llm_text_fixture(mock_output: TextGenModelOutput):
    with _mock_generate(
        "ai_gateway.models.litellm.LiteLlmTextGenModel", mock_output
    ) as mock:
        yield mock


@pytest.fixture(name="mock_agent_model")
def mock_agent_model_fixture(mock_output: TextGenModelOutput):
    with _mock_generate(
        "ai_gateway.models.agent_model.AgentModel", mock_output
    ) as mock:
        yield mock


@pytest.fixture(name="mock_amazon_q_model")
def mock_amazon_q_model_fixture(mock_output: TextGenModelOutput):
    with _mock_generate("ai_gateway.models.amazon_q.AmazonQModel", mock_output) as mock:
        yield mock


@pytest.fixture(name="mock_suggestions_output_text")
def mock_suggestions_output_text_fixture():
    return "def search"


@pytest.fixture(name="mock_suggestions_model")
def mock_suggestions_model_fixture():
    return "claude-3-haiku-20240307"


@pytest.fixture(name="mock_suggestions_engine")
def mock_suggestions_engine_fixture():
    return "anthropic"


@pytest.fixture(name="mock_suggestions_output")
def mock_suggestions_output_fixture(
    mock_suggestions_output_text: str,
    mock_suggestions_model: str,
    mock_suggestions_engine: str,
):
    return CodeSuggestionsOutput(
        text=mock_suggestions_output_text,
        score=0,
        model_metadata=LegacyModelMetadata(
            name=mock_suggestions_model, engine=mock_suggestions_engine
        ),
        lang_id=LanguageId.PYTHON,
        metadata=CodeSuggestionsOutput.Metadata(),  # type: ignore[attr-defined]
    )


@contextmanager
def _mock_execute(klass: str, mock_suggestions_output: CodeSuggestionsOutput):
    with patch(f"{klass}.execute", return_value=mock_suggestions_output) as mock:
        yield mock


@pytest.fixture(name="mock_generations")
def mock_generations_fixture(mock_suggestions_output: CodeSuggestionsOutput):
    with _mock_execute(
        "ai_gateway.code_suggestions.CodeGenerations", mock_suggestions_output
    ) as mock:
        yield mock


@pytest.fixture(name="mock_completions")
def mock_completions_fixture(mock_suggestions_output: CodeSuggestionsOutput):
    with _mock_execute(
        "ai_gateway.code_suggestions.CodeCompletions", mock_suggestions_output
    ) as mock:
        yield mock


@contextmanager
def _mock_async_execute(klass: str, mock_suggestions_output: CodeSuggestionsOutput):
    async def _stream(
        *_args: Any, **_kwargs: Any
    ) -> AsyncIterator[CodeSuggestionsChunk]:
        for c in list(mock_suggestions_output.text):
            yield CodeSuggestionsChunk(text=c)

    with patch(f"{klass}.execute", side_effect=_stream) as mock:
        yield mock


@pytest.fixture(name="mock_generations_stream")
def mock_generations_stream_fixture(mock_suggestions_output: CodeSuggestionsOutput):
    with (  # pylint: disable=contextmanager-generator-missing-cleanup
        _mock_async_execute(
            "ai_gateway.code_suggestions.CodeGenerations", mock_suggestions_output
        ) as mock
    ):
        yield mock


@pytest.fixture(name="mock_completions_stream")
def mock_completions_stream_fixture(mock_suggestions_output: CodeSuggestionsOutput):
    with (  # pylint: disable=contextmanager-generator-missing-cleanup
        _mock_async_execute(
            "ai_gateway.code_suggestions.CodeCompletions", mock_suggestions_output
        ) as mock
    ):
        yield mock


@pytest.fixture(name="mock_with_prompt_prepared")
def mock_with_prompt_prepared_fixture():
    with patch(
        "ai_gateway.code_suggestions.CodeGenerations.with_prompt_prepared"
    ) as mock:
        yield mock


@pytest.fixture(name="mock_litellm_atext_completion")
def mock_litellm_atext_completion_fixture():
    with patch("litellm.atext_completion") as mock_acompletion:
        mock_acompletion.return_value = AsyncMock(
            choices=[
                AsyncMock(
                    text="Test text completion response",
                    logprobs=AsyncMock(token_logprobs=[999]),
                ),
            ],
            usage={"prompt_tokens": 1, "completion_tokens": 999, "total_tokens": 1000},
        )

        yield mock_acompletion
