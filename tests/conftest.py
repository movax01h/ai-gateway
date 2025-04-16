from contextlib import contextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Type
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from gitlab_cloud_connector import CloudConnectorUser, GitLabUnitPrimitive, UserClaims
from langchain.chat_models.fake import FakeListChatModel
from langchain_core.messages import BaseMessage
from langchain_core.messages.ai import AIMessage, UsageMetadata
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from starlette.middleware import Middleware
from starlette_context.middleware import RawContextMiddleware

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.api.middleware import AccessLogMiddleware, MiddlewareAuthentication
from ai_gateway.code_suggestions.base import CodeSuggestionsChunk, CodeSuggestionsOutput
from ai_gateway.code_suggestions.processing.base import ModelEngineOutput
from ai_gateway.code_suggestions.processing.typing import (
    LanguageId,
    MetadataCodeContent,
    MetadataPromptBuilder,
)
from ai_gateway.config import Config
from ai_gateway.container import ContainerApplication
from ai_gateway.internal_events.client import InternalEventsClient
from ai_gateway.model_metadata import TypeModelMetadata, current_model_metadata_context
from ai_gateway.models.base import ModelMetadata, TokensConsumptionMetadata
from ai_gateway.models.base_text import (
    TextGenModelBase,
    TextGenModelChunk,
    TextGenModelOutput,
)
from ai_gateway.prompts import Prompt
from ai_gateway.prompts.config.base import ModelConfig, PromptConfig, PromptParams
from ai_gateway.prompts.config.models import ChatLiteLLMParams, TypeModelParams
from ai_gateway.prompts.typing import Model, TypeModelFactory
from ai_gateway.safety_attributes import SafetyAttributes
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    Plan,
    WorkflowState,
    WorkflowStatusEnum,
)

pytest_plugins = ("pytest_asyncio",)


@pytest.fixture
def assets_dir() -> Path:
    return Path(__file__).parent / "_assets"


@pytest.fixture
def tpl_assets_codegen_dir(assets_dir) -> Path:
    tpl_dir = assets_dir / "tpl"
    return tpl_dir / "codegen"


@pytest.fixture
def text_gen_base_model():
    model = Mock(spec=TextGenModelBase)
    type(model).input_token_limit = PropertyMock(return_value=1_000)
    return model


@pytest.fixture(scope="class")
def stub_auth_provider():
    class StubKeyAuthProvider:
        def authenticate(self, token):
            return None

    return StubKeyAuthProvider()


@pytest.fixture(scope="class")
def test_client(fast_api_router, stub_auth_provider, request):
    middlewares = [
        Middleware(RawContextMiddleware),
        Middleware(AccessLogMiddleware, skip_endpoints=[]),
        MiddlewareAuthentication(stub_auth_provider, False, None),
    ]
    app = FastAPI(middleware=middlewares)
    app.include_router(fast_api_router)
    client = TestClient(app)

    return client


@pytest.fixture
def model_metadata_context():
    current_model_metadata_context.set(None)
    yield current_model_metadata_context


@pytest.fixture
def mock_track_internal_event():
    with patch("ai_gateway.internal_events.InternalEventsClient.track_event") as mock:
        yield mock


@pytest.fixture
def mock_detect_abuse():
    with patch("ai_gateway.abuse_detection.AbuseDetector.detect") as mock:
        yield mock


@pytest.fixture
def mock_client(
    test_client, stub_auth_provider, auth_user, mock_container, model_metadata_context
):
    """Setup all the needed mocks to perform requests in the test environment"""
    with patch.object(stub_auth_provider, "authenticate", return_value=auth_user):
        yield test_client


@pytest.fixture
def mock_connect_vertex():
    with patch("ai_gateway.models.base.PredictionServiceAsyncClient"):
        yield


@pytest.fixture
def mock_connect_vertex_search():
    with patch(
        "ai_gateway.searches.container.discoveryengine.SearchServiceAsyncClient"
    ):
        yield


@pytest.fixture
def config_values():
    return {}


@pytest.fixture
def mock_config(config_values: dict[str, Any]):
    return Config(_env_file=None, _env_prefix="AIGW_TEST", **config_values)


@pytest.fixture
def mock_container(
    mock_config: Config, mock_connect_vertex: Mock, mock_connect_vertex_search: Mock
):
    container_application = ContainerApplication()
    container_application.config.from_dict(mock_config.model_dump())

    return container_application


@pytest.fixture
def mock_output_text():
    return "test completion"


@pytest.fixture
def mock_output(mock_output_text: str):
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
    async def _stream(*args: Any, **kwargs: Any) -> AsyncIterator[TextGenModelChunk]:
        for c in list(mock_output.text):
            yield TextGenModelChunk(text=c)

    with patch(f"{klass}.generate", side_effect=_stream) as mock:
        yield mock


@pytest.fixture
def mock_code_bison(mock_output: TextGenModelOutput):
    with _mock_generate(
        "ai_gateway.models.vertex_text.PalmCodeBisonModel", mock_output
    ) as mock:
        yield mock


@pytest.fixture
def mock_code_gecko(mock_output: TextGenModelOutput):
    with _mock_generate(
        "ai_gateway.models.vertex_text.PalmCodeGeckoModel", mock_output
    ) as mock:
        yield mock


@pytest.fixture
def mock_anthropic(mock_output: TextGenModelOutput):
    with _mock_generate(
        "ai_gateway.models.anthropic.AnthropicModel", mock_output
    ) as mock:
        yield mock


@pytest.fixture
def mock_anthropic_chat(mock_output: TextGenModelOutput):
    with _mock_generate(
        "ai_gateway.models.anthropic.AnthropicChatModel", mock_output
    ) as mock:
        yield mock


@pytest.fixture
def mock_anthropic_stream(mock_output: TextGenModelOutput):
    with _mock_async_generate(  # pylint: disable=contextmanager-generator-missing-cleanup
        "ai_gateway.models.anthropic.AnthropicModel", mock_output
    ) as mock:
        yield mock


@pytest.fixture
def mock_anthropic_chat_stream(mock_output: TextGenModelOutput):
    with _mock_async_generate(  # pylint: disable=contextmanager-generator-missing-cleanup
        "ai_gateway.models.anthropic.AnthropicChatModel", mock_output
    ) as mock:
        yield mock


@pytest.fixture
def mock_llm_chat(mock_output: TextGenModelOutput):
    with _mock_generate(
        "ai_gateway.models.litellm.LiteLlmChatModel", mock_output
    ) as mock:
        yield mock


@pytest.fixture
def mock_llm_text(mock_output: TextGenModelOutput):
    with _mock_generate(
        "ai_gateway.models.litellm.LiteLlmTextGenModel", mock_output
    ) as mock:
        yield mock


@pytest.fixture
def mock_agent_model(mock_output: TextGenModelOutput):
    with _mock_generate(
        "ai_gateway.models.agent_model.AgentModel", mock_output
    ) as mock:
        yield mock


@pytest.fixture
def mock_amazon_q_model(mock_output: TextGenModelOutput):
    with _mock_generate("ai_gateway.models.amazon_q.AmazonQModel", mock_output) as mock:
        yield mock


@pytest.fixture
def mock_completions_legacy_output_texts():
    return ["def search"]


@pytest.fixture
def mock_completions_legacy_output(mock_completions_legacy_output_texts: str):
    output = []
    for text in mock_completions_legacy_output_texts:
        output.append(
            ModelEngineOutput(
                text=text,
                score=0,
                model=ModelMetadata(name="code-gecko", engine="vertex-ai"),
                lang_id=LanguageId.PYTHON,
                metadata=MetadataPromptBuilder(
                    components={
                        "prefix": MetadataCodeContent(length=10, length_tokens=2),
                        "suffix": MetadataCodeContent(length=10, length_tokens=2),
                    },
                ),
                tokens_consumption_metadata=TokensConsumptionMetadata(
                    input_tokens=1, output_tokens=2
                ),
            )
        )

    return output


@pytest.fixture
def mock_suggestions_output_text():
    return "def search"


@pytest.fixture
def mock_suggestions_model():
    return "claude-3-haiku-20240307"


@pytest.fixture
def mock_suggestions_engine():
    return "anthropic"


@pytest.fixture
def mock_suggestions_output(
    mock_suggestions_output_text: str,
    mock_suggestions_model: str,
    mock_suggestions_engine: str,
):
    return CodeSuggestionsOutput(
        text=mock_suggestions_output_text,
        score=0,
        model=ModelMetadata(
            name=mock_suggestions_model, engine=mock_suggestions_engine
        ),
        lang_id=LanguageId.PYTHON,
        metadata=CodeSuggestionsOutput.Metadata(),  # type: ignore[attr-defined]
    )


@pytest.fixture
def mock_completions_legacy(mock_completions_legacy_output: list[ModelEngineOutput]):
    with patch(
        "ai_gateway.code_suggestions.CodeCompletionsLegacy.execute",
        return_value=mock_completions_legacy_output,
    ) as mock:
        yield mock


@contextmanager
def _mock_execute(klass: str, mock_suggestions_output: CodeSuggestionsOutput):
    with patch(f"{klass}.execute", return_value=mock_suggestions_output) as mock:
        yield mock


@pytest.fixture
def mock_generations(mock_suggestions_output: CodeSuggestionsOutput):
    with _mock_execute(
        "ai_gateway.code_suggestions.CodeGenerations", mock_suggestions_output
    ) as mock:
        yield mock


@pytest.fixture
def mock_completions(mock_suggestions_output: CodeSuggestionsOutput):
    with _mock_execute(
        "ai_gateway.code_suggestions.CodeCompletions", mock_suggestions_output
    ) as mock:
        yield mock


@contextmanager
def _mock_async_execute(klass: str, mock_suggestions_output: CodeSuggestionsOutput):
    async def _stream(*args: Any, **kwargs: Any) -> AsyncIterator[CodeSuggestionsChunk]:
        for c in list(mock_suggestions_output.text):
            yield CodeSuggestionsChunk(text=c)

    with patch(f"{klass}.execute", side_effect=_stream) as mock:
        yield mock


@pytest.fixture
def mock_generations_stream(mock_suggestions_output: CodeSuggestionsOutput):
    with _mock_async_execute(  # pylint: disable=contextmanager-generator-missing-cleanup
        "ai_gateway.code_suggestions.CodeGenerations", mock_suggestions_output
    ) as mock:
        yield mock


@pytest.fixture
def mock_completions_stream(mock_suggestions_output: CodeSuggestionsOutput):
    with _mock_async_execute(  # pylint: disable=contextmanager-generator-missing-cleanup
        "ai_gateway.code_suggestions.CodeCompletions", mock_suggestions_output
    ) as mock:
        yield mock


@pytest.fixture
def mock_with_prompt_prepared():
    with patch(
        "ai_gateway.code_suggestions.CodeGenerations.with_prompt_prepared"
    ) as mock:
        yield mock


@pytest.fixture
def mock_litellm_acompletion():
    with patch("ai_gateway.models.litellm.acompletion") as mock_acompletion:
        mock_acompletion.return_value = AsyncMock(
            choices=[
                AsyncMock(
                    message=AsyncMock(content="Test response"),
                    text="Test text completion response",
                    logprobs=AsyncMock(token_logprobs=[999]),
                ),
            ],
            usage=AsyncMock(completion_tokens=999),
        )

        yield mock_acompletion


@pytest.fixture
def mock_litellm_acompletion_streamed():
    with patch("ai_gateway.models.litellm.acompletion") as mock_acompletion:
        streamed_response = AsyncMock()
        streamed_response.__aiter__.return_value = iter(
            [
                AsyncMock(
                    choices=[AsyncMock(delta=AsyncMock(content="Streamed content"))]
                )
            ]
        )

        mock_acompletion.return_value = streamed_response

        yield mock_acompletion


@pytest.fixture
def model_response():
    return "Hello there!"


@pytest.fixture
def model_engine():
    return "fake-engine"


@pytest.fixture
def model_name():
    return "fake-model"


@pytest.fixture
def model_error():
    return None


@pytest.fixture
def usage_metadata():
    return None


class FakeModel(FakeListChatModel):
    model_engine: str
    model_name: str
    model_error: Optional[Exception] = None
    usage_metadata: Optional[UsageMetadata] = None

    @property
    def _llm_type(self) -> str:
        return self.model_engine

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {**super()._identifying_params, **{"model": self.model_name}}

    def _generate(self, *args, **kwargs) -> ChatResult:
        result = super()._generate(*args, **kwargs)

        self._set_usage_metadata(result.generations[0].message)

        return result

    async def _astream(
        self,
        *args,
        **kwargs,
    ) -> AsyncIterator[ChatGenerationChunk]:
        usage_metadata_sent = False
        async for c in super()._astream(*args, **kwargs):
            # Send usage metadata only once
            if not usage_metadata_sent:
                self._set_usage_metadata(c.message)
                usage_metadata_sent = True

            yield c

        if self.model_error:
            raise self.model_error  # pylint: disable=raising-bad-type

    def _set_usage_metadata(self, message: BaseMessage):
        if not self.usage_metadata or not isinstance(message, AIMessage):
            return

        message.usage_metadata = self.usage_metadata
        message.response_metadata = {"model_name": self.model_name}


@pytest.fixture
def model(
    model_response: str,
    model_engine: str,
    model_name: str,
    model_error: Exception,
    usage_metadata: Optional[UsageMetadata],
):
    # our default Assistant prompt template already contains "Thought: "
    text = model_response.removeprefix("Thought: ") if model_response else ""

    return FakeModel(
        model_engine=model_engine,
        model_name=model_name,
        responses=[text],
        model_error=model_error,
        usage_metadata=usage_metadata,
    )


@pytest.fixture
def model_factory(model: Model):
    return lambda *args, **kwargs: model


@pytest.fixture
def model_params():
    return ChatLiteLLMParams(model_class_provider="litellm")


@pytest.fixture
def model_config(model_params: TypeModelParams):
    return ModelConfig(name="test_model", params=model_params)


@pytest.fixture
def prompt_template():
    return {"system": "Hi, I'm {{name}}", "user": "{{content}}"}


@pytest.fixture
def unit_primitives():
    return ["complete_code", "generate_code"]


@pytest.fixture
def prompt_params():
    return PromptParams()


@pytest.fixture
def prompt_config(
    model_config: ModelConfig,
    unit_primitives: list[GitLabUnitPrimitive],
    prompt_template: dict[str, str],
    prompt_params: PromptParams,
):
    return PromptConfig(
        name="test_prompt",
        model=model_config,
        unit_primitives=unit_primitives,
        prompt_template=prompt_template,
        params=prompt_params,
    )


@pytest.fixture
def model_metadata():
    return None


@pytest.fixture
def prompt_class():
    return Prompt


@pytest.fixture
def prompt(
    prompt_class: Type[Prompt],
    model_factory: TypeModelFactory,
    prompt_config: PromptConfig,
    model_metadata: TypeModelMetadata | None,
):
    return prompt_class(model_factory, prompt_config, model_metadata)


@pytest.fixture
def internal_event_client():
    return Mock(spec=InternalEventsClient)


@pytest.fixture
def scopes():
    return []


@pytest.fixture
def user_is_debug():
    return False


@pytest.fixture
def user(user_is_debug: bool, scopes: list[str]):
    return StarletteUser(
        CloudConnectorUser(
            authenticated=True, is_debug=user_is_debug, claims=UserClaims(scopes=scopes)
        )
    )


@pytest.fixture(scope="function")
def workflow_state():
    return WorkflowState(
        plan=Plan(steps=[]),
        status=WorkflowStatusEnum.NOT_STARTED,
        conversation_history={},
        handover=[],
        last_human_input=None,
        ui_chat_log=[
            {
                "message_type": MessageTypeEnum.AGENT,
                "content": "This is a test message",
                "timestamp": "2025-01-08T12:00:00Z",
                "status": None,
                "correlation_id": None,
                "tool_info": None,
            }
        ],
    )
