import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Optional, Union
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import litellm
import pytest
import structlog
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from gitlab_cloud_connector import CloudConnectorUser, GitLabUnitPrimitive, UserClaims
from langchain.tools import BaseTool
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.messages import BaseMessage
from langchain_core.messages.ai import AIMessage, UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from starlette.middleware import Middleware
from starlette_context.middleware import RawContextMiddleware

from ai_gateway import structured_logging
from ai_gateway.api.middleware import (
    AccessLogMiddleware,
    MiddlewareAuthentication,
    ModelConfigMiddleware,
)
from ai_gateway.api.middleware.internal_event import InternalEventMiddleware
from ai_gateway.api.middleware.route import usage_quota
from ai_gateway.api.server import CONTAINER_APPLICATION_MODULES
from ai_gateway.code_suggestions.base import CodeSuggestionsChunk, CodeSuggestionsOutput
from ai_gateway.code_suggestions.processing.typing import LanguageId
from ai_gateway.config import Config, ConfigModelLimits
from ai_gateway.container import ContainerApplication
from ai_gateway.model_metadata import ModelMetadata, TypeModelMetadata
from ai_gateway.model_selection.model_selection_config import (
    ChatLiteLLMDefinition,
    LLMDefinition,
    ModelSelectionConfig,
    PromptParams,
)
from ai_gateway.model_selection.models import BaseModelParams, ModelClassProvider
from ai_gateway.models.base import ModelMetadata as LegacyModelMetadata
from ai_gateway.models.base_text import (
    TextGenModelBase,
    TextGenModelChunk,
    TextGenModelOutput,
)
from ai_gateway.prompts import Prompt
from ai_gateway.prompts.config.base import ModelConfig, PromptConfig
from ai_gateway.prompts.typing import Model, TypeModelFactory, TypePromptTemplateFactory
from ai_gateway.safety_attributes import SafetyAttributes
from duo_workflow_service.entities.event import WorkflowEvent
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    Plan,
    UiChatLog,
    WorkflowState,
    WorkflowStatusEnum,
)
from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.server import CONTAINER_APPLICATION_PACKAGES
from duo_workflow_service.workflows.type_definitions import AdditionalContext
from lib.billing_events.client import BillingEventsClient
from lib.context import (
    StarletteUser,
    current_model_metadata_context,
    llm_operations,
    token_usage,
)
from lib.events.contextvar import self_hosted_dap_billing_enabled
from lib.feature_flags.context import current_feature_flag_context
from lib.internal_events.client import InternalEventsClient
from lib.prompts.caching import current_prompt_cache_context
from lib.usage_quota import UsageQuotaService

pytest_plugins = ("pytest_asyncio",)


# LANGCHAIN_TRACING_V2 is how Langchain decides whether to send traces
# We do not want to send traces when running tests so we set it to false
# see https://github.com/langchain-ai/langchain/issues/16429#issuecomment-1907051834
# pylint: disable=direct-environment-variable-reference
os.environ["LANGCHAIN_TRACING_V2"] = "false"
# pylint: enable=direct-environment-variable-reference


@pytest.fixture(name="assets_dir")
def assets_dir_fixture() -> Path:
    return Path(__file__).parent / "_assets"


@pytest.fixture(name="tpl_assets_codegen_dir")
def tpl_assets_codegen_dir_fixture(assets_dir) -> Path:
    tpl_dir = assets_dir / "tpl"
    return tpl_dir / "codegen"


@pytest.fixture(name="text_gen_base_model")
def text_gen_base_model_fixture():
    model = Mock(spec=TextGenModelBase)
    type(model).input_token_limit = PropertyMock(return_value=1_000)
    return model


@pytest.fixture(name="stub_auth_provider", scope="class")
def stub_auth_provider_fixture():
    class StubKeyAuthProvider:
        def authenticate(self, *_args):
            return None

    return StubKeyAuthProvider()


@pytest.fixture(name="test_client", scope="class")
def test_client_fixture(fast_api_router, stub_auth_provider):
    middlewares = [
        Middleware(RawContextMiddleware),
        Middleware(AccessLogMiddleware, skip_endpoints=[]),
        MiddlewareAuthentication(stub_auth_provider, False, None),
        Middleware(
            InternalEventMiddleware, skip_endpoints=[], enabled=True, environment="test"
        ),
        Middleware(ModelConfigMiddleware),
    ]
    app = FastAPI(middleware=middlewares)
    app.include_router(fast_api_router)
    client = TestClient(app)

    return client


@pytest.fixture(name="model_metadata_context")
def model_metadata_context_fixture():
    current_model_metadata_context.set(None)
    yield current_model_metadata_context


@pytest.fixture(name="mock_track_internal_event")
def mock_track_internal_event_fixture():
    with patch("lib.internal_events.InternalEventsClient.track_event") as mock:
        yield mock


@pytest.fixture(name="mock_client")
def mock_client_fixture(
    test_client,
    stub_auth_provider,
    auth_user,
    mock_ai_gateway_container,  # pylint: disable=unused-argument
    model_metadata_context,  # pylint: disable=unused-argument
    mock_config,
):
    """Setup all the needed mocks to perform requests in the test environment."""
    # Set the config on the app in the nested structure: extra['extra']['config']
    test_client.app.extra["extra"] = {"config": mock_config}

    # Initialize usage quota service for tests
    test_client.app.state.usage_quota_service = UsageQuotaService(
        customersdot_url=mock_config.customer_portal_url,
        customersdot_api_user=None,
        customersdot_api_token=None,
    )

    with patch.object(stub_auth_provider, "authenticate", return_value=auth_user):
        yield test_client


@pytest.fixture(name="mock_connect_vertex")
def mock_connect_vertex_fixture():
    with patch("ai_gateway.models.base.PredictionServiceAsyncClient"):
        yield


@pytest.fixture(name="mock_connect_vertex_search")
def mock_connect_vertex_search_fixture():
    with patch(
        "ai_gateway.searches.container.discoveryengine.SearchServiceAsyncClient"
    ):
        yield


@pytest.fixture(name="config_values")
def config_values_fixture():
    return {}


@pytest.fixture(name="mock_config")
def mock_config_fixture(config_values: dict[str, Any]):
    return Config(_env_file=None, _env_prefix="AIGW_TEST", **config_values)


@pytest.fixture(name="mock_container")
def mock_container_fixture(
    mock_config: Config,
    mock_connect_vertex: Mock,  # pylint: disable=unused-argument
    mock_connect_vertex_search: Mock,  # pylint: disable=unused-argument
):
    container_application = ContainerApplication()
    container_application.config.from_dict(mock_config.model_dump())

    return container_application


@pytest.fixture(name="mock_ai_gateway_container")
def mock_ai_gateway_container_fixture(
    mock_container: ContainerApplication,
) -> ContainerApplication:

    mock_container.wire(modules=CONTAINER_APPLICATION_MODULES + [usage_quota])

    return mock_container


@pytest.fixture(name="mock_duo_workflow_service_container")
def mock_duo_workflow_service_container_fixture(
    mock_container: ContainerApplication,
) -> ContainerApplication:
    mock_container.wire(packages=CONTAINER_APPLICATION_PACKAGES)

    return mock_container


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


@pytest.fixture(name="mock_anthropic_stream")
def mock_anthropic_stream_fixture(mock_output: TextGenModelOutput):
    with _mock_async_generate(  # pylint: disable=contextmanager-generator-missing-cleanup
        "ai_gateway.models.anthropic.AnthropicModel", mock_output
    ) as mock:
        yield mock


@pytest.fixture(name="mock_anthropic_chat_stream")
def mock_anthropic_chat_stream_fixture(mock_output: TextGenModelOutput):
    with _mock_async_generate(  # pylint: disable=contextmanager-generator-missing-cleanup
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


# Legacy completions output fixtures removed as CodeCompletionsLegacy no longer exists


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
        model=LegacyModelMetadata(
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


@pytest.fixture(name="mock_litellm_acompletion")
def mock_litellm_acompletion_fixture():
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
            usage=AsyncMock(completion_tokens=999),
        )

        yield mock_acompletion


@pytest.fixture(name="mock_litellm_acompletion_streamed")
def mock_litellm_acompletion_streamed_fixture():
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


@pytest.fixture(name="mock_litellm_completion")
def mock_litellm_completion_fixture():
    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(content="test completion"),
                    text="test completion",
                    logprobs=Mock(token_logprobs=[999]),
                ),
            ],
            usage=Mock(completion_tokens=999),
        )
        yield mock_completion


@pytest.fixture(name="mock_litellm_acompletion_for_vertex")
def mock_litellm_acompletion_for_vertex_fixture():
    with patch("litellm.acompletion") as mock_acompletion:
        mock_acompletion.return_value = AsyncMock(
            choices=[
                AsyncMock(
                    message=AsyncMock(content="test completion"),
                    text="test completion",
                    logprobs=AsyncMock(token_logprobs=[999]),
                ),
            ],
            usage=AsyncMock(completion_tokens=999),
        )
        yield mock_acompletion


@pytest.fixture(name="model_response")
def model_response_fixture():
    return "Hello there!"


@pytest.fixture(name="model_engine")
def model_engine_fixture():
    return "fake-engine"


@pytest.fixture(name="model_name")
def model_name_fixture():
    return "fake-model"


@pytest.fixture(name="model_error")
def model_error_fixture():
    return None


@pytest.fixture(name="usage_metadata")
def usage_metadata_fixture():
    return None


@pytest.fixture(name="model_disable_streaming")
def model_disable_streaming_fixture():
    return False


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
        if all(isinstance(response, BaseMessage) for response in self.responses):
            return ChatResult(
                generations=[
                    ChatGeneration(message=response) for response in self.responses
                ]
            )

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
            raise self.model_error

    def _set_usage_metadata(self, message: BaseMessage):
        if not self.usage_metadata or not isinstance(message, AIMessage):
            return

        message.usage_metadata = self.usage_metadata
        message.response_metadata = {"model_name": self.model_name}


@pytest.fixture(name="model")
def model_fixture(
    model_response: str | list,
    model_engine: str,
    model_name: str,
    model_error: Exception,
    usage_metadata: Optional[UsageMetadata],
    model_disable_streaming: Union[bool, Literal["tool_calling"]],
):
    # our default Assistant prompt template already contains "Thought: "
    if isinstance(model_response, str):
        responses = [model_response.removeprefix("Thought: ")]
    else:
        responses = model_response

    return FakeModel(
        model_engine=model_engine,
        model_name=model_name,
        responses=responses,
        model_error=model_error,
        usage_metadata=usage_metadata,
        disable_streaming=model_disable_streaming,
    )


@pytest.fixture(name="model_factory")
def model_factory_fixture(model: Model):
    return lambda *args, **kwargs: model


@pytest.fixture(name="model_provider")
def model_provider_fixture():
    return ModelClassProvider.LITE_LLM


@pytest.fixture(name="model_params")
def model_params_fixture():
    return BaseModelParams(model="test_model")


@pytest.fixture(name="model_config")
def model_config_fixture(model_params: BaseModelParams):
    return ModelConfig(params=model_params)


@pytest.fixture(name="prompt_template")
def prompt_template_fixture():
    return {"system": "Hi, I'm {{name}}", "user": "{{content}}"}


@pytest.fixture(name="unit_primitive")
def unit_primitive_fixture():
    return GitLabUnitPrimitive.COMPLETE_CODE


@pytest.fixture(name="prompt_params")
def prompt_params_fixture():
    return PromptParams()


@pytest.fixture(name="prompt_name")
def prompt_name_fixture():
    return "test_prompt"


@pytest.fixture(name="prompt_config")
def prompt_config_fixture(
    prompt_name: str,
    model_config: ModelConfig,
    unit_primitive: GitLabUnitPrimitive,
    prompt_template: dict[str, str],
    prompt_params: PromptParams,
):
    return PromptConfig(
        name=prompt_name,
        model=model_config,
        unit_primitive=unit_primitive,
        prompt_template=prompt_template,
        params=prompt_params,
    )


@pytest.fixture(name="llm_definition")
def llm_definition_fixture():
    return ChatLiteLLMDefinition(
        name="Mistral",
        gitlab_identifier="mistral",
        max_context_tokens=128000,
        family=["mistral"],
        params={"model": "mistral", "temperature": 0.0, "max_tokens": 4096},
    )


@pytest.fixture(name="model_metadata")
def model_metadata_fixture(llm_definition: LLMDefinition):
    return ModelMetadata(
        provider="gitlab",
        name="mistral",
        family=["mistral"],
        llm_definition=llm_definition,
        friendly_name="Mistral",
    )


@pytest.fixture(name="prompt_template_factory")
def prompt_template_factory_fixture():
    return None


@pytest.fixture(name="internal_event_extra")
def internal_event_extra_fixture():
    return {}


@pytest.fixture(name="prompt")
def prompt_fixture(
    model_provider: ModelClassProvider,
    model_factory: TypeModelFactory,
    prompt_config: PromptConfig,
    model_metadata: TypeModelMetadata | None,
    prompt_template_factory: TypePromptTemplateFactory | None,
    internal_event_extra: dict[str, Any],
):
    return Prompt(
        model_provider,
        model_factory,
        prompt_config,
        model_metadata,
        prompt_template_factory,
        internal_event_extra=internal_event_extra,
    )


@pytest.fixture(name="internal_event_client")
def internal_event_client_fixture():
    return Mock(spec=InternalEventsClient)


@pytest.fixture(name="model_limits")
def model_limits_fixture():
    return ConfigModelLimits()


@pytest.fixture(name="scopes")
def scopes_fixture():
    return []


@pytest.fixture(name="user_is_debug")
def user_is_debug_fixture():
    return False


@pytest.fixture(name="auth_user")
def auth_user_fixture(user_is_debug: bool, scopes: list[str]):
    return CloudConnectorUser(
        authenticated=True,
        is_debug=user_is_debug,
        claims=UserClaims(scopes=scopes, gitlab_instance_uid="unique-instance-uid"),
    )


@pytest.fixture(name="user")
def user_fixture(auth_user: CloudConnectorUser) -> StarletteUser | None:
    return StarletteUser(auth_user)


@pytest.fixture(name="ui_chat_log")
def ui_chat_log_fixture() -> list[UiChatLog]:
    return [
        {
            "message_type": MessageTypeEnum.AGENT,
            "content": "This is a test message",
            "timestamp": "2025-01-08T12:00:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "message_sub_type": None,
            "additional_context": None,
            "message_id": None,
        }
    ]


@pytest.fixture(name="goal")
def goal_fixture() -> str:
    return "Make the world a better place"


@pytest.fixture(name="last_human_input")
def last_human_input_fixture() -> WorkflowEvent | None:
    return None


@pytest.fixture(name="project")
def project_fixture() -> Project:
    return {
        "id": 123,
        "name": "test-project",
        "description": "This is a test project",
        "languages": None,
        "http_url_to_repo": "https://gitlab.com/test/repo",
        "web_url": "https://gitlab.com/test/repo",
        "default_branch": "main",
        "exclusion_rules": None,
    }


@pytest.fixture(name="additional_context")
def additional_context_fixture() -> list[AdditionalContext] | None:
    return None


@pytest.fixture(name="workflow_state", scope="function")
def workflow_state_fixture(
    project: Project,
    goal: str | None,
    last_human_input: WorkflowEvent | None,
    ui_chat_log: list[UiChatLog],
    additional_context: list[AdditionalContext] | None,
):
    return WorkflowState(
        plan=Plan(steps=[]),
        status=WorkflowStatusEnum.NOT_STARTED,
        conversation_history={},
        handover=[],
        last_human_input=last_human_input,
        ui_chat_log=ui_chat_log,
        project=project,
        goal=goal,
        additional_context=additional_context,
    )


@pytest.fixture(autouse=True)
def disable_cached_logger():
    structlog.configure(cache_logger_on_first_use=False)


def reset_context_vars():
    current_feature_flag_context.set(set[str]())
    current_model_metadata_context.set(None)
    token_usage.set(None)
    llm_operations.set(None)
    current_prompt_cache_context.set(None)
    ModelSelectionConfig._instance = None
    structured_logging.ENABLE_REQUEST_LOGGING = False
    self_hosted_dap_billing_enabled.set(False)


@pytest.fixture(autouse=True)
def reset_litellm_settings():
    # This fixture will reset the litellm settings before and after each test
    original_module_level_aclient = litellm.module_level_aclient

    yield

    litellm.module_level_aclient = original_module_level_aclient


@pytest.fixture(autouse=True)
def reset_context():
    # This fixture will reset the context before and after each test
    reset_context_vars()

    yield

    reset_context_vars()


@pytest.fixture(name="vertex_project")
def vertex_project_fixture():
    return "vertex-project"


@pytest.fixture(name="tools")
def tools_fixture() -> list[BaseTool]:
    return [Mock(spec=BaseTool)]


@pytest.fixture(name="end_message")
def end_message_fixture():
    return AIMessage(
        content="Done with the execution, over to handover agent",
        tool_calls=[
            {
                "id": "1",
                "name": "handover_tool",
                "args": {"summary": "done"},
            }
        ],
    )


@pytest.fixture(name="mock_request")
def mock_request_fixture(user: StarletteUser | None):
    request = Mock(spec=Request)
    request.headers = {}
    request.user = user
    request.state = Mock(spec=[])
    return request


@pytest.fixture(name="billing_event_client")
def billing_event_client_fixture():
    return Mock(spec=BillingEventsClient)
