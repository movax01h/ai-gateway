# pylint: disable=too-many-lines
from pathlib import Path
from typing import Any, Sequence, Type, cast
from unittest.mock import Mock, patch

import pytest
from langchain.tools import BaseTool
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import MessageLikeRepresentation
from langchain_core.runnables import (
    Runnable,
    RunnableBinding,
    RunnableLambda,
    RunnableSequence,
)
from pydantic import BaseModel, HttpUrl
from pyfakefs.fake_filesystem import FakeFilesystem

from ai_gateway.config import ConfigModelLimits
from ai_gateway.integrations.amazon_q.chat import ChatAmazonQ
from ai_gateway.integrations.amazon_q.client import AmazonQClientFactory
from ai_gateway.model_metadata import ModelMetadata, create_model_metadata
from ai_gateway.model_selection import LLMDefinition
from ai_gateway.models.litellm import KindLiteLlmModel
from ai_gateway.models.v2.completion_litellm import CompletionLiteLLM
from ai_gateway.models.v2.embedding_litellm import EmbeddingLiteLLM
from ai_gateway.prompts import LocalPromptRegistry, Prompt
from ai_gateway.prompts.base import TemplateNotFoundError
from ai_gateway.prompts.config import ModelClassProvider
from ai_gateway.prompts.config.base import PromptConfig
from ai_gateway.prompts.typing import Model, TypeModelFactory, TypePromptTemplateFactory
from ai_gateway.vendor.langchain_litellm.litellm import ChatLiteLLM

_SECURITY_BLOCK = "<test security block>test</test security block>"


@pytest.fixture(autouse=True)
def mock_render_security_block():
    with patch(
        "lib.prompts.utilities.render_security_block", return_value=_SECURITY_BLOCK
    ):
        yield


class MockPromptTemplateClass(Runnable):
    def __init__(self, model_provider: ModelClassProvider, config: PromptConfig):
        pass

    def invoke(self, *_args, **_kwargs):
        pass


# Clear the cache before and after each test
@pytest.fixture(autouse=True)
def clear_prompt_cache():
    """Clear cache before and after each test to ensure test isolation."""
    cache_clear = getattr(
        LocalPromptRegistry._load_prompt_definition, "cache_clear", None
    )

    if cache_clear is not None:
        cache_clear()

    yield

    if cache_clear is not None:
        cache_clear()


# editorconfig-checker-disable
@pytest.fixture(name="mock_fs")
def mock_fs_fixture(fs: FakeFilesystem):
    ai_gateway_dir = Path(__file__).parent.parent.parent / "ai_gateway"
    model_selection_dir = ai_gateway_dir / "model_selection"
    prompts_definitions_dir = ai_gateway_dir / "prompts" / "definitions"

    fs.create_file(
        model_selection_dir / "models.yml",
        contents="""---
models:
  - name: Test
    gitlab_identifier: test
    max_context_tokens: 200000
    model_class_provider: litellm
    params:
        model: claude-3-5-sonnet-20241022
  - name: Haiku
    gitlab_identifier: haiku
    max_context_tokens: 200000
    model_class_provider: anthropic
    params:
        model: claude-3-haiku-20240307
  - name: Codestral
    gitlab_identifier: codestral
    max_context_tokens: 200000
    model_class_provider: litellm
    params:
        model: codestral
  - name: Completion Test
    gitlab_identifier: completion_test
    max_context_tokens: 200000
    model_class_provider: litellm_completion
    family:
        - completion_fim
    params:
        model: completion_test
        completion_type: fim
        fim_format: "<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"
  - name: Custom
    gitlab_identifier: custom
    max_context_tokens: 200000
    model_class_provider: litellm
    family:
        - custom
    params:
        model: custom
  - name: General
    gitlab_identifier: general
    max_context_tokens: 200000
    model_class_provider: litellm
    family:
        - claude_3
    params:
        model: general
  - name: Amazon Q
    gitlab_identifier: amazon_q
    max_context_tokens: 200000
    model_class_provider: amazon_q
    family:
        - amazon_q
    params:
        model: amazon_q
  - name: Multi family
    gitlab_identifier: multi_family
    max_context_tokens: 200000
    model_class_provider: litellm
    family:
        - non_existing
        - custom
    params:
        model: custom
  - name: Claude Sonnet 4.5
    gitlab_identifier: claude_sonnet_4_5
    max_context_tokens: 200000
    model_class_provider: litellm
    family:
        - claude_4_5
        - claude_3
    params:
        model: claude-sonnet-4-5-20250929
  - name: Claude Sonnet 4.5 Vertex
    gitlab_identifier: claude_sonnet_4_5_vertex
    max_context_tokens: 200000
    model_class_provider: litellm
    family:
        - claude_vertex_4_5
        - vertex
    params:
        model: claude-sonnet-4-5@20250929
        custom_llm_provider: vertex_ai
  - name: Embedding Test
    gitlab_identifier: embedding_test
    max_context_tokens: 200000
    model_class_provider: litellm_embedding
    family:
        - embedding
    params:
        model: embedding_test
        custom_llm_provider: vertex_ai
""",
    )
    fs.create_file(
        model_selection_dir / "unit_primitives.yml",
        contents="""---
configurable_unit_primitives:
  - feature_setting: "test"
    unit_primitives:
      - "duo_chat"
    default_models:
      - "test"
    selectable_models:
      - "test"
  - feature_setting: "duo_chat"
    unit_primitives:
      - "duo_chat"
    default_models:
      - "haiku"
    selectable_models:
      - "haiku"
  - feature_setting: "empty_prompt"
    unit_primitives:
      - "duo_chat"
    default_models:
      - "test"
    selectable_models:
      - "test"
  - feature_setting: "no_up"
    unit_primitives:
      - "duo_chat"
    default_models:
      - "test"
    selectable_models:
      - "test"
  - feature_setting: "duo_agent_platform"
    unit_primitives:
      - "duo_agent_platform"
    default_models:
      - "test"
    selectable_models:
      - "test"
""",
    )
    fs.create_file(
        prompts_definitions_dir / "test" / "base" / "0.0.1.yml",
        contents="""
---
name: Test prompt 0.0.1
model:
  params:
    top_p: 0.1
    top_k: 50
    max_tokens: 256
    max_retries: 10
unit_primitive: explain_code
prompt_template:
  system: Template1
""",
    )
    fs.create_file(
        prompts_definitions_dir
        / "code_suggestions"
        / "completions"
        / "completion_fim"
        / "1.0.0.yml",
        contents="""
---
name: Completion prompt
model:
  params:
    temperature: 0.32
    max_tokens: 64
unit_primitive: complete_code
prompt_template:
  user: "{{prefix}}"
params:
  timeout: 60
  stop:
    - "[PREFIX]"
    - "[MIDDLE]"
    - "[SUFFIX]"
    - "</s>[SUFFIX]"
""",
    )
    fs.create_file(
        prompts_definitions_dir / "test" / "base" / "1.0.0.yml",
        contents="""
---
name: Test prompt 1.0.0
model:
  params:
    top_p: 0.1
    top_k: 50
    max_tokens: 256
    max_retries: 10
unit_primitive: explain_code
prompt_template:
  system: Template1
""",
    )
    fs.create_file(
        prompts_definitions_dir / "test" / "base" / "1.0.1.yml",
        contents="""
---
name: Test prompt 1.0.1
model:
  params:
    top_p: 0.1
    top_k: 50
    max_tokens: 256
    max_retries: 10
unit_primitive: explain_code
prompt_template:
  system: Template1
""",
    )
    fs.create_file(
        prompts_definitions_dir / "test" / "base" / "1.0.2-dev.yml",
        contents="""
---
name: Test prompt 1.0.2-dev
model:
  params:
    top_p: 0.1
    top_k: 50
    max_tokens: 256
    max_retries: 10
unit_primitive: explain_code
prompt_template:
  system: Template1
""",
    )
    fs.create_file(
        prompts_definitions_dir / "chat" / "react" / "base" / "1.0.0.yml",
        contents="""
---
name: Chat react prompt
model:
  params:
    temperature: 0.1
    top_p: 0.8
    top_k: 40
    max_tokens: 256
    max_retries: 6
unit_primitive: duo_chat
prompt_template:
  system: Template1
  user: Template2
params:
  timeout: 60
  stop:
    - Foo
    - Bar
""",
    )
    fs.create_file(
        prompts_definitions_dir / "chat" / "react" / "amazon_q" / "1.0.0.yml",
        contents="""
---
name: Amazon Q React prompt
unit_primitive: amazon_q_integration
prompt_template:
  system: Template1
  user: Template2
params:
  timeout: 60
  stop:
    - Foo
    - Bar
""",
    )
    fs.create_file(
        prompts_definitions_dir / "chat" / "react" / "custom" / "1.0.0.yml",
        contents="""
---
name: Chat react custom prompt
model:
  params:
    temperature: 0.1
    top_p: 0.8
    top_k: 40
    max_tokens: 256
    max_retries: 6
unit_primitive: duo_chat
prompt_template:
  system: Template1
  user: Template2
params:
  vertex_location: us-east1
  timeout: 60
  stop:
    - Foo
    - Bar
""",
    )
    fs.create_file(
        prompts_definitions_dir / "chat" / "react" / "claude_3" / "1.0.0.yml",
        contents="""
---
name: Chat react claude_3 prompt
model:
  params:
    temperature: 0.1
    top_p: 0.8
    top_k: 40
    max_tokens: 256
    max_retries: 6
unit_primitive: duo_chat
prompt_template:
  system: Template1
  user: Template2
params:
  timeout: 60
  stop:
    - Foo
    - Bar
""",
    )
    fs.create_file(
        prompts_definitions_dir / "chat" / "react" / "claude_4_5" / "1.0.0.yml",
        contents="""
---
name: Chat react claude_4_5 prompt
model:
  params:
    temperature: 0.2
    top_p: 0.9
    top_k: 50
    max_tokens: 512
    max_retries: 8
unit_primitive: duo_chat
prompt_template:
  system: Claude 4.5 Template
  user: Template2
params:
  timeout: 60
  stop:
    - Foo
    - Bar
""",
    )
    fs.create_file(
        prompts_definitions_dir / "chat" / "react" / "claude_vertex_4_5" / "1.0.0.yml",
        contents="""
---
name: Chat react claude_vertex_4_5 prompt
model:
  params:
    temperature: 0.2
    top_p: 0.9
    top_k: 50
    max_tokens: 512
    max_retries: 8
unit_primitive: duo_chat
prompt_template:
  system: Claude Vertex 4.5 Template
  user: Template2
params:
  vertex_location: global
  timeout: 60
  stop:
    - Foo
    - Bar
""",
    )
    fs.create_file(
        prompts_definitions_dir / "embeddings_code" / "base" / "1.0.0.yml",
        contents="""
---
name: Code Embeddings
unit_primitive: generate_embeddings_codebase
prompt_template:
  user: "{{contents}}"
""",
    )

    with patch(
        "ai_gateway.prompts.registry.LEGACY_MODEL_MAPPING", {"test": {"0.0.1": "haiku"}}
    ):
        yield


# editorconfig-checker-enable


@pytest.fixture(name="model_factories")
def model_factories_fixture():
    return {
        ModelClassProvider.ANTHROPIC: lambda model, **kwargs: ChatAnthropic(
            model=model, **kwargs
        ),
        ModelClassProvider.LITE_LLM: lambda model, **kwargs: ChatLiteLLM(
            model=model, **kwargs
        ),
        ModelClassProvider.AMAZON_Q: lambda model, **kwargs: ChatAmazonQ(
            model=model,
            amazon_q_client_factory=Mock(spec=AmazonQClientFactory),
            **kwargs,
        ),
        ModelClassProvider.LITE_LLM_COMPLETION: lambda model, **kwargs: (
            CompletionLiteLLM(model=model, **kwargs)
        ),
        ModelClassProvider.LITE_LLM_EMBEDDING: lambda model, **kwargs: EmbeddingLiteLLM(
            model=model, **kwargs
        ),
    }


@pytest.fixture(name="custom_models_enabled")
def custom_models_enabled_fixture():
    return True


@pytest.fixture(name="disable_streaming")
def disable_streaming_fixture():
    return True


@pytest.fixture(name="registry")
def registry_fixture(
    mock_fs: None,  # pylint: disable=unused-argument
    model_factories: dict[ModelClassProvider, TypeModelFactory],
    internal_event_client: Mock,
    model_limits: ConfigModelLimits,
    custom_models_enabled: bool,
    disable_streaming: bool,
):
    # Use from_local_yaml for lazy loading version
    return LocalPromptRegistry.from_local_yaml(
        prompt_template_factories={},
        model_factories=model_factories,
        internal_event_client=internal_event_client,
        model_limits=model_limits,
        custom_models_enabled=custom_models_enabled,
        disable_streaming=disable_streaming,
    )


class TestLocalPromptRegistry:  # pylint: disable=too-many-public-methods
    @pytest.mark.usefixtures("mock_fs")
    @pytest.mark.parametrize(
        ("override_key", "prompt_template_factory"),
        [
            ("chat/react", MockPromptTemplateClass),
            (
                "chat/react",
                "tests.prompts.test_registry.MockPromptTemplateClass",
            ),
        ],
    )
    def test_from_local_yaml(
        self,
        model_factories: dict[ModelClassProvider, TypeModelFactory],
        internal_event_client: Mock,
        model_limits: ConfigModelLimits,
        override_key: str,
        prompt_template_factory: str | TypePromptTemplateFactory,
    ):
        registry = LocalPromptRegistry.from_local_yaml(
            prompt_template_factories={override_key: prompt_template_factory},
            model_factories=model_factories,
            internal_event_client=internal_event_client,
            model_limits=model_limits,
            custom_models_enabled=False,
            disable_streaming=False,
        )

        # Test behavior instead of checking internal state
        prompt_with_override = registry.get("chat/react", "^1.0.0")
        assert isinstance(prompt_with_override, Prompt)
        assert isinstance(prompt_with_override.prompt_tpl, MockPromptTemplateClass)

        prompt_without_override = registry.get("test", "^1.0.0")
        assert isinstance(prompt_without_override, Prompt)

    @pytest.mark.usefixtures("mock_fs")
    @pytest.mark.parametrize(
        ("override_key", "override_class", "error_class", "error_message"),
        [
            (
                "chat/react",
                "tests.prompts.test_registry",
                TypeError,
                "'module' object is not callable",
            ),
            (
                "chat/react",
                "tests.prompts.test_registry.UnknownClass",
                AttributeError,
                "module 'tests.prompts.test_registry' has no attribute 'UnknownClass'",
            ),
            (
                "chat/react",
                "tests.unknown",
                AttributeError,
                "module 'tests' has no attribute 'unknown'",
            ),
        ],
    )
    def test_from_local_yaml_failure(
        self,
        model_factories: dict[ModelClassProvider, TypeModelFactory],
        internal_event_client: Mock,
        model_limits: ConfigModelLimits,
        override_key: str,
        override_class: str,
        error_class: Type[Exception],
        error_message: str,
    ):
        registry = LocalPromptRegistry.from_local_yaml(
            prompt_template_factories={override_key: override_class},
            model_factories=model_factories,
            internal_event_client=internal_event_client,
            model_limits=model_limits,
            custom_models_enabled=False,
            disable_streaming=False,
        )

        with pytest.raises(error_class, match=error_message):
            registry.get("chat/react", "^1.0.0")

    @pytest.mark.usefixtures("mock_fs")
    def test_get_prompt_config_no_compatible_versions(
        self,
        registry: LocalPromptRegistry,
    ):
        # Try to get a version 2.0.0 which doesn't exist
        with pytest.raises(ValueError) as exc_info:
            registry.get("test", "2.0.0")

        assert "No version matching '2.0.0'" in str(exc_info.value)

    @pytest.mark.usefixtures("mock_fs")
    @pytest.mark.parametrize(
        ("model_metadata", "expected_identifier"),
        [
            (None, "haiku"),
            (
                {
                    "name": "custom",
                    "endpoint": HttpUrl("http://localhost:4000/"),
                    "api_key": "token",
                    "provider": "custom_openai",
                    "identifier": "custom_model_id",
                },
                "custom_model_id",
            ),
            (
                {
                    "name": "amazon_q",
                    "provider": "amazon_q",
                    "role_arn": "role-arn",
                },
                None,
            ),
            (
                {
                    "provider": "gitlab",
                    "identifier": "test",
                },
                "test",
            ),
            (
                {
                    "provider": "gitlab",
                    "feature_setting": "duo_chat",
                },
                "haiku",
            ),
        ],
    )
    def test_logging_with_model_identifier(
        self,
        registry: LocalPromptRegistry,
        model_metadata: dict[str, Any] | None,
        expected_identifier: str,
    ):
        with patch("ai_gateway.prompts.registry.log") as mock_log:
            registry.get(
                "chat/react",
                "^1.0.0",
                model_metadata=(
                    create_model_metadata(model_metadata) if model_metadata else None
                ),
            )

            call_dict = mock_log.info.call_args[1]
            assert call_dict["model_identifier"] == expected_identifier

    @pytest.mark.usefixtures("mock_fs")
    def test_logging_with_feature_enabled_by_namespace_ids(
        self,
        registry: LocalPromptRegistry,
    ):
        """Test that feature_enabled_by_namespace_ids is correctly logged."""
        with (
            patch("ai_gateway.prompts.registry.log") as mock_log,
            patch("ai_gateway.prompts.registry.current_event_context") as mock_context,
        ):
            mock_event_context = Mock()
            mock_event_context.feature_enabled_by_namespace_ids = [123, 456]
            mock_context.get.return_value = mock_event_context

            registry.get(
                "chat/react",
                "^1.0.0",
            )

            call_dict = mock_log.info.call_args[1]
            assert call_dict["gitlab_feature_enabled_by_namespace_ids"] == [123, 456]

    @pytest.mark.usefixtures("mock_fs")
    def test_logging_with_missing_feature_enabled_by_namespace_ids(
        self,
        registry: LocalPromptRegistry,
    ):
        """Test that logging works when feature_enabled_by_namespace_ids is missing from context."""
        with (
            patch("ai_gateway.prompts.registry.log") as mock_log,
            patch("ai_gateway.prompts.registry.current_event_context") as mock_context,
        ):
            mock_event_context = Mock(spec=[])
            mock_context.get.return_value = mock_event_context

            registry.get(
                "chat/react",
                "^1.0.0",
            )

            call_dict = mock_log.info.call_args[1]
            assert call_dict["gitlab_feature_enabled_by_namespace_ids"] is None

    @pytest.mark.usefixtures("mock_fs")
    @pytest.mark.parametrize("tool_choice", ["auto", "any", None])
    def test_get_with_tool_choice(
        self,
        registry: LocalPromptRegistry,
        tools: list[BaseTool],
        tool_choice: str | None,
    ):
        """Test that tool_choice parameter is correctly passed from get method to Prompt constructor."""
        with patch("ai_gateway.prompts.registry.Prompt") as prompt_class:
            registry.get(
                prompt_id="test",
                prompt_version="^1.0.0",
                tools=tools,
                tool_choice=tool_choice,
            )

        kwargs = prompt_class.call_args.kwargs
        assert kwargs.get("tool_choice") == tool_choice
        assert kwargs.get("tools") == tools

    @pytest.mark.usefixtures("mock_fs")
    def test_file_not_found_error(
        self,
        registry: LocalPromptRegistry,
    ):
        """Test that appropriate error is raised when prompt definition is not found."""
        with pytest.raises(ValueError, match="Failed to load prompt definition"):
            registry.get("nonexistent", "1.0.0")

    @pytest.mark.usefixtures("mock_fs")
    @pytest.mark.parametrize(
        (
            "prompt_id",
            "prompt_version",
            "model_metadata",
            "disable_streaming",
            "expected_name",
            "expected_messages",
            "expected_model",
            "expected_kwargs",
            "expected_model_params",
            "expected_model_class",
        ),
        [
            (
                "test",
                "^1.0.0",
                None,
                True,
                "Test prompt 1.0.1",
                [
                    ("system", _SECURITY_BLOCK + "Template1"),
                    MessagesPlaceholder("history", optional=True),
                ],
                "claude-3-5-sonnet-20241022",
                {
                    "cache_control_injection_points": [
                        {"location": "message", "index": 0}
                    ],
                },
                {
                    "top_p": 0.1,
                    "top_k": 50,
                    "max_tokens": 256,
                    "max_retries": 10,
                },
                ChatLiteLLM,
            ),
            (
                "test",
                "1.0.2-dev",
                None,
                True,
                "Test prompt 1.0.2-dev",
                [
                    ("system", _SECURITY_BLOCK + "Template1"),
                    MessagesPlaceholder("history", optional=True),
                ],
                "claude-3-5-sonnet-20241022",
                {
                    "cache_control_injection_points": [
                        {"location": "message", "index": 0}
                    ],
                },
                {
                    "top_p": 0.1,
                    "top_k": 50,
                    "max_tokens": 256,
                    "max_retries": 10,
                },
                ChatLiteLLM,
            ),
            (
                "test",
                "=1.0.0",
                None,
                True,
                "Test prompt 1.0.0",
                [
                    ("system", _SECURITY_BLOCK + "Template1"),
                    MessagesPlaceholder("history", optional=True),
                ],
                "claude-3-5-sonnet-20241022",
                {
                    "cache_control_injection_points": [
                        {"location": "message", "index": 0}
                    ],
                },
                {
                    "top_p": 0.1,
                    "top_k": 50,
                    "max_tokens": 256,
                    "max_retries": 10,
                },
                ChatLiteLLM,
            ),
            (
                "test",
                "0.0.1",
                None,
                True,
                "Test prompt 0.0.1",
                [
                    ("system", _SECURITY_BLOCK + "Template1"),
                    MessagesPlaceholder("history", optional=True),
                ],
                "claude-3-haiku-20240307",
                {},
                {
                    "top_p": 0.1,
                    "top_k": 50,
                    "max_tokens": 256,
                    "max_retries": 10,
                },
                ChatAnthropic,
            ),
            (
                "chat/react",
                "^1.0.0",
                None,
                False,
                "Chat react prompt",
                [
                    ("system", _SECURITY_BLOCK + "Template1"),
                    ("user", "Template2"),
                    MessagesPlaceholder("history", optional=True),
                ],
                "claude-3-haiku-20240307",
                {"stop": ["Foo", "Bar"], "timeout": 60},
                {
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_tokens": 256,
                    "max_retries": 6,
                },
                ChatAnthropic,
            ),
            (
                "chat/react",
                "^1.0.0",
                {
                    "name": "amazon_q",
                    "provider": "amazon_q",
                    "role_arn": "role-arn",
                },
                False,
                "Amazon Q React prompt",
                [
                    ("system", _SECURITY_BLOCK + "Template1"),
                    ("user", "Template2"),
                    MessagesPlaceholder("history", optional=True),
                ],
                "amazon_q",
                {
                    "role_arn": "role-arn",
                    "stop": ["Foo", "Bar"],
                    "timeout": 60.0,
                    "user": None,
                },
                {},
                ChatAmazonQ,
            ),
            (
                "chat/react",
                "^1.0.0",
                {
                    "name": "custom",
                    "endpoint": HttpUrl("http://localhost:4000/"),
                    "api_key": "token",
                    "provider": "custom_openai",
                    "identifier": "anthropic/claude-3-haiku-20240307",
                },
                True,
                "Chat react custom prompt",
                [
                    ("system", _SECURITY_BLOCK + "Template1"),
                    ("user", "Template2"),
                    MessagesPlaceholder("history", optional=True),
                ],
                "custom",
                {
                    "stop": ["Foo", "Bar"],
                    "timeout": 60,
                    "model": "claude-3-haiku-20240307",
                    "custom_llm_provider": "anthropic",
                    "api_key": "token",
                    "api_base": "http://localhost:4000",
                    "vertex_location": "us-east1",
                    "cache_control_injection_points": [
                        {"location": "message", "index": 0}
                    ],
                },
                {
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_tokens": 256,
                    "max_retries": 6,
                },
                ChatLiteLLM,
            ),
            (
                "chat/react",
                "^1.0.0",
                {
                    "name": "custom",
                    "endpoint": HttpUrl("http://localhost:4000/"),
                    "api_key": "token",
                    "provider": "custom_openai",
                    "identifier": "custom_openai/mistralai/Mistral-7B-Instruct-v0.3",
                },
                False,
                "Chat react custom prompt",
                [
                    ("system", _SECURITY_BLOCK + "Template1"),
                    ("user", "Template2"),
                    MessagesPlaceholder("history", optional=True),
                ],
                "custom",
                {
                    "stop": ["Foo", "Bar"],
                    "timeout": 60,
                    "model": "mistralai/Mistral-7B-Instruct-v0.3",
                    "custom_llm_provider": "custom_openai",
                    "api_key": "token",
                    "api_base": "http://localhost:4000",
                    "vertex_location": "us-east1",
                    "cache_control_injection_points": [
                        {"location": "message", "index": 0}
                    ],
                },
                {
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_tokens": 256,
                    "max_retries": 6,
                },
                ChatLiteLLM,
            ),
            (
                "chat/react",
                "^1.0.0",
                {
                    "name": "multi_family",
                    "endpoint": HttpUrl("http://localhost:4000/"),
                    "api_key": "token",
                    "provider": "custom_openai",
                    "identifier": "custom_openai/mistralai/Mistral-7B-Instruct-v0.3",
                },
                False,
                "Chat react custom prompt",
                [
                    ("system", _SECURITY_BLOCK + "Template1"),
                    ("user", "Template2"),
                    MessagesPlaceholder("history", optional=True),
                ],
                "custom",
                {
                    "stop": ["Foo", "Bar"],
                    "timeout": 60,
                    "model": "mistralai/Mistral-7B-Instruct-v0.3",
                    "custom_llm_provider": "custom_openai",
                    "api_key": "token",
                    "api_base": "http://localhost:4000",
                    "vertex_location": "us-east1",
                    "cache_control_injection_points": [
                        {"location": "message", "index": 0}
                    ],
                },
                {
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_tokens": 256,
                    "max_retries": 6,
                },
                ChatLiteLLM,
            ),
            (
                "chat/react",
                "^1.0.0",
                {
                    "name": KindLiteLlmModel.GENERAL,
                    "provider": "litellm",
                },
                False,
                "Chat react claude_3 prompt",  # Should map to claude_3 variant
                [
                    ("system", _SECURITY_BLOCK + "Template1"),
                    ("user", "Template2"),
                    MessagesPlaceholder("history", optional=True),
                ],
                "general",  # The model_metadata.name overrides the prompt file's model name
                {
                    "stop": ["Foo", "Bar"],
                    "timeout": 60,
                    "cache_control_injection_points": [
                        {"location": "message", "index": 0}
                    ],
                },
                {
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_tokens": 256,
                    "max_retries": 6,
                },
                ChatLiteLLM,
            ),
            (
                "chat/react",
                "^1.0.0",
                {
                    "name": "claude_sonnet_4_5",
                    "provider": "gitlab",
                },
                False,
                "Chat react claude_4_5 prompt",  # Should use claude_4_5 specific prompt
                [
                    ("system", _SECURITY_BLOCK + "Claude 4.5 Template"),
                    ("user", "Template2"),
                    MessagesPlaceholder("history", optional=True),
                ],
                "claude-sonnet-4-5-20250929",
                {
                    "stop": ["Foo", "Bar"],
                    "timeout": 60,
                    "cache_control_injection_points": [
                        {"location": "message", "index": 0}
                    ],
                },
                {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "top_k": 50,
                    "max_tokens": 512,
                    "max_retries": 8,
                },
                ChatLiteLLM,
            ),
            (
                "chat/react",
                "^1.0.0",
                {
                    "name": "claude_sonnet_4_5_vertex",
                    "provider": "gitlab",
                },
                False,
                "Chat react claude_vertex_4_5 prompt",  # Should use claude_vertex_4_5 specific prompt
                [
                    (
                        "system",
                        _SECURITY_BLOCK + "Claude Vertex 4.5 Template",
                    ),
                    ("user", "Template2"),
                    MessagesPlaceholder("history", optional=True),
                ],
                "claude-sonnet-4-5@20250929",
                {
                    "stop": ["Foo", "Bar"],
                    "timeout": 60,
                    "vertex_location": "global",
                    "cache_control_injection_points": [
                        {"location": "message", "index": 0}
                    ],
                },
                {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "top_k": 50,
                    "max_tokens": 512,
                    "max_retries": 8,
                },
                ChatLiteLLM,
            ),
        ],
    )
    def test_get(
        self,
        registry: LocalPromptRegistry,
        prompt_id: str,
        prompt_version: str,
        model_metadata: dict[str, Any] | None,
        disable_streaming: bool,
        expected_name: str,
        expected_messages: Sequence[MessageLikeRepresentation],
        expected_model: str,
        expected_kwargs: dict,
        expected_model_params: dict,
        expected_model_class: Type[Model],
    ):
        prompt = registry.get(
            prompt_id,
            prompt_version=prompt_version,
            model_metadata=(
                create_model_metadata(model_metadata) if model_metadata else None
            ),
        )

        chain = cast(RunnableSequence, prompt.bound)
        actual_messages = cast(ChatPromptTemplate, chain.first).messages
        binding = cast(RunnableBinding, chain.last)
        actual_model = cast(BaseModel, binding.bound)

        assert prompt.name == expected_name
        assert isinstance(prompt, Prompt)
        assert (
            actual_messages
            == ChatPromptTemplate.from_messages(
                expected_messages, template_format="jinja2"
            ).messages
        )
        assert prompt.model_name == expected_model
        assert prompt.model.disable_streaming == disable_streaming
        assert prompt.internal_event_client == registry.internal_event_client
        assert binding.kwargs == expected_kwargs

        actual_model_params = {
            key: value
            for key, value in dict(actual_model).items()
            if key in expected_model_params
        }
        assert actual_model_params == expected_model_params
        assert isinstance(prompt.model, expected_model_class)

    @pytest.mark.usefixtures("mock_fs")
    def test_get_prompt_directory_without_yaml_files(
        self,
        fs: FakeFilesystem,
        registry: LocalPromptRegistry,
    ):
        """Test that appropriate error is raised when prompt directory exists but has no YAML files."""
        prompts_definitions_dir = (
            Path(__file__).parent.parent.parent
            / "ai_gateway"
            / "prompts"
            / "definitions"
        )
        empty_prompt_dir = prompts_definitions_dir / "empty_prompt" / "base"
        fs.create_dir(empty_prompt_dir)

        fs.create_file(
            empty_prompt_dir / "README.md", contents="This directory has no YAML files"
        )

        with pytest.raises(ValueError) as exc_info:
            registry.get("empty_prompt", "1.0.0")

        assert (
            str(exc_info.value)
            == "Failed to load prompt definition for 'empty_prompt': No version YAML files found for prompt id: "
            "empty_prompt"
        )

    @pytest.mark.parametrize(
        ("tool_choice", "model_identifier", "expected_tool_choice"),
        [
            # Bedrock models: 'any' should be converted to 'required'
            ("any", "bedrock/anthropic.claude-v2", "required"),
            ("any", "bedrock/anthropic.claude-3-sonnet-20240229-v1:0", "required"),
            ("any", "bedrock/meta.llama3-70b-instruct-v1:0", "required"),
            # Azure models: 'any' should be converted to 'required'
            ("any", "azure/gpt-4", "required"),
            ("any", "azure/gpt-35-turbo", "required"),
            ("any", "azure/claude-3-opus", "required"),
            # Non-bedrock/azure models: 'any' should remain 'any'
            ("any", "anthropic/claude-3-opus-20240229", "any"),
            ("any", "openai/gpt-4", "any"),
            ("any", "vertex_ai/claude-3-sonnet", "any"),
            # Other tool_choice values should remain unchanged
            ("auto", "bedrock/anthropic.claude-v2", "auto"),
            ("required", "bedrock/anthropic.claude-v2", "required"),
            ("none", "bedrock/anthropic.claude-v2", "none"),
            (None, "bedrock/anthropic.claude-v2", None),
            ("auto", "azure/gpt-4", "auto"),
            # Edge cases
            ("any", None, "any"),  # No model identifier
            (
                "any",
                "vertex_ai/bedrock-model",
                "any",
            ),  # "bedrock" not in provider position
            (
                "any",
                "vertex_ai/azure-model",
                "any",
            ),  # "azure" not in provider position
        ],
    )
    def test_adjust_tool_choice_for_model(
        self,
        registry: LocalPromptRegistry,
        tool_choice: str | None,
        model_identifier: str | None,
        expected_tool_choice: str | None,
        llm_definition: LLMDefinition,
    ):
        """Test that tool_choice is adjusted correctly based on model identifier."""
        model_metadata = None
        if model_identifier:
            model_metadata = ModelMetadata(
                provider="custom",
                name="test_model",
                identifier=model_identifier,
                llm_definition=llm_definition,
            )

        result = registry._adjust_tool_choice_for_model(tool_choice, model_metadata)
        assert result == expected_tool_choice

    def test_adjust_tool_choice_for_model_without_metadata(
        self,
        registry: LocalPromptRegistry,
    ):
        """Test that tool_choice remains unchanged when model_metadata is None."""
        result = registry._adjust_tool_choice_for_model("any", None)
        assert result == "any"

    @pytest.mark.usefixtures("mock_fs")
    def test_build_prompt_with_bedrock_model_adjusts_tool_choice(
        self,
        registry: LocalPromptRegistry,
        prompt_config: PromptConfig,
        tools: list[BaseTool],
        llm_definition: LLMDefinition,
    ):
        """Test that tool_choice is automatically adjusted when getting a prompt with a Bedrock model."""
        bedrock_metadata = ModelMetadata(
            provider="custom",
            name="bedrock_model",
            identifier="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            llm_definition=llm_definition,
        )

        with patch("ai_gateway.prompts.registry.Prompt") as prompt_class:
            registry._build_prompt(
                model_class_provider=ModelClassProvider.ANTHROPIC,
                config=prompt_config,
                prompt_id="test",
                prompt_version="^1.0.0",
                model_metadata=bedrock_metadata,
                tools=tools,
                tool_choice="any",
            )

        kwargs = prompt_class.call_args.kwargs
        # tool_choice should be converted from 'any' to 'required'
        assert kwargs.get("tool_choice") == "required"
        assert kwargs.get("tools") == tools

    def test_completion_prompt_uses_passthrough_template(
        self,
        registry: LocalPromptRegistry,
    ):
        prompt = registry.get(
            "code_suggestions/completions",
            "^1.0.0",
            model_metadata=create_model_metadata(
                {"provider": "gitlab", "identifier": "completion_test"}
            ),
        )

        assert isinstance(prompt, Prompt)
        assert isinstance(prompt.prompt_tpl, RunnableLambda)

    def test_embedding_prompt_uses_passthrough_template(
        self,
        registry: LocalPromptRegistry,
    ):
        prompt = registry.get(
            "embeddings_code",
            "^1.0.0",
            model_metadata=create_model_metadata(
                {"provider": "gitlab", "identifier": "embedding_test"}
            ),
        )

        assert isinstance(prompt, Prompt)
        assert isinstance(prompt.prompt_tpl, RunnableLambda)

    @pytest.mark.usefixtures("mock_fs")
    def test_build_prompt_with_azure_model_adjusts_tool_choice(
        self,
        registry: LocalPromptRegistry,
        prompt_config: PromptConfig,
        tools: list[BaseTool],
        llm_definition: LLMDefinition,
    ):
        """Test that tool_choice is automatically adjusted when getting a prompt with an Azure model."""
        azure_metadata = ModelMetadata(
            provider="custom",
            name="azure_model",
            identifier="azure/gpt-4",
            llm_definition=llm_definition,
        )

        with patch("ai_gateway.prompts.registry.Prompt") as prompt_class:
            registry._build_prompt(
                model_class_provider=ModelClassProvider.ANTHROPIC,
                config=prompt_config,
                prompt_id="test",
                prompt_version="^1.0.0",
                model_metadata=azure_metadata,
                tools=tools,
                tool_choice="any",
            )

        kwargs = prompt_class.call_args.kwargs
        # tool_choice should be converted from 'any' to 'required'
        assert kwargs.get("tool_choice") == "required"
        assert kwargs.get("tools") == tools

    def test_build_prompt_calls_prompt_initializer_with_expected_params(
        self,
        registry: LocalPromptRegistry,
        prompt_config: PromptConfig,
        model_metadata: ModelMetadata,
        model_factories: dict[ModelClassProvider, TypeModelFactory],
    ):
        with patch("ai_gateway.prompts.registry.Prompt") as mock_prompt_class:
            registry._build_prompt(
                model_class_provider=ModelClassProvider.ANTHROPIC,
                config=prompt_config,
                model_metadata=model_metadata,
                tool_choice="auto",
                extra_kwarg="value",
            )

        # Verify Prompt was called with expected parameters
        mock_prompt_class.assert_called_once_with(
            ModelClassProvider.ANTHROPIC,
            model_factories[ModelClassProvider.ANTHROPIC],
            prompt_config,
            model_metadata,
            disable_streaming=registry.disable_streaming,
            custom_models_extra_headers=registry.custom_models_extra_headers,
            tool_choice="auto",
            internal_event_client=registry.internal_event_client,
            extra_kwarg="value",
        )

    def test_build_prompt_raises_error_for_unrecognized_model_class_provider(
        self,
        registry: LocalPromptRegistry,
        prompt_config: PromptConfig,
        model_metadata: ModelMetadata,
    ):
        """Test that _build_prompt raises ValueError for unrecognized model class provider."""
        # Create an unrecognized model class provider
        unrecognized_provider = "unrecognized_provider"

        with pytest.raises(ValueError) as exc_info:
            registry._build_prompt(
                model_class_provider=unrecognized_provider,  # type: ignore
                config=prompt_config,
                model_metadata=model_metadata,
                tool_choice=None,
            )

        assert "unrecognized model class provider" in str(exc_info.value)
        assert unrecognized_provider in str(exc_info.value)

    @pytest.mark.usefixtures("mock_fs")
    def test_duo_chat_max_tokens_override_applied_to_chat_prompts(
        self,
        model_factories: dict[ModelClassProvider, TypeModelFactory],
        internal_event_client: Mock,
        model_limits: ConfigModelLimits,
    ):
        """Test that duo_chat_max_tokens override is passed to Prompt for chat/* prompts."""
        registry = LocalPromptRegistry.from_local_yaml(
            prompt_template_factories={},
            model_factories=model_factories,
            internal_event_client=internal_event_client,
            model_limits=model_limits,
            custom_models_enabled=False,
            disable_streaming=False,
            duo_chat_max_tokens=8192,
        )

        with patch("ai_gateway.prompts.registry.Prompt") as prompt_class:
            registry.get(
                prompt_id="chat/react",
                prompt_version="^1.0.0",
            )

        kwargs = prompt_class.call_args.kwargs
        assert kwargs.get("max_tokens_override") == 8192

    @pytest.mark.usefixtures("mock_fs")
    def test_duo_chat_max_tokens_override_not_applied_to_non_chat_prompts(
        self,
        model_factories: dict[ModelClassProvider, TypeModelFactory],
        internal_event_client: Mock,
        model_limits: ConfigModelLimits,
    ):
        """Test that duo_chat_max_tokens override is NOT passed to Prompt for non-chat prompts."""
        registry = LocalPromptRegistry.from_local_yaml(
            prompt_template_factories={},
            model_factories=model_factories,
            internal_event_client=internal_event_client,
            model_limits=model_limits,
            custom_models_enabled=False,
            disable_streaming=False,
            duo_chat_max_tokens=8192,
        )

        with patch("ai_gateway.prompts.registry.Prompt") as prompt_class:
            registry.get(
                prompt_id="test",
                prompt_version="^1.0.0",
            )

        kwargs = prompt_class.call_args.kwargs
        assert kwargs.get("max_tokens_override") is None

    @pytest.mark.usefixtures("mock_fs")
    def test_duo_chat_max_tokens_none_when_not_configured(
        self,
        registry: LocalPromptRegistry,
    ):
        """Test that max_tokens_override is None when duo_chat_max_tokens is not configured."""
        with patch("ai_gateway.prompts.registry.Prompt") as prompt_class:
            registry.get(
                prompt_id="chat/react",
                prompt_version="^1.0.0",
            )

        kwargs = prompt_class.call_args.kwargs
        assert kwargs.get("max_tokens_override") is None


class TestGetRequiredVariables:

    _PROMPT_BASE_DIR = (
        Path(__file__).parent.parent.parent / "ai_gateway" / "prompts" / "definitions"
    )

    def _create_prompt_file(
        self, fs: FakeFilesystem, prompt_id: str, template: str
    ) -> None:
        """Create a minimal prompt YAML file with the given template string."""
        path = self._PROMPT_BASE_DIR / prompt_id / "base" / "1.0.0.yml"
        yaml_content = "\n".join(
            [
                "---",
                "name: Test prompt",
                "unit_primitive: duo_chat",
                "prompt_template:",
                f"  system: {repr(template)}",
                "",
            ]
        )
        fs.create_file(path, contents=yaml_content)

    def test_no_version_raises(self, registry: LocalPromptRegistry):
        """File-based prompts without a version constraint raise TemplateNotFoundError."""
        with pytest.raises(TemplateNotFoundError, match="no prompt_version provided"):
            registry.get_required_variables("any_prompt", prompt_version=None)

    @pytest.mark.usefixtures("mock_fs")
    def test_returns_variables_from_template(self, registry: LocalPromptRegistry):
        """File-based prompt with matching version returns template variables."""
        result = registry.get_required_variables("chat/react", prompt_version="^1.0.0")
        assert isinstance(result, set)

    def test_unknown_prompt_raises(self, registry: LocalPromptRegistry):
        """Non-existent prompt ID raises TemplateNotFoundError."""
        with pytest.raises(TemplateNotFoundError):
            registry.get_required_variables("nonexistent", prompt_version="^1.0.0")

    def test_simple_variables_extracted(
        self, fs: FakeFilesystem, registry: LocalPromptRegistry
    ):
        """Variables referenced in a template are returned as a set."""
        self._create_prompt_file(
            fs, "test/simple_vars", "Hello {{ name }}, goal: {{ goal }}"
        )
        result = registry.get_required_variables(
            "test/simple_vars", prompt_version="^1.0.0"
        )
        assert result == {"name", "goal"}

    def test_no_variables_returns_empty_set(
        self, fs: FakeFilesystem, registry: LocalPromptRegistry
    ):
        """A template with no Jinja2 variables returns an empty set."""
        self._create_prompt_file(fs, "test/no_vars", "Hello world")
        result = registry.get_required_variables(
            "test/no_vars", prompt_version="^1.0.0"
        )
        assert result == set()

    def test_duplicate_variables_deduplicated(
        self, fs: FakeFilesystem, registry: LocalPromptRegistry
    ):
        """Each variable name appears only once even if used multiple times."""
        self._create_prompt_file(fs, "test/dup_vars", "{{ x }} and {{ x }}")
        result = registry.get_required_variables(
            "test/dup_vars", prompt_version="^1.0.0"
        )
        assert result == {"x"}

    def test_variables_with_filters_extracted(
        self, fs: FakeFilesystem, registry: LocalPromptRegistry
    ):
        """Filter expressions do not obscure the underlying variable name."""
        self._create_prompt_file(fs, "test/filter_vars", "{{ name | upper }}")
        result = registry.get_required_variables(
            "test/filter_vars", prompt_version="^1.0.0"
        )
        assert result == {"name"}

    def test_variables_in_conditionals_extracted(
        self, fs: FakeFilesystem, registry: LocalPromptRegistry
    ):
        """Variables inside conditional blocks are included in the result."""
        self._create_prompt_file(
            fs,
            "test/cond_vars",
            "{% if flag %}{{ a }}{% else %}{{ b }}{% endif %}",
        )
        result = registry.get_required_variables(
            "test/cond_vars", prompt_version="^1.0.0"
        )
        assert result == {"flag", "a", "b"}

    def test_include_collects_variables_from_nested_template(
        self, fs: FakeFilesystem, registry: LocalPromptRegistry
    ):
        """Variables from included templates are collected transitively."""
        self._create_prompt_file(
            fs, "test/include_vars", '{{ goal }}\n{%- include "fake.jinja" %}'
        )
        fake_loader = patch(
            "ai_gateway.prompts.base.jinja_loader.get_source",
            return_value=("Nested: {{ nested_var }}", "fake.jinja", lambda: True),
        )
        with fake_loader:
            result = registry.get_required_variables(
                "test/include_vars", prompt_version="^1.0.0"
            )
        assert result == {"goal", "nested_var"}

    def test_include_unknown_template_raises(
        self, fs: FakeFilesystem, registry: LocalPromptRegistry
    ):
        """An unresolvable include propagates an exception from get_required_variables."""
        self._create_prompt_file(
            fs, "test/bad_include", '{% include "no_such_template.txt" %}'
        )
        with pytest.raises(Exception):
            registry.get_required_variables("test/bad_include", prompt_version="^1.0.0")

    def test_graph_node_falls_back_to_duo_agent_platform(
        self, registry: LocalPromptRegistry
    ):
        """Graph nodes with no matching feature setting fall back to duo_agent_platform model."""
        metadata = registry._default_model_metadata(
            "unknown_graph_step", "1.0.0", is_graph_node=True
        )
        assert (
            metadata.name == "test"
        )  # duo_agent_platform default_model in test fixture

    @pytest.mark.usefixtures("mock_fs")
    def test_non_graph_node_raises_for_unknown_feature_setting(
        self, registry: LocalPromptRegistry
    ):
        """Non-graph-node callers get a ValueError for unknown feature settings."""
        with pytest.raises(ValueError, match="Invalid feature setting: unknown_step"):
            registry._default_model_metadata(
                "unknown_step", "1.0.0", is_graph_node=False
            )
