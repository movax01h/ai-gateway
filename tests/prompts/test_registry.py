# pylint: disable=too-many-lines
from pathlib import Path
from typing import Any, Sequence, Type, cast
from unittest.mock import Mock, patch

import pytest
from langchain.tools import BaseTool
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessageLikeRepresentation
from langchain_core.runnables import Runnable, RunnableBinding, RunnableSequence
from pydantic import BaseModel, HttpUrl
from pyfakefs.fake_filesystem import FakeFilesystem

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.config import ConfigModelLimits
from ai_gateway.integrations.amazon_q.chat import ChatAmazonQ
from ai_gateway.integrations.amazon_q.client import AmazonQClientFactory
from ai_gateway.model_metadata import ModelMetadata, create_model_metadata
from ai_gateway.models.litellm import KindLiteLlmModel
from ai_gateway.prompts import LocalPromptRegistry, Prompt
from ai_gateway.prompts.config import ModelClassProvider
from ai_gateway.prompts.config.base import PromptConfig
from ai_gateway.prompts.typing import Model, TypeModelFactory, TypePromptTemplateFactory


class MockPromptTemplateClass(Runnable):
    def __init__(self, config: PromptConfig):
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
    params:
        model: claude-3-5-sonnet-20241022
  - name: Haiku
    gitlab_identifier: haiku
    params:
        model: claude-3-haiku-20240307
  - name: Codestral
    gitlab_identifier: codestral
    params:
        model: codestral
  - name: Custom
    gitlab_identifier: custom
    family:
        - custom
    params:
        model: custom
  - name: General
    gitlab_identifier: general
    family:
        - claude_3
    params:
        model: general
  - name: Amazon Q
    gitlab_identifier: amazon_q
    family:
        - amazon_q
    params:
        model: amazon_q
  - name: Multi family
    gitlab_identifier: multi_family
    family:
        - non_existing
        - custom
    params:
        model: custom
  - name: Claude Sonnet 4.5
    gitlab_identifier: claude_sonnet_4_5
    family:
        - claude_4_5
        - claude_3
    params:
        model: claude-sonnet-4-5-20250929
  - name: Claude Sonnet 4.5 Vertex
    gitlab_identifier: claude_sonnet_4_5_vertex
    family:
        - claude_vertex_4_5
        - vertex
    params:
        model: claude-sonnet-4-5@20250929
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
    default_model: "test"
    selectable_models:
      - "test"
  - feature_setting: "duo_chat"
    unit_primitives:
      - "duo_chat"
    default_model: "haiku"
    selectable_models:
      - "haiku"
  - feature_setting: "empty_prompt"
    unit_primitives:
      - "duo_chat"
    default_model: "test"
    selectable_models:
      - "test"
  - feature_setting: "no_up"
    unit_primitives:
      - "duo_chat"
    default_model: "test"
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
    model_class_provider: litellm
    top_p: 0.1
    top_k: 50
    max_tokens: 256
    max_retries: 10
    custom_llm_provider: vllm
unit_primitives:
  - explain_code
prompt_template:
  system: Template1
""",
    )
    fs.create_file(
        prompts_definitions_dir / "test" / "base" / "1.0.0.yml",
        contents="""
---
name: Test prompt 1.0.0
model:
  params:
    model_class_provider: litellm
    top_p: 0.1
    top_k: 50
    max_tokens: 256
    max_retries: 10
    custom_llm_provider: vllm
unit_primitives:
  - explain_code
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
    model_class_provider: litellm
    top_p: 0.1
    top_k: 50
    max_tokens: 256
    max_retries: 10
    custom_llm_provider: vllm
unit_primitives:
  - explain_code
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
    model_class_provider: litellm
    top_p: 0.1
    top_k: 50
    max_tokens: 256
    max_retries: 10
    custom_llm_provider: vllm
unit_primitives:
  - explain_code
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
    model_class_provider: anthropic
    temperature: 0.1
    top_p: 0.8
    top_k: 40
    max_tokens: 256
    max_retries: 6
    default_headers:
      header1: "Header1 value"
      header2: "Header2 value"
unit_primitives:
  - duo_chat
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
model:
  params:
    model_class_provider: amazon_q
unit_primitives:
  - amazon_q_integration
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
    model_class_provider: litellm
    temperature: 0.1
    top_p: 0.8
    top_k: 40
    max_tokens: 256
    max_retries: 6
unit_primitives:
  - duo_chat
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
    model_class_provider: litellm
    temperature: 0.1
    top_p: 0.8
    top_k: 40
    max_tokens: 256
    max_retries: 6
unit_primitives:
  - duo_chat
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
    model_class_provider: litellm
    temperature: 0.2
    top_p: 0.9
    top_k: 50
    max_tokens: 512
    max_retries: 8
unit_primitives:
  - duo_chat
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
    model_class_provider: litellm
    temperature: 0.2
    top_p: 0.9
    top_k: 50
    max_tokens: 512
    max_retries: 8
unit_primitives:
  - duo_chat
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

    with patch(
        "ai_gateway.prompts.registry.LEGACY_MODEL_MAPPING", {"test": {"0.0.1": "haiku"}}
    ):
        yield


# editorconfig-checker-enable


@pytest.fixture(name="model_metadata")
def model_metadata_fixture():
    return ModelMetadata(provider="gitlab", name="test")


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
    }


@pytest.fixture(name="custom_models_enabled")
def custom_models_enabled_fixture():
    return True


@pytest.fixture(name="disable_streaming")
def disable_streaming_fixture():
    return True


@pytest.fixture(name="registry")
def registry_fixture(
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


class TestLocalPromptRegistry:
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

        assert (
            str(exc_info.value) == "No prompt version found matching the query: 2.0.0"
        )

    @pytest.mark.usefixtures("mock_fs")
    def test_load_prompt_without_unit_primitive(
        self,
        fs: FakeFilesystem,
        model_factories,
        internal_event_client: Mock,
        model_limits: ConfigModelLimits,
    ):
        yaml_content = """
---
name: TestPrompt No UP
model:
    params:
        model: test_model
        model_class_provider: litellm
prompt_template:
    system: test
"""

        prompts_definitions_dir = (
            Path(__file__).parent.parent.parent
            / "ai_gateway"
            / "prompts"
            / "definitions"
        )
        fs.create_file(
            prompts_definitions_dir / "no_up" / "base" / "1.0.0.yml",
            contents=yaml_content,
        )

        registry = LocalPromptRegistry.from_local_yaml(
            prompt_template_factories={},
            model_factories=model_factories,
            custom_models_enabled=True,
            internal_event_client=internal_event_client,
            model_limits=model_limits,
        )

        prompt = registry.get("no_up", "1.0.0")
        assert prompt.unit_primitives == []

        # Test with model metadata
        fs.create_file(
            prompts_definitions_dir / "no_up" / "codestral" / "1.0.0.yml",
            contents=yaml_content,
        )

        prompt = registry.get(
            "no_up",
            "1.0.0",
            ModelMetadata(
                name="codestral",
                endpoint=HttpUrl("http://localhost:4000/"),
                provider="custom_openai",
            ),
        )
        assert prompt.unit_primitives == []

    @pytest.mark.usefixtures("mock_fs")
    def test_get_on_behalf_no_unit_primitive(
        self,
        user: StarletteUser,
        prompt: Prompt,
        internal_event_client: Mock,
        model_limits: ConfigModelLimits,
    ):
        test_registry = LocalPromptRegistry.from_local_yaml(
            prompt_template_factories={},
            model_factories={},
            internal_event_client=internal_event_client,
            model_limits=model_limits,
        )
        prompt.unit_primitives = []

        with patch.object(test_registry, "get", return_value=prompt):
            result_prompt = test_registry.get_on_behalf(user, prompt_id="test")

            assert result_prompt == prompt

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
                model_metadata=create_model_metadata(model_metadata),
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
                [("system", "Template1")],
                "claude-3-5-sonnet-20241022",
                {},
                {
                    "top_p": 0.1,
                    "top_k": 50,
                    "max_tokens": 256,
                    "max_retries": 10,
                    "custom_llm_provider": "vllm",
                },
                ChatLiteLLM,
            ),
            (
                "test",
                "1.0.2-dev",
                None,
                True,
                "Test prompt 1.0.2-dev",
                [("system", "Template1")],
                "claude-3-5-sonnet-20241022",
                {},
                {
                    "top_p": 0.1,
                    "top_k": 50,
                    "max_tokens": 256,
                    "max_retries": 10,
                    "custom_llm_provider": "vllm",
                },
                ChatLiteLLM,
            ),
            (
                "test",
                "=1.0.0",
                None,
                True,
                "Test prompt 1.0.0",
                [("system", "Template1")],
                "claude-3-5-sonnet-20241022",
                {},
                {
                    "top_p": 0.1,
                    "top_k": 50,
                    "max_tokens": 256,
                    "max_retries": 10,
                    "custom_llm_provider": "vllm",
                },
                ChatLiteLLM,
            ),
            (
                "test",
                "0.0.1",
                None,
                True,
                "Test prompt 0.0.1",
                [("system", "Template1")],
                "claude-3-haiku-20240307",
                {},
                {
                    "top_p": 0.1,
                    "top_k": 50,
                    "max_tokens": 256,
                    "max_retries": 10,
                    "custom_llm_provider": "vllm",
                },
                ChatLiteLLM,
            ),
            (
                "chat/react",
                "^1.0.0",
                None,
                False,
                "Chat react prompt",
                [("system", "Template1"), ("user", "Template2")],
                "claude-3-haiku-20240307",
                {"stop": ["Foo", "Bar"], "timeout": 60},
                {
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_tokens": 256,
                    "max_retries": 6,
                    "default_headers": {
                        "header1": "Header1 value",
                        "header2": "Header2 value",
                    },
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
                [("system", "Template1"), ("user", "Template2")],
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
                [("system", "Template1"), ("user", "Template2")],
                "custom",
                {
                    "stop": ["Foo", "Bar"],
                    "timeout": 60,
                    "model": "claude-3-haiku-20240307",
                    "custom_llm_provider": "anthropic",
                    "api_key": "token",
                    "api_base": "http://localhost:4000",
                    "vertex_location": "us-east1",
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
                [("system", "Template1"), ("user", "Template2")],
                "custom",
                {
                    "stop": ["Foo", "Bar"],
                    "timeout": 60,
                    "model": "mistralai/Mistral-7B-Instruct-v0.3",
                    "custom_llm_provider": "custom_openai",
                    "api_key": "token",
                    "api_base": "http://localhost:4000",
                    "vertex_location": "us-east1",
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
                [("system", "Template1"), ("user", "Template2")],
                "custom",
                {
                    "stop": ["Foo", "Bar"],
                    "timeout": 60,
                    "model": "mistralai/Mistral-7B-Instruct-v0.3",
                    "custom_llm_provider": "custom_openai",
                    "api_key": "token",
                    "api_base": "http://localhost:4000",
                    "vertex_location": "us-east1",
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
                [("system", "Template1"), ("user", "Template2")],
                "general",  # The model_metadata.name overrides the prompt file's model name
                {
                    "stop": ["Foo", "Bar"],
                    "timeout": 60,
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
                [("system", "Claude 4.5 Template"), ("user", "Template2")],
                "claude-sonnet-4-5-20250929",
                {
                    "stop": ["Foo", "Bar"],
                    "timeout": 60,
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
                [("system", "Claude Vertex 4.5 Template"), ("user", "Template2")],
                "claude-sonnet-4-5@20250929",
                {
                    "stop": ["Foo", "Bar"],
                    "timeout": 60,
                    "vertex_location": "global",
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
            model_metadata=create_model_metadata(model_metadata),
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
    ):
        """Test that tool_choice is adjusted correctly based on model identifier."""
        model_metadata = None
        if model_identifier:
            model_metadata = ModelMetadata(
                provider="custom",
                name="test_model",
                identifier=model_identifier,
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
    def test_get_with_bedrock_model_adjusts_tool_choice(
        self,
        registry: LocalPromptRegistry,
        tools: list[BaseTool],
    ):
        """Test that tool_choice is automatically adjusted when getting a prompt with a Bedrock model."""
        bedrock_metadata = ModelMetadata(
            provider="custom",
            name="bedrock_model",
            identifier="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
        )

        with patch("ai_gateway.prompts.registry.Prompt") as prompt_class:
            registry.get(
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

    @pytest.mark.usefixtures("mock_fs")
    def test_get_with_azure_model_adjusts_tool_choice(
        self,
        registry: LocalPromptRegistry,
        tools: list[BaseTool],
    ):
        """Test that tool_choice is automatically adjusted when getting a prompt with an Azure model."""
        azure_metadata = ModelMetadata(
            provider="custom",
            name="azure_model",
            identifier="azure/gpt-4",
        )

        with patch("ai_gateway.prompts.registry.Prompt") as prompt_class:
            registry.get(
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
