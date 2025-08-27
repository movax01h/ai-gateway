# pylint: disable=too-many-lines
from pathlib import Path
from typing import Sequence, Type, cast
from unittest.mock import Mock, patch

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessageLikeRepresentation
from langchain_core.runnables import RunnableBinding, RunnableSequence
from pydantic import BaseModel, HttpUrl
from pyfakefs.fake_filesystem import FakeFilesystem

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.config import ConfigModelLimits
from ai_gateway.integrations.amazon_q.chat import ChatAmazonQ
from ai_gateway.integrations.amazon_q.client import AmazonQClientFactory
from ai_gateway.model_metadata import (
    AmazonQModelMetadata,
    ModelMetadata,
    TypeModelMetadata,
)
from ai_gateway.models.litellm import KindLiteLlmModel
from ai_gateway.prompts import LocalPromptRegistry, Prompt
from ai_gateway.prompts.config import ModelClassProvider
from ai_gateway.prompts.typing import Model, TypeModelFactory


class MockPromptClass(Prompt):
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
    prompts_definitions_dir = (
        Path(__file__).parent.parent.parent / "ai_gateway" / "prompts" / "definitions"
    )
    fs.create_file(
        prompts_definitions_dir / "test" / "base" / "1.0.0.yml",
        contents="""
---
name: Test prompt 1.0.0
model:
  name: claude-3-5-sonnet-20241022
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
  config_file: conversation_quick
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
  config_file: conversation_quick
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
  name: claude-3-haiku-20240307
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
  name: amazon_q
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
  name: custom
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
  name: general
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
    model_configs_dir = (
        Path(__file__).parent.parent.parent / "ai_gateway" / "prompts" / "model_configs"
    )
    fs.create_file(
        model_configs_dir / "conversation_quick.yml",
        contents="""
---
name: claude-3-5-sonnet-20241022
params:
  temperature: 0.9
  max_tokens: 200
  model_class_provider: test
""",
    )


# editorconfig-checker-enable


@pytest.fixture(name="model_factories")
def model_factories_fixture():
    return {
        # type: ignore[call-arg]
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
        class_overrides={"chat/react": MockPromptClass},
        model_factories=model_factories,
        internal_event_client=internal_event_client,
        model_limits=model_limits,
        custom_models_enabled=custom_models_enabled,
        disable_streaming=disable_streaming,
    )


class TestLocalPromptRegistry:
    @pytest.mark.usefixtures("mock_fs")
    @pytest.mark.parametrize(
        ("override_key", "override_class"),
        [
            ("chat/react", MockPromptClass),
            ("chat/react", "tests.prompts.test_registry.MockPromptClass"),
        ],
    )
    def test_from_local_yaml(
        self,
        model_factories: dict[ModelClassProvider, TypeModelFactory],
        internal_event_client: Mock,
        model_limits: ConfigModelLimits,
        override_key: str,
        override_class: str | Type[Prompt],
    ):
        registry = LocalPromptRegistry.from_local_yaml(
            class_overrides={
                override_key: override_class,
            },
            model_factories=model_factories,
            internal_event_client=internal_event_client,
            model_limits=model_limits,
            custom_models_enabled=False,
            disable_streaming=False,
        )

        # Test behavior instead of checking internal state
        prompt_with_override = registry.get("chat/react", "^1.0.0")
        assert isinstance(prompt_with_override, MockPromptClass)

        prompt_without_override = registry.get("test", "^1.0.0")
        assert isinstance(prompt_without_override, Prompt)
        assert not isinstance(prompt_without_override, MockPromptClass)

    @pytest.mark.usefixtures("mock_fs")
    @pytest.mark.parametrize(
        ("override_key", "override_class", "error_class", "error_message"),
        [
            (
                "chat/react",
                "tests.prompts.test_registry",
                ValueError,
                "The specified klass must be a subclass of Prompt",
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
            class_overrides={
                override_key: override_class,
            },
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
    name: claude-3.5
    params:
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
            class_overrides={},
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
            class_overrides={},
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
            (None, None),
            (
                ModelMetadata(
                    name="custom",
                    endpoint=HttpUrl("http://localhost:4000/"),
                    api_key="token",
                    provider="custom_openai",
                    identifier="custom_model_id",
                ),
                "custom_model_id",
            ),
            (
                AmazonQModelMetadata(
                    name="amazon_q",
                    provider="amazon_q",
                    role_arn="role-arn",
                ),
                None,
            ),
        ],
    )
    def test_logging_with_model_identifier(
        self,
        registry: LocalPromptRegistry,
        model_metadata: TypeModelMetadata,
        expected_identifier: str,
    ):
        with patch("ai_gateway.prompts.registry.log") as mock_log:

            registry.get(
                "chat/react",
                "^1.0.0",
                model_metadata=model_metadata,
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
    @pytest.mark.parametrize(
        ("tool_choice", "prompt_class"),
        [
            ("auto", Mock(spec=Prompt)),
            ("any", Mock(spec=Prompt)),
            (None, Mock(spec=Prompt)),
        ],
    )
    def test_get_with_tool_choice(
        self,
        prompt_class: Mock,
        registry: LocalPromptRegistry,
        tool_choice: str | None,
    ):
        """Test that tool_choice parameter is correctly passed from get method to Prompt constructor."""

        # We have custom BaseTool in ai_gateway.chat.tools.
        # To avoid potential collisions, we import BaseTool from LangChain locally.
        from langchain_core.tools.base import (  # pylint: disable=import-outside-toplevel
            BaseTool,
        )

        tools: list[BaseTool] = [Mock(spec=BaseTool)]

        # Mock the _load_prompt_definition to return a mock prompt
        with patch.object(
            registry,
            "_load_prompt_definition",
            return_value=Mock(klass=prompt_class, versions={"1.0.0": Mock()}),
        ):
            with patch.object(
                registry,
                "_get_prompt_config",
                return_value=Mock(
                    model=Mock(
                        params=Mock(model_class_provider=ModelClassProvider.LITE_LLM)
                    )
                ),
            ):
                _ = registry.get(
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
            "expected_class",
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
                Prompt,
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
                Prompt,
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
                Prompt,
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
                "chat/react",
                "^1.0.0",
                None,
                False,
                "Chat react prompt",
                MockPromptClass,
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
                AmazonQModelMetadata(
                    name="amazon_q",
                    provider="amazon_q",
                    role_arn="role-arn",
                ),
                False,
                "Amazon Q React prompt",
                Prompt,
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
                ModelMetadata(
                    name="custom",
                    endpoint=HttpUrl("http://localhost:4000/"),
                    api_key="token",
                    provider="custom_openai",
                    identifier="anthropic/claude-3-haiku-20240307",
                ),
                True,
                "Chat react custom prompt",
                MockPromptClass,
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
                ModelMetadata(
                    name="custom",
                    endpoint=HttpUrl("http://localhost:4000/"),
                    api_key="token",
                    provider="custom_openai",
                    identifier="custom_openai/mistralai/Mistral-7B-Instruct-v0.3",
                ),
                False,
                "Chat react custom prompt",
                MockPromptClass,
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
                ModelMetadata(
                    name=KindLiteLlmModel.GENERAL,
                    provider="litellm",
                ),
                False,
                "Chat react claude_3 prompt",  # Should map to claude_3 variant
                MockPromptClass,
                [("system", "Template1"), ("user", "Template2")],
                "general",  # The model_metadata.name overrides the prompt file's model name
                {
                    "stop": ["Foo", "Bar"],
                    "timeout": 60,
                    "custom_llm_provider": "litellm",
                    "model": "general",
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
        ],
    )
    def test_get(
        self,
        registry: LocalPromptRegistry,
        prompt_id: str,
        prompt_version: str,
        model_metadata: ModelMetadata | None,
        disable_streaming: bool,
        expected_name: str,
        expected_class: Type[Prompt],
        expected_messages: Sequence[MessageLikeRepresentation],
        expected_model: str,
        expected_kwargs: dict,
        expected_model_params: dict,
        expected_model_class: Type[Model],
    ):
        prompt = registry.get(
            prompt_id,
            prompt_version=prompt_version,
            model_metadata=model_metadata,
        )

        chain = cast(RunnableSequence, prompt.bound)
        actual_messages = cast(ChatPromptTemplate, chain.first).messages
        binding = cast(RunnableBinding, chain.last)
        actual_model = cast(BaseModel, binding.bound)

        assert prompt.name == expected_name
        assert isinstance(prompt, expected_class)
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
            == "Failed to load prompt definition for 'empty_prompt/base': No version YAML files found for prompt id: "
            "empty_prompt/base"
        )
