# pylint: disable=too-many-lines
from pathlib import Path
from textwrap import dedent
from typing import Sequence, Type, cast
from unittest.mock import Mock, patch

import pytest
import yaml
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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
from ai_gateway.prompts import LocalPromptRegistry, Prompt, PromptRegistered
from ai_gateway.prompts.config import (
    ChatAmazonQParams,
    ChatAnthropicParams,
    ChatLiteLLMParams,
    ModelClassProvider,
    ModelConfig,
    PromptConfig,
)
from ai_gateway.prompts.typing import Model, TypeModelFactory


class MockPromptClass(Prompt):
    pass


# editorconfig-checker-disable
@pytest.fixture
def mock_fs(fs: FakeFilesystem):
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


@pytest.fixture
def model_factories():
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


@pytest.fixture
def prompts_registered():
    return {
        "test/base": PromptRegistered(
            klass=Prompt,
            versions={
                "1.0.0": PromptConfig(
                    name="Test prompt 1.0.0",
                    model=ModelConfig(
                        name="claude-3-5-sonnet-20241022",
                        params=ChatLiteLLMParams(
                            model_class_provider=ModelClassProvider.LITE_LLM,
                            top_p=0.1,
                            top_k=50,
                            max_tokens=256,
                            max_retries=10,
                            custom_llm_provider="vllm",
                        ),
                    ),
                    unit_primitives=["explain_code"],
                    prompt_template={"system": "Template1"},
                ),
                "1.0.1": PromptConfig(
                    name="Test prompt 1.0.1",
                    model=ModelConfig(
                        name="claude-3-5-sonnet-20241022",
                        params=ChatLiteLLMParams(
                            model_class_provider=ModelClassProvider.LITE_LLM,
                            top_p=0.1,
                            top_k=50,
                            max_tokens=256,
                            max_retries=10,
                            custom_llm_provider="vllm",
                            temperature=0.9,
                        ),
                    ),
                    unit_primitives=["explain_code"],
                    prompt_template={"system": "Template1"},
                ),
                "1.0.2-dev": PromptConfig(
                    name="Test prompt 1.0.2-dev",
                    model=ModelConfig(
                        name="claude-3-5-sonnet-20241022",
                        params=ChatLiteLLMParams(
                            model_class_provider=ModelClassProvider.LITE_LLM,
                            top_p=0.1,
                            top_k=50,
                            max_tokens=256,
                            max_retries=10,
                            custom_llm_provider="vllm",
                            temperature=0.9,
                        ),
                    ),
                    unit_primitives=["explain_code"],
                    prompt_template={"system": "Template1"},
                ),
            },
        ),
        "chat/react/base": PromptRegistered(
            klass=MockPromptClass,
            versions={
                "1.0.0": PromptConfig(
                    name="Chat react prompt",
                    model=ModelConfig(
                        name="claude-3-haiku-20240307",
                        params=ChatAnthropicParams(
                            model_class_provider=ModelClassProvider.ANTHROPIC,
                            temperature=0.1,
                            top_p=0.8,
                            top_k=40,
                            max_tokens=256,
                            max_retries=6,
                            default_headers={
                                "header1": "Header1 value",
                                "header2": "Header2 value",
                            },
                        ),
                    ),
                    unit_primitives=["duo_chat"],
                    prompt_template={"system": "Template1", "user": "Template2"},
                    params={"timeout": 60, "stop": ["Foo", "Bar"]},
                ),
            },
        ),
        "chat/react/amazon_q": PromptRegistered(
            klass=MockPromptClass,
            versions={
                "1.0.0": PromptConfig(
                    name="Amazon Q React prompt",
                    model=ModelConfig(
                        name="amazon_q",
                        params=ChatAmazonQParams(
                            model_class_provider=ModelClassProvider.AMAZON_Q,
                        ),
                    ),
                    unit_primitives=["amazon_q_integration"],
                    prompt_template={"system": "Template1", "user": "Template2"},
                    params={
                        "timeout": 60,
                        "stop": ["Foo", "Bar"],
                    },
                ),
            },
        ),
        "chat/react/custom": PromptRegistered(
            klass=MockPromptClass,
            versions={
                "1.0.0": PromptConfig(
                    name="Chat react custom prompt",
                    model=ModelConfig(
                        name="custom",
                        params=ChatLiteLLMParams(
                            model_class_provider=ModelClassProvider.LITE_LLM,
                            temperature=0.1,
                            top_p=0.8,
                            top_k=40,
                            max_tokens=256,
                            max_retries=6,
                        ),
                    ),
                    unit_primitives=["duo_chat"],
                    prompt_template={"system": "Template1", "user": "Template2"},
                    params={
                        "timeout": 60,
                        "stop": ["Foo", "Bar"],
                        "vertex_location": "us-east1",
                    },
                ),
            },
        ),
    }


@pytest.fixture
def default_prompts():
    return {}


@pytest.fixture
def custom_models_enabled():
    return True


@pytest.fixture
def disable_streaming():
    return True


@pytest.fixture
def registry(
    prompts_registered: dict[str, PromptRegistered],
    model_factories: dict[ModelClassProvider, TypeModelFactory],
    default_prompts: dict[str, str],
    internal_event_client: Mock,
    model_limits: ConfigModelLimits,
    custom_models_enabled: bool,
    disable_streaming: bool,
):
    return LocalPromptRegistry(
        model_factories=model_factories,
        prompts_registered=prompts_registered,
        default_prompts=default_prompts,
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
        prompts_registered: dict[str, PromptRegistered],
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
            default_prompts={},
            internal_event_client=internal_event_client,
            model_limits=model_limits,
            custom_models_enabled=False,
            disable_streaming=False,
        )

        assert registry.prompts_registered == prompts_registered

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
        with pytest.raises(error_class, match=error_message):
            LocalPromptRegistry.from_local_yaml(
                class_overrides={
                    override_key: override_class,
                },
                model_factories=model_factories,
                default_prompts={},
                internal_event_client=internal_event_client,
                model_limits=model_limits,
                custom_models_enabled=False,
                disable_streaming=False,
            )

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

    @pytest.mark.parametrize(
        (
            "prompt_id",
            "expected_name",
            "expected_class",
            "expected_model",
            "expected_prompt_version",
            "expected_model_class",
            "expected_kwargs",
            "default_prompt_env_config",
        ),
        [
            (
                "code_suggestions/generations",
                "Claude 3.7 Vertex Code Generations Agent",
                Prompt,
                "claude-3-7-sonnet@20250219",
                "2.0.2",
                ChatLiteLLM,
                {"stop": ["</new_code>"], "vertex_location": "us-east5"},
                {"code_suggestions/generations": "base"},
            ),
            (
                "code_suggestions/generations",
                "Claude Sonnet 4 Code Generations Agent",
                Prompt,
                "claude-sonnet-4-20250514",
                "^1.0.0",
                ChatAnthropic,
                {"stop": ["</new_code>"]},
                {},
            ),
        ],
    )
    def test_get_code_generations_base(
        self,
        model_factories: dict[ModelClassProvider, TypeModelFactory],
        internal_event_client: Mock,
        model_limits: ConfigModelLimits,
        prompt_id: str,
        expected_name: str,
        expected_prompt_version: str,
        expected_class: Type[Prompt],
        expected_model: str,
        expected_model_class: Type[Model],
        expected_kwargs: dict,
        default_prompt_env_config: dict[str, str],
    ):
        registry = LocalPromptRegistry.from_local_yaml(
            class_overrides={},
            model_factories=model_factories,
            default_prompts=default_prompt_env_config,
            internal_event_client=internal_event_client,
            model_limits=model_limits,
        )
        prompt = registry.get(
            prompt_id,
            prompt_version=expected_prompt_version,
        )
        chain = cast(RunnableSequence, prompt.bound)
        binding = cast(RunnableBinding, chain.last)

        params = {
            "language": "Go",
            "file_name": "test.go",
            "examples_array": [
                {
                    "example": "// calculate the square root of a number",
                    "response": "<new_code>if isPrime { primes = append(primes, num) }\n}",
                    "trigger_type": "empty_function",
                }
            ],
            "trimmed_content_above_cursor": "write a function to find min abs value from an array",
            "trimmed_content_below_cursor": "\n",
            "related_files": [
                '<file_content file_name="client/gitlabnet.go"></file_content>\n',
                '<file_content file_name="client/client_test.go"></file_content>\n',
            ],
            "related_snippets": [],
            "libraries": [],
            "user_instruction": "// write a function to find min abs value from an array",
        }
        expected_rendered_prompt = [
            # pylint: disable=line-too-long
            SystemMessage(
                dedent(
                    """\
            You are a tremendously accurate and skilled coding autocomplete agent. We want to generate new Go code inside the
            file 'test.go' based on instructions from the user.
            Here are a few examples of successfully generated code:
            <examples>
            <example>
            H: <existing_code>
            // calculate the square root of a number
            </existing_code>

            A: <new_code>if isPrime { primes = append(primes, num) }
            }</new_code>
            </example>


            </examples>
            <existing_code>
            write a function to find min abs value from an array{{cursor}}

            </existing_code>

            The existing code is provided in <existing_code></existing_code> tags.
            Here are some files and code snippets that could be related to the current code.
            The files provided in <related_files><related_files> tags.
            The code snippets provided in <related_snippets><related_snippets> tags.
            Please use existing functions from these files and code snippets if possible when suggesting new code.
            <related_files>
            <file_content file_name="client/gitlabnet.go"></file_content>

            <file_content file_name="client/client_test.go"></file_content>

            </related_files>

            The new code you will generate will start at the position of the cursor, which is currently indicated by the {{cursor}} tag.
            In your process, first, review the existing code to understand its logic and format. Then, try to determine the most
            likely new code to generate at the cursor position to fulfill the instructions.

            The comment directly before the {{cursor}} position is the instruction,
            all other comments are not instructions.

            When generating the new code, please ensure the following:
            1. It is valid Go code.
            2. It matches the existing code's variable, parameter and function names.
            3. It does not repeat any existing code. Do not repeat code that comes before or after the cursor tags. This includes cases where the cursor is in the middle of a word.
            4. If the cursor is in the middle of a word, it finishes the word instead of repeating code before the cursor tag.
            5. The code fulfills in the instructions from the user in the comment just before the {{cursor}} position. All other comments are not instructions.
            6. Do not add any comments that duplicates any of the already existing comments, including the comment with instructions.

            Return new code enclosed in <new_code></new_code> tags. We will then insert this at the {{cursor}} position.
            If you are not able to write code based on the given instructions return an empty result like <new_code></new_code>."""
                )
            ),
            HumanMessage("// write a function to find min abs value from an array"),
            AIMessage("<new_code>"),
        ]
        assert (
            prompt.prompt_tpl.invoke(params).to_messages() == expected_rendered_prompt
        )
        assert prompt.name == expected_name
        assert isinstance(prompt, expected_class)
        assert isinstance(prompt.model, expected_model_class)
        assert prompt.model_name == expected_model
        assert binding.kwargs == expected_kwargs

    @pytest.mark.usefixtures("mock_fs")
    def test_default_prompts(
        self,
        model_factories: dict[ModelClassProvider, TypeModelFactory],
        internal_event_client: Mock,
        model_limits: ConfigModelLimits,
    ):
        registry = LocalPromptRegistry.from_local_yaml(
            class_overrides={
                "chat/react": MockPromptClass,
            },
            model_factories=model_factories,
            default_prompts={"chat/react": "custom"},
            internal_event_client=internal_event_client,
            model_limits=model_limits,
            custom_models_enabled=False,
        )

        assert registry.get("chat/react", "^1.0.0").name == "Chat react custom prompt"

    def test_get_prompt_config_no_compatible_versions(
        self,
        model_factories: dict[ModelClassProvider, TypeModelFactory],
        internal_event_client: Mock,
        model_limits: ConfigModelLimits,
    ):
        # Create a registry with a prompt that has versions 1.0.0 and 1.0.1
        registry = LocalPromptRegistry(
            prompts_registered={
                "test/base": PromptRegistered(
                    klass=Prompt,
                    versions={
                        "1.0.0": PromptConfig(
                            name="Test prompt 1.0.0",
                            model=ModelConfig(
                                name="claude-3-5-sonnet-20241022",
                                params=ChatLiteLLMParams(
                                    model_class_provider=ModelClassProvider.LITE_LLM,
                                    top_p=0.1,
                                    top_k=50,
                                    max_tokens=256,
                                    max_retries=10,
                                    custom_llm_provider="vllm",
                                ),
                            ),
                            unit_primitives=["explain_code"],
                            prompt_template={"system": "Template1"},
                        ),
                        "1.0.1": PromptConfig(
                            name="Test prompt 1.0.1",
                            model=ModelConfig(
                                name="claude-3-5-sonnet-20241022",
                                params=ChatLiteLLMParams(
                                    model_class_provider=ModelClassProvider.LITE_LLM,
                                    top_p=0.1,
                                    top_k=50,
                                    max_tokens=256,
                                    max_retries=10,
                                    custom_llm_provider="vllm",
                                ),
                            ),
                            unit_primitives=["explain_code"],
                            prompt_template={"system": "Template1"},
                        ),
                    },
                ),
            },
            model_factories=model_factories,
            default_prompts={},
            internal_event_client=internal_event_client,
            model_limits=model_limits,
            custom_models_enabled=True,
            disable_streaming=True,
        )

        # Try to get a version 2.0.0 which doesn't exist
        with pytest.raises(ValueError) as exc_info:
            registry.get("test", "2.0.0")

        assert (
            str(exc_info.value) == "No prompt version found matching the query: 2.0.0"
        )

    @pytest.mark.usefixtures("mock_fs")
    def test_load_prompt_without_unit_primitive(
        self,
        model_factories,
        internal_event_client: Mock,
        model_limits: ConfigModelLimits,
    ):
        registry = LocalPromptRegistry.from_local_yaml(
            class_overrides={},
            model_factories=model_factories,
            default_prompts={},
            custom_models_enabled=True,
            internal_event_client=internal_event_client,
            model_limits=model_limits,
        )

        yaml_content = """
            name: TestPrompt No UP
            model:
                name: claude-3.5
                params:
                    model_class_provider: litellm
            prompt_template:
                system: test
            """

        with open("/tmp/test_prompt_no_up.yml", "w") as f:
            f.write(yaml_content)

        registry.prompts_registered.update(
            {
                "test/base": PromptRegistered(
                    klass=Prompt,
                    versions={"1.0.0": PromptConfig(**yaml.safe_load(yaml_content))},
                ),  # type:ignore
                "test/codestral": PromptRegistered(
                    klass=Prompt,
                    versions={"1.0.0": PromptConfig(**yaml.safe_load(yaml_content))},
                ),  # type:ignore
            }
        )

        prompt = registry.get("test", "1.0.0")
        assert prompt.unit_primitives == []

        prompt = registry.get(
            "test",
            "1.0.0",
            ModelMetadata(
                name="codestral",
                endpoint=HttpUrl("http://localhost:4000/"),
                provider="custom_openai",
            ),
        )
        assert prompt.unit_primitives == []

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
            default_prompts={},
            internal_event_client=internal_event_client,
            model_limits=model_limits,
        )
        prompt.unit_primitives = []

        with patch.object(test_registry, "get", return_value=prompt):
            result_prompt = test_registry.get_on_behalf(user, prompt_id="test")

            assert result_prompt == prompt

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
