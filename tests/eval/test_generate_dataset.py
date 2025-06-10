import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest
from eli5.datasets.generator import DatasetGenerator, ModelConfig, PromptConfig
from eli5.datasets.serializers import JsonFileSerializer, LangSmithSerializer
from langsmith import Client

from ai_gateway.prompts.base import BasePromptRegistry
from eval.generate_dataset import (
    create_langsmith_client,
    get_message_source,
    get_prompt_source,
    run,
)


class TestGetMessageSource:
    def mock_ast(self, template_path):
        mock_ast = MagicMock()
        mock_ast.body = [MagicMock()]
        mock_ast.body[0].template = MagicMock()
        mock_ast.body[0].template.value = template_path
        return mock_ast

    @pytest.fixture
    def mock_system_ast(self):
        return self.mock_ast("chat/explain_code/system/1.0.0.jinja")

    @pytest.fixture
    def mock_user_ast(self):
        return self.mock_ast("chat/explain_code/user/1.0.0.jinja")

    @pytest.fixture
    def mock_env(self, mock_system_ast, mock_user_ast):
        def parse_side_effect(template_str):
            if "system" in template_str:
                return mock_system_ast
            return mock_user_ast

        with patch("eval.generate_dataset.SandboxedEnvironment") as mock_env_class:
            mock_env = MagicMock()
            mock_env_class.return_value = mock_env
            mock_env.parse.side_effect = parse_side_effect

            yield mock_env

    @pytest.fixture
    def mock_loader(self, mock_env, mock_user_ast):
        mock_loader = MagicMock()
        mock_env.loader = mock_loader
        mock_env.parse.return_value = mock_user_ast
        return mock_loader

    def test_get_message_source_with_include_statement(self, mock_env, mock_loader):
        template_content = "Test template content"
        template_path = "chat/explain_code/user/1.0.0.jinja"
        prompt_template = {"user": f"{{% include '{template_path}' %}}\n"}

        mock_loader.get_source.return_value = (template_content, None, None)

        result = get_message_source(prompt_template)

        assert result == {"user": template_content}
        mock_env.parse.assert_called_once_with(prompt_template["user"])
        mock_loader.get_source.assert_called_once_with(mock_env, template_path)

    def test_get_message_source_without_include_statement(self):
        template_content = "Direct template content"
        prompt_template = {"user": template_content}

        result = get_message_source(prompt_template)

        assert result == {"user": template_content}

    def test_get_message_source_with_include_error(self, mock_loader):
        prompt_template = {"user": "{% include 'non_existent_template.jinja' %}\n"}

        mock_loader.get_source.side_effect = Exception("Template not found")

        with pytest.raises(ValueError) as exc_info:
            get_message_source(prompt_template)

        assert "Template not found" in str(exc_info.value)
        assert "non_existent_template.jinja" in str(exc_info.value)

    def test_get_message_source_multiple_roles(self, mock_loader):
        system_content = "System template content"
        user_content = "User template content"
        prompt_template = {
            "system": "{% include 'chat/explain_code/system/1.0.0.jinja' %}\n",
            "user": "{% include 'chat/explain_code/user/1.0.0.jinja' %}\n",
        }

        def get_source_side_effect(_, template_path):
            if "system" in template_path:
                return (system_content, None, None)
            return (user_content, None, None)

        mock_loader.get_source.side_effect = get_source_side_effect

        result = get_message_source(prompt_template)

        assert result == {"system": system_content, "user": user_content}
        assert mock_loader.get_source.call_count == 2


class TestGetPromptSource:
    @patch("eval.generate_dataset.get_message_source")
    def test_get_prompt_source(self, mock_get_message_source):
        prompt_id = "chat/explain_code"
        prompt_version = "1.0.0"

        mock_prompt = Mock()
        mock_message_system = Mock()
        mock_message_system.__class__.__name__ = "SystemMessagePromptTemplate"
        mock_message_system.prompt.template = "Source system template content"

        mock_message_user = Mock()
        mock_message_user.__class__.__name__ = "HumanMessagePromptTemplate"
        mock_message_user.prompt.template = "Source user template content"

        mock_prompt_tpl = Mock()
        mock_prompt_tpl.messages = [mock_message_system, mock_message_user]
        mock_prompt.prompt_tpl = mock_prompt_tpl
        mock_prompt.name = "Test Prompt"

        mock_registry = Mock(spec=BasePromptRegistry)
        mock_registry.get.return_value = mock_prompt

        mock_get_message_source.return_value = {
            "system": "Processed system template",
            "user": "Processed user template",
        }

        result = get_prompt_source(
            prompt_id, prompt_version, prompt_registry=mock_registry
        )

        assert result["name"] == "Test Prompt"
        assert result["prompt_template"]["system"] == "Processed system template"
        assert "Processed user template" in result["prompt_template"]["user"]
        mock_registry.get.assert_called_once_with(prompt_id, prompt_version)
        mock_get_message_source.assert_called_once_with(
            {
                "system": "Source system template content",
                "user": "Source user template content",
            }
        )

    @patch("eval.generate_dataset.get_message_source")
    def test_message_extraction_with_different_types(self, mock_get_message_source):
        prompt_id = "chat/complex_test"
        prompt_version = "1.0.0"

        mock_prompt = Mock()

        mock_message_system = Mock()
        mock_message_system.__class__.__name__ = "SystemMessagePromptTemplate"
        mock_message_system.prompt.template = "Source system template content"

        mock_message_user = Mock()
        mock_message_user.__class__.__name__ = "HumanMessagePromptTemplate"
        mock_message_user.prompt.template = "Source user template content"

        mock_message_ai = Mock()
        mock_message_ai.__class__.__name__ = "AIMessagePromptTemplate"
        mock_message_ai.prompt.template = "Source AI template content"

        mock_message_custom = Mock()
        mock_message_custom.__class__.__name__ = "CustomMessagePromptTemplate"
        mock_message_custom.prompt.template = "Source custom template content"

        mock_prompt_tpl = Mock()
        mock_prompt_tpl.messages = [
            mock_message_system,
            mock_message_user,
            mock_message_ai,
            mock_message_custom,
        ]
        mock_prompt.prompt_tpl = mock_prompt_tpl
        mock_prompt.name = "Test Prompt"

        mock_registry = Mock(spec=BasePromptRegistry)
        mock_registry.get.return_value = mock_prompt

        mock_get_message_source.return_value = {
            "system": "Processed system message",
            "user": "Processed user message",
            "assistant": "Processed assistant message",
            "CustomMessagePromptTemplate": "Processed custom message",
        }

        result = get_prompt_source(
            prompt_id, prompt_version, prompt_registry=mock_registry
        )

        assert result["name"] == "Test Prompt"
        assert result["prompt_template"]["system"] == "Processed system message"
        assert "Processed user message" in result["prompt_template"]["user"]
        assert (
            "[assistant]: Processed assistant message"
            in result["prompt_template"]["user"]
        )
        assert (
            "[CustomMessagePromptTemplate]: Processed custom message"
            in result["prompt_template"]["user"]
        )

        mock_get_message_source.assert_called_once()
        args = mock_get_message_source.call_args[0][0]
        assert args["system"] == "Source system template content"
        assert args["user"] == "Source user template content"
        assert args["assistant"] == "Source AI template content"
        assert args["CustomMessagePromptTemplate"] == "Source custom template content"

    @patch("eval.generate_dataset.get_message_source")
    def test_get_prompt_source_without_user_message(self, mock_get_message_source):
        mock_prompt = Mock()
        mock_prompt_tpl = Mock()
        mock_prompt_tpl.messages = []
        mock_prompt.prompt_tpl = mock_prompt_tpl

        mock_registry = Mock(spec=BasePromptRegistry)
        mock_registry.get.return_value = mock_prompt

        mock_get_message_source.return_value = {
            "system": "Processed system",
        }

        with pytest.raises(ValueError) as exc_info:
            get_prompt_source("foo", "bar", prompt_registry=mock_registry)

        assert "Prompt must include a user message" in str(exc_info.value)


class TestCreateLangsmithClient:
    # pylint: disable=direct-environment-variable-reference
    @patch.dict(os.environ, {}, clear=True)
    def test_create_langsmith_client_missing_api_key(self):
        with pytest.raises(Exception):
            create_langsmith_client()

    @patch.dict(os.environ, {"LANGCHAIN_API_KEY": "test_key"})
    @patch("eval.generate_dataset.Client")
    def test_create_langsmith_client_success(self, mock_client_class):
        mock_client = Mock(spec=Client)
        mock_client_class.return_value = mock_client

        result = create_langsmith_client()

        assert result == mock_client
        mock_client_class.assert_called_once_with(api_key="test_key")

    @patch.dict(os.environ, {"LANGCHAIN_API_KEY": "test_key"})
    @patch("eval.generate_dataset.Client")
    def test_create_langsmith_client_error(self, mock_client_class):
        mock_client_class.side_effect = Exception("Connection error")

        with pytest.raises(Exception):
            create_langsmith_client()

    # pylint: enable=direct-environment-variable-reference


class TestRun:
    @pytest.fixture
    def mock_container_application(self):
        with patch(
            "eval.generate_dataset.ContainerApplication"
        ) as mock_container_class:
            mock_container = Mock()
            mock_container_class.return_value = mock_container
            mock_config = Mock()
            mock_container.config = mock_config
            yield mock_container

    @patch("eval.generate_dataset.DatasetGenerator")
    @patch("eval.generate_dataset.JsonFileSerializer")
    @patch("eval.generate_dataset.PromptConfig")
    @patch("eval.generate_dataset.ModelConfig")
    @patch("eval.generate_dataset.create_langsmith_client")
    @patch("eval.generate_dataset.get_prompt_source")
    @patch("eval.generate_dataset.typer.echo")
    def test_run_basic(
        self,
        mock_echo,
        mock_get_prompt_source,
        mock_create_client,
        mock_model_config_class,
        mock_prompt_config_class,
        mock_json_serializer_class,
        mock_dataset_generator_class,
        mock_container_application,
    ):
        prompt_id = "chat/explain_code"
        prompt_version = "1.0.0"
        dataset_name = "test_dataset"

        mock_create_client.return_value = None
        mock_prompt_source = {"name": "Test Prompt", "prompt_template": {}}
        mock_get_prompt_source.return_value = mock_prompt_source

        mock_prompt_config = Mock(spec=PromptConfig)
        mock_prompt_config_class.from_source.return_value = mock_prompt_config

        mock_model_config = Mock(spec=ModelConfig)
        mock_model_config_class.return_value = mock_model_config

        mock_json_serializer = Mock(spec=JsonFileSerializer)
        mock_json_serializer.output_path = Path("/path/to/output/dataset.json")
        mock_json_serializer_class.return_value = mock_json_serializer

        mock_generator = Mock(spec=DatasetGenerator)
        mock_dataset_generator_class.return_value = mock_generator

        run(
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            dataset_name=dataset_name,
            num_examples=10,
            temperature=0.7,
            upload=False,
        )

        mock_container_application.wire.assert_called_once_with(
            modules=["eval.generate_dataset"]
        )
        mock_create_client.assert_not_called()
        mock_get_prompt_source.assert_called_once_with(prompt_id, prompt_version)
        mock_prompt_config_class.from_source.assert_called_once_with(mock_prompt_source)
        mock_model_config_class.assert_called_once_with(temperature=0.7)
        mock_json_serializer_class.assert_called_once_with(dataset_name, Path.cwd())
        mock_dataset_generator_class.assert_called_once_with(
            prompt_config=mock_prompt_config,
            model_config=mock_model_config,
            serializers=[mock_json_serializer],
        )
        mock_generator.generate.assert_called_once_with(num_examples=10)

        assert mock_echo.call_args_list == [
            call("Generating dataset with 10 examples from prompt: chat/explain_code"),
            call("Dataset generated successfully: /path/to/output/dataset.json"),
        ]

    @patch("eval.generate_dataset.DatasetGenerator")
    @patch("eval.generate_dataset.LangSmithSerializer")
    @patch("eval.generate_dataset.JsonFileSerializer")
    @patch("eval.generate_dataset.PromptConfig")
    @patch("eval.generate_dataset.ModelConfig")
    @patch("eval.generate_dataset.create_langsmith_client")
    @patch("eval.generate_dataset.get_prompt_source")
    @patch("eval.generate_dataset.typer.echo")
    def test_run_with_upload(
        self,
        mock_echo,
        mock_get_prompt_source,
        mock_create_client,
        mock_model_config_class,
        mock_prompt_config_class,
        mock_json_serializer_class,
        mock_langsmith_serializer_class,
        mock_dataset_generator_class,
    ):
        prompt_id = "chat/explain_code"
        prompt_version = "1.0.0"
        dataset_name = "test_dataset"
        description = "Test dataset description"
        output_dir = "/custom/output"

        mock_client = Mock(spec=Client)
        mock_create_client.return_value = mock_client
        mock_prompt_source = {"name": "Test Prompt", "prompt_template": {}}
        mock_get_prompt_source.return_value = mock_prompt_source

        mock_prompt_config = Mock(spec=PromptConfig)
        mock_prompt_config.name = "Test Prompt"
        mock_prompt_config_class.from_source.return_value = mock_prompt_config

        mock_model_config = Mock(spec=ModelConfig)
        mock_model_config_class.return_value = mock_model_config

        mock_json_serializer = Mock(spec=JsonFileSerializer)
        mock_json_serializer.output_path = Path("/path/to/output/dataset.json")
        mock_json_serializer_class.return_value = mock_json_serializer

        mock_langsmith_serializer = Mock(spec=LangSmithSerializer)
        mock_langsmith_serializer_class.return_value = mock_langsmith_serializer

        mock_generator = Mock(spec=DatasetGenerator)
        mock_dataset_generator_class.return_value = mock_generator

        run(
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            dataset_name=dataset_name,
            output_dir=output_dir,
            num_examples=5,
            temperature=0.5,
            upload=True,
            description=description,
        )

        mock_create_client.assert_called_once()
        mock_prompt_config_class.from_source.assert_called_once_with(mock_prompt_source)
        mock_model_config_class.assert_called_once_with(temperature=0.5)
        mock_json_serializer_class.assert_called_once_with(dataset_name, output_dir)
        mock_langsmith_serializer_class.assert_called_once_with(
            client=mock_client,
            dataset_name=dataset_name,
            dataset_description=description,
        )
        mock_dataset_generator_class.assert_called_once_with(
            prompt_config=mock_prompt_config,
            model_config=mock_model_config,
            serializers=[mock_json_serializer, mock_langsmith_serializer],
        )
        mock_generator.generate.assert_called_once_with(num_examples=5)
        assert any(
            "uploaded to LangSmith" in call.args[0] for call in mock_echo.call_args_list
        )
        assert mock_echo.call_args_list == [
            call("Generating dataset with 5 examples from prompt: chat/explain_code"),
            call("Dataset generated successfully: /path/to/output/dataset.json"),
            call("Dataset 'test_dataset' uploaded to LangSmith"),
        ]
