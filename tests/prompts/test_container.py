import sys
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
from dependency_injector import containers, providers
from dependency_injector.providers import Factory
from pydantic import AnyUrl

from ai_gateway.config import ConfigModelLimits
from ai_gateway.model_metadata import create_model_metadata
from ai_gateway.model_selection.model_selection_config import ModelSelectionConfig
from ai_gateway.prompts.config import ChatOpenAIParams, ModelClassProvider
from ai_gateway.prompts.registry import (
    LocalPromptRegistry,
    feature_setting_for_prompt_id,
)
from duo_workflow_service import agents as workflow
from duo_workflow_service.agents import prompt_adapter
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from lib.internal_events.event_enum import CategoryEnum


@pytest.fixture(autouse=True, scope="module")
def patch_env():
    mp = pytest.MonkeyPatch()
    mp.setenv("OPENAI_API_KEY", "test-key")
    yield
    mp.undo()


@pytest.fixture(name="config_values")
def config_values_fixture(assets_dir):
    return {
        "custom_models": {"enabled": True, "disable_streaming": True},
        "self_signed_jwt": {
            "signing_key": open(assets_dir / "keys" / "signing_key.pem").read()
        },
        "amazon_q": {
            "region": "us-west-2",
        },
        "model_engine_limits": {
            "anthropic-chat": {
                "claude-3-7-sonnet-20250219": {
                    "input_tokens": 5,
                    "output_tokens": 10,
                    "concurrency": 15,
                }
            }
        },
    }


def _kwargs_for_class(klass):
    match klass:
        case workflow.Agent:
            return {
                "workflow_id": "123",
                "workflow_type": CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
                "http_client": Mock(spec=GitlabHttpClient),
            }
        case workflow.ChatAgent:
            return {
                "workflow_id": "123",
                "workflow_type": CategoryEnum.WORKFLOW_CHAT,
            }
        case prompt_adapter.ChatPrompt:
            return {
                "workflow_id": "123",
                "workflow_type": CategoryEnum.WORKFLOW_CHAT,
            }

    return {}


def test_container(mock_ai_gateway_container: containers.DeclarativeContainer):
    prompts = cast(providers.Container, mock_ai_gateway_container.pkg_prompts)
    registry = cast(LocalPromptRegistry, prompts.prompt_registry())

    assert registry.model_limits == ConfigModelLimits(
        {
            "anthropic-chat": {
                "claude-3-7-sonnet-20250219": {
                    "input_tokens": 5,
                    "output_tokens": 10,
                    "concurrency": 15,
                }
            }
        }
    )

    unit_primitive_config_map = ModelSelectionConfig().get_unit_primitive_config_map()
    prompts_dir = Path(
        sys.modules[LocalPromptRegistry.__module__].__file__ or ""
    ).parent
    prompts_definitions_dir = prompts_dir / "definitions"
    # Iterate over every file in the prompts definitions directory. Make sure
    # they're loaded into the registry and that the resulting Prompts are valid.
    for path in prompts_definitions_dir.glob("**"):
        versions = [version.stem for version in path.glob("*.yml")]

        if not versions:
            continue

        prompt_id_with_model_name = path.relative_to(prompts_definitions_dir)
        prompt_id = str(prompt_id_with_model_name.parent)
        model_name = prompt_id_with_model_name.name

        if model_name == "base":
            # The base model is requested when no model metadata is passed
            model_metadata = None
        else:
            model_metadata = create_model_metadata(
                {
                    "name": str(model_name),
                    "endpoint": AnyUrl("http://localhost:4000"),
                    "provider": "gitlab",
                }
            )

        # Load the prompt definition to get the class
        prompt_registered = registry._load_prompt_definition(prompt_id, path)
        klass = prompt_registered.klass
        kwargs = _kwargs_for_class(klass)

        # Check every existing version
        for version in versions:
            prompt = registry.get(
                prompt_id,
                f"={version}",
                model_metadata=model_metadata,
                **kwargs,
            )
            assert isinstance(prompt, klass)
            assert prompt.model.disable_streaming

        # Check that at least one version is available and loads for each selectable model
        selectable_model_metadata = [
            create_model_metadata({"provider": "gitlab", "identifier": model})
            for model in unit_primitive_config_map[
                feature_setting_for_prompt_id(prompt_id)
            ].selectable_models
        ]
        for model in selectable_model_metadata:
            prompt = registry.get(
                prompt_id,
                "*",  # Grab the latest version
                model_metadata=model,
                **kwargs,
            )
            assert isinstance(prompt, klass)


def test_container_openai_model_factory_exists(
    mock_ai_gateway_container: containers.DeclarativeContainer,
):
    from langchain_openai import ChatOpenAI  # pylint: disable=import-outside-toplevel

    prompts = cast(providers.Container, mock_ai_gateway_container.pkg_prompts)
    registry = cast(LocalPromptRegistry, prompts.prompt_registry())

    # Test that the OpenAI provider is registered in the model_factory_mapping
    assert ModelClassProvider.OPENAI in registry.model_factories

    factory = registry.model_factories[ModelClassProvider.OPENAI]
    assert isinstance(factory, Factory)

    params = ChatOpenAIParams(
        temperature=1,
        max_tokens=1_028,
        max_retries=1,
        model_class_provider=ModelClassProvider.OPENAI,
    )
    model: ChatOpenAI = factory(
        model="gpt-4",
        **params.model_dump(exclude_none=True, exclude={"model_class_provider"}),
    )

    assert model.model_name == "gpt-4"
    assert model.temperature == 1.0
    assert model.max_tokens == 1_028
    assert model.max_retries == 1
    assert model.output_version == "responses/v1"
