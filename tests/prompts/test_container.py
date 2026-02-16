import sys
from pathlib import Path
from typing import cast

import pytest
import yaml
from dependency_injector import containers, providers
from dependency_injector.providers import Factory
from pydantic import AnyUrl

from ai_gateway.config import ConfigModelLimits
from ai_gateway.model_metadata import create_model_metadata
from ai_gateway.model_selection.model_selection_config import ModelSelectionConfig
from ai_gateway.prompts import Prompt
from ai_gateway.prompts.config import ChatOpenAIParams, ModelClassProvider, PromptConfig
from ai_gateway.prompts.registry import (
    LEGACY_MODEL_MAPPING,
    LocalPromptRegistry,
    feature_setting_for_prompt_id,
)


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

        # Check every existing version
        for version in versions:
            if model_name == "base":
                # Test that the legacy prompts where the version was tied to the model work with no model metadata
                if LEGACY_MODEL_MAPPING.get(prompt_id, {}).get(version, None):
                    model_metadata = None
                else:
                    # To be able to test `base` prompt versions that aren't tied to a specific model, we select an
                    # arbitrary model identifier that we know will fetch the `base` prompt
                    model_metadata = create_model_metadata(
                        {"provider": "gitlab", "identifier": "claude_sonnet_4_20250514"}
                    )
            else:
                # Skip directories that are prompt families (like claude_4_5) rather than model families
                # Prompt families are resolved through the model's family field, not as standalone models
                try:
                    model_metadata = create_model_metadata(
                        {
                            "name": str(model_name),
                            "endpoint": AnyUrl("http://localhost:4000"),
                            "provider": "gitlab",
                        }
                    )
                except ValueError:
                    # This is a prompt family directory, not a model family directory - skip it
                    continue
            prompt = registry.get(
                prompt_id,
                version,
                model_metadata=model_metadata,
            )
            assert isinstance(prompt, Prompt)
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
            )
            assert isinstance(prompt, Prompt)


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


def test_prompt_family_configs_are_valid():
    """Test that all prompt family YAML files are valid PromptConfig objects.

    Prompt families (like claude_4_5) are directories that contain prompt definitions but aren't directly instantiable
    as models. They're used by models that have them in their 'family' field. The main test_container skips these
    because it can't create model_metadata for them, so we test them separately here by directly validating their YAML
    files against the PromptConfig schema.
    """
    prompts_dir = Path(
        sys.modules[LocalPromptRegistry.__module__].__file__ or ""
    ).parent
    prompts_definitions_dir = prompts_dir / "definitions"

    # Known prompt family directories that should be validated
    # These are directories that contain prompts but aren't standalone model identifiers
    prompt_families_to_test = [
        "claude_4_5",
        "gpt_5",
    ]

    validation_errors = []

    for path in prompts_definitions_dir.glob("**"):
        # Check if this is a prompt family directory we want to test
        if path.name not in prompt_families_to_test:
            continue

        # Find all YAML version files in this directory
        version_files = list(path.glob("*.yml"))

        if not version_files:
            continue

        prompt_id_with_model_name = path.relative_to(prompts_definitions_dir)
        prompt_id = str(prompt_id_with_model_name.parent)
        model_name = prompt_id_with_model_name.name

        # Validate each version file
        for version_file in version_files:
            try:
                with open(version_file, "r") as fp:
                    yaml_content = yaml.safe_load(fp)
                    # This will raise ValidationError if the YAML doesn't match the schema
                    PromptConfig(**yaml_content)
            except Exception as e:
                validation_errors.append(
                    f"Validation failed for {prompt_id}/{model_name}/{version_file.stem}: {e}"
                )

    # Assert that there were no validation errors
    if validation_errors:
        error_message = "\n".join(validation_errors)
        pytest.fail(f"Prompt family config validation errors:\n{error_message}")
