from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pytest
from gitlab_cloud_connector import GitLabUnitPrimitive
from pyfakefs.fake_filesystem import FakeFilesystem

from ai_gateway.model_selection.model_selection_config import (
    LLMDefinition,
    ModelSelectionConfig,
    UnitPrimitiveConfig,
)


# editorconfig-checker-disable
@pytest.fixture(name="mock_fs")
def mock_fs_fixture(fs: FakeFilesystem):
    print(Path(__file__).parent.parent.parent / "ai_gateway" / "model_selection")
    model_selection_dir = (
        Path(__file__).parent.parent.parent / "ai_gateway" / "model_selection"
    )
    fs.create_file(
        model_selection_dir / "models.yml",
        contents=dedent(
            """
            models:
              - name: Model One
                gitlab_identifier: gitlab-model-1
                max_context_tokens: 200000
                params:
                  model: provider-model-1
                  param1: value1
              - name: Model Two
                gitlab_identifier: gitlab-model-2
                max_context_tokens: 200000
                params:
                  model: provider-model-2
            """
        ),
    )

    fs.create_file(
        model_selection_dir / "unit_primitives.yml",
        contents=dedent(
            """
            configurable_unit_primitives:
              - feature_setting: "test_config"
                unit_primitives:
                  - "ask_commit"
                  - "ask_epic"
                default_model: "gitlab-model-1"
                selectable_models:
                  - "gitlab-model-1"
                beta_models:
                  - "gitlab-model-2"
            """
        ),
    )


# editorconfig-checker-enable


@pytest.fixture(name="selection_config")
def selection_config_fixture(mock_fs):  # pylint: disable=unused-argument
    return ModelSelectionConfig()


def test_load_llm_definitions(selection_config):
    assert selection_config.get_llm_definitions() == {
        "gitlab-model-1": LLMDefinition(
            name="Model One",
            gitlab_identifier="gitlab-model-1",
            max_context_tokens=200000,
            params={"model": "provider-model-1", "param1": "value1"},
        ),
        "gitlab-model-2": LLMDefinition(
            name="Model Two",
            gitlab_identifier="gitlab-model-2",
            max_context_tokens=200000,
            params={"model": "provider-model-2"},
        ),
    }


def test_get_unit_primitive_config(selection_config):
    assert list(selection_config.get_unit_primitive_config()) == [
        UnitPrimitiveConfig(
            feature_setting="test_config",
            unit_primitives=[
                GitLabUnitPrimitive.ASK_COMMIT,
                GitLabUnitPrimitive.ASK_EPIC,
            ],
            default_model="gitlab-model-1",
            selectable_models=["gitlab-model-1"],
            beta_models=["gitlab-model-2"],
            dev=None,
        )
    ]


def test_get_unit_primitive_config_map(selection_config):
    assert selection_config.get_unit_primitive_config_map() == {
        "test_config": UnitPrimitiveConfig(
            feature_setting="test_config",
            unit_primitives=[
                GitLabUnitPrimitive.ASK_COMMIT,
                GitLabUnitPrimitive.ASK_EPIC,
            ],
            default_model="gitlab-model-1",
            selectable_models=["gitlab-model-1"],
            beta_models=["gitlab-model-2"],
            dev=None,
        )
    }


@pytest.mark.usefixtures("mock_fs")
def test_is_singleton():
    config_instance_1 = ModelSelectionConfig.instance()
    config_instance_2 = ModelSelectionConfig.instance()

    assert config_instance_1 is config_instance_2


@pytest.mark.usefixtures("mock_fs")
def test_singleton_caches_yaml_loading():
    """Test that YAML is only loaded once when using the module-level singleton.

    This is a regression test for a bug where __init__ was resetting the cache to None on every instantiation, causing
    YAML to be re-parsed on every request. Now we use a module-level singleton to ensure YAML is loaded only once.
    """
    ModelSelectionConfig._instance = None
    # Track how many times yaml.safe_load is called
    original_safe_load = __import__("yaml").safe_load
    call_count = 0

    def counting_safe_load(*args, **kwargs):
        nonlocal call_count  # Allow modifying outer scope variable
        call_count += 1
        return original_safe_load(*args, **kwargs)

    with patch("yaml.safe_load", side_effect=counting_safe_load):
        # First access - should load YAML (2 files: models.yml + unit_primitives.yml)
        config = ModelSelectionConfig.instance()
        llm_defs1 = config.get_llm_definitions()
        unit_primitives1 = config.get_unit_primitive_config_map()

        first_load_count = call_count
        assert first_load_count == 2, "Should load 2 YAML files on first access"

        # Second access - should NOT reload YAML (uses cached data)
        llm_defs2 = config.get_llm_definitions()
        unit_primitives2 = config.get_unit_primitive_config_map()

        second_load_count = call_count

        assert second_load_count == first_load_count, (
            f"YAML was reloaded! Expected {first_load_count} calls, "
            f"but got {second_load_count}. The cache was not preserved."
        )

        assert llm_defs2 is llm_defs1
        assert unit_primitives2 is unit_primitives1


def test_get_model(selection_config):
    assert selection_config.get_model("gitlab-model-1") == LLMDefinition(
        name="Model One",
        gitlab_identifier="gitlab-model-1",
        max_context_tokens=200000,
        params={"model": "provider-model-1", "param1": "value1"},
    )


def test_get_model_missing_key(selection_config):
    with pytest.raises(ValueError):
        selection_config.get_model("non-existing-model")


def test_get_model_for_feature(selection_config):
    assert selection_config.get_model_for_feature("test_config") == LLMDefinition(
        name="Model One",
        gitlab_identifier="gitlab-model-1",
        max_context_tokens=200000,
        params={"model": "provider-model-1", "param1": "value1"},
    )


def test_get_model_for_feature_no_feature(selection_config):
    with pytest.raises(ValueError, match="Invalid feature setting: random-feature"):
        selection_config.get_model_for_feature("random-feature")


@pytest.mark.usefixtures("mock_fs")
def test_validate_without_error():
    assert ModelSelectionConfig().validate() is None


def test_validate_with_error(fs: FakeFilesystem):
    model_selection_dir = (
        Path(__file__).parent.parent.parent / "ai_gateway" / "model_selection"
    )

    # editorconfig-checker-disable
    fs.create_file(
        model_selection_dir / "unit_primitives.yml",
        contents=dedent(
            """
            configurable_unit_primitives:
              - feature_setting: "test_config"
                unit_primitives:
                  - "ask_commit"
                default_model: "non_existent_model"
                selectable_models:
                  - "model_1"
                  - "another_non_existent_model"
                beta_models:
                  - "third_non_existent_model"
            """
        ),
    )

    # Create a models.yml file with valid models
    fs.create_file(
        model_selection_dir / "models.yml",
        contents=dedent(
            """
            models:
              - name: Model One
                gitlab_identifier: model_1
                max_context_tokens: 200000
                params:
                  model: provider-model-1
            """
        ),
    )
    # editorconfig-checker-enable

    with pytest.raises(ValueError) as excinfo:
        ModelSelectionConfig().validate()

    error_message = str(excinfo.value)
    assert "non_existent_model" in error_message
    assert "another_non_existent_model" in error_message
    assert "third_non_existent_model" in error_message


def test_validate_default_model_not_in_selectable_models(fs: FakeFilesystem):
    """Test that validation fails when default models are not in selectable_models."""
    model_selection_dir = (
        Path(__file__).parent.parent.parent / "ai_gateway" / "model_selection"
    )

    # editorconfig-checker-disable
    fs.create_file(
        model_selection_dir / "models.yml",
        contents=dedent(
            """
            models:
              - name: Model One
                gitlab_identifier: model_1
                max_context_tokens: 200000
                params:
                  model: provider-model-1
              - name: Model Two
                gitlab_identifier: model_2
                max_context_tokens: 200000
                params:
                  model: provider-model-2
              - name: Model Three
                gitlab_identifier: model_3
                max_context_tokens: 200000
                params:
                  model: provider-model-3
            """
        ),
    )

    fs.create_file(
        model_selection_dir / "unit_primitives.yml",
        contents=dedent(
            """
            configurable_unit_primitives:
              - feature_setting: "test_config"
                unit_primitives:
                  - "ask_commit"
                default_model: "model_1"
                selectable_models:
                  - "model_2"
                  - "model_3"
              - feature_setting: "another_config"
                unit_primitives:
                  - "generate_code"
                default_model: "model_2"
                selectable_models:
                  - "model_1"
                  - "model_3"
            """
        ),
    )
    # editorconfig-checker-enable

    with pytest.raises(ValueError) as excinfo:
        ModelSelectionConfig().validate()

    error_message = str(excinfo.value)
    expected_error = (
        "Default models must be included in selectable_models:\n"
        "  - Feature 'test_config' has default model 'model_1' that is not in selectable_models.\n"
        "  - Feature 'another_config' has default model 'model_2' that is not in selectable_models."
    )
    assert error_message == expected_error


def test_get_proxy_models_for_provider(fs: FakeFilesystem):
    """Test that get_proxy_models_for_provider returns models with matching proxy_provider."""
    model_selection_dir = (
        Path(__file__).parent.parent.parent / "ai_gateway" / "model_selection"
    )

    # editorconfig-checker-disable
    fs.create_file(
        model_selection_dir / "models.yml",
        contents=dedent(
            """
            models:
              - name: Claude Sonnet
                gitlab_identifier: claude_sonnet
                max_context_tokens: 200000
                proxy_provider: anthropic
                params:
                  model: claude-sonnet-4-5-20250929
              - name: Claude Opus
                gitlab_identifier: claude_opus
                max_context_tokens: 200000
                proxy_provider: anthropic
                params:
                  model: claude-opus-4-5-20251101
              - name: GPT Model
                gitlab_identifier: gpt_model
                max_context_tokens: 128000
                proxy_provider: openai
                params:
                  model: gpt-5
              - name: Non-proxy Model
                gitlab_identifier: non_proxy
                max_context_tokens: 200000
                params:
                  model: some-model
            """
        ),
    )

    fs.create_file(
        model_selection_dir / "unit_primitives.yml",
        contents=dedent(
            """
            configurable_unit_primitives: []
            """
        ),
    )
    # editorconfig-checker-enable

    config = ModelSelectionConfig()

    # Test anthropic provider
    anthropic_models = config.get_proxy_models_for_provider("anthropic")
    assert set(anthropic_models) == {
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-5-20251101",
    }

    # Test openai provider
    openai_models = config.get_proxy_models_for_provider("openai")
    assert set(openai_models) == {"gpt-5"}

    # Test unknown provider returns empty list
    unknown_models = config.get_proxy_models_for_provider("unknown")
    assert not unknown_models
