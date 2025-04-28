from pathlib import Path
from textwrap import dedent

import pytest
from gitlab_cloud_connector import GitLabUnitPrimitive
from pyfakefs.fake_filesystem import FakeFilesystem

from ai_gateway.model_selection.model_selection_config import (
    LLMDefinition,
    ModelSelectionConfig,
    UnitPrimitiveConfig,
)


@pytest.fixture
def mock_fs(fs: FakeFilesystem):
    # Create the directory structure
    print(Path(__file__).parent.parent.parent / "ai_gateway" / "model_selection")
    model_selection_dir = (
        Path(__file__).parent.parent.parent / "ai_gateway" / "model_selection"
    )
    fs.create_file(
        model_selection_dir / "models.yml",
        contents=dedent(
            """
            models:
              - model_identifier: model1
                name: Model One
                gitlab_identifier: gitlab-model-1
                provider: provider1
                provider_identifier: provider-model-1
                params:
                  param1: value1
              - model_identifier: model2
                name: Model Two
                gitlab_identifier: gitlab-model-2
                provider: provider2
                provider_identifier: provider-model-2
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
                default_model: "model_1"
                selectable_models:
                  - "model_1"
                beta_models:
                  - "model_2"
            """
        ),
    )


def test_load_llm_definitions(mock_fs):
    configs = ModelSelectionConfig().get_llm_definitions()

    assert configs == {
        "gitlab-model-1": LLMDefinition(
            name="Model One",
            gitlab_identifier="gitlab-model-1",
            provider="provider1",
            provider_identifier="provider-model-1",
            params={"param1": "value1"},
        ),
        "gitlab-model-2": LLMDefinition(
            name="Model Two",
            gitlab_identifier="gitlab-model-2",
            provider="provider2",
            provider_identifier="provider-model-2",
            params={},
        ),
    }


def test_get_unit_primitive_config(mock_fs):
    configs = ModelSelectionConfig().get_unit_primitive_config()

    assert configs == [
        UnitPrimitiveConfig(
            feature_setting="test_config",
            unit_primitives=[
                GitLabUnitPrimitive.ASK_COMMIT,
                GitLabUnitPrimitive.ASK_EPIC,
            ],
            default_model="model_1",
            selectable_models=["model_1"],
            beta_models=["model_2"],
        )
    ]


def test_is_singleton(mock_fs):
    config_instance_1 = ModelSelectionConfig()
    config_instance_2 = ModelSelectionConfig()

    assert config_instance_1 is config_instance_2
