from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from gitlab_cloud_connector import GitLabUnitPrimitive

from ai_gateway.api.v1.models.get_definitions import router
from ai_gateway.model_selection import LLMDefinition, UnitPrimitiveConfig


@pytest.fixture
def client():
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def mock_model_config():
    with patch(
        "ai_gateway.api.v1.models.get_definitions.ModelSelectionConfig",
    ) as mock:
        mock_configs = MagicMock()
        mock.return_value = mock_configs
        mock_configs.get_llm_definitions.return_value = {
            "model1": LLMDefinition(
                name="Model 1",
                gitlab_identifier="model1",
                provider="provider1",
                provider_identifier="provider_id_1",
                params={},
            ),
            "model2": LLMDefinition(
                name="Model 2",
                gitlab_identifier="model2",
                provider="provider2",
                provider_identifier="provider_id_2",
                params={},
            ),
        }
        mock_configs.get_unit_primitive_config.return_value = [
            UnitPrimitiveConfig(
                feature_setting="config1",
                unit_primitives=[
                    GitLabUnitPrimitive.ASK_ISSUE,
                    GitLabUnitPrimitive.ASK_EPIC,
                ],
                default_model="model1",
                selectable_models=["model1", "model2"],
                beta_models=[],
            ),
            UnitPrimitiveConfig(
                feature_setting="config2",
                unit_primitives=[
                    GitLabUnitPrimitive.DUO_CHAT,
                ],
                default_model="model2",
                selectable_models=["model2"],
                beta_models=["model1"],
            ),
        ]

        yield mock_configs


def test_get_models_returns_correct_data(mock_model_config, client):
    response = client.get("/definitions")

    assert response.status_code == 200
    data = response.json()

    assert data["models"][0] == {"name": "Model 1", "identifier": "model1"}
    assert data["models"][1] == {"name": "Model 2", "identifier": "model2"}
    assert data["unit_primitives"][0] == {
        "feature_setting": "config1",
        "unit_primitives": ["ask_issue", "ask_epic"],
        "default_model": "model1",
        "selectable_models": ["model1", "model2"],
        "beta_models": [],
    }
    assert data["unit_primitives"][1] == {
        "feature_setting": "config2",
        "unit_primitives": ["duo_chat"],
        "default_model": "model2",
        "selectable_models": ["model2"],
        "beta_models": ["model1"],
    }
