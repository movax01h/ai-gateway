from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from gitlab_cloud_connector import GitLabUnitPrimitive

from ai_gateway.api.v1.models.get_definitions import router
from ai_gateway.model_selection import UnitPrimitiveConfig
from ai_gateway.model_selection.model_selection_config import ChatLiteLLMDefinition


@pytest.fixture(name="client")
def client_fixture():
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture(name="mock_model_config")
def mock_model_config_fixture():
    with patch(
        "ai_gateway.api.v1.models.get_definitions.ModelSelectionConfig",
    ) as mock:
        mock_configs = MagicMock()
        mock.return_value = mock_configs
        mock_configs.get_llm_definitions.return_value = {
            "model1": ChatLiteLLMDefinition(
                name="Model 1",
                gitlab_identifier="model1",
                provider="Anthropic",
                max_context_tokens=200000,
                description="Fast, cost-effective responses.",
                cost_indicator="$",
                params={},
            ),
            "model2": ChatLiteLLMDefinition(
                name="Model 2",
                gitlab_identifier="model2",
                provider="Vertex",
                max_context_tokens=200000,
                description="Fast, cost-effective responses.",
                cost_indicator="$$",
                params={},
            ),
            "model3": ChatLiteLLMDefinition(
                name="Model 3",
                gitlab_identifier="model3",
                max_context_tokens=200000,
                params={},
                deprecation={
                    "deprecation_date": "2025-10-28",
                    "removal_version": "18.8",
                },
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
            UnitPrimitiveConfig(
                feature_setting="config3",
                unit_primitives=[
                    GitLabUnitPrimitive.DUO_CHAT,
                ],
                default_model="model3",
                selectable_models=["model3"],
                beta_models=[],
            ),
        ]

        yield mock_configs


def test_get_models_returns_correct_data(mock_model_config, client):
    response = client.get("/definitions")

    assert response.status_code == 200
    data = response.json()

    assert data["models"][0] == {
        "name": "Model 1",
        "identifier": "model1",
        "provider": "Anthropic",
        "deprecation": None,
        "description": "Fast, cost-effective responses.",
        "cost_indicator": "$",
    }
    assert data["models"][1] == {
        "name": "Model 2",
        "identifier": "model2",
        "provider": "Vertex",
        "deprecation": None,
        "description": "Fast, cost-effective responses.",
        "cost_indicator": "$$",
    }
    assert data["models"][2] == {
        "name": "Model 3",
        "identifier": "model3",
        "provider": None,
        "deprecation": {"deprecation_date": "2025-10-28", "removal_version": "18.8"},
        "description": None,
        "cost_indicator": None,
    }

    resp0 = data["unit_primitives"][0]
    expected0 = {
        "feature_setting": "config1",
        "unit_primitives": ["ask_issue", "ask_epic"],
        "default_model": "model1",
        "selectable_models": ["model1", "model2"],
        "beta_models": [],
    }
    # subset match for dict keys/values (tolerates dev_* extras)
    assert expected0.items() <= resp0.items()
    assert set(resp0["unit_primitives"]) == set(expected0["unit_primitives"])

    resp1 = data["unit_primitives"][1]
    expected1 = {
        "feature_setting": "config2",
        "unit_primitives": ["duo_chat"],
        "default_model": "model2",
        "selectable_models": ["model2"],
        "beta_models": ["model1"],
    }

    assert expected1.items() <= resp1.items()
    assert set(resp1["unit_primitives"]) == set(expected1["unit_primitives"])

    resp2 = data["unit_primitives"][2]
    expected2 = {
        "feature_setting": "config3",
        "unit_primitives": ["duo_chat"],
        "default_model": "model3",
        "selectable_models": ["model3"],
        "beta_models": [],
    }
    # subset match for dict keys/values (tolerates dev_* extras)
    assert expected2.items() <= resp2.items()
    assert set(resp2["unit_primitives"]) == set(expected2["unit_primitives"])
