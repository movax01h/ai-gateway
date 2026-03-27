from __future__ import annotations

from unittest.mock import Mock

import pytest
import yaml

from ai_gateway.prompts.registry import LocalPromptRegistry
from duo_workflow_service.agent_platform.utils.validation import FlowValidator


def _build_yaml(
    inputs: list[dict] | None = None,
    template_vars: str = "Hello {{ goal }}",
) -> str:
    if inputs is None:
        inputs = [{"from": "context:goal", "as": "goal"}]

    return yaml.dump(
        {
            "version": "v1",
            "environment": "ambient",
            "flow": {"entry_point": "comp"},
            "components": [
                {
                    "name": "comp",
                    "type": "AgentComponent",
                    "prompt_id": "test_prompt",
                    "toolset": [],
                    "inputs": inputs,
                }
            ],
            "routers": [{"from": "comp", "to": "end"}],
            "prompts": [
                {
                    "prompt_id": "test_prompt",
                    "name": "test_prompt",
                    "unit_primitives": ["duo_agent_platform"],
                    "prompt_template": {"system": template_vars},
                }
            ],
        }
    )


@pytest.fixture(name="validator")
def validator_fixture() -> FlowValidator:
    return FlowValidator(
        prompt_registry=LocalPromptRegistry(
            prompt_template_factories={},
            model_factories={},
            internal_event_client=Mock(),
            model_limits=Mock(),
            custom_models_enabled=False,
        )
    )


class TestFlowConfigStructure:
    def test_yaml_syntax_error(self, validator: FlowValidator):
        invalid_yaml = "key: [unclosed bracket"
        with pytest.raises(yaml.YAMLError):
            validator.validate(invalid_yaml)

    def test_missing_version(self, validator: FlowValidator):
        yml = _build_yaml()
        yml_dict = yaml.safe_load(yml)
        del yml_dict["version"]
        yml_without_version = yaml.dump(yml_dict)
        with pytest.raises(ValueError, match="Missing required field 'version'"):
            validator.validate(yml_without_version)

    def test_missing_environment(self, validator: FlowValidator):
        yml = _build_yaml()
        yml_dict = yaml.safe_load(yml)
        del yml_dict["environment"]
        yml_without_environment = yaml.dump(yml_dict)
        with pytest.raises(ValueError, match="Missing required field 'environment'"):
            validator.validate(yml_without_environment)


class TestPromptVariableValidation:
    def test_happy_path(self, validator: FlowValidator):
        yml = _build_yaml(
            inputs=[{"from": "context:goal", "as": "goal"}],
            template_vars="{{ goal }}",
        )
        validator.validate(yml)

    def test_missing_input_variable(self, validator: FlowValidator):
        yml = _build_yaml(
            inputs=[],
            template_vars="{{ goal }}",
        )
        with pytest.raises(ValueError, match="missing input variables"):
            validator.validate(yml)

    def test_extra_input_variable(self, validator: FlowValidator):
        yml = _build_yaml(
            inputs=[
                {"from": "context:goal", "as": "goal"},
                {"from": "context:extra", "as": "extra"},
            ],
            template_vars="{{ goal }}",
        )
        with pytest.raises(ValueError, match="extra input variables"):
            validator.validate(yml)
