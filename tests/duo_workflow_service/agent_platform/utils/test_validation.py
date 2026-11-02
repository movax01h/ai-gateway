from __future__ import annotations

from unittest.mock import Mock

import pytest
import yaml

from ai_gateway.prompts.registry import LocalPromptRegistry
from duo_workflow_service.agent_platform.utils.validation import (
    FlowConfigLimitExceededError,
    FlowValidationError,
    FlowValidator,
)


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


def _build_prompt_registry() -> LocalPromptRegistry:
    return LocalPromptRegistry(
        prompt_template_factories={},
        model_factories={},
        internal_event_client=Mock(),
        model_limits=Mock(),
        custom_models_enabled=False,
    )


@pytest.fixture(name="validator")
def validator_fixture() -> FlowValidator:
    return FlowValidator(prompt_registry=_build_prompt_registry())


def _chain_config(component_count: int) -> dict:
    """A valid v1 ambient flow with ``component_count`` agent components routed in a chain."""
    components = []
    routers = []
    for index in range(component_count):
        name = f"comp{index}"
        components.append(
            {
                "name": name,
                "type": "AgentComponent",
                "prompt_id": "test_prompt",
                "toolset": [],
                "inputs": [{"from": "context:goal", "as": "goal"}],
            }
        )
        next_name = f"comp{index + 1}" if index + 1 < component_count else "end"
        routers.append({"from": name, "to": next_name})
    return {
        "version": "v1",
        "environment": "ambient",
        "flow": {"entry_point": "comp0"},
        "components": components,
        "routers": routers,
        "prompts": [
            {
                "prompt_id": "test_prompt",
                "name": "test_prompt",
                "unit_primitives": ["duo_agent_platform"],
                "prompt_template": {"system": "Hello {{ goal }}"},
            }
        ],
    }


class TestComponentCountLimit:
    @pytest.mark.parametrize(
        ("max_components", "component_count"),
        [(None, 3), (3, 3), (3, 1)],
        ids=["no_limit", "at_limit", "under_limit"],
    )
    def test_within_limit_is_valid(self, max_components, component_count):
        validator = FlowValidator(
            prompt_registry=_build_prompt_registry(), max_components=max_components
        )

        validator.validate_dict(_chain_config(component_count))

    def test_over_limit_is_rejected(self):
        validator = FlowValidator(
            prompt_registry=_build_prompt_registry(), max_components=2
        )

        with pytest.raises(FlowConfigLimitExceededError) as exc_info:
            validator.validate_dict(_chain_config(3))

        assert isinstance(exc_info.value, FlowValidationError)
        assert exc_info.value.errors == [
            "Flow config has 3 components; the maximum is 2"
        ]

    def test_limit_is_checked_before_schema_parsing(self):
        """An oversized config is rejected on the count alone, without paying for Pydantic or compilation."""
        validator = FlowValidator(
            prompt_registry=_build_prompt_registry(), max_components=1
        )
        config = _chain_config(2)
        del config["routers"]  # would fail Pydantic parsing if it got that far

        with pytest.raises(FlowValidationError) as exc_info:
            validator.validate_dict(config)

        assert exc_info.value.errors == [
            "Flow config has 2 components; the maximum is 1"
        ]

    def test_non_list_components_is_left_to_schema_validation(self):
        validator = FlowValidator(
            prompt_registry=_build_prompt_registry(), max_components=1
        )
        config = _chain_config(1)
        config["components"] = "not-a-list"

        with pytest.raises(FlowValidationError) as exc_info:
            validator.validate_dict(config)

        assert all("maximum" not in err for err in exc_info.value.errors)


class TestToolsetOptional:
    @pytest.mark.parametrize("environment", ["ambient", "chat-partial"])
    def test_omitted_toolset_is_valid(self, validator: FlowValidator, environment: str):
        yml_dict = yaml.safe_load(_build_yaml())
        del yml_dict["components"][0]["toolset"]
        yml_dict["environment"] = environment
        if environment == "chat-partial":
            del yml_dict["flow"]
            del yml_dict["routers"]

        validator.validate(yaml.dump(yml_dict))


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
