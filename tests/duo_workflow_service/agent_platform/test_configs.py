from pathlib import Path
from unittest.mock import Mock

import pytest

from ai_gateway.prompts.registry import LocalPromptRegistry
from duo_workflow_service.agent_platform.utils.validation import (
    ExtraInputVariablesError,
    FlowValidator,
    MissingInputVariablesError,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig

V1_CONFIGS = sorted(FlowConfig.DIRECTORY_PATH.glob("**/*.yml"))


def _make_local_prompt_registry() -> LocalPromptRegistry:
    return LocalPromptRegistry(
        prompt_template_factories={},
        model_factories={},
        internal_event_client=Mock(),
        model_limits=Mock(),
        custom_models_enabled=False,
    )


class TestValidateFlowConfigs:
    @pytest.mark.parametrize(
        "config_path",
        V1_CONFIGS,
        ids=lambda p: f"{p.parent.name}/{p.stem}",
    )
    def test_v1_configs(self, config_path: Path):
        self._test_flow_config(config_path)

    @staticmethod
    def _test_flow_config(config_path: Path):
        yaml_content = config_path.read_text()
        registry = _make_local_prompt_registry()
        validator = FlowValidator(prompt_registry=registry)

        error = None
        try:
            validator.validate(yaml_content)
        except (
            MissingInputVariablesError,
            ExtraInputVariablesError,
            ValueError,
        ) as exc:
            error = exc

        if error is not None:
            pytest.fail(f"validate_flow raised:\n{error}", pytrace=False)
