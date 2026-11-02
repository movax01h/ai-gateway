# pylint: disable=file-naming-for-tests
"""Tests for ai_gateway.prompts.config.base module."""

from pathlib import Path

import pytest
import yaml
from gitlab_cloud_connector import GitLabUnitPrimitive

from ai_gateway.model_selection.models import CompletionLiteLLMParams, CompletionType
from ai_gateway.prompts.config.base import InMemoryPromptConfig, PromptConfig


class TestInMemoryPromptConfig:
    """Test InMemoryPromptConfig class functionality."""

    def test_inmemory_prompt_config_creation(self):
        """Test that guards stable API of InMemoryPromptConfig.

        The InMemoryPromptConfig is exposed via gRPC public API, therefore any change to its structure must be backward
        compatible!
        """
        config_data = {
            "prompt_id": "test/prompt",
            "name": "test_prompt",
            "model": {
                "params": {
                    "model": "claude-3-sonnet",
                }
            },
            "unit_primitives": [GitLabUnitPrimitive.DUO_CHAT],
            "prompt_template": {
                "system": "You are a helpful assistant.",
                "user": "{{user_input}}",
            },
        }

        config = InMemoryPromptConfig(**config_data)

        assert config.prompt_id == "test/prompt"
        assert config.name == "test_prompt"
        assert config.model.params.model == "claude-3-sonnet"
        assert config.unit_primitives == [GitLabUnitPrimitive.DUO_CHAT]
        assert config.prompt_template == {
            "system": "You are a helpful assistant.",
            "user": "{{user_input}}",
        }

    @pytest.mark.parametrize(
        "unit_primitives,expected_unit_primitive",
        [
            ([GitLabUnitPrimitive.DUO_CHAT], GitLabUnitPrimitive.DUO_CHAT),
            ([], GitLabUnitPrimitive.DUO_AGENT_PLATFORM),
        ],
    )
    @pytest.mark.parametrize(
        "extra_config,expected_operation_type",
        [
            ({}, "standard"),
            ({"operation_type": "compaction_auto"}, "compaction_auto"),
        ],
        ids=["default_operation_type", "explicit_compaction_auto"],
    )
    def test_to_prompt_data(
        self,
        unit_primitives: list[GitLabUnitPrimitive],
        expected_unit_primitive: GitLabUnitPrimitive,
        extra_config: dict,
        expected_operation_type: str,
    ):
        """Test converting InMemoryPromptConfig to PromptConfig."""
        config_data = {
            "prompt_id": "test/prompt",
            "name": "test_prompt",
            "model": {
                "params": {
                    "model": "claude-3-sonnet",
                }
            },
            "unit_primitives": unit_primitives,
            "prompt_template": {
                "system": "You are a helpful assistant.",
                "user": "{{user_input}}",
            },
            "params": {
                "stop": ["Human:", "Assistant:"],
                "timeout": 30.0,
            },
            **extra_config,
        }

        in_memory_config = InMemoryPromptConfig.model_validate(config_data)
        prompt_data = in_memory_config.to_prompt_data()

        assert isinstance(prompt_data, dict)
        assert prompt_data["operation_type"] == expected_operation_type

        prompt_config = PromptConfig(**prompt_data)

        # Should be a PromptConfig instance
        assert isinstance(prompt_config, PromptConfig)
        assert not isinstance(prompt_config, InMemoryPromptConfig)

        # Should have all relevant fields
        assert prompt_config.name == "test_prompt"
        assert prompt_config.model.params.model == "claude-3-sonnet"
        assert prompt_config.unit_primitive == expected_unit_primitive
        assert prompt_config.operation_type == expected_operation_type
        assert prompt_config.prompt_template == {
            "system": "You are a helpful assistant.",
            "user": "{{user_input}}",
        }
        assert prompt_config.params
        assert prompt_config.params.stop == ["Human:", "Assistant:"]
        assert prompt_config.params.timeout == 30.0

        # Should not have excluded fields
        assert not hasattr(prompt_config, "prompt_id")
        assert not hasattr(prompt_config, "unit_primitives")


class TestPromptConfig:
    _repo_root = Path(__file__).parents[4]

    @staticmethod
    def _config(**fields) -> PromptConfig:
        return PromptConfig(
            name="Test", prompt_template={"user": "{{ input }}"}, **fields
        )

    def test_models_from_is_optional(self):
        config = self._config(unit_primitive=GitLabUnitPrimitive.ASK_ISSUE)

        assert config.models_from is None

    def test_accepts_models_from_that_lists_the_unit_primitive(self):
        config = self._config(
            unit_primitive=GitLabUnitPrimitive.ASK_ISSUE, models_from="duo_chat"
        )

        assert config.models_from == "duo_chat"

    @pytest.mark.parametrize(
        ("models_from", "message"),
        [
            ("no_such_feature", "is not a feature setting in unit_primitives.yml"),
            ("generate_commit_message", "does not list the unit primitive 'ask_issue'"),
        ],
    )
    def test_rejects_models_from_that_does_not_list_the_unit_primitive(
        self, models_from, message
    ):
        with pytest.raises(ValueError, match=message):
            self._config(
                unit_primitive=GitLabUnitPrimitive.ASK_ISSUE, models_from=models_from
            )

    def test_every_shipped_prompt_version_is_valid(self):
        version_files = [
            *(self._repo_root / "ai_gateway" / "prompts" / "definitions").rglob(
                "*.yml"
            ),
            *(self._repo_root / "ai").glob("features/*/*/prompts/**/*.yml"),
            *(self._repo_root / "ai").glob("shared/*/prompts/**/*.yml"),
        ]

        invalid = []
        for version_file in sorted(version_files):
            try:
                PromptConfig(**yaml.safe_load(version_file.read_text()))
            except ValueError as error:
                invalid.append(f"{version_file.relative_to(self._repo_root)}: {error}")

        assert version_files
        assert not invalid


class TestCompletionLiteLLMParams:
    def test_validate_fim_format_requires_format(self):
        with pytest.raises(ValueError, match="fim_format is required"):
            CompletionLiteLLMParams(
                model="codestral-2501",
                completion_type=CompletionType.FIM,
            )

    def test_validate_fim_format_accepts_format(self):
        params = CompletionLiteLLMParams(
            model="codestral-2501",
            completion_type=CompletionType.FIM,
            fim_format="</s>[SUFFIX]{suffix}[PREFIX]{prefix}[MIDDLE]",
        )

        assert params.fim_format == "</s>[SUFFIX]{suffix}[PREFIX]{prefix}[MIDDLE]"
