import re
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from ai_gateway.prompts.registry import LocalPromptRegistry
from duo_workflow_service.agent_platform.utils.validation import (
    ExtraInputVariablesError,
    FlowValidator,
    MissingInputVariablesError,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig

V1_CONFIGS = sorted(FlowConfig.DIRECTORY_PATH.glob("**/*.yml"))

TOOL_NAME_PATTERN = re.compile(r"[a-z0-9_]+")


def _make_local_prompt_registry() -> LocalPromptRegistry:
    return LocalPromptRegistry(
        prompt_template_factories={},
        model_factories={},
        internal_event_client=Mock(),
        model_limits=Mock(),
        custom_models_enabled=False,
    )


def _declared_toolset_names(config_path: Path) -> list[str]:
    """Collect every tool name referenced by a config's component ``toolset`` lists.

    Entries may be plain strings or single-key ``{"tool_name": {options}}`` mappings, matching
    ``Flow._parse_toolset``.
    """
    config = yaml.safe_load(config_path.read_text()) or {}

    names: list[str] = []
    for component in config.get("components") or []:
        for entry in component.get("toolset") or []:
            if isinstance(entry, dict):
                names.extend(entry.keys())
            else:
                names.append(entry)

    return names


class TestValidateFlowConfigs:
    @pytest.mark.parametrize(
        "config_path",
        V1_CONFIGS,
        ids=lambda p: f"{p.parent.name}/{p.stem}",
    )
    def test_v1_configs(self, config_path: Path):
        self._test_flow_config(config_path)

    @pytest.mark.parametrize(
        "config_path",
        V1_CONFIGS,
        ids=lambda p: f"{p.parent.name}/{p.stem}",
    )
    def test_v1_config_toolset_entries_are_well_formed(self, config_path: Path):
        """Every ``toolset`` entry must be a syntactically valid tool identifier.

        A comma-less YAML flow sequence::

            toolset: [
              read_file
              grep
            ]

        is folded by YAML into a *single* string, ``"read_file grep"``. ``ToolsRegistry.toolset``
        then skips the unrecognised name without raising, so the agent silently starts with no
        tools at all. Nothing else catches this: ``components`` is typed as ``list[dict]`` with no
        per-entry schema, and ``chat-partial`` configs bypass dry-run validation entirely.

        This asserts only on the *shape* of each name. Names are not checked against the tool
        registry because MCP-provided tools (for example ``orbit_*``) are registered at runtime and
        are legitimately absent from the static registry.
        """
        malformed = [
            name
            for name in _declared_toolset_names(config_path)
            if not TOOL_NAME_PATTERN.fullmatch(name)
        ]

        assert not malformed, (
            f"{config_path.parent.name}/{config_path.stem} has malformed toolset entries: "
            f"{malformed}. Tool names must match {TOOL_NAME_PATTERN.pattern}. An entry containing "
            f"spaces usually means the toolset was written as a comma-less YAML flow sequence "
            f"(`toolset: [a\\n b]`), which folds into one string — use a block sequence instead."
        )

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


def _write_config(tmp_path: Path, config) -> Path:
    config_path = tmp_path / "flow.yml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


class TestDeclaredToolsetNames:
    """Direct coverage for ``_declared_toolset_names``.

    The sweep above only ever sees the entry forms that shipped configs happen to use. No config
    currently uses the single-key mapping form, so the ``isinstance(entry, dict)`` branch — and the
    malformed-name path for mapping keys — would otherwise never execute, and a regression in either
    would go unnoticed until a config started using that form.
    """

    @pytest.mark.parametrize(
        "components,expected",
        [
            pytest.param(
                [{"toolset": ["read_file", "grep"]}],
                ["read_file", "grep"],
                id="string_entries",
            ),
            pytest.param(
                [{"toolset": [{"read_file": {"max_bytes": 1024}}, {"grep": {}}]}],
                ["read_file", "grep"],
                id="mapping_entries",
            ),
            pytest.param(
                [{"toolset": ["read_file", {"grep": {"flags": "-i"}}]}],
                ["read_file", "grep"],
                id="mixed_entries",
            ),
            pytest.param(
                [{"toolset": ["read_file"]}, {"toolset": [{"grep": {}}]}],
                ["read_file", "grep"],
                id="across_components",
            ),
            pytest.param([{"name": "no_toolset_key"}], [], id="toolset_absent"),
            pytest.param([{"toolset": None}], [], id="toolset_null"),
            pytest.param([], [], id="no_components"),
            pytest.param(None, [], id="components_null"),
        ],
    )
    def test_collects_names_from_both_entry_forms(
        self, tmp_path: Path, components, expected
    ):
        config_path = _write_config(tmp_path, {"components": components})

        assert _declared_toolset_names(config_path) == expected

    def test_empty_document_yields_no_names(self, tmp_path: Path):
        """A file that parses to ``None`` must not raise, exercising the ``or {}`` fallback."""
        config_path = tmp_path / "flow.yml"
        config_path.write_text("")

        assert _declared_toolset_names(config_path) == []

    @pytest.mark.parametrize(
        "entry",
        [
            pytest.param("read_file grep", id="folded_string_entry"),
            pytest.param({"read_file grep": {}}, id="folded_mapping_key"),
        ],
    )
    def test_folded_entries_are_detected_as_malformed(self, tmp_path: Path, entry):
        """A comma-less flow sequence folds into one space-separated name, in either entry form."""
        config_path = _write_config(tmp_path, {"components": [{"toolset": [entry]}]})

        names = _declared_toolset_names(config_path)

        assert [name for name in names if not TOOL_NAME_PATTERN.fullmatch(name)] == [
            "read_file grep"
        ]

    @pytest.mark.parametrize(
        "entry",
        [
            pytest.param("read_file", id="string_entry"),
            pytest.param({"read_file": {"max_bytes": 1024}}, id="mapping_entry"),
        ],
    )
    def test_well_formed_entries_are_not_flagged(self, tmp_path: Path, entry):
        config_path = _write_config(tmp_path, {"components": [{"toolset": [entry]}]})

        names = _declared_toolset_names(config_path)

        assert names == ["read_file"]
        assert all(TOOL_NAME_PATTERN.fullmatch(name) for name in names)


class TestFixPipelineConfig:
    @staticmethod
    def _config() -> FlowConfig:
        return FlowConfig.from_yaml_config("fix_pipeline", "1.0.2")

    def test_merge_request_author_id_is_optional(self):
        config = self._config()
        schema = config.input_json_schemas_by_category()["merge_request"]

        assert schema["properties"]["author_id"] == {
            "type": "string",
            "description": "ID of the Merge Request author",
        }
        assert "author_id" not in schema["required"]

    def test_create_new_mr_receives_merge_request_author_id(self):
        config = self._config()
        component = next(
            component
            for component in config.components
            if component["name"] == "fix_pipeline_create_new_mr"
        )

        assert {
            "from": "context:inputs.merge_request.author_id",
            "as": "merge_request_author_id",
            "optional": True,
        } in component["inputs"]

    def test_changed_prompts_use_exact_versions(self):
        config = self._config()
        components = {component["name"]: component for component in config.components}

        assert components["fix_pipeline_create_new_mr"]["prompt_version"] == "1.0.2"
        assert components["fix_pipeline_new_mr_comment"]["prompt_version"] == "1.0.2"

    def test_new_mr_comment_reuses_existing_context_without_user_lookup(self):
        config = self._config()
        component = next(
            component
            for component in config.components
            if component["name"] == "fix_pipeline_new_mr_comment"
        )

        assert component["toolset"] == ["create_merge_request_note"]
        assert {
            "from": "context:inputs.merge_request.url",
            "as": "merge_request_url",
            "optional": True,
        } in component["inputs"]
        assert not any(
            component_input["as"] == "session_owner_id"
            for component_input in component["inputs"]
        )

    @pytest.mark.parametrize("version", ["1.0.0", "1.0.1"])
    def test_historical_flows_pin_changed_prompts(self, version: str):
        config = FlowConfig.from_yaml_config("fix_pipeline", version)
        components = {component["name"]: component for component in config.components}

        assert components["fix_pipeline_create_new_mr"]["prompt_version"] == "1.0.0"
        assert components["fix_pipeline_new_mr_comment"]["prompt_version"] == "1.0.0"
