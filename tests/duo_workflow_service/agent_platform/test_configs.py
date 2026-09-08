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
from duo_workflow_service.agent_platform.v1.catalog import (
    CatalogItems,
    WorkspaceAgent,
    bind_catalog_items,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    FlowConfig,
    _default_features_dir,
)
from duo_workflow_service.agent_platform.v1.flows.validation import DryRunFlowValidator
from duo_workflow_service.components.tools_registry import ToolsRegistry

# Legacy root plus moved features' config/ dirs, so a moved flow keeps validation.
# Reuse the loader's own root derivation so this sweep cannot silently diverge.
_FEATURES_DIR = _default_features_dir()
V1_CONFIGS = sorted(FlowConfig.DIRECTORY_PATH.glob("**/*.yml")) + sorted(
    _FEATURES_DIR.glob("*/*/config/*.yml")
)


def _config_id(config_path: Path) -> str:
    flow = (
        config_path.parent.parent.name
        if config_path.parent.name == "config"
        else config_path.parent.name
    )
    return f"{flow}/{config_path.stem}"


# Configs that accept catalog items. Derived from the configs themselves so a
# flow that adds an `include` section is covered without touching this file.
V1_CATALOG_ITEM_CONFIGS = [
    path
    for path in V1_CONFIGS
    if (yaml.safe_load(path.read_text()) or {}).get("include")
]

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
    ``FlowGraphBuilder._parse_toolset``.
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
        ids=_config_id,
    )
    def test_v1_configs(self, config_path: Path):
        self._test_flow_config(config_path)

    @pytest.mark.parametrize(
        "config_path",
        V1_CONFIGS,
        ids=_config_id,
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

    def test_at_least_one_config_declares_include(self):
        """Guard the parametrisation below: an empty list would silently cover nothing."""
        assert V1_CATALOG_ITEM_CONFIGS

    @pytest.mark.parametrize(
        "config_path",
        V1_CATALOG_ITEM_CONFIGS,
        ids=lambda p: f"{p.parent.name}/{p.stem}",
    )
    def test_v1_configs_bind_catalog_items(self, config_path: Path):
        """Items reach the graph: one component per item, claimed by one coordinator.

        Compiling proves the components build, not that they were attached. A config
        that lost its wildcard `subagents` entry drops the items and still compiles
        unless its prompt happens to read `has_workspace_agents`.
        """
        config = FlowConfig(**yaml.safe_load(config_path.read_text()))
        items = CatalogItems(
            workspace_agents=[
                WorkspaceAgent(
                    name="tester", description="Runs tests.", prompt="Be terse."
                )
            ]
        )
        authored = {component["name"] for component in config.components}

        bound = bind_catalog_items(
            config.components, config.include, items, Mock(spec=ToolsRegistry)
        )

        synthesized = [c["name"] for c in bound if c["name"] not in authored]
        assert len(synthesized) == 1

        coordinators = [
            c["name"]
            for c in bound
            if {"name": synthesized[0]} in (c.get("subagents") or [])
        ]
        assert len(coordinators) == 1

    @pytest.mark.parametrize(
        "config_path",
        V1_CATALOG_ITEM_CONFIGS,
        ids=lambda p: f"{p.parent.name}/{p.stem}",
    )
    def test_v1_configs_compile_with_catalog_items(self, config_path: Path):
        """A config declaring `include` must also compile with items attached.

        ``test_v1_configs`` only ever builds the authored components. The synthesized
        subagents and the promoted supervisor exist only once a request carries items,
        so a flow whose prompt cannot build them would otherwise reach production
        unbuilt.
        """
        # Tools the config already declares, so resolution is exercised against names
        # this flow is known to allow.
        toolset = _declared_toolset_names(config_path)[:1]
        items = CatalogItems(
            workspace_agents=[
                WorkspaceAgent(
                    name="tester",
                    description="Runs tests.",
                    toolset=toolset,
                    prompt="Be terse.",
                ),
                WorkspaceAgent(
                    name="reviewer",
                    description="Reviews changes.",
                    prompt="Be picky.",
                ),
            ]
        )

        DryRunFlowValidator(
            config=FlowConfig(**yaml.safe_load(config_path.read_text())),
            prompt_registry=_make_local_prompt_registry(),
            internal_event_client=Mock(),
            catalog_items=items,
        ).validate()

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


def _component(config: FlowConfig, name: str) -> dict:
    return next(
        component for component in config.components if component["name"] == name
    )


class TestFixPipelineConfigVersions:
    @pytest.mark.parametrize("version", ["1.0.0", "1.0.1", "1.0.2"])
    def test_existing_versions_keep_original_bootstrap(self, version: str):
        config = FlowConfig.from_yaml_config("fix_pipeline", version)
        context_component = _component(config, "fix_pipeline_context")

        assert config.flow.entry_point == "fetch_failing_bridge_jobs"
        assert all(
            component["name"] != "fetch_failing_jobs" for component in config.components
        )
        assert context_component["prompt_version"] == "^1.0.0"
        assert all(
            input_["as"] != "failing_jobs" for input_ in context_component["inputs"]
        )

    @pytest.mark.parametrize("version", ["1.0.3", "1.0.4"])
    def test_patch_version_uses_deterministic_failing_jobs_bootstrap(
        self, version: str
    ):
        config = FlowConfig.from_yaml_config("fix_pipeline", version)
        context_component = _component(config, "fix_pipeline_context")

        assert config.flow.entry_point == "fetch_failing_jobs"
        assert _component(config, "fetch_failing_jobs")["tool_name"] == (
            "get_pipeline_failing_jobs"
        )
        assert {
            "from": "context:fetch_failing_jobs.tool_responses",
            "as": "failing_jobs",
        } in context_component["inputs"]
        assert {
            "from": "fetch_failing_jobs",
            "to": "fetch_failing_bridge_jobs",
        } in config.routers

    @pytest.mark.parametrize(
        ("version", "expected_prompt_version"),
        [("1.0.3", "2.0.0"), ("1.0.4", "2.0.1")],
    )
    def test_flow_versions_pin_context_prompt_exactly(
        self, version: str, expected_prompt_version: str
    ):
        """Each flow version must resolve to one prompt version.

        A caret range would let a newer prompt leak into an older flow, so per-version LLM call counts could no longer
        be attributed to the flow version that produced them.
        """
        context_component = _component(
            FlowConfig.from_yaml_config("fix_pipeline", version),
            "fix_pipeline_context",
        )

        assert context_component["prompt_version"] == expected_prompt_version

    @pytest.mark.parametrize(
        ("prompt", "expected_categories_version"),
        [
            ("fix_pipeline_context/system/2.0.1.jinja", "1.0.1"),
            ("fix_pipeline_context/system/2.0.0.jinja", "1.0.0"),
            ("fix_pipeline_context/system/1.0.0.jinja", "1.0.0"),
            ("fix_pipeline_experiment/system/1.0.0.jinja", "1.0.0"),
        ],
    )
    def test_failure_categories_partial_is_versioned_per_prompt(
        self, prompt: str, expected_categories_version: str
    ):
        """The partial is shared, so a new revision must be cut rather than edited in place."""
        include = f"fix_pipeline_failure_categories/{expected_categories_version}.jinja"

        assert include in (Path("ai_gateway/prompts/definitions") / prompt).read_text()

    @pytest.mark.parametrize(
        ("prompt_version", "has_failing_jobs"),
        [("^1.0.0", False), ("2.0.0", True), ("2.0.1", True)],
    )
    def test_prompt_versions_have_expected_contract(
        self, prompt_version: str, has_failing_jobs: bool
    ):
        variables = _make_local_prompt_registry().get_required_variables(
            "fix_pipeline_context", prompt_version=prompt_version
        )

        assert ("failing_jobs" in variables) is has_failing_jobs


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
