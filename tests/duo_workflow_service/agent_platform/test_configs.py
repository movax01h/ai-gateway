import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast, get_args
from unittest.mock import Mock

import pytest
import yaml
from pydantic import ValidationError

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
    MCP_AUTO_INJECT_ENVIRONMENTS,
    FlowConfig,
    FlowConfigMetadata,
    _default_features_dir,
)
from duo_workflow_service.agent_platform.v1.flows.validation import (
    DryRunFlowValidator,
    _make_validation_tools_registry,
)
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

MCP_TOOL_NAME_PREFIXES = ("orbit_",)

# Read off the model rather than restated here, so adding a flow environment automatically
# extends the tests below instead of leaving the new value silently uncovered.
FLOW_ENVIRONMENTS: frozenset[str] = frozenset(
    get_args(FlowConfig.model_fields["environment"].annotation)
)
NON_MCP_ENVIRONMENTS: frozenset[str] = FLOW_ENVIRONMENTS - MCP_AUTO_INJECT_ENVIRONMENTS


def _make_local_prompt_registry() -> LocalPromptRegistry:
    return LocalPromptRegistry(
        prompt_template_factories={},
        model_factories={},
        internal_event_client=Mock(),
        model_limits=Mock(),
        custom_models_enabled=False,
    )


def _load_flow_config(config_path: Path) -> FlowConfig:
    """Parse a config file into the model the flow platform itself uses.

    Deliberately not ``from_yaml_config``: that resolves ``(flow_id, version)`` against
    ``DIRECTORY_PATH``, which neither the synthetic configs in these tests nor a config living
    outside that root can satisfy. Constructing the model directly keeps the sweep working on
    whatever path it was handed, while still going through ``FlowConfig`` validation rather than
    reading loose YAML.
    """
    return FlowConfig(**(yaml.safe_load(config_path.read_text()) or {}))


def _iter_entry_names(entries: Any) -> Iterator[str]:
    """Yield each tool name in a ``toolset`` or ``pre_approved_tools`` list.

    An entry is either a plain string or a single-key ``{"tool_name": {options}}`` mapping
    (``FlowGraphBuilder._parse_toolset``, ``strip_ask_listed_pre_approvals``).
    """
    for entry in entries or []:
        if isinstance(entry, dict):
            yield from entry
        else:
            yield entry


def _iter_declared_tool_names(config: FlowConfig) -> Iterator[tuple[str, str]]:
    """Yield ``(component_name, tool_name)`` for every tool a config names.

    Reads ``FlowConfig.components`` rather than raw YAML, so a change to how a config is parsed is
    absorbed by the model instead of silently bypassing this sweep. Component bodies are still
    ``dict`` — the model does not type them — so the three key paths below have to be walked by
    hand, matching the three branches of ``FlowGraphBuilder._build_component_params``:

    * ``toolset`` — entries are plain strings or single-key ``{"tool_name": {options}}``
      mappings (``FlowGraphBuilder._parse_toolset``);
    * ``pre_approved_tools`` — same entry forms (``strip_ask_listed_pre_approvals``);
    * ``tool_name`` — the scalar tool of a ``DeterministicStepComponent``.

    Collecting only ``toolset`` would leave the other two paths unchecked.
    """
    for component in config.components:
        component_name = component.get("name") or "<unnamed>"

        for key in ("toolset", "pre_approved_tools"):
            for name in _iter_entry_names(component.get(key)):
                yield component_name, name

        tool_name = component.get("tool_name")
        if tool_name:
            yield component_name, tool_name


def _declared_tool_names(config: FlowConfig) -> list[str]:
    return [name for _, name in _iter_declared_tool_names(config)]


def _declared_toolset_names(config: FlowConfig) -> list[str]:
    """Only the names a component lists under ``toolset``.

    Deliberately narrower than :func:`_declared_tool_names`: a deterministic step resolves its own
    scalar ``tool_name``, which is not necessarily a tool an agent in the same flow is granted, so
    it is not a safe toolset to hand a synthesized catalog agent.
    """
    return [
        name
        for component in config.components
        for name in _iter_entry_names(component.get("toolset"))
    ]


def _is_runtime_injected(tool_name: str) -> bool:
    """Whether a name belongs to a tool family registered at runtime rather than statically.

    MCP tools are supplied per-session by Rails and so are absent from a statically built registry. Only flows that opt
    into MCP auto-injection may use them.
    """
    return tool_name.startswith(MCP_TOOL_NAME_PREFIXES)


def _make_config(components: list[dict], environment: str = "ambient") -> FlowConfig:
    """Build a minimal valid ``FlowConfig`` for tests that need a specific component shape.

    ``environment`` is taken as ``str`` because callers parametrize over ``FLOW_ENVIRONMENTS``,
    which is derived from the model at runtime and so is not a static ``Literal`` to mypy.
    """
    return FlowConfig(
        flow=FlowConfigMetadata(entry_point="a_component"),
        components=components,
        routers=[],
        environment=cast(Any, environment),
        version="v1",
    )


def _unresolved_tool_names(
    config: FlowConfig, tools_registry: ToolsRegistry
) -> list[tuple[str, str]]:
    """Return ``(component_name, tool_name)`` for each named tool that resolves to nothing.

    Exactly one name is excused: a runtime-injected tool in a flow whose ``environment`` opts into
    MCP auto-injection, which is supplied per session and so is expected to be absent from a
    statically built registry.

    Everything else is a tool that exists nowhere.
    """
    allows_mcp_tools = config.environment in MCP_AUTO_INJECT_ENVIRONMENTS

    return [
        (component_name, tool_name)
        for component_name, tool_name in _iter_declared_tool_names(config)
        if tools_registry.get(tool_name) is None
        and not (allows_mcp_tools and _is_runtime_injected(tool_name))
    ]


@pytest.fixture(scope="module", name="tools_registry")
def tools_registry_fixture() -> ToolsRegistry:
    """The registry the dry-run flow validator builds: all privileges, all capabilities, no I/O.

    Reused across every config so the tool instances are constructed once. Membership in this
    registry is the definition of "this tool exists" — see
    ``test_v1_config_tool_names_are_defined_in_the_registry``.
    """
    return _make_validation_tools_registry()


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

        This asserts only on the *shape* of each name.
        ``test_v1_config_tool_names_are_defined_in_the_registry`` checks the names themselves.
        """
        malformed = [
            name
            for name in _declared_tool_names(_load_flow_config(config_path))
            if not TOOL_NAME_PATTERN.fullmatch(name)
        ]

        assert not malformed, (
            f"{_config_id(config_path)} declares malformed tool names: "
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
    def test_v1_configs_bind_catalog_items(
        self, config_path: Path, workspace_agents_flag
    ):
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
    def test_v1_configs_compile_with_catalog_items(
        self, config_path: Path, workspace_agents_flag
    ):
        """A config declaring `include` must also compile with items attached.

        ``test_v1_configs`` only ever builds the authored components. The synthesized
        subagents and the promoted supervisor exist only once a request carries items,
        so a flow whose prompt cannot build them would otherwise reach production
        unbuilt.
        """
        # Tools the config already declares, so resolution is exercised against names
        # this flow is known to allow.
        config = _load_flow_config(config_path)
        toolset = _declared_toolset_names(config)[:1]
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
            config=config,
            prompt_registry=_make_local_prompt_registry(),
            internal_event_client=Mock(),
            catalog_items=items,
        ).validate()

    @pytest.mark.parametrize(
        "config_path",
        V1_CONFIGS,
        ids=_config_id,
    )
    def test_v1_config_tool_names_are_defined_in_the_registry(
        self, config_path: Path, tools_registry: ToolsRegistry
    ):
        """Every tool a config names must be a tool that actually exists.

        ``ToolsRegistry.toolset`` skips a name it does not recognise instead of raising, so a
        misspelled tool is dropped at flow-build time with no error and no warning — the flow ships
        and runs, just without that capability. That is how ``security_review`` ran for months
        declaring ``blob_search`` when the registered name is ``gitlab_blob_search``
        (gitlab-org/gitlab#627166), and how ``project_activity`` came to declare a
        ``gitlab_api_post`` that was never written (#2802, fixed by !6789 — this sweep is what
        found it).

        Nothing else catches this class of typo. ``DeterministicStepComponent`` does raise on an
        unresolvable ``tool_name``, but ``AgentComponent`` performs no declared-vs-resolved check,
        and dry-run validation does not compare declared names against the registry.

        The registry here is the one ``DryRunFlowValidator`` uses: every agent privilege granted
        and every client capability enabled, so an unresolved name means "no such tool anywhere",
        never "this tool was not granted to this flow".
        """
        unresolved = _unresolved_tool_names(
            _load_flow_config(config_path), tools_registry
        )

        offenders = ", ".join(
            f"{tool_name!r} (component {component_name!r})"
            for component_name, tool_name in unresolved
        )

        assert not unresolved, (
            f"{_config_id(config_path)} names tools that are not defined in the registry: "
            f"{offenders}. "
            f"Each is silently dropped at flow-build time, so the component runs without it. "
            f"Fix the spelling to match the tool's registered `name`. If instead this is an "
            f"MCP tool registered at runtime, add its prefix to MCP_TOOL_NAME_PREFIXES — and "
            f"note that only flows whose environment is one of "
            f"{sorted(MCP_AUTO_INJECT_ENVIRONMENTS)} may use one."
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


class TestRuntimeInjectedToolExemption:
    """Direct coverage for the escape hatch that lets MCP tool names go unresolved.

    The sweep exercises this branch only through whichever shipped configs happen to declare
    ``orbit_*`` tools today — currently ``orbit_agent`` and ``analytics_agent``. If those configs
    are renamed, retired, or moved to a different environment, the branch would keep passing while
    testing nothing. These synthetic configs pin the contract regardless of what ships.
    """

    UNKNOWN_MCP_TOOL = "orbit_query_graph"
    UNKNOWN_PLAIN_TOOL = "read_fil"
    DEFINED_TOOL = "read_file"

    @staticmethod
    def _config(environment: str, tool_name: str) -> FlowConfig:
        return _make_config(
            [{"name": "a_component", "toolset": [tool_name]}], environment=environment
        )

    @pytest.mark.parametrize("environment", sorted(MCP_AUTO_INJECT_ENVIRONMENTS))
    def test_mcp_name_is_excused_in_a_flow_that_injects_mcp_tools(
        self, tools_registry: ToolsRegistry, environment: str
    ):
        config = self._config(environment, self.UNKNOWN_MCP_TOOL)

        assert _unresolved_tool_names(config, tools_registry) == []

    @pytest.mark.parametrize("environment", sorted(NON_MCP_ENVIRONMENTS))
    def test_mcp_name_is_still_flagged_in_a_flow_that_does_not(
        self, tools_registry: ToolsRegistry, environment: str
    ):
        """The exemption is scoped to the environment, not granted to the prefix globally."""
        config = self._config(environment, self.UNKNOWN_MCP_TOOL)

        assert _unresolved_tool_names(config, tools_registry) == [
            ("a_component", self.UNKNOWN_MCP_TOOL)
        ]

    @pytest.mark.parametrize("environment", sorted(FLOW_ENVIRONMENTS))
    def test_ordinary_typo_is_flagged_in_every_environment(
        self, tools_registry: ToolsRegistry, environment: str
    ):
        """Opting into MCP injection must not exempt a flow's non-MCP tool names."""
        config = self._config(environment, self.UNKNOWN_PLAIN_TOOL)

        assert _unresolved_tool_names(config, tools_registry) == [
            ("a_component", self.UNKNOWN_PLAIN_TOOL)
        ]

    @pytest.mark.parametrize("environment", sorted(FLOW_ENVIRONMENTS))
    def test_a_real_tool_is_never_flagged(
        self, tools_registry: ToolsRegistry, environment: str
    ):
        """Positive control: the check must not simply flag everything it is shown."""
        config = self._config(environment, self.DEFINED_TOOL)

        assert _unresolved_tool_names(config, tools_registry) == []

    def test_environment_sets_are_derived_from_the_model(self):
        """The split must stay exhaustive, or a whole environment goes untested in silence.

        Both halves are asserted non-empty because both are parametrize sources in this class: an
        empty one collects zero cases and reports as a pass, so the set emptying out is precisely
        the failure this guard has to catch.
        """
        assert MCP_AUTO_INJECT_ENVIRONMENTS
        assert NON_MCP_ENVIRONMENTS
        assert MCP_AUTO_INJECT_ENVIRONMENTS <= FLOW_ENVIRONMENTS
        assert NON_MCP_ENVIRONMENTS | MCP_AUTO_INJECT_ENVIRONMENTS == FLOW_ENVIRONMENTS

    @pytest.mark.parametrize(
        ("tool_name", "expected"),
        [
            pytest.param("orbit_query_graph", True, id="mcp_prefix"),
            pytest.param("orbit_", True, id="bare_prefix"),
            pytest.param("read_file", False, id="ordinary_tool"),
            pytest.param("gitlab_api_post", False, id="unknown_ordinary_tool"),
            pytest.param("my_orbit_tool", False, id="prefix_must_be_leading"),
            pytest.param("Orbit_query_graph", False, id="prefix_is_case_sensitive"),
        ],
    )
    def test_is_runtime_injected(self, tool_name: str, expected: bool):
        assert _is_runtime_injected(tool_name) is expected


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


class TestDeclaredToolNames:
    """Direct coverage for ``_declared_tool_names``.

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
            pytest.param(
                [{"tool_name": "get_pipeline_failing_jobs"}],
                ["get_pipeline_failing_jobs"],
                id="deterministic_step_tool_name",
            ),
            pytest.param([{"tool_name": None}], [], id="tool_name_null"),
            pytest.param(
                [{"pre_approved_tools": ["todo_write", {"grep": {"flags": "-i"}}]}],
                ["todo_write", "grep"],
                id="pre_approved_tools",
            ),
            pytest.param(
                [{"pre_approved_tools": None}], [], id="pre_approved_tools_null"
            ),
            pytest.param(
                [
                    {
                        "toolset": ["read_file"],
                        "pre_approved_tools": ["todo_write"],
                        "tool_name": "grep",
                    }
                ],
                ["read_file", "todo_write", "grep"],
                id="all_three_key_paths",
            ),
        ],
    )
    def test_collects_names_from_every_key_path(self, components, expected):
        assert _declared_tool_names(_make_config(components)) == expected

    @pytest.mark.parametrize(
        ("document", "missing"),
        [
            pytest.param("", "flow", id="empty_document"),
            pytest.param("components: null\n", "flow", id="components_null"),
            pytest.param(
                "flow: {}\nrouters: []\nenvironment: ambient\nversion: v1\n",
                "components",
                id="components_absent",
            ),
        ],
    )
    def test_malformed_documents_are_rejected_by_the_model(
        self, tmp_path: Path, document: str, missing: str
    ):
        """A config that cannot be a flow fails loudly at parse time.

        Previously these produced an empty tool list and the sweep passed vacuously. Routing
        through ``FlowConfig`` turns them into a validation error naming the missing field.
        """
        config_path = tmp_path / "flow.yml"
        config_path.write_text(document)

        with pytest.raises(ValidationError) as excinfo:
            _load_flow_config(config_path)

        assert missing in str(excinfo.value)

    @pytest.mark.parametrize(
        "entry",
        [
            pytest.param("read_file grep", id="folded_string_entry"),
            pytest.param({"read_file grep": {}}, id="folded_mapping_key"),
        ],
    )
    def test_folded_entries_are_detected_as_malformed(self, entry):
        """A comma-less flow sequence folds into one space-separated name, in either entry form."""
        names = _declared_tool_names(_make_config([{"toolset": [entry]}]))

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
    def test_well_formed_entries_are_not_flagged(self, entry):
        names = _declared_tool_names(_make_config([{"toolset": [entry]}]))

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
