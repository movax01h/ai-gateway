# These tests cover a YAML flow config, so there is no module to name the file after.
# pylint: disable=file-naming-for-tests
"""Guards for the shipped ``bl_security/1.0.0.yml`` flow config.

Generic ``FlowConfig`` behaviour is covered in ``test_flow_config.py``. These
tests cover what this config wires together: the ``scan_effort`` dial and its
fallbacks, the discovery dials, the stage graph and toolsets, the answer
schemas the pipeline parses, and the joint between the prompts' declared scope
and ``bl_report.IN_SCOPE_CWES``. Prompt wording is not pinned; only the triage
schema descriptions that steer the verdict out of ``reasoning`` are.
"""

import asyncio
import functools
import json
import re
from typing import Any
from unittest.mock import Mock

import pytest
import yaml
from pydantic import ValidationError as PydanticValidationError

import duo_workflow_service.tools.bl_discovery as bl
import duo_workflow_service.tools.bl_report as report
from ai_gateway.prompts import InMemoryPromptRegistry
from ai_gateway.prompts.registry import LocalPromptRegistry
from ai_gateway.response_schemas import InlineResponseSchemaRegistry
from ai_gateway.response_schemas.converter import json_schema_to_pydantic
from duo_workflow_service.agent_platform.experimental.components.agent.component import (
    AgentComponent,
    AgentComponentBase,
)
from duo_workflow_service.agent_platform.experimental.components.for_each import (
    PUBLISHED_SUBKEYS,
)
from duo_workflow_service.agent_platform.experimental.components.for_each.config import (
    ForEachConfig,
)
from duo_workflow_service.agent_platform.experimental.flows.flow_config import (
    FlowConfig,
)
from duo_workflow_service.agent_platform.experimental.state import IOKey, IOKeyTemplate
from duo_workflow_service.components.tools_registry import _AGENT_PRIVILEGES
from duo_workflow_service.tools.bl_collect_and_flatten import BlCollectAndFlatten
from duo_workflow_service.tools.bl_prioritize_and_cap import (
    resolve_effective_max_units,
)
from duo_workflow_service.tools.search import AdvanceBlobSearchInput, BlobSearchInput
from duo_workflow_service.tools.toolset import Toolset
from lib.events import GLReportingEventContext
from lib.version import resolve_version

FLOW_NAME = "bl_security"
FLOW_VERSION = "1.0.0"
CONFIG_PATH = FlowConfig.DIRECTORY_PATH / FLOW_NAME / f"{FLOW_VERSION}.yml"

DEFAULT_FAN_OUT_CAP = 150
# The scan dials come in their own additional-context category.
BL_CONTEXT_CATEGORY = "agent_platform_bl_security_context"
BL_CONTEXT = f"context:inputs.{BL_CONTEXT_CATEGORY}"
EFFORT_KEY = f"{BL_CONTEXT}.scan_effort"
SCAN_DIALS = ("scan_effort", "target_files")

# The two passes that fan out over discovery units, and every fan-out stage.
SCAN_STAGES = ("map_reviews", "sibling_scan")
FAN_OUT_STAGES = (*SCAN_STAGES, "triage")
FAN_OUT_TYPE = "AgentComponent"

SHIPPED_MAX_CONCURRENCY = {"map_reviews": 32, "sibling_scan": 32, "triage": 40}

DETECT_PROMPT_ID = "security_scan_detect"

SCOPE_TAG = re.compile(r"<scope>(.*?)</scope>", re.DOTALL)
# The answer tool each findings prompt ends on, by prompt id.
UNIT_SCHEMA_ID = "bl_unit_findings"
TRIAGE_SCHEMA_ID = "bl_triage_verdict"
TUNER_SCHEMA_ID = "bl_tuner_patterns"
STAGE_SCHEMA = {
    "map_reviews": UNIT_SCHEMA_ID,
    "sibling_scan": UNIT_SCHEMA_ID,
    "triage": TRIAGE_SCHEMA_ID,
}
FINDING_KEYS = ("file", "new_line", "code_excerpt", "tier", "severity", "cwe", "body")
VERDICT_KEYS = ("verdict", "clause", "evidence", "triage_evidence")


@functools.cache
def _config() -> FlowConfig:
    return FlowConfig.from_yaml_config(FLOW_NAME, FLOW_VERSION)


@functools.cache
def _config_source() -> str:
    """Raw YAML, comments included: the parsed config cannot see comments."""
    return CONFIG_PATH.read_text(encoding="utf-8")


def test_the_flow_version_resolves_as_a_semver_constraint():
    available = [p.stem for p in CONFIG_PATH.parent.glob("*.yml")]
    assert resolve_version(available, FLOW_VERSION) == FLOW_VERSION
    assert resolve_version(available, "^1.0.0") == FLOW_VERSION


def _component(name: str) -> dict:
    return next(c for c in _config().components if c.get("name") == name)


def _literals(name: str) -> dict:
    return {i["as"]: i["from"] for i in _component(name)["inputs"] if i.get("literal")}


def _input(component: str, name: str) -> dict:
    return next(i for i in _component(component)["inputs"] if i.get("as") == name)


def _fan_out_stages() -> list[dict]:
    return [
        c
        for c in _config().components
        if c.get("type") == FAN_OUT_TYPE and "for_each" in c
    ]


def _stage_cap(stage: str) -> int:
    return int(_literals(f"{stage}_prep")["max_units"])


def _stage_tiers(stage: str) -> dict:
    return json.loads(_literals(f"{stage}_prep")["max_units_tiers"])


def _prompt_system(prompt_id: str) -> str:
    prompt = next(p for p in (_config().prompts or []) if p.prompt_id == prompt_id)
    return str(prompt.prompt_template["system"])


def _cwes(text: str) -> set[str]:
    """CWE numbers named in ``text``, expanding runs such as ``CWE-284/285/287``."""
    found: set[str] = set()
    for run in re.findall(r"CWE-(\d+(?:/\d+)*)", text):
        found.update(run.split("/"))
    return found


def _schema(schema_id: str) -> dict:
    return next(
        s.to_schema_dict()
        for s in (_config().response_schemas or [])
        if s.schema_id == schema_id
    )


def _registered_tools() -> dict[str, type]:
    """Every tool registered under any agent privilege group, by tool name."""
    return {
        tool.model_fields["name"].default: tool
        for group in _AGENT_PRIVILEGES.values()
        for tool in group
    }


# One fixture call per deterministic tool whose output a later step reads; the
# keys it returns are the keys that step may name.
_TOOL_FIXTURE_ARGS: dict[str, dict[str, Any]] = {
    "bl_discover_and_cluster": {
        "files_per_unit": 8,
        "max_units": 1,
        "target_files": "a.rb",
    },
    "bl_prioritize_and_cap": {"units": []},
    "bl_collect_and_flatten": {"results": [], "emitted_count": 0},
    "bl_dedup_findings": {"batches": []},
}


def _published_paths(component: dict) -> set[str]:
    """The ``<component>.<key>`` paths a component really writes to context."""
    name = component["name"]
    if component.get("for_each"):
        return {f"{name}.{key}" for key in PUBLISHED_SUBKEYS}
    if component.get("tool_name"):
        # Every deterministic step publishes its outcome; the closing routers read it.
        outcome = {f"{name}.execution_result"}
        if component["tool_name"] not in _TOOL_FIXTURE_ARGS:
            return outcome  # no later step reads its response
        tool = _registered_tools()[component["tool_name"]](metadata={"outbox": None})
        out = asyncio.run(tool._execute(**_TOOL_FIXTURE_ARGS[component["tool_name"]]))
        if not isinstance(out, dict):
            return outcome | {f"{name}.tool_responses"}  # a list: read whole
        return outcome | {f"{name}.tool_responses.{key}" for key in out}
    # The only other component type the flow uses.
    outputs = {"AgentComponent": AgentComponentBase}[component["type"]]._outputs
    return {
        ".".join([name, *t.subkeys[1:]])
        for t in outputs
        if t.target == "context"
        and t.subkeys
        and t.subkeys[0] == IOKeyTemplate.COMPONENT_NAME_TEMPLATE
    }


class TestContextWiring:
    def test_every_read_of_a_step_names_a_key_that_step_publishes(self):
        published = {path for c in _config().components for path in _published_paths(c)}
        steps = {c["name"] for c in _config().components}
        reads = {
            m.group(1)
            for m in re.finditer(r"context:([a-z_]+\.[a-z_.]+)", _config_source())
            if m.group(1).split(".")[0] in steps
        }
        assert reads, "found no step output reads"
        assert not reads - published


class TestFanOutCaps:
    """The fan-out caps decide how much of a repository is reviewed.

    ``scan_effort`` may only raise or lower them as an opt-in: a run without it fans out at the literal default.
    """

    def test_every_fan_out_stage_is_tiered_off_scan_effort(self):
        assert {c["name"] for c in _fan_out_stages()} == set(FAN_OUT_STAGES)
        for stage in FAN_OUT_STAGES:
            dial = _input(f"{stage}_prep", "scan_effort")
            assert dial["from"] == EFFORT_KEY, stage
            assert dial.get("optional") is True, stage
            assert set(_stage_tiers(stage)) == set(bl.SCAN_EFFORT_TIERS), stage

    def test_both_scan_stages_share_one_tier_table(self):
        assert _stage_tiers("map_reviews") == _stage_tiers("sibling_scan")

    def test_standard_is_the_default_cap_and_tiers_are_monotone(self):
        tiers = _stage_tiers("map_reviews")
        assert tiers["standard"] == DEFAULT_FAN_OUT_CAP
        assert tiers["low"] < tiers["standard"] < tiers["high"]

    @pytest.mark.parametrize("stage", FAN_OUT_STAGES)
    @pytest.mark.parametrize("scan_effort", [None, "unrecognized"])
    def test_no_or_unknown_scan_effort_falls_back_to_the_default(
        self, stage, scan_effort
    ):
        resolved = resolve_effective_max_units(
            max_units=_stage_cap(stage),
            max_units_tiers=_stage_tiers(stage),
            scan_effort=scan_effort,
        )
        assert resolved == DEFAULT_FAN_OUT_CAP

    def test_every_discovery_pool_is_at_least_its_fan_out_cap(self):
        # A pool smaller than the cap makes a tier narrower than it advertises.
        for tier, cap in _stage_tiers("map_reviews").items():
            pool = bl.SCAN_EFFORT_TIERS[tier][1]
            assert pool >= cap, tier

    def test_triage_never_caps_tighter_than_the_scan_that_fed_it(self):
        # Otherwise raising effort surfaces findings that triage then discards.
        triage = _stage_tiers("triage")
        for tier, scan_cap in _stage_tiers("map_reviews").items():
            assert triage[tier] >= scan_cap, tier

    def test_standard_discovery_pool_equals_the_no_scan_effort_pool(self):
        literals = {
            name: int(value)
            for name, value in _literals("discover_units").items()
            if name in ("files_per_unit", "max_units")
        }
        assert (
            bl.resolve_scan_effort("standard", **literals)[:2]
            == bl.resolve_scan_effort(None, **literals)[:2]
        )

    @pytest.mark.parametrize("stage", FAN_OUT_STAGES)
    def test_each_fan_out_ships_its_in_flight_bound(self, stage):
        block = ForEachConfig(**_component(stage)["for_each"])
        assert block.max_concurrency == SHIPPED_MAX_CONCURRENCY[stage]

    @pytest.mark.parametrize("stage", FAN_OUT_STAGES)
    def test_each_fan_out_iterates_its_own_prep_step(self, stage):
        block = _component(stage)["for_each"]
        assert block["items"] == f"context:{stage}_prep.tool_responses.units"


class TestDeterministicStepInputs:
    """A deterministic step silently drops an input its tool does not declare."""

    @pytest.mark.parametrize(
        "component",
        [
            c["name"]
            for c in _config().components
            if c["type"] == "DeterministicStepComponent"
        ],
    )
    def test_every_input_is_a_tool_argument(self, component):
        spec = _component(component)
        tool = _registered_tools()[spec["tool_name"]]
        arguments = set(tool.model_fields["args_schema"].default.model_fields)
        assert {i["as"] for i in spec["inputs"]} <= arguments


class TestCoverageIsReported:
    """Each fan-out stage publishes how much it reviewed, and the report states it."""

    def test_the_report_reads_every_stage_coverage_optionally(self):
        wired = {
            i["from"]: i
            for i in _component("write_report")["inputs"]
            if i["from"].endswith(".coverage")
        }
        assert set(wired) == {
            f"context:{stage}_collect.tool_responses.coverage"
            for stage in FAN_OUT_STAGES
        }
        assert all(i.get("optional") is True for i in wired.values())


class TestDiscoveryDials:
    """The discovery literals, resolved through the tool's own resolvers."""

    @pytest.mark.parametrize(
        ("dial", "resolver", "expected"),
        [
            ("coverage_first", bl.resolve_coverage_first, True),
            ("content_filter", bl.resolve_content_filter, True),
            ("cap_aware_backfill", bl.resolve_cap_aware_backfill, True),
            ("saturation_stop", bl.resolve_saturation_stop, True),
        ],
    )
    def test_each_switch_resolves_to_its_shipped_state(self, dial, resolver, expected):
        # Every resolver maps an unrecognized spelling to off, so assert the reading.
        assert resolver(_literals("discover_units")[dial]) is expected

    @pytest.mark.parametrize(
        ("scan_effort", "pair"),
        [
            ("low", (4, 4)),
            ("standard", (2, 1)),
            ("high", (4, 1)),
            (None, (1, 1)),
            ("unrecognized", (1, 1)),
        ],
    )
    def test_the_context_dials_resolve_per_tier(self, scan_effort, pair):
        literals = _literals("discover_units")
        configured = (int(literals["files_per_unit"]), int(literals["max_units"]))
        assert bl.resolve_scan_effort(scan_effort, *configured)[2:] == pair


class TestOptionalRunInputs:
    """``scan_effort`` and ``target_files`` are opt-in: omitting them keeps a full default scan."""

    @pytest.mark.parametrize("name", SCAN_DIALS)
    def test_declared_optional_on_the_bl_security_context(self, name):
        schema = _config().input_json_schemas_by_category()[BL_CONTEXT_CATEGORY]
        assert schema["properties"][name]["type"] == "string"
        assert name not in schema["required"]

    def test_the_bl_security_context_declares_exactly_the_scan_dials(self):
        schema = _config().input_json_schemas_by_category()[BL_CONTEXT_CATEGORY]
        assert set(schema["properties"]) == set(SCAN_DIALS)
        assert schema["required"] == []

    def test_every_scan_dial_is_read_from_the_bl_security_context(self):
        readers = {
            (c["name"], i["as"]): i["from"]
            for c in _config().components
            for i in c.get("inputs", [])
            if i.get("as") in SCAN_DIALS
        }
        assert readers, "no component reads a scan dial"
        for (component, name), source in readers.items():
            assert source == f"{BL_CONTEXT}.{name}", component


class TestScanDialsAtRuntime:
    """The dials resolve from their input category, and degrade to ``None`` without it."""

    @staticmethod
    def _resolve(component: str, name: str, inputs: dict) -> Any:
        key = IOKey.parse_key(_input(component, name))
        # A partial state is enough here: only `context.inputs` is read.
        return key.value_from_state({"context": {"inputs": inputs}})  # type: ignore[typeddict-item]

    def test_the_dials_resolve_from_the_category(self):
        inputs = {BL_CONTEXT_CATEGORY: {"scan_effort": "high", "target_files": "a.rb"}}
        assert self._resolve("discover_units", "scan_effort", inputs) == "high"
        assert self._resolve("discover_units", "target_files", inputs) == "a.rb"
        assert self._resolve("write_report", "target_files", inputs) == "a.rb"

    @pytest.mark.parametrize(
        "inputs",
        [
            {BL_CONTEXT_CATEGORY: {"scan_effort": "low"}},  # whole-repo scan
            {},  # the category is not sent at all
        ],
    )
    def test_a_missing_target_files_resolves_to_none(self, inputs):
        assert self._resolve("discover_units", "target_files", inputs) is None
        assert self._resolve("write_report", "target_files", inputs) is None

    def test_scan_effort_resolves_to_none_without_the_category(self):
        for stage in (*FAN_OUT_STAGES, "discover_units"):
            component = stage if stage == "discover_units" else f"{stage}_prep"
            assert self._resolve(component, "scan_effort", {}) is None


class TestStageGraph:
    def test_the_component_list(self):
        assert [c["name"] for c in _config().components] == [
            "grounding",
            "tuner",
            "discover_units",
            "map_reviews_prep",
            "map_reviews",
            "map_reviews_collect",
            "sibling_scan_prep",
            "sibling_scan",
            "sibling_scan_collect",
            "adjudicate",
            "triage_prep",
            "triage",
            "triage_collect",
            "adjudicate_post",
            "write_report",
        ]

    def test_one_linear_path_from_entry_to_end_runs_every_component(self):
        successors: dict[str, list[str]] = {}
        for route in _config().routers:
            # A closing step's only route is its "success" one.
            target = route.get("to") or route["condition"]["routes"]["success"]
            successors.setdefault(route["from"], []).append(target)
        assert all(len(targets) == 1 for targets in successors.values())

        node, walk = _config().flow.entry_point, []
        while node != "end":
            assert node not in walk, f"cycle at {node}"
            walk.append(node)
            node = successors[node][0]

        assert walk[:3] == ["grounding", "tuner", "discover_units"]
        assert set(walk) == {c["name"] for c in _config().components}

    def test_discovery_reads_both_halves_of_the_tuner_answer(self):
        # The structured answer, whole; each resolver takes its own key.
        answer = {"globs": ["lib/gate/**/*"], "anchor_patterns": [r"_gate\.rb$"]}
        for name, resolve, want in (
            ("extra_globs", bl.resolve_extra_globs, ["lib/gate/**/*"]),
            ("extra_anchor_patterns", bl.resolve_extra_anchor_patterns, None),
        ):
            entry = _input("discover_units", name)
            assert entry["from"] == "context:tuner.final_answer", name
            # Optional, so unrouting the tuner turns the additions off.
            assert entry.get("optional") is True, name
            got = resolve(answer)
            assert (got if want else [p.pattern for p in got]) == (
                want or [r"_gate\.rb$"]
            )

    def test_every_declared_prompt_is_consumed_by_a_component(self):
        declared = {p.prompt_id for p in (_config().prompts or [])}
        used = {c["prompt_id"] for c in _config().components if c.get("prompt_id")}
        assert declared == used

    def test_no_component_or_prompt_declares_its_own_model(self):
        # The model comes from the `bl_security` feature setting. Read the raw YAML:
        # pydantic drops keys FlowConfig does not declare.
        raw = yaml.safe_load(_config_source())
        assert "model" not in raw
        for entry in raw["components"] + raw["prompts"]:
            assert "model" not in entry, entry["name"]
            assert "model_tags" not in entry, entry["name"]


def _tool_names(toolset: list) -> list[str]:
    """Tool names in a toolset, whose entries are names or ``{name: options}`` maps."""
    return [
        name
        for entry in toolset
        for name in (entry if isinstance(entry, dict) else [entry])
    ]


def _tool_options(component: str, tool: str) -> dict:
    return next(
        (
            entry[tool]
            for entry in _component(component)["toolset"]
            if isinstance(entry, dict) and tool in entry
        ),
        {},
    )


class TestToolsets:
    def test_every_tool_the_flow_names_is_registered(self):
        # tools_registry skips an unknown toolset name silently.
        named = {
            tool
            for c in _config().components
            for tool in [*_tool_names(c.get("toolset", [])), c.get("tool_name")]
            if tool
        }
        assert named <= set(_registered_tools())

    @pytest.mark.parametrize("schema", [BlobSearchInput, AdvanceBlobSearchInput])
    def test_the_pinned_option_is_a_blob_search_argument(self, schema):
        # Mirrors Toolset._validate_tool_options: a renamed field fails here.
        assert set(_tool_options("grounding", "gitlab_blob_search")) <= set(
            schema.model_fields
        )

    @pytest.mark.parametrize(
        "agent", [c["name"] for c in _config().components if "toolset" in c]
    )
    def test_no_agent_reading_untrusted_code_holds_a_write_tool(self, agent):
        read_only = {
            tool.model_fields["name"].default
            for group in ("read_only_files", "read_only_gitlab")
            for tool in _AGENT_PRIVILEGES[group]
        }
        assert set(_tool_names(_component(agent)["toolset"])) <= read_only


class TestScopeMatchesTheReportAllowlist:
    """The prompts' ``<scope>`` and ``bl_report.IN_SCOPE_CWES`` must declare the same classes.

    The prompt is static YAML and cannot import the constant, so this is the joint that holds them together.
    """

    def test_the_requested_half_names_exactly_the_allowlist(self):
        scope = SCOPE_TAG.search(_prompt_system(DETECT_PROMPT_ID))
        assert scope, "the detect prompt has no <scope> block"
        requested, _, excluded = scope.group(1).partition("EXCLUDE")
        assert excluded, "<scope> has no EXCLUDE clause"
        assert _cwes(requested) == set(report.IN_SCOPE_CWES)


class TestResponseSchemas:
    """The answer tools the fan-out stages end on, and the fields the pipeline reads from them."""

    @pytest.mark.parametrize(("stage", "schema_id"), sorted(STAGE_SCHEMA.items()))
    def test_each_fan_out_stage_answers_through_its_schema(self, stage, schema_id):
        assert _component(stage)["response_schema_id"] == schema_id

    def test_the_tuner_answers_through_its_schema(self):
        assert _component("tuner")["response_schema_id"] == TUNER_SCHEMA_ID
        schema = _schema(TUNER_SCHEMA_ID)
        assert tuple(schema["properties"]) == ("globs", "anchor_patterns")
        assert set(schema["required"]) == {"globs", "anchor_patterns"}
        for field in schema["properties"].values():
            assert (field["type"], field["items"]["type"]) == ("array", "string")
        model = json_schema_to_pydantic(schema, title_fallback=TUNER_SCHEMA_ID)
        assert model.tool_title == TUNER_SCHEMA_ID
        assert TUNER_SCHEMA_ID not in _tool_names(_component("tuner")["toolset"])
        assert TUNER_SCHEMA_ID not in _registered_tools()

    def test_only_triage_collect_merges_verdicts_onto_the_items_triage_judged(self):
        def verdict_items(name):
            inputs = _component(name)["inputs"]
            return [i["from"] for i in inputs if i.get("as") == "verdict_items"]

        assert verdict_items("triage_collect") == [
            _component("triage")["for_each"]["items"]
        ]
        assert verdict_items("map_reviews_collect") == []
        assert verdict_items("sibling_scan_collect") == []

    def test_the_triage_reasoning_is_the_first_required_field(self):
        schema = _schema(TRIAGE_SCHEMA_ID)
        assert next(iter(schema["properties"])) == "reasoning"
        assert "reasoning" in schema["required"]

    def test_each_triage_audit_field_says_it_is_its_own_field(self):
        """Agents put the whole verdict inside `reasoning`; each field's description steers it out."""
        properties = _schema(TRIAGE_SCHEMA_ID)["properties"]
        assert "complete" not in properties["reasoning"]["description"]
        assert "own fields" in properties["reasoning"]["description"]
        assert "not inside reasoning" in properties["verdict"]["description"]
        for key in ("clause", "evidence", "triage_evidence"):
            assert properties[key]["description"]

    def test_unit_findings_come_first_then_a_bounded_summary(self):
        schema = _schema(UNIT_SCHEMA_ID)
        assert tuple(schema["properties"]) == ("findings", "summary")
        assert set(schema["required"]) == {"findings", "summary"}
        summary = schema["properties"]["summary"]
        assert 20 <= summary["minLength"] < summary["maxLength"]
        assert summary["maxLength"] >= 2000
        model = json_schema_to_pydantic(schema, title_fallback=UNIT_SCHEMA_ID)
        with pytest.raises(PydanticValidationError):
            model(summary="test", findings=[])

    def test_a_unit_finding_carries_the_keys_the_pipeline_reads(self):
        findings = _schema(UNIT_SCHEMA_ID)["properties"]["findings"]
        assert findings["type"] == "array"
        item = findings["items"]
        assert tuple(item["properties"]) == FINDING_KEYS
        assert set(item["required"]) == set(FINDING_KEYS)
        assert item["properties"]["new_line"]["type"] == "integer"
        assert item["properties"]["tier"]["type"] == "integer"

    def test_a_triage_verdict_carries_the_audit_keys_the_report_reads(self):
        schema = _schema(TRIAGE_SCHEMA_ID)
        assert tuple(schema["properties"])[1:] == VERDICT_KEYS
        assert schema["properties"]["verdict"]["enum"] == ["KEEP", "DROP"]
        assert set(schema["required"]) == {"reasoning", "verdict", "clause", "evidence"}

    @pytest.mark.parametrize("key", ["file", "code_excerpt", "severity", "cwe", "body"])
    def test_the_answer_tool_rejects_a_finding_with_an_empty_field(self, key):
        """The agent re-prompts on a rejected answer, so an empty field never reaches the pipeline."""
        model = json_schema_to_pydantic(
            _schema(UNIT_SCHEMA_ID), title_fallback=UNIT_SCHEMA_ID
        )
        finding = dict.fromkeys(FINDING_KEYS, "x") | {"new_line": 3, "tier": 1}
        summary = "One handler, one finding."
        model(summary=summary, findings=[finding])

        with pytest.raises(PydanticValidationError):
            model(summary=summary, findings=[finding | {key: ""}])

    @pytest.mark.parametrize(
        ("key", "value"),
        [("reasoning", "x" * 19), ("clause", ""), ("evidence", "")],
    )
    def test_the_verdict_tool_rejects_short_reasoning_and_empty_fields(
        self, key, value
    ):
        model = json_schema_to_pydantic(
            _schema(TRIAGE_SCHEMA_ID), title_fallback=TRIAGE_SCHEMA_ID
        )
        verdict = {
            "reasoning": "x" * 20,
            "verdict": "KEEP",
            "clause": "KEEP-1",
            "evidence": "a.rb:3",
        }
        model(**verdict)

        with pytest.raises(PydanticValidationError):
            model(**verdict | {key: value})

    @pytest.mark.parametrize(("stage", "schema_id"), sorted(STAGE_SCHEMA.items()))
    def test_the_stage_agent_is_built_with_its_answer_tool(
        self, stage, schema_id, user
    ):
        """Built the way the flow builds it: the flow's inline prompts and schemas, the stage's own config."""
        config = _config()
        prompts = InMemoryPromptRegistry(
            LocalPromptRegistry(
                prompt_template_factories={},
                model_factories={},
                internal_event_client=Mock(),
                model_limits=Mock(),
                custom_models_enabled=False,
            )
        )
        for prompt in config.prompts or []:
            prompts.register_prompt(
                prompt_id=prompt.prompt_id, prompt_data=prompt.to_prompt_data()
            )
        schemas = InlineResponseSchemaRegistry(Mock())
        for schema in config.response_schemas or []:
            schemas.register_schema(schema.schema_id, schema.to_schema_dict())
        params = {
            k: v
            for k, v in _component(stage).items()
            if k not in ("type", "for_each", "toolset")
        }
        agent = AgentComponent(
            **params,
            flow_id="1",
            flow_type=GLReportingEventContext.from_workflow_definition(
                "software_development"
            ),
            user=user,
            prompt_registry=prompts,
            schema_registry=schemas,
            toolset=Toolset(pre_approved=set(), all_tools={}),
        )

        assert agent._response_schema.tool_title == schema_id

    def test_a_structured_unit_answer_flattens_to_its_findings(self):
        model = json_schema_to_pydantic(
            _schema(UNIT_SCHEMA_ID), title_fallback=UNIT_SCHEMA_ID
        )
        finding = dict.fromkeys(FINDING_KEYS, "x") | {"new_line": 3, "tier": 1}
        answer = model(
            summary="One handler, one finding.", findings=[finding]
        ).to_output()

        # The collect step publishes the flattened findings inline.
        out = asyncio.run(
            BlCollectAndFlatten(metadata=None)._execute(
                results=[{"final_answer": answer}]
            )
        )

        assert json.loads(out["final_answer"]) == [finding]
