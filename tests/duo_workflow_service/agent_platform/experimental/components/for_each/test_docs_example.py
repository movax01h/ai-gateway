"""Checks that the documented ``for_each`` example is the one this code runs.

The doc is the surface a flow author works from, so what it shows has to keep parsing, keep naming real components, and
keep matching the defaults and node names the implementation produces.
"""

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import yaml
from langgraph.graph import StateGraph

from ai_gateway.prompts.registry import LocalPromptRegistry
from duo_workflow_service.agent_platform.experimental.components import (
    ComponentRegistry,
)
from duo_workflow_service.agent_platform.experimental.components.agent.component import (
    AgentComponent,
)
from duo_workflow_service.agent_platform.experimental.components.for_each import (
    ERRORS_SUBKEY,
    ITEM_ERROR_SUBKEY,
    ITEM_INDEX_CONTEXT_KEY,
    MAX_CONCURRENCY_CEILING,
    MAX_ITEMS_CEILING,
    PUBLISHED_SUBKEYS,
    RESULTS_SUBKEY,
    TERMINAL_EXCEPTIONS,
    AllItemsFailedError,
    ForEachComponent,
    ForEachConfig,
    TerminalRouter,
    failed_item_errors,
    item_error_record,
)
from duo_workflow_service.agent_platform.experimental.flows.flow_config import (
    FlowConfig,
)
from duo_workflow_service.agent_platform.experimental.state import FlowState, IOKey
from duo_workflow_service.tools.toolset import Toolset
from lib.events import GLReportingEventContext
from lib.feature_flags.context import FeatureFlag
from lib.internal_events import InternalEventsClient

_DOC = (
    Path(__file__).resolve().parents[6] / "docs" / "flow_registry" / "experimental.md"
)

#: The section heading the example lives under, kept as a literal so that
#: renaming it in the doc fails here rather than silently skipping the checks.
_EXAMPLE_HEADING = "#### Complete for_each Example"

#: The component in the example that carries ``for_each``.
_FAN_OUT_STAGE = "review_one"

#: A config with nothing but the two required fields, so every other value
#: it carries is the default the doc has to agree with.
_DEFAULT_CONFIG = ForEachConfig.model_validate(
    {"items": "context:discover.files", "as": "context:item"}
)


def _section(heading: str) -> str:
    """The doc text from ``heading`` up to the next heading of any level."""
    text = _DOC.read_text()
    assert text.count(f"\n{heading}\n") == 1, (
        f"'{heading}' is not a unique heading of {_DOC}, so the text scraped for "
        "it is whichever section happens to come first"
    )
    after = text.split(heading, 1)[1]
    match = re.search(r"\n#{1,6} ", after)
    return after[: match.start()] if match else after


def _paths(keys: Iterable[IOKey]) -> set[str]:
    """The ``target:sub.keys`` strings a flow config names these keys by."""
    return {":".join([key.target, ".".join(key.subkeys or [])]) for key in keys}


def _stage(example: dict[str, Any], name: str) -> dict[str, Any]:
    stage = next((c for c in example["components"] if c["name"] == name), None)
    assert stage is not None, f"the example no longer defines a '{name}' component"
    return stage


@pytest.fixture(name="example")
def example_fixture() -> dict[str, Any]:
    """The first YAML block under the example heading, parsed."""
    match = re.search(r"```yaml\n(.*?)\n```", _section(_EXAMPLE_HEADING), re.DOTALL)
    assert match is not None, "no YAML block follows the example heading"
    return yaml.safe_load(match.group(1))


@pytest.fixture(name="fan_out_stage")
def fan_out_stage_fixture(example) -> dict[str, Any]:
    stage = _stage(example, _FAN_OUT_STAGE)
    assert "for_each" in stage, f"'{_FAN_OUT_STAGE}' no longer carries a for_each block"
    return stage


@pytest.fixture(name="fanned_component")
def fanned_component_fixture(user) -> ForEachComponent:
    """A fan-out over an ``AgentComponent``, the type the example fans out.

    The wrapped type matters to what the checks below can catch: an agent
    declares outputs of its own, so a published set that leaked them would look
    different from one that does not.
    """
    return ForEachComponent.wrapping(
        AgentComponent(
            name=_FAN_OUT_STAGE,
            flow_id="flow-1",
            flow_type=GLReportingEventContext.from_workflow_definition(
                "software_development"
            ),
            user=user,
            prompt_id="review_one_file",
            toolset=Toolset(pre_approved=set(), all_tools={}),
            prompt_registry=Mock(spec=LocalPromptRegistry),
            internal_event_client=Mock(spec=InternalEventsClient),
        ),
        _DEFAULT_CONFIG,
    )


class TestTheExampleIsLoadable:
    def test_it_validates_as_a_flow_config(self, example):
        config = FlowConfig(**example)

        names = {component["name"] for component in config.components}
        assert config.flow.entry_point in names
        for router in config.routers:
            assert router["from"] in names
            assert router["to"] in names | {"end"}

    def test_every_component_type_it_shows_is_registered(self, example):
        registry = ComponentRegistry.instance()

        for component in example["components"]:
            assert registry[component["type"]] is not None

    def test_its_for_each_block_validates(self, fan_out_stage):
        config = ForEachConfig.model_validate(fan_out_stage["for_each"])

        assert config.items == "context:discover.final_answer.files"
        assert config.as_ == "context:item"

    def test_every_optional_value_it_shows_differs_from_the_default(
        self, fan_out_stage
    ):
        """Otherwise a misspelled optional key in the example is undetectable."""
        shown = ForEachConfig.model_validate(fan_out_stage["for_each"])
        defaults = ForEachConfig.model_validate({"items": shown.items, "as": shown.as_})

        for field in shown.model_fields_set - {"items", "as_"}:
            assert getattr(shown, field) != getattr(defaults, field), (
                f"the example sets '{field}' to its default, so a typo in that key "
                "would leave the example passing"
            )

    def test_the_fanned_stage_reads_its_item(self, fan_out_stage):
        """The body has to declare the item as an input like any other."""
        item_path = fan_out_stage["for_each"]["as"]

        assert {"from": item_path} in fan_out_stage["inputs"]

    def test_the_items_path_addresses_an_array_field_of_a_response_schema(
        self, example
    ):
        items_path = _stage(example, _FAN_OUT_STAGE)["for_each"]["items"]
        producer_name, _, field = items_path.removeprefix("context:").partition(
            ".final_answer."
        )
        producer = _stage(example, producer_name)

        schema = next(
            entry["definition"]
            for entry in example["response_schemas"]
            if entry["schema_id"] == producer["response_schema_id"]
        )
        assert schema["properties"][field]["type"] == "array"

    def test_the_downstream_stage_reads_only_what_the_fan_out_publishes(
        self, example, fanned_component
    ):
        """Every input the next stage declares has to name a key that resolves.

        What the wrapped component declares is not among them: a branch's key
        is readable only inside its own ``results`` entry, so an input naming it
        at the top level would find nothing there.
        """
        published = _paths(fanned_component.outputs)
        delegated = _paths(fanned_component.component.outputs)

        assert delegated
        assert published.isdisjoint(delegated)
        for declared in _stage(example, "summarize")["inputs"]:
            assert declared["from"] in published


class TestTheProseMatchesTheCode:
    def test_the_documented_max_items_default_and_ceiling_are_real(self):
        row = next(
            line
            for line in _section("#### Configuration").splitlines()
            if line.startswith("| `max_items` |")
        )
        default = _DEFAULT_CONFIG.max_items

        assert f"`{default}`" in row
        assert f"`{MAX_ITEMS_CEILING}`" in row

    def test_the_documented_max_concurrency_default_and_ceiling_are_real(self):
        row = next(
            line
            for line in _section("#### Configuration").splitlines()
            if line.startswith("| `max_concurrency` |")
        )

        assert f"`{_DEFAULT_CONFIG.max_concurrency}`" in row
        assert f"`{MAX_CONCURRENCY_CEILING}`" in row

    def test_the_documented_node_names_are_the_ones_attached(self, fanned_component):
        graph = StateGraph(FlowState)
        fanned_component.attach(graph, TerminalRouter())
        section = _section("#### How a fan-out runs")

        for node in graph.nodes:
            documented = f"`{node.replace(_FAN_OUT_STAGE, '<name>')}`"
            assert documented in section, f"{documented} is not in the node table"

    def test_every_published_key_is_documented(self, fanned_component):
        """The table has to name each key, and each name has to be one that exists."""
        section = _section("#### Collected outputs")
        published = _paths(fanned_component.outputs)

        for subkey in PUBLISHED_SUBKEYS:
            assert f"`context:<name>.{subkey}`" in section
            assert f"context:{_FAN_OUT_STAGE}.{subkey}" in published

    def test_the_documented_index_key_is_the_one_a_branch_is_given(self):
        section = _section("#### Where each item lands")

        assert f"`context:<name>.{ITEM_INDEX_CONTEXT_KEY}`" in section

    def test_the_documented_feature_flag_is_the_one_checked(self):
        assert f"`{FeatureFlag.DAP_FOR_EACH.value}`" in _section("#### Feature flag")

    @pytest.mark.parametrize(
        "claim",
        [
            "The concurrency cap is per fan-out, not fleet-wide.",
            "There is no per-item fail-fast.",
        ],
    )
    def test_the_limitations_say_plainly_what_is_absent(self, claim):
        assert claim in _section("#### Limitations")

    def test_the_documented_error_sub_key_is_the_one_a_record_carries(self):
        assert f"`{ITEM_ERROR_SUBKEY}`" in _section("#### When an item fails")

    def test_the_documented_error_record_parses_and_has_the_shape_produced(self):
        """The doc shows both shapes a downstream reader may branch on."""
        match = re.search(
            r"```yaml\n(.*?)\n```", _section("#### When an item fails"), re.DOTALL
        )
        assert match is not None, "no results example follows the heading"
        shown = yaml.safe_load(match.group(1))
        results = shown[f"context:{_FAN_OUT_STAGE}.{RESULTS_SUBKEY}"]

        by_index = {str(index): entry for index, entry in enumerate(results)}
        failed = failed_item_errors(by_index)

        assert set(failed["1"]) == set(
            item_error_record(ValueError("x"))[ITEM_ERROR_SUBKEY]
        )
        assert shown[f"context:{_FAN_OUT_STAGE}.{ERRORS_SUBKEY}"] == [
            {"index": int(index), **record} for index, record in failed.items()
        ]

    @pytest.mark.parametrize("terminal", TERMINAL_EXCEPTIONS, ids=lambda c: c.__name__)
    def test_every_terminal_exception_is_documented(self, terminal):
        assert f"`{terminal.__name__}`" in _section("##### Failures that stay terminal")

    def test_the_documented_all_failed_error_is_the_one_raised(self):
        section = _section("##### Every item failing is a bug, not a result")

        assert f"`{AllItemsFailedError.__name__}`" in section

    def test_the_outputs_section_warns_that_the_ui_log_is_not_in_item_order(self):
        assert "completion order, not item order" in _section("#### Collected outputs")
