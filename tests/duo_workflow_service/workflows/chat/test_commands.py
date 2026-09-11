# pylint: disable=file-naming-for-tests
import json

import pytest

from duo_workflow_service.tools.start_flow import StartFlow
from duo_workflow_service.workflows.chat.commands import (
    CHAT_COMMAND_CATEGORY,
    ForcedToolCall,
    parse_forced_tool_call,
    strip_command_context,
)
from duo_workflow_service.workflows.type_definitions import AdditionalContext


def command_context(payload, category=CHAT_COMMAND_CATEGORY):
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return AdditionalContext(category=category, content=content, metadata={})


def test_parses_a_flow_command():
    context = [command_context({"command": "flow", "ai_catalog_item_consumer_id": 42})]

    assert parse_forced_tool_call(context) == ForcedToolCall(
        name="start_flow",
        args={"flow": {"name": "catalog_flow", "ai_catalog_item_consumer_id": 42}},
    )


def test_carries_the_goal_when_one_is_given():
    context = [
        command_context(
            {
                "command": "flow",
                "ai_catalog_item_consumer_id": 42,
                "goal": "check the auth module",
            }
        )
    ]

    forced = parse_forced_tool_call(context)

    assert forced is not None
    assert forced.args["flow"]["goal"] == "check the auth module"


@pytest.mark.parametrize("goal", [None, ""], ids=["absent", "empty"])
def test_omits_an_absent_goal_rather_than_sending_null(goal):
    """Rails falls back to the flow's own description only when goal is absent."""
    context = [
        command_context(
            {"command": "flow", "ai_catalog_item_consumer_id": 42, "goal": goal}
        )
    ]

    forced = parse_forced_tool_call(context)

    assert forced is not None
    assert "goal" not in forced.args["flow"]


@pytest.mark.parametrize(
    "context",
    [
        [],
        None,
        [AdditionalContext(category="file", content="x", metadata={})],
        [
            command_context(
                {"command": "flow", "ai_catalog_item_consumer_id": 42}, "file"
            )
        ],
        [command_context("not json")],
        [command_context([1, 2, 3])],
        [command_context({"command": "definitely_not_a_command"})],
        [command_context({"command": "flow"})],
        [command_context({"command": "flow", "ai_catalog_item_consumer_id": "42"})],
        [command_context({"command": "flow", "ai_catalog_item_consumer_id": True})],
        [AdditionalContext(category=CHAT_COMMAND_CATEGORY, content=None, metadata={})],
    ],
    ids=[
        "no_context",
        "null_context",
        "unrelated_category",
        "payload_under_the_wrong_category",
        "unparsable_content",
        "content_is_not_an_object",
        "unknown_command",
        "missing_consumer_id",
        "consumer_id_is_a_string",
        "consumer_id_is_a_bool",
        "no_content",
    ],
)
def test_returns_nothing_for_anything_it_does_not_recognise(context):
    assert parse_forced_tool_call(context) is None


def test_strips_only_the_command_envelope():
    keep = AdditionalContext(category="file", content="x", metadata={})
    context = [command_context({"command": "flow"}), keep]

    assert strip_command_context(context) == [keep]


@pytest.mark.parametrize("context", [[], None], ids=["empty", "null"])
def test_stripping_tolerates_no_context(context):
    assert strip_command_context(context) is None


def test_stripping_reports_an_envelope_only_turn_as_no_context():
    """``[]`` would change what every workflow reports, since ``with_attachment_references`` branches on ``None``."""
    context = [command_context({"command": "flow"})]

    assert strip_command_context(context) is None


@pytest.mark.parametrize(
    "goal", [42, ["check"], {"text": "check"}], ids=["int", "list", "object"]
)
def test_rejects_a_goal_that_is_not_a_string(goal):
    context = [
        command_context(
            {"command": "flow", "ai_catalog_item_consumer_id": 42, "goal": goal}
        )
    ]

    assert parse_forced_tool_call(context) is None


@pytest.mark.parametrize(
    "features",
    [
        {},
        {"foundational_flows": {"enabled": True, "enabled_flows": None}},
        {"foundational_flows": {"enabled": False, "enabled_flows": None}},
        {"foundational_flows": {"enabled": True, "enabled_flows": ["code_review/v1"]}},
        {"foundational_flows": {"enabled": True, "enabled_flows": []}},
    ],
    ids=[
        "no_features",
        "all_foundational_flows",
        "foundational_flows_disabled",
        "some_foundational_flows",
        "no_foundational_flows",
    ],
)
@pytest.mark.parametrize(
    "goal", ["check the auth module", None], ids=["goal", "no_goal"]
)
def test_the_arguments_it_builds_satisfy_the_tool(features, goal):
    """The built arguments are handed to start_flow unvalidated, so they have to fit its schema.

    start_flow narrows that schema to the flows a project has enabled, and the catalog member has to survive every
    narrowing because catalog flows are enabled per project instead.
    """
    forced = parse_forced_tool_call(
        [
            command_context(
                {"command": "flow", "ai_catalog_item_consumer_id": 42, "goal": goal}
            )
        ]
    )
    assert forced is not None

    tool = StartFlow(metadata={"features": features})
    flow = tool.args_schema(**forced.args).flow

    assert flow.name == "catalog_flow"
    assert flow.ai_catalog_item_consumer_id == 42
    assert flow.goal == goal
