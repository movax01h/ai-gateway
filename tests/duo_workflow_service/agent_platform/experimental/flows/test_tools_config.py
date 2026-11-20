import pytest

from duo_workflow_service.agent_platform.experimental.flows.tools_config import (
    parse_component_tools,
)
from duo_workflow_service.components.tools_registry import ToolSelector


def _parse(tools):
    return parse_component_tools({"name": "agent", "tools": tools})


def test_a_component_without_tools_has_nothing_to_parse():
    assert parse_component_tools({"name": "agent", "toolset": ["read_file"]}) is None


def test_selectors_keep_their_order():
    parsed = _parse(
        [
            {"server": "gitlab", "name": ["read_file", "get_merge_request"]},
            {"origin": "instance", "server": "orbit", "name": ["list_commands"]},
            {"server": "slack", "name": ["slack_read_thread"]},
        ]
    )

    assert parsed == [
        ToolSelector("gitlab", ["read_file", "get_merge_request"]),
        ToolSelector("orbit", ["list_commands"]),
        ToolSelector("slack", ["slack_read_thread"]),
    ]


def test_an_empty_list_means_no_tools():
    assert _parse([]) == []


@pytest.mark.parametrize(
    "tools, message",
    [
        (None, '"tools" must be a list'),
        ("read_file", '"tools" must be a list'),
        # A plain name and a name with options are not selectors.
        (["read_file"], 'must be a selector with "server" and "name"'),
        ([42], 'must be a selector with "server" and "name"'),
        (
            [{"get_pipeline_failing_jobs": {"exclude_allow_failure": True}}],
            'stays on "toolset:"',
        ),
        ([{"name": ["get_issue"]}], '"server" must be "gitlab", "orbit" or "slack"'),
        (
            [{"server": "jira", "name": ["create_issue"]}],
            '"server" must be "gitlab", "orbit" or "slack"',
        ),
        (
            [{"server": "*", "name": ["get_issue"]}],
            '"server" must be "gitlab", "orbit" or "slack"',
        ),
        (
            [{"server": "gitlab", "origin": "client", "name": ["a"]}],
            '"origin" can only be "instance"',
        ),
        ([{"server": "gitlab"}], '"name" is required'),
        ([{"server": "gitlab", "name": "get_issue"}], '"name" is required'),
        ([{"server": "gitlab", "name": []}], '"name" must list at least one tool'),
        ([{"server": "gitlab", "name": ["list_*"]}], '"name" takes exact tool names'),
        ([{"server": "gitlab", "name": ["*"]}], '"name" takes exact tool names'),
        (
            [{"server": "gitlab", "name": ["a"], "include": ["b"]}],
            'selector key "include" is not supported',
        ),
        (
            [{"server": "gitlab", "name": ["a"], "exclude": ["b"]}],
            'selector key "exclude" is not supported',
        ),
        (
            [{"server": "gitlab", "name": ["a"], "ops": {}}],
            'selector key "ops" is not supported',
        ),
        (
            [{"server": "gitlab", "name": ["a"], "version": "^1.0"}],
            'selector key "version" is not supported',
        ),
    ],
)
def test_a_config_error_says_what_is_wrong(tools, message):
    with pytest.raises(ValueError, match='Component "agent"') as exc:
        _parse(tools)

    assert message in str(exc.value)
