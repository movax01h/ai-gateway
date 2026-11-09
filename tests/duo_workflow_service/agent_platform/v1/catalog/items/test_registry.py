import pytest
from pydantic import ValidationError
from structlog.testing import capture_logs

from duo_workflow_service.agent_platform.v1.catalog import (
    MAX_NAME_LENGTH,
    CatalogItems,
    CatalogItemsError,
)
from duo_workflow_service.agent_platform.v1.catalog.items.registry import _describe


def _agent(**overrides) -> dict:
    return {
        "name": "tester",
        "description": "Runs tests.",
        "prompt": "You are a specialist.",
        **overrides,
    }


def _catalog_agent(**overrides) -> dict:
    return {
        "item_id": "42",
        "version": "1.0.0",
        "name": "tester",
        "description": "Runs tests.",
        "prompt": "You are a specialist.",
        **overrides,
    }


class TestCatalogItems:
    def test_defaults_to_no_items(self):
        """A request that carries no catalog section builds the flow exactly as authored."""
        assert CatalogItems().workspace_agents == []
        assert CatalogItems().ai_catalog_agents == []

    @pytest.mark.parametrize(
        "names",
        [("tester", "tester"), ("tester", "workspace/agents/tester")],
        ids=["identical", "differ_only_by_the_prefix"],
    )
    def test_duplicate_agent_names_raise(self, names):
        """Each name becomes one graph component and one delegate_task choice.

        Uniqueness is judged after namespacing, which is where a collision lands.
        """
        with pytest.raises(ValidationError, match="Duplicate workspace agent name"):
            CatalogItems(workspace_agents=[_agent(name=name) for name in names])

    def test_the_ceiling_itself_is_accepted(self):
        payload = [
            _agent(name=f"a{i}") for i in range(CatalogItems.MAX_WORKSPACE_AGENTS)
        ]

        assert len(CatalogItems(workspace_agents=payload).workspace_agents) == len(
            payload
        )

    def test_one_item_over_the_ceiling_raises(self):
        payload = [
            _agent(name=f"a{i}") for i in range(CatalogItems.MAX_WORKSPACE_AGENTS + 1)
        ]

        with pytest.raises(ValidationError, match="at most 10 items"):
            CatalogItems(workspace_agents=payload)


class TestAiCatalogAgents:
    def test_no_ceiling_and_no_duplicate_check(self):
        """GitLab enforces the per-flow limit and the one-reference-per-item rule; this side does not repeat them."""
        agents = [_catalog_agent(item_id=str(i)) for i in range(25)]
        agents.append(_catalog_agent(item_id="0", version="2.0.0"))

        assert len(CatalogItems(ai_catalog_agents=agents).ai_catalog_agents) == 26

    def test_a_workspace_name_does_not_collide_with_a_catalog_name(self):
        items = CatalogItems(
            workspace_agents=[_agent()], ai_catalog_agents=[_catalog_agent()]
        )

        assert items.workspace_agents[0].name == "workspace/agents/tester"
        assert items.ai_catalog_agents[0].name == "ai-catalog/agents/tester (42)"


class TestFromPayload:
    """Converting a client payload reports its first failure as one typed error.

    The constructor reports through pydantic, which discards the type of anything a validator raises and words its
    messages in request paths. The user sees this message, so it names the agent and the field as the user declared
    them.
    """

    def test_a_valid_payload_builds_items(self):
        items = CatalogItems.from_payload({"workspace_agents": [_agent()]})

        assert [agent.name for agent in items.workspace_agents] == [
            "workspace/agents/tester"
        ]

    def test_an_absent_agents_key_builds_no_items(self):
        assert CatalogItems.from_payload({}).workspace_agents == []

    @pytest.mark.parametrize(
        "payload",
        [
            {"workspace_agents": [_agent(name="  ")]},
            {"workspace_agents": [{"name": "tester"}]},
            {"workspace_agents": [_agent(toolset=["read_file", " "])]},
            {"workspace_agents": [_agent(), _agent()]},
            {
                "workspace_agents": [
                    _agent(name=f"a{i}")
                    for i in range(CatalogItems.MAX_WORKSPACE_AGENTS + 1)
                ]
            },
            {"workspace_agents": "not-a-list"},
        ],
        ids=[
            "blank_name",
            "missing_required_fields",
            "blank_tool_name",
            "duplicate_names",
            "over_the_ceiling",
            "malformed_shape",
        ],
    )
    def test_every_failure_is_an_items_error(self, payload):
        """Field constraints, cross-item rules and shape errors all arrive as one type."""
        with pytest.raises(CatalogItemsError):
            CatalogItems.from_payload(payload)

    @pytest.mark.parametrize(
        ("agents", "message"),
        [
            (
                [_agent(name="reviewer", description="x" * 1184)],
                "Workspace agent 'reviewer': the description has 1184 characters, "
                "the limit is 1024.",
            ),
            ([_agent(name="  ")], "Workspace agent 1: the name is missing."),
            (
                [{"name": "reviewer", "description": "Reviews."}],
                "Workspace agent 'reviewer': the prompt is missing.",
            ),
            (
                [_agent(name="reviewer", toolset=["read_file", " "])],
                "Workspace agent 'reviewer': tool 2 has no name.",
            ),
            (
                [_agent(name="reviewer", toolset="read_file")],
                "Workspace agent 'reviewer': the tools are not valid.",
            ),
            (
                [_agent(name="reviewer"), _agent(name="workspace/agents/reviewer")],
                "Duplicate workspace agent name 'reviewer'.",
            ),
            (
                [_agent(name=f"a{i}") for i in range(11)],
                "Too many workspace agents: 11, the limit is 10.",
            ),
            (
                [_agent(name="a"), _agent(name="b"), _agent(name="n" * 70)],
                "Workspace agent 3: the name has 70 characters, the limit is 64.",
            ),
            (["not-an-agent"], "Workspace agent 1 is not valid."),
            ("not-a-list", "The workspace agents are not valid."),
        ],
        ids=[
            "description_too_long",
            "blank_name",
            "missing_prompt",
            "blank_tool_name",
            "malformed_tools",
            "duplicate_names",
            "over_the_ceiling",
            "name_too_long_falls_back_to_position",
            "malformed_agent",
            "malformed_shape",
        ],
    )
    def test_the_message_is_in_the_users_terms(self, agents, message):
        """The user knows agents by name and writes ``tools``, not request paths."""
        with pytest.raises(CatalogItemsError) as exc_info:
            CatalogItems.from_payload({"workspace_agents": agents})

        assert str(exc_info.value) == message
        for internal in ("workspace_agents", "toolset", "Value error", "workspace/"):
            assert internal not in message

    def test_the_ceiling_is_checked_before_the_items(self):
        """An oversized request reports its size, not the first broken item in it."""
        agents = [_agent(name="  ")] * (CatalogItems.MAX_WORKSPACE_AGENTS + 1)

        with pytest.raises(CatalogItemsError, match="Too many workspace agents"):
            CatalogItems.from_payload({"workspace_agents": agents})

    def test_only_the_first_problem_is_raised_and_all_are_logged(self):
        agents = [
            _agent(name="tester"),
            _agent(name="reviewer", prompt=" "),
            _agent(name="  "),
        ]

        with capture_logs() as logs, pytest.raises(CatalogItemsError) as exc_info:
            CatalogItems.from_payload({"workspace_agents": agents})

        assert str(exc_info.value) == (
            "Workspace agent 'reviewer': the prompt is missing."
        )
        assert logs == [
            {
                "event": "Rejected catalog items",
                "log_level": "info",
                "errors": [
                    "Workspace agent 'reviewer': the prompt is missing.",
                    "Workspace agent 3: the name is missing.",
                ],
            }
        ]

    def test_the_longest_message_fits_a_websocket_close_frame(self):
        """Workhorse forwards the message as a close reason, which it cuts at 123 bytes."""
        agent = _agent(name="é" * MAX_NAME_LENGTH, description="x" * 9_999_999)

        with pytest.raises(CatalogItemsError) as exc_info:
            CatalogItems.from_payload({"workspace_agents": [agent]})

        assert len(str(exc_info.value).encode()) <= 123

    def test_the_validation_error_is_chained(self):
        """The structured pydantic error stays reachable for logging."""
        with pytest.raises(CatalogItemsError) as exc_info:
            CatalogItems.from_payload({"workspace_agents": [_agent(name="  ")]})

        assert isinstance(exc_info.value.__cause__, ValidationError)

    def test_the_constructor_still_reports_through_pydantic(self):
        """``from_payload`` is the boundary door; the constructor stays pydantic-native."""
        with pytest.raises(ValidationError):
            CatalogItems(workspace_agents=[_agent(), _agent()])


class TestWithAiCatalogAgents:
    def test_keeps_the_workspace_agents(self):
        items = CatalogItems(workspace_agents=[_agent()])
        fetched = CatalogItems(ai_catalog_agents=[_catalog_agent()]).ai_catalog_agents

        updated = items.with_ai_catalog_agents(fetched)

        assert updated.workspace_agents == items.workspace_agents
        assert updated.ai_catalog_agents == fetched
        assert updated.ai_catalog_skipped_reason is None
        assert items.ai_catalog_agents == []

    def test_records_the_skip_reason(self):
        updated = CatalogItems().with_ai_catalog_agents([], "duo_features_disabled")

        assert updated.ai_catalog_agents == []
        assert updated.ai_catalog_skipped_reason == "duo_features_disabled"


class TestDescribe:
    @pytest.mark.parametrize(
        ("err", "message"),
        [
            (
                {
                    "type": "string_too_long",
                    "loc": ("workspace_agents", 0, "prompt"),
                    "input": "long",
                },
                "Workspace agent 'tester': the prompt is not valid.",
            ),
            (
                {
                    "type": "missing",
                    "loc": ("workspace_agents", 0, "unlisted_field"),
                    "input": None,
                },
                "Workspace agent 'tester' is not valid.",
            ),
            (
                {"type": "too_long", "loc": ("workspace_agents",), "input": []},
                "The workspace agents are not valid.",
            ),
            (
                {"type": "model_type", "loc": (), "input": "nope"},
                "The catalog items are not valid.",
            ),
        ],
        ids=[
            "string_too_long",
            "field_without_a_message",
            "too_many_agents",
            "not_about_one_kind",
        ],
    )
    def test_an_error_without_its_context_falls_back(self, err, message):
        """Runs while handling a rejection, so a missing key must not turn it into a crash."""
        assert _describe(err, {"workspace_agents": [_agent()]}) == message
