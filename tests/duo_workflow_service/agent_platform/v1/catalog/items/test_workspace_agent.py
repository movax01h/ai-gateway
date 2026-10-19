import pytest
from pydantic import ValidationError

from duo_workflow_service.agent_platform.v1.catalog import (
    WorkspaceAgent,
)
from duo_workflow_service.agent_platform.v1.catalog.items import (
    MAX_DESCRIPTION_LENGTH,
    MAX_NAME_LENGTH,
    MAX_PROMPT_LENGTH,
)

# Every bounded text field and the ceiling it carries, so the two ceiling tests below
# cannot drift apart.
BOUNDED_FIELDS = [
    ("name", MAX_NAME_LENGTH),
    ("description", MAX_DESCRIPTION_LENGTH),
    ("prompt", MAX_PROMPT_LENGTH),
]


def _agent(**overrides) -> dict:
    return {
        "name": "tester",
        "description": "Runs tests.",
        "prompt": "You are a specialist.",
        **overrides,
    }


class TestWorkspaceAgent:
    @pytest.mark.parametrize(
        "missing",
        ["name", "description", "prompt"],
    )
    def test_a_missing_required_field_is_rejected(self, missing):
        """Each is load-bearing on its own, so exactly one is dropped per case.

        The name becomes a graph node, the description is what the coordinator delegates on, and the prompt is the
        agent's own instructions.
        """
        payload = {k: v for k, v in _agent().items() if k != missing}

        with pytest.raises(ValidationError, match=missing):
            WorkspaceAgent(**payload)

    @pytest.mark.parametrize("field", ["name", "description", "prompt"])
    @pytest.mark.parametrize("value", ["", "   "], ids=["empty", "whitespace"])
    def test_a_blank_required_field_is_rejected(self, field, value):
        """Present but blank is no better than absent.

        The flow's prompt is only a placeholder for the item's own, so a blank prompt would leave the agent with no
        instructions at all.
        """
        with pytest.raises(ValidationError):
            WorkspaceAgent(**_agent(**{field: value}))

    @pytest.mark.parametrize("value", ["", "   "], ids=["empty", "whitespace"])
    def test_a_blank_tool_name_is_rejected(self, value):
        """One unusable entry fails the item rather than being quietly dropped."""
        with pytest.raises(ValidationError):
            WorkspaceAgent(**_agent(toolset=["read_file", value]))

    def test_surrounding_whitespace_is_stripped(self):
        """Stripped, not just rejected: the raw value reaches the graph verbatim."""
        agent = WorkspaceAgent(
            **_agent(name="  tester  ", description="  Runs tests.  ")
        )

        assert agent.name == "workspace/agents/tester"  # stripped, then namespaced
        assert agent.description == "Runs tests."

    @pytest.mark.parametrize(
        "name",
        ["tester", "developer_agent", "end"],
        ids=["plain", "authored_component_name", "reserved_graph_node"],
    )
    def test_names_are_namespaced(self, name):
        """Client-supplied names are namespaced, whatever the flow author called things.

        A name that would land on a component, or on a graph node, is a different name.
        """
        assert WorkspaceAgent(**_agent(name=name)).name == f"workspace/agents/{name}"

    def test_an_already_namespaced_name_is_not_prefixed_twice(self):
        """The prefix is idempotent, so a client that echoes a name back is not renamed."""
        agent = WorkspaceAgent(**_agent(name="workspace/agents/tester"))

        assert agent.name == "workspace/agents/tester"

    @pytest.mark.parametrize("field,max_length", BOUNDED_FIELDS)
    def test_the_ceiling_itself_is_accepted(self, field, max_length):
        """The value survives whole, so the bound is a limit rather than a truncation."""
        value = "a" * max_length

        agent = WorkspaceAgent(**_agent(**{field: value}))

        assert getattr(agent, field).endswith(value)

    @pytest.mark.parametrize("field,max_length", BOUNDED_FIELDS)
    def test_one_character_over_the_ceiling_is_rejected(self, field, max_length):
        """Bounded because each value reaches the model, so it costs context and is attack surface."""
        with pytest.raises(ValidationError, match="at most"):
            WorkspaceAgent(**_agent(**{field: "a" * (max_length + 1)}))

    def test_a_tool_name_shares_the_name_ceiling(self):
        with pytest.raises(ValidationError, match="at most"):
            WorkspaceAgent(**_agent(toolset=["t" * (MAX_NAME_LENGTH + 1)]))

    def test_the_ceiling_applies_before_namespacing(self):
        """The limit is on what the client sent; the prefix this service adds is not its problem."""
        agent = WorkspaceAgent(**_agent(name="a" * MAX_NAME_LENGTH))

        assert len(agent.name) > MAX_NAME_LENGTH

    def test_the_toolset_is_optional(self):
        """An agent that only reasons is a legitimate delegation target."""
        agent = WorkspaceAgent(**_agent())

        assert agent.toolset == []
