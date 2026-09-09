import pytest

from duo_workflow_service.agent_platform.v1.catalog.errors import (
    CatalogItemConfigError,
)
from duo_workflow_service.agent_platform.v1.catalog.sources.reference import (
    CatalogItemSource,
    CatalogItemType,
)
from duo_workflow_service.agent_platform.v1.catalog.sources.workspace_agent import (
    AgentComponentTemplate,
    WorkspaceAgentSource,
)
from lib.feature_flags.context import FeatureFlag, current_feature_flag_context


class TestTheDefaultTemplate:
    """The component config every workspace agent template is built from."""

    def test_each_call_returns_a_fresh_template(self):
        """Built per item, so a rewrite of one cannot reach the next."""
        first = AgentComponentTemplate.default()

        assert AgentComponentTemplate.default() is not first

    def test_it_names_the_prompt_the_platform_ships(self):
        """Named after the source and kind, so two sources including the same kind never share one."""
        assert (
            AgentComponentTemplate.default().prompt_id
            == "workspace_agent_template_prompt"
        )

    def test_it_declares_only_the_delegated_goal(self):
        """A subagent reads nothing else from state; its own prompt is injected per item.

        Declared even though ``bind_to_supervisor`` later swaps it, because prompt-variable coverage is
        checked at construction.
        """
        assert AgentComponentTemplate.default().inputs == [
            {"from": "context:goal", "as": "goal"}
        ]

    def test_ui_log_events_are_forwarded_rather_than_declared(self):
        """``ui_log_events`` is the component's field, so the template only carries it."""
        assert AgentComponentTemplate.default().model_dump(exclude_unset=True)[
            "ui_log_events"
        ] == [
            "on_agent_reasoning",
            "on_tool_execution_success",
            "on_tool_execution_failed",
            "on_agent_final_answer",
        ]


class TestWhatTheSourceOffers:
    def test_it_serves_the_workspace_source(self):
        assert WorkspaceAgentSource.SOURCE is CatalogItemSource.WORKSPACE

    def test_it_builds_agent_templates(self):
        assert WorkspaceAgentSource.ITEM_TYPE is CatalogItemType.AGENT_TEMPLATE

    def test_it_is_experimental_behind_the_workspace_agents_flag(self):
        assert WorkspaceAgentSource.EXPERIMENT_FLAG is FeatureFlag.DAP_WORKSPACE_AGENTS

    @pytest.mark.parametrize(
        "enabled_flags, expected",
        [(set(), False), ({"dap_workspace_agents"}, True)],
        ids=["off", "on"],
    )
    def test_it_is_enabled_only_while_the_flag_is(self, enabled_flags, expected):
        """Spelled as the raw flag name: it is what GitLab pushes in the header."""
        current_feature_flag_context.set(enabled_flags)

        assert WorkspaceAgentSource().is_enabled is expected


class TestValidateRef:
    """Its items arrive with the request, so there is no id or version to select by."""

    def test_the_wildcard_reference_is_accepted(self, ref):
        WorkspaceAgentSource().validate_ref(ref())

    def test_a_concrete_id_is_rejected(self, ref):
        with pytest.raises(CatalogItemConfigError, match="no id to address one by"):
            WorkspaceAgentSource().validate_ref(ref(item_id="123"))

    def test_a_version_is_rejected(self, ref):
        with pytest.raises(CatalogItemConfigError, match="no version to select"):
            WorkspaceAgentSource().validate_ref(ref(version="1.2.0"))


class TestAgentComponentTemplate:
    """The template validates what binding touches and forwards the rest verbatim.

    It does not mirror ``AgentComponent``'s config surface, so anything the component accepts must
    reach it unchanged, including keys this model has never heard of.
    """

    MINIMAL = {"type": "AgentComponent", "prompt_id": "workspace_agent_template_prompt"}

    @pytest.mark.parametrize(
        "overrides,expected",
        [
            ({"type": None}, "type"),
            ({"type": "DeterministicStepComponent"}, "AgentComponent"),
            ({"prompt_id": None}, "prompt_id"),
            # `inputs` is the one key binding rewrites, so its shape is checked
            # here rather than assumed.
            ({"inputs": "context:goal"}, "inputs"),
        ],
        ids=["missing_type", "non_agent_type", "missing_prompt_id", "malformed_inputs"],
    )
    def test_invalid_templates_raise(self, overrides, expected):
        template = {
            key: value
            for key, value in {**self.MINIMAL, **overrides}.items()
            if value is not None
        }

        with pytest.raises(ValueError, match=expected):
            AgentComponentTemplate.model_validate(template)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("max_cycles", 42),
            ("require_tool_approval", True),
            # Falsy values must survive too: they are explicitly set, and dropping
            # one would silently restore the component's default.
            ("require_tool_approval", False),
            ("model_tags", ["fast"]),
            # A field this model has no knowledge of whatsoever.
            ("some_future_agent_field", "value"),
        ],
    )
    def test_component_fields_are_forwarded(self, field, value):
        template = AgentComponentTemplate.model_validate({**self.MINIMAL, field: value})

        assert template.model_dump(exclude_unset=True)[field] == value

    def test_unset_fields_are_left_to_the_component(self):
        """The template invents no defaults for keys it was not given."""
        template = AgentComponentTemplate.model_validate(self.MINIMAL)

        assert set(template.model_dump(exclude_unset=True)) == {"type", "prompt_id"}
