import pytest

from duo_workflow_service.agent_platform.v1.catalog.errors import (
    CatalogItemConfigError,
)
from duo_workflow_service.agent_platform.v1.catalog.sources.reference import (
    CatalogItemSource,
    CatalogItemType,
)
from duo_workflow_service.agent_platform.v1.catalog.sources.workspace_agent import (
    WorkspaceAgentSource,
    agent_template_prompt,
)
from lib.feature_flags.context import FeatureFlag, current_feature_flag_context

_PROMPT_ID = "workspace_agent_template_prompt"


class TestTheShippedPrompt:
    def test_it_is_named_after_the_source_and_kind(self):
        """Derived from the source and kind, so two sources including the same kind never share one."""
        assert agent_template_prompt().prompt_id == _PROMPT_ID


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
