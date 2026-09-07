from unittest.mock import Mock

import pytest

from duo_workflow_service.agent_platform.v1.catalog.binding import bind_catalog_items
from duo_workflow_service.agent_platform.v1.catalog.errors import (
    CatalogItemConfigError,
)
from duo_workflow_service.agent_platform.v1.catalog.items import CatalogItems
from duo_workflow_service.agent_platform.v1.catalog.sources import registry
from duo_workflow_service.agent_platform.v1.catalog.sources.base import (
    BindRequest,
    CatalogSource,
)
from duo_workflow_service.agent_platform.v1.catalog.sources.reference import (
    CatalogItemRef,
    CatalogItemSource,
    CatalogItemType,
)
from duo_workflow_service.agent_platform.v1.catalog.sources.workspace_agent import (
    WorkspaceAgentSource,
)


class TestSourceFor:
    def test_it_returns_the_strategy_for_the_pair(self, ref):
        assert isinstance(registry.source_for(ref()), WorkspaceAgentSource)

    @pytest.mark.parametrize(
        "overrides",
        [{"source": "ai-catalog"}, {"item_type": "flow"}],
        ids=["source_with_no_strategy", "kind_with_no_strategy"],
    )
    def test_a_pair_with_no_strategy_is_rejected(self, overrides, ref):
        """Rejected here rather than reaching a strategy that would have to disown it."""
        with pytest.raises(CatalogItemConfigError, match="not supported yet"):
            registry.source_for(ref(**overrides))

    def test_the_error_names_every_pair_that_would_work(self, ref):
        with pytest.raises(
            CatalogItemConfigError, match=r"Supported: workspace/agent_template\."
        ):
            registry.source_for(ref(source="ai-catalog"))


class TestAddingAKind:
    """The extension path: a new kind is a strategy plus a registry entry.

    Nothing in ``binding`` or ``reference`` knows about kinds, so this proves the shared code needs no edit.
    """

    class _FlowSource(CatalogSource):
        SOURCE = CatalogItemSource.WORKSPACE
        ITEM_TYPE = CatalogItemType.FLOW

        def validate_ref(self, ref: CatalogItemRef) -> None:
            pass

        def bind(self, request: BindRequest) -> list[dict]:
            return request.components_config + [{"name": "built-by-the-new-kind"}]

    @pytest.fixture(name="with_flow_source")
    def with_flow_source_fixture(self, monkeypatch):
        strategy = self._FlowSource()
        monkeypatch.setitem(
            registry.SOURCES, (strategy.SOURCE, strategy.ITEM_TYPE), strategy
        )
        return strategy

    def test_the_new_pair_resolves(self, with_flow_source, ref):
        assert registry.source_for(ref(item_type="flow")) is with_flow_source

    def test_errors_offer_it_without_being_told_to(self, with_flow_source, ref):
        """The message is derived from what is registered, so it cannot drift from it."""
        with pytest.raises(
            CatalogItemConfigError,
            match=r"Supported: workspace/agent_template, workspace/flow\.",
        ):
            registry.source_for(ref(source="ai-catalog"))

    def test_binding_dispatches_to_it_untouched(self, with_flow_source):
        # Spelled as a raw dict, the way a flow config carries it.
        entry = {"source": "workspace", "item_type": "flow", "item_id": "*"}

        bound = bind_catalog_items(
            [{"name": "coordinator", "subagents": [dict(entry)]}],
            [CatalogItemRef.model_validate(entry)],
            CatalogItems(),
            Mock(),
        )

        assert [component["name"] for component in bound] == [
            "coordinator",
            "built-by-the-new-kind",
        ]
