"""Where catalog items come from, and how each kind becomes part of a flow.

One strategy per source and item kind, because sources have little in common: a workspace agent template arrives in full
and becomes an ``AgentComponent``, an AI Catalog item would be fetched by id and version, and a flow item would not
become a single component at all.

Adding a kind:

1. Name it in ``CatalogItemType``, and its source in ``CatalogItemSource`` if that is new too.
2. Add its payload model to ``catalog.items``, and a field for it on ``CatalogItems``.
3. Write a ``CatalogSource`` in a module here, naming its ``SOURCE`` and ``ITEM_TYPE``.
4. List it in ``registry._STRATEGIES``.
5. Name its ``EXPERIMENT_FLAG`` if it should roll out per request; binding checks the flag, so the strategy need not.

:mod:`~duo_workflow_service.agent_platform.v1.catalog.binding` needs no change. It does assume a claim appears in a
component's ``subagents`` list, which suits kinds that become subagents; a kind claimed some other way is where that
assumption would have to move into the strategy.
"""

# flake8: noqa

from duo_workflow_service.agent_platform.v1.catalog.sources.base import *
from duo_workflow_service.agent_platform.v1.catalog.sources.reference import *
from duo_workflow_service.agent_platform.v1.catalog.sources.registry import *
from duo_workflow_service.agent_platform.v1.catalog.sources.workspace_agent import *
