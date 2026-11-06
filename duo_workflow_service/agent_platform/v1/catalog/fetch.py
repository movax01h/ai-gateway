"""Fetch the AI Catalog items a run's flow references.

GitLab resolves the saved flow's ``include`` entries through its dependency rows and checks each one for the
current user. The response names the root item and, under ``sub_items``, each referenced item with a definition or
with an error that says why it was refused. When a setting turns the sub-items off, ``sub_items`` is empty and
``sub_items_skipped_reason`` names the setting. The run then proceeds without them.

Only ``agent`` sub-items are read here. ``flow`` sub-items are left for a later iteration.

For a saved flow that the client asked the service to fetch, the root item's ``definition`` is the flow itself.
:func:`fetch_catalog_flow` reads it and the agents from the same response, so the run makes one request.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import structlog
from pydantic import ValidationError

from duo_workflow_service.agent_platform.v1.catalog.errors import (
    CatalogFlowUnavailableError,
    CatalogItemsError,
    CatalogItemsFetchError,
)
from duo_workflow_service.agent_platform.v1.catalog.items import AiCatalogAgent
from duo_workflow_service.gitlab.http_client import GitlabHttpClient

__all__ = [
    "FetchedCatalogAgents",
    "FetchedCatalogFlow",
    "fetch_catalog_agents",
    "fetch_catalog_flow",
]

logger = structlog.stdlib.get_logger(__name__)

_AGENT_ITEM_TYPE = "agent"


@dataclass(frozen=True)
class FetchedCatalogAgents:
    """What GitLab served for one run.

    Attributes:
        agents: One agent for each ``agent`` sub-item. Empty when the flow references none, or when they were skipped.
        skipped_reason: The setting that turned the sub-items off, or ``None`` when they were served.
    """

    agents: list[AiCatalogAgent] = field(default_factory=list)
    skipped_reason: Optional[str] = None


@dataclass(frozen=True)
class FetchedCatalogFlow:
    """A saved flow and its agents, read from one catalog items response.

    Attributes:
        definition: The root item's ``definition``, which is the flow config.
        agents: The agents and skip reason from the same response.
    """

    definition: dict[str, Any]
    agents: FetchedCatalogAgents


async def fetch_catalog_agents(
    client: GitlabHttpClient, workflow_id: str
) -> FetchedCatalogAgents:
    """Return the AI Catalog agents the run's flow references, or the reason they were skipped.

    Args:
        client: The run's GitLab client. The call becomes a RunHTTPRequest that Workhorse serves with the run's
            token.
        workflow_id: The run.

    Returns:
        The served agents, or an empty set with the skip reason.

    Raises:
        CatalogItemsFetchError: If GitLab answers with an error status, or the body is not an object.
        CatalogItemsError: If a sub-item was refused, or an agent does not pass field validation.
    """
    item = await _fetch_root_item(client, workflow_id)
    return _agents_of(item, workflow_id)


async def fetch_catalog_flow(
    client: GitlabHttpClient, workflow_id: str
) -> FetchedCatalogFlow:
    """Return the saved flow of the run and its AI Catalog agents, from one request.

    Args:
        client: The run's GitLab client.
        workflow_id: The run. GitLab reads the flow version from the workflow record.

    Returns:
        The root definition and the agents.

    Raises:
        CatalogFlowUnavailableError: If GitLab answers 404, or the response has no root definition.
        CatalogItemsFetchError: If GitLab answers with another error status, or the body is not an object.
        CatalogItemsError: If a sub-item was refused, or an agent does not pass field validation.
    """
    try:
        item = await _fetch_root_item(client, workflow_id)
    except _NotFoundError as exc:
        raise CatalogFlowUnavailableError(str(exc)) from exc

    definition = item.get("definition")
    if not isinstance(definition, dict) or not definition:
        # A 2xx that structurally lacks the root definition is a malformed body, not a
        # clean 404 "no saved flow". Raise with a CatalogItemsFetchError cause (no status,
        # so is_catalog_refusal is False) so the server maps it to INTERNAL. Without a cause
        # the NotifiableAgentException leaves last_error None and the run ends as UNKNOWN.
        raise CatalogFlowUnavailableError(
            f"Catalog items response for workflow {workflow_id} has no root definition"
        ) from CatalogItemsFetchError(
            f"Catalog items response for workflow {workflow_id} has no root definition"
        )

    return FetchedCatalogFlow(
        definition=definition, agents=_agents_of(item, workflow_id)
    )


class _NotFoundError(CatalogItemsFetchError):
    """GitLab answered 404: an older GitLab without the endpoint, or a workflow it does not serve."""


async def _fetch_root_item(
    client: GitlabHttpClient, workflow_id: str
) -> dict[str, Any]:
    """Send the one GET and return the root ``item``, or an empty dict when the body has none.

    Raises:
        _NotFoundError: If GitLab answers 404.
        CatalogItemsFetchError: If GitLab answers with another error status, or the body is not an object.
    """
    # No timeout of our own: the client retries 401 and 429 for up to 39 seconds to ride out replica lag.
    response = await client.aget(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}/catalog_items",
        parse_json=True,
    )

    if not response.is_success():
        logger.error(
            "Failed to fetch catalog items",
            workflow_id=workflow_id,
            status_code=response.status_code,
        )
        error_cls = (
            _NotFoundError if response.status_code == 404 else CatalogItemsFetchError
        )
        raise error_cls(
            f"GitLab refused the catalog items request for workflow {workflow_id}: "
            f"HTTP {response.status_code}: {_message_of(response.body)}",
            status_code=response.status_code,
        )

    if not isinstance(response.body, dict):
        raise CatalogItemsFetchError(
            f"Unexpected catalog items response for workflow {workflow_id}: "
            f"expected object, got {type(response.body).__name__}"
        )

    item = response.body.get("item")
    return item if isinstance(item, dict) else {}


def _agents_of(item: dict[str, Any], workflow_id: str) -> FetchedCatalogAgents:
    """Build the agents from the root item's ``sub_items``.

    Raises:
        CatalogItemsError: If a sub-item was refused, or an agent does not pass field validation.
    """
    sub_items = item.get("sub_items")
    if not isinstance(sub_items, list):
        sub_items = []
    skipped_reason = item.get("sub_items_skipped_reason") or None
    agents = [_agent_from(node) for node in sub_items if _is_agent(node)]

    logger.info(
        "Fetched catalog items",
        workflow_id=workflow_id,
        agent_count=len(agents),
        sub_item_count=len(sub_items),
        skipped_reason=skipped_reason,
    )

    return FetchedCatalogAgents(agents=agents, skipped_reason=skipped_reason)


def _is_agent(node: Any) -> bool:
    return isinstance(node, dict) and node.get("item_type") == _AGENT_ITEM_TYPE


def _agent_from(node: dict) -> AiCatalogAgent:
    """Build one agent from a ``sub_items`` node, or refuse it the way GitLab did.

    Raises:
        CatalogItemsError: If the node is an error node, or its definition is not a valid agent.
    """
    error = node.get("error")
    if error:
        detail = error if isinstance(error, dict) else {"code": str(error)}
        # GitLab leaves the name out when the user may not see it. The id is
        # always present, because the flow's own include entry shows it.
        subject = node.get("name") or node.get("item_id")
        raise CatalogItemsError(
            f"AI Catalog agent '{subject}' ({node.get('item_id')}) was refused: "
            f"{detail.get('message') or detail.get('code')}"
        )

    definition = node.get("definition")
    if not isinstance(definition, dict):
        # A definition of the wrong shape has no prompt, so validation names it.
        definition = {}
    try:
        return AiCatalogAgent.model_validate(
            {
                "item_id": node.get("item_id"),
                "version": node.get("version"),
                "name": node.get("name"),
                "description": node.get("description"),
                "prompt": definition.get("prompt"),
                "toolset": definition.get("toolset") or [],
            }
        )
    except ValidationError as exc:
        raise CatalogItemsError.from_validation_error(exc) from exc


def _message_of(body: Any) -> str:
    if isinstance(body, dict) and body.get("message"):
        return str(body["message"])
    return str(body)
