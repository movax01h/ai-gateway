"""The catalog items one run binds, one field per kind.

Workspace agents arrive in the start request. AI Catalog agents are fetched from GitLab before the graph is built.
"""

from typing import Any, ClassVar, Mapping, Optional, Self

import structlog
from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_core import ErrorDetails

from duo_workflow_service.agent_platform.v1.catalog.errors import CatalogItemsError
from duo_workflow_service.agent_platform.v1.catalog.items.ai_catalog_agent import (
    AiCatalogAgent,
)
from duo_workflow_service.agent_platform.v1.catalog.items.workspace_agent import (
    WorkspaceAgent,
)

__all__ = ["CatalogItems"]

logger = structlog.stdlib.get_logger(__name__)


class CatalogItems(BaseModel):
    """Catalog items a run binds, keyed by kind.

    Kinds are not interchangeable: each has its own model, ceiling and way of reaching the graph. A new kind is a new
    field beside the existing ones.

    Attributes:
        workspace_agents: Agent templates authored in the customer's workspace. The ceiling is checked before any item
            is, so an oversized request is rejected without validating its items.
        ai_catalog_agents: Agents published to the AI Catalog, fetched from GitLab before the graph is built.
        ai_catalog_skipped_reason: The GitLab setting that turned the AI Catalog agents off, or ``None``.
    """

    # Held here rather than in the flow config, so a client-supplied config cannot raise it.
    MAX_WORKSPACE_AGENTS: ClassVar[int] = 10

    workspace_agents: list[WorkspaceAgent] = Field(
        default_factory=list, max_length=MAX_WORKSPACE_AGENTS
    )
    # No ceiling and no duplicate check here: GitLab resolves these from the saved flow's
    # dependency rows and enforces the per-flow limit, so the set arrives already checked.
    ai_catalog_agents: list[AiCatalogAgent] = Field(default_factory=list)
    ai_catalog_skipped_reason: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        """Build from a client payload.

        The constructor reports through pydantic, whose messages name request fields and positions. This rewrites them
        in the user's terms and raises the first as one type, so a caller at the request boundary catches one type and
        the user sees one problem. Every problem is logged.

        Args:
            payload: The catalog items section of a start request.

        Returns:
            The validated items.

        Raises:
            CatalogItemsError: If the payload does not describe valid items.
        """
        try:
            return cls.model_validate(payload)
        except ValidationError as exc:
            # Logged as rewritten: pydantic's own text carries the input, which can be
            # a whole client prompt.
            problems = [_describe(err, payload) for err in exc.errors()]
            logger.info("Rejected catalog items", errors=problems)
            raise CatalogItemsError(problems[0]) from exc

    def with_ai_catalog_agents(
        self, agents: list[AiCatalogAgent], skipped_reason: Optional[str] = None
    ) -> Self:
        """Return a copy with what GitLab served for this run's AI Catalog agents.

        Args:
            agents: The agents fetched from GitLab. Empty when they were skipped.
            skipped_reason: The setting that turned them off, or ``None`` when they were served.

        Returns:
            New items with the same workspace agents.
        """
        return self.model_copy(
            update={
                "ai_catalog_agents": list(agents),
                "ai_catalog_skipped_reason": skipped_reason,
            }
        )

    @model_validator(mode="after")
    def validate_unique_workspace_agent_names(self) -> Self:
        """Check that no two workspace agents share a name.

        Names must be unique because each becomes one graph component and one ``delegate_task`` choice, so duplicates
        would collapse into one.

        Raises:
            CatalogItemsError: If two agents share a name.
        """
        seen: set[str] = set()
        for agent in self.workspace_agents:
            if agent.name in seen:
                raise CatalogItemsError(
                    f"Duplicate workspace agent name '{agent.display_name}'."
                )
            seen.add(agent.name)

        return self


def _describe(err: ErrorDetails, payload: Mapping[str, Any]) -> str:
    """Rewrite one pydantic error in the user's terms.

    Errors about one item go to its kind, which knows how the user wrote it. A new kind adds its cases here.

    This runs while a rejection is handled, so it reads the error defensively: a crash here would replace the user's
    message.

    Args:
        err: One error from validating ``payload``.
        payload: The payload as sent, for the name of the item at fault.

    Returns:
        A message for the user.
    """
    ctx = err.get("ctx", {})

    # Raised by a model validator, which pydantic wraps as "Value error, ...".
    if isinstance(ctx.get("error"), CatalogItemsError):
        return str(ctx["error"])

    match err["loc"]:
        case ("workspace_agents",) if (
            err["type"] == "too_long" and "actual_length" in ctx
        ):
            return (
                f"Too many workspace agents: {ctx['actual_length']}, "
                f"the limit is {CatalogItems.MAX_WORKSPACE_AGENTS}."
            )
        case ("workspace_agents", int(index), *field_loc):
            return WorkspaceAgent.describe_error(
                err, tuple(field_loc), payload["workspace_agents"][index], index + 1
            )
        case ("workspace_agents", *_):
            return "The workspace agents are not valid."
        case _:
            return "The catalog items are not valid."
