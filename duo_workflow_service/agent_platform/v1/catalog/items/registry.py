"""The catalog items one run binds, one field per kind.

Workspace agents arrive in the start request. AI Catalog agents are fetched from GitLab before the graph is built.
"""

from typing import Any, ClassVar, Mapping, Optional, Self

from pydantic import BaseModel, Field, ValidationError, model_validator

from duo_workflow_service.agent_platform.v1.catalog.errors import CatalogItemsError
from duo_workflow_service.agent_platform.v1.catalog.items.ai_catalog_agent import (
    AiCatalogAgent,
)
from duo_workflow_service.agent_platform.v1.catalog.items.workspace_agent import (
    WorkspaceAgent,
)

__all__ = ["CatalogItems"]


class CatalogItems(BaseModel):
    """Catalog items a run binds, keyed by kind.

    Kinds are not interchangeable: each has its own model, ceiling and way of reaching the graph. A new kind is a new
    field beside the existing ones.

    Attributes:
        workspace_agents: Agent templates authored in the customer's workspace.
        ai_catalog_agents: Agents published to the AI Catalog, fetched from GitLab before the graph is built.
        ai_catalog_skipped_reason: The GitLab setting that turned the AI Catalog agents off, or ``None``.
    """

    # Held here rather than in the flow config, so a client-supplied config cannot raise it.
    MAX_WORKSPACE_AGENTS: ClassVar[int] = 10

    workspace_agents: list[WorkspaceAgent] = Field(default_factory=list)
    # No ceiling and no duplicate check here: GitLab resolves these from the saved flow's
    # dependency rows and enforces the per-flow limit, so the set arrives already checked.
    ai_catalog_agents: list[AiCatalogAgent] = Field(default_factory=list)
    ai_catalog_skipped_reason: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        """Build from a client payload.

        The constructor reports through pydantic, which discards the type of anything a validator raises. This converts
        it back, so a caller at the request boundary catches one type.

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
            raise CatalogItemsError.from_validation_error(exc) from exc

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
    def validate_workspace_agents(self) -> Self:
        """Check the rules that span the whole set.

        Names must be unique because each becomes one graph component and one ``delegate_task`` choice, so duplicates
        would collapse into one.

        Raises:
            CatalogItemsError: If the ceiling is exceeded, or two agents share a name.
        """
        if len(self.workspace_agents) > self.MAX_WORKSPACE_AGENTS:
            raise CatalogItemsError(
                f"Too many workspace agents declared: {len(self.workspace_agents)} "
                f"(max {self.MAX_WORKSPACE_AGENTS})."
            )

        seen: set[str] = set()
        for agent in self.workspace_agents:
            if agent.name in seen:
                raise CatalogItemsError(
                    f"Duplicate workspace agent name '{agent.name}'."
                )
            seen.add(agent.name)

        return self
