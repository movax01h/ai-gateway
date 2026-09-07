"""The catalog items one start request carries, one field per kind."""

from typing import Any, ClassVar, Mapping, Self

from pydantic import BaseModel, Field, ValidationError, model_validator

from duo_workflow_service.agent_platform.v1.catalog.errors import CatalogItemsError
from duo_workflow_service.agent_platform.v1.catalog.items.workspace_agent import (
    WorkspaceAgent,
)

__all__ = ["CatalogItems"]


class CatalogItems(BaseModel):
    """Catalog items sent with a start request, keyed by kind.

    Kinds are not interchangeable: each has its own model, ceiling and way of reaching the graph. A new kind is a new
    field beside the existing ones.

    Attributes:
        workspace_agents: Agent templates authored in the customer's workspace.
    """

    # Held here rather than in the flow config, so a client-supplied config cannot raise it.
    MAX_WORKSPACE_AGENTS: ClassVar[int] = 10

    workspace_agents: list[WorkspaceAgent] = Field(default_factory=list)

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
