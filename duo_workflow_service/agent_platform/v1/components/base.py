from abc import ABC, abstractmethod
from typing import Annotated, Any, ClassVar, Optional, Protocol, Self

from gitlab_cloud_connector import CloudConnectorUser
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator

from duo_workflow_service.agent_platform.v1.state import (
    FlowState,
    FlowStateKeys,
    IOKey,
    IOKeyTemplate,
)
from duo_workflow_service.entities.state import WorkflowStatusEnum
from lib.internal_events.event_enum import CategoryEnum

__all__ = ["RouterProtocol", "BaseComponent", "EndComponent"]


class RouterProtocol(Protocol):
    """Protocol defining the interface for routers used by components."""

    def attach(self, graph: StateGraph) -> None:
        """Attach the router to a StateGraph."""

    def route(self, state: FlowState) -> Annotated[str, "Next node"]:
        """Determine the next node based on the current state."""


class BaseComponent(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _outputs: ClassVar[tuple[IOKeyTemplate, ...]] = ()
    _allowed_input_targets: ClassVar[tuple[str, ...]] = ()

    supported_environments: ClassVar[tuple[str, ...]] = ()

    inputs: list[IOKey] = Field(default_factory=list)
    name: str
    flow_id: str
    flow_type: CategoryEnum
    user: CloudConnectorUser

    @model_validator(mode="before")
    @classmethod
    def build_base_component(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "inputs" in data:
            data["inputs"] = IOKey.parse_keys(data["inputs"])

        return data

    @model_validator(mode="after")
    def validate_base_fields(self) -> Self:
        for inp in self.inputs:
            if inp.literal:
                continue

            if inp.target not in self._allowed_input_targets:
                raise ValueError(
                    f"The '{self.__class__.__name__}' component doesn't support the input target '{inp.target}'."
                )

        return self

    @abstractmethod
    def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
        pass

    @abstractmethod
    def __entry_hook__(self) -> Annotated[str, "Components entry node name"]:
        pass

    @property
    def outputs(self) -> tuple[IOKey, ...]:
        replacements = {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
        return tuple(output.to_iokey(replacements) for output in self._outputs)


class EndComponent(BaseComponent):
    def __entry_hook__(self) -> Annotated[str, "Components entry node name"]:
        return "terminate_flow"

    def attach(
        self, graph: StateGraph, router: Optional[RouterProtocol] = None
    ) -> None:
        graph.add_node(self.__entry_hook__(), self._terminate_flow)
        graph.add_edge(self.__entry_hook__(), END)

    async def _terminate_flow(self, _state: FlowState) -> dict:
        return {FlowStateKeys.STATUS: WorkflowStatusEnum.COMPLETED.value}
