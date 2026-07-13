from abc import ABC, abstractmethod
from typing import Annotated, Any, ClassVar, Optional, Protocol, Self, override

from gitlab_cloud_connector import CloudConnectorUser
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator

from duo_workflow_service.agent_platform.v1.state import (
    FlowState,
    FlowStateKeys,
    IOKey,
    IOKeyTemplate,
    RuntimeIOKey,
)
from duo_workflow_service.entities.state import WorkflowStatusEnum
from lib.events import GLReportingEventContext

__all__ = [
    "AbortComponent",
    "BaseComponent",
    "EndComponent",
    "ExtraInputVariablesError",
    "MissingInputVariablesError",
    "RouterProtocol",
]


class MissingInputVariablesError(ValueError):
    """Raised when a component's inputs do not cover all required prompt template variables."""


class ExtraInputVariablesError(ValueError):
    """Raised when a component provides input variables not present in the prompt template."""


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

    inputs: list[IOKey | RuntimeIOKey] = Field(default_factory=list)
    name: str
    flow_id: str
    flow_type: GLReportingEventContext
    user: CloudConnectorUser
    environment: str | None = None
    model_tags: list[str] | str | None = None
    # Deprecated alias for model_tags, retained for backward compatibility with
    # Flow Registry v1 configs/clients that still set model_size_preference.
    # Mapped onto model_tags in build_base_component; excluded from serialization.
    model_size_preference: str | None = Field(
        default=None,
        exclude=True,
        description="Deprecated: use model_tags instead.",
    )
    strict_validation: bool = Field(default=False, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def build_base_component(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "inputs" in data:
            data["inputs"] = IOKey.parse_keys(data["inputs"])

        # Backward compatibility: map the deprecated model_size_preference onto
        # model_tags when the new field isn't provided.
        if data.get("model_size_preference") is not None and not data.get("model_tags"):
            data["model_tags"] = data["model_size_preference"]

        return data

    @model_validator(mode="after")
    def validate_base_fields(self) -> Self:
        for inp in self.inputs:
            if isinstance(inp, RuntimeIOKey):
                continue

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
    @override
    def __entry_hook__(self) -> Annotated[str, "Components entry node name"]:
        return "terminate_flow"

    @override
    def attach(
        self, graph: StateGraph, router: Optional[RouterProtocol] = None
    ) -> None:
        graph.add_node(self.__entry_hook__(), self._terminate_flow)
        graph.add_edge(self.__entry_hook__(), END)

    async def _terminate_flow(
        self,
        state: FlowState,  # pylint: disable=unused-argument
    ) -> dict:
        return {FlowStateKeys.STATUS: WorkflowStatusEnum.COMPLETED.value}


class AbortComponent(BaseComponent):
    def __entry_hook__(self) -> Annotated[str, "Components entry node name"]:
        return "abort_flow"

    def attach(
        self, graph: StateGraph, router: Optional[RouterProtocol] = None
    ) -> None:
        graph.add_node(self.__entry_hook__(), self._abort_flow)
        graph.add_edge(self.__entry_hook__(), END)

    async def _abort_flow(
        self,
        state: FlowState,  # pylint: disable=unused-argument
    ) -> dict:
        return {FlowStateKeys.STATUS: WorkflowStatusEnum.ERROR.value}
