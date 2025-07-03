from abc import ABC, abstractmethod
from typing import Annotated, Any, ClassVar, Optional, Self

from langgraph.graph import StateGraph
from pydantic import BaseModel, model_validator

from duo_workflow_service.agent_platform.experimental.state import FlowState, IOKey

__all__ = ["BaseRouter"]


class BaseRouter(BaseModel, ABC):
    DEFAULT_ROUTE: ClassVar[str] = "default_route"
    _allowed_input_targets: ClassVar[tuple[str, ...]] = ()

    input: Optional[IOKey] = None

    @model_validator(mode="before")
    @classmethod
    def build_base_router(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "input" in data:
            data["input"] = IOKey.parse_key(data["input"])

        return data

    @model_validator(mode="after")
    def validate_input_field(self) -> Self:
        if self.input and self.input.target not in self._allowed_input_targets:
            raise ValueError(
                f"The '{self.__class__.__name__}' router doesn't support the input target '{self.input.target}'."
            )

        return self

    @abstractmethod
    def attach(self, graph: StateGraph):
        pass

    @abstractmethod
    def route(self, state: FlowState) -> Annotated[str, "Next node"]:
        pass
