from typing import (
    Annotated,
    Any,
    ClassVar,
    Final,
    Literal,
    Optional,
    Self,
    TypedDict,
    get_args,
    get_origin,
)

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, model_validator

# TODO: Remove dependency on legacy duo workflow packages
from duo_workflow_service.entities.state import (
    UiChatLog,
    WorkflowStatusEnum,
    _conversation_history_reducer,
    _ui_chat_log_reducer,
)

__all__ = [
    "FlowState",
    "FlowStateKeys",
    "merge_nested_dict",
    "create_nested_dict",
    "merge_nested_dict_reducer",
    "IOKey",
    "get_vars_from_state",
]


def merge_nested_dict(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(existing, dict):
        existing = {}
    if not isinstance(new, dict):
        return new

    result = existing.copy()

    for key, value in new.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_nested_dict(result[key], value)
        else:
            # Overwrite or add new key-value pair
            result[key] = value

    return result


def create_nested_dict(keys: list[str], value: Any) -> dict[str, Any]:
    if not keys:
        return {}

    result: dict[str, Any] = {}
    current = result

    # Navigate through all keys except the last one
    for key in keys[:-1]:
        current[key] = {}
        current = current[key]

    # Set the value at the last key
    current[keys[-1]] = value

    return result


def merge_nested_dict_reducer(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    """Reducer specifically for nested dictionary fields."""
    return merge_nested_dict(left or {}, right or {})


class FlowStateKeys:
    STATUS: Literal["status"] = "status"
    CONVERSATION_HISTORY: Literal["conversation_history"] = "conversation_history"
    UI_CHAT_LOG: Final[str] = "ui_chat_log"
    CONTEXT: Final[str] = "context"


class FlowState(TypedDict):
    status: WorkflowStatusEnum
    conversation_history: Annotated[
        dict[str, list[BaseMessage]], _conversation_history_reducer
    ]
    ui_chat_log: Annotated[list[UiChatLog], _ui_chat_log_reducer]
    context: Annotated[dict[str, Any], merge_nested_dict_reducer]


class IOKey(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    target: str
    subkeys: Optional[list[str]] = None

    _target_separator: ClassVar[str] = ":"
    _key_separator: ClassVar[str] = "."

    @model_validator(mode="after")
    def parse_valid_target(self) -> Self:

        allowed_targets = FlowState.__annotations__.keys()
        if self.target not in allowed_targets:
            raise ValueError(
                f"Invalid target: {self.target} allowed targets are {allowed_targets}"
            )

        targets_with_subkeys: set[str] = set([])

        for attribute, annotation in FlowState.__annotations__.items():
            annontation_type = get_origin(annotation)

            if annontation_type is None:
                continue

            if annontation_type is dict:
                targets_with_subkeys.add(attribute)
            elif (
                annontation_type is Annotated
                and get_origin(get_args(annotation)[0]) is dict
            ):
                targets_with_subkeys.add(attribute)

        if self.target not in targets_with_subkeys and self.subkeys:
            raise ValueError(f"{self.target} does not support subkeys")

        return self

    @classmethod
    def parse_keys(cls, keys: list[str]) -> list[Self]:
        return [cls.parse_key(key) for key in keys]

    @classmethod
    def parse_key(cls, key: str) -> Self:
        target, _, remaining = key.partition(cls._target_separator)

        if not remaining:
            subkeys = None
        else:
            subkeys = remaining.split(cls._key_separator)

        return cls(target=target, subkeys=subkeys)

    def template_variable_from_state(self, state: FlowState) -> dict[str, Any]:
        # self.target presence in state is validated in parse_valid_target
        # thereby state[self.target] will always succeed
        current = state[self.target]  # type: ignore[literal-required]
        if not self.subkeys:
            return {self.target: current}

        for key in self.subkeys:  # pylint: disable=not-an-iterable
            current = current[key]

        return {self.subkeys[-1]: current}  # pylint: disable=unsubscriptable-object

    def value_from_state(self, state: FlowState) -> Any:
        # self.target presence in state is validated in parse_valid_target
        # thereby state[self.target] will always succeed
        current = state[self.target]  # type: ignore[literal-required]
        if self.subkeys:
            for key in self.subkeys:  # pylint: disable=not-an-iterable
                current = current[key]
        return current


def get_vars_from_state(inputs: list[IOKey], state: FlowState) -> dict[str, Any]:
    variables: dict[str, Any] = {}

    for inp in inputs:
        variables = merge_nested_dict(
            variables, inp.template_variable_from_state(state)
        )

    return variables
