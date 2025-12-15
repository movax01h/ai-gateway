from enum import StrEnum
from typing import (
    Annotated,
    Any,
    ClassVar,
    Final,
    Literal,
    NotRequired,
    Optional,
    Self,
    TypedDict,
    get_args,
    get_origin,
)

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator

# TODO: Remove dependency on legacy duo workflow packages
from duo_workflow_service.entities.state import (
    UiChatLog,
    WorkflowStatusEnum,
    _conversation_history_reducer,
    _ui_chat_log_reducer,
)

__all__ = [
    "FlowEvent",
    "FlowEventType",
    "FlowState",
    "FlowStateKeys",
    "merge_nested_dict",
    "create_nested_dict",
    "merge_nested_dict_reducer",
    "IOKey",
    "IOKeyTemplate",
    "get_vars_from_state",
]


class FlowEventType(StrEnum):
    RESPONSE = "response"
    APPROVE = "approve"
    REJECT = "reject"


class FlowEvent(TypedDict):
    event_type: FlowEventType
    message: NotRequired[str]


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
    alias: Optional[str] = None
    literal: Optional[bool] = False
    optional: Optional[bool] = False

    _target_separator: ClassVar[str] = ":"
    _key_separator: ClassVar[str] = "."

    class _AliasedIOKeyConfig(BaseModel):
        from_: str = Field(alias="from")
        as_: Optional[str] = Field(default=None, alias="as")
        literal_: Optional[bool] = Field(default=False, alias="literal")
        optional_: Optional[bool] = Field(default=False, alias="optional")

    @model_validator(mode="after")
    def parse_valid_target(self) -> Self:
        if self.literal:
            if not self.alias or self.alias.strip() == "":
                raise ValueError("Field 'as' is required when using 'literal: true'")
        else:
            allowed_targets = FlowState.__annotations__.keys()
            if self.target not in allowed_targets:
                raise ValueError(
                    f"Invalid target: {self.target} allowed targets are {allowed_targets}"
                )

            targets_with_subkeys: set[str] = set([])

            for attribute, annotation in FlowState.__annotations__.items():
                annotation_type = get_origin(annotation)

                if annotation_type is None:
                    continue

                if annotation_type is dict:
                    targets_with_subkeys.add(attribute)
                elif (
                    annotation_type is Annotated
                    and get_origin(get_args(annotation)[0]) is dict
                ):
                    targets_with_subkeys.add(attribute)

            if self.target not in targets_with_subkeys and self.subkeys:
                raise ValueError(f"{self.target} does not support subkeys")

        return self

    @classmethod
    def parse_keys(cls, keys: list[str | dict]) -> list[Self]:
        return [cls.parse_key(key) for key in keys]

    @classmethod
    def parse_key(cls, key: str | dict) -> Self:
        alias: Optional[str] = None
        literal: Optional[bool] = False
        optional: Optional[bool] = False

        if isinstance(key, dict):
            key_config = cls._AliasedIOKeyConfig(**key)
            key = key_config.from_
            alias = key_config.as_
            literal = key_config.literal_
            optional = key_config.optional_

        subkeys = None
        if literal:
            target = key
        else:
            target, _, remaining = key.partition(cls._target_separator)

            if remaining:
                subkeys = remaining.split(cls._key_separator)

        return cls(
            target=target,
            subkeys=subkeys,
            alias=alias,
            literal=literal,
            optional=optional,
        )

    def template_variable_from_state(self, state: FlowState) -> dict[str, Any]:
        # self.target presence in state is validated in parse_valid_target
        # thereby state[self.target] will always succeed
        if self.literal:
            return {self.alias: self.target}  # type: ignore[dict-item]

        value = self.value_from_state(state)

        if self.alias:
            return {self.alias: value}

        if not self.subkeys:
            return {self.target: value}

        return {self.subkeys[-1]: value}  # pylint: disable=unsubscriptable-object

    def value_from_state(self, state: FlowState) -> Any:
        # self.target presence in state is validated in parse_valid_target
        # thereby state[self.target] will always succeed
        current = state[self.target]  # type: ignore[literal-required]
        if self.subkeys:
            for key in self.subkeys:  # pylint: disable=not-an-iterable
                current = current.get(key) if self.optional else current[key]
        return current

    def to_nested_dict(self, value: Any) -> dict[str, Any]:
        """Generate nested dictionary matching target and subkeys list, with value supplied as argument.

        Args:
            value: The value to be placed at the nested location

        Returns:
            A nested dictionary with the structure matching target and subkeys

        Examples:
            IOKey(target="context", subkeys=["project", "name"]).to_nested_dict("test")
            # Returns: {"context": {"project": {"name": "test"}}}

            IOKey(target="status").to_nested_dict("active")
            # Returns: {"status": "active"}
        """
        if self.subkeys:
            # Create nested structure: target -> subkeys -> value
            keys = [self.target] + self.subkeys
        else:
            # Simple structure: target -> value
            keys = [self.target]

        return create_nested_dict(keys, value)


class IOKeyTemplate(IOKey):
    COMPONENT_NAME_TEMPLATE: ClassVar[str] = "<name>"
    SENDS_RESPONSE_TO_COMPONENT_NAME_TEMPLATE: ClassVar[str] = (
        "<sends_response_to_component>"
    )

    def to_iokey(self, replacements: dict[str, str]) -> IOKey:
        return IOKey(target=self.target, subkeys=self._resolved_subkeys(replacements))

    def _resolved_subkeys(self, replacements: dict[str, str]) -> list[str] | None:
        if not self.subkeys:
            return None

        return [
            replacements.get(subkey, subkey)
            for subkey in self.subkeys  # pylint: disable=not-an-iterable
        ]


def get_vars_from_state(inputs: list[IOKey], state: FlowState) -> dict[str, Any]:
    variables: dict[str, Any] = {}

    for inp in inputs:
        variables = merge_nested_dict(
            variables, inp.template_variable_from_state(state)
        )

    return variables
