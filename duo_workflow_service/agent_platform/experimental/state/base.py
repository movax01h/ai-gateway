from enum import StrEnum
from typing import (
    Annotated,
    Any,
    Callable,
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

import structlog
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator

from duo_workflow_service.agent_platform.v1.state.base import (
    conversation_history_replace_reducer,
)
from duo_workflow_service.entities.state import (
    UiChatLog,
    WorkflowStatusEnum,
    _ui_chat_log_reducer,
)

logger = structlog.stdlib.get_logger("experimental_state")

__all__ = [
    "FlowEvent",
    "FlowEventType",
    "FlowState",
    "FlowStateKeys",
    "merge_nested_dict",
    "create_nested_dict",
    "merge_nested_dict_reducer",
    "BaseIOKey",
    "IOKey",
    "IOKeyTemplate",
    "RuntimeIOKey",
    "get_vars_from_state",
    "conversation_history_replace_reducer",
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
        dict[str, list[BaseMessage]], conversation_history_replace_reducer
    ]
    ui_chat_log: Annotated[list[UiChatLog], _ui_chat_log_reducer]
    context: Annotated[dict[str, Any], merge_nested_dict_reducer]


class BaseIOKey(BaseModel):
    """Shared base for all IOKey variants.

    Holds the fields and class-level configuration that are common to both
    ``IOKey`` (build-time, fully resolved) and ``RuntimeIOKey`` (resolved at
    graph-execution time).  Concrete subclasses add the fields and validators
    that are specific to their resolution tier.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

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


class IOKey(BaseIOKey):
    target: str

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
            for key in self.subkeys:
                if self.optional:
                    if current is None:
                        return None
                    current = current.get(key)
                else:
                    current = current[key]
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
    SUPERVISOR_NAME_TEMPLATE: ClassVar[str] = "<supervisor_name>"
    SUBAGENT_NAME_TEMPLATE: ClassVar[str] = "<subagent_name>"
    SUBSESSION_ID_TEMPLATE: ClassVar[str] = "<subsession_id>"

    def to_iokey(self, replacements: dict[str, str]) -> IOKey:
        return IOKey(
            target=self.target,
            subkeys=self._resolved_subkeys(replacements),
            optional=self.optional,
        )

    def _resolved_subkeys(self, replacements: dict[str, str]) -> list[str] | None:
        if not self.subkeys:
            return None

        return [
            replacements.get(subkey, subkey)
            for subkey in self.subkeys  # pylint: disable=not-an-iterable
        ]


IOKeyFactory = Callable[[FlowState], IOKey]


class RuntimeIOKey(BaseIOKey):
    """An ``IOKey`` whose concrete identity is resolved at graph-execution time.

    ``IOKey`` and ``IOKeyTemplate`` are both resolved at graph-build time (the
    latter via ``to_iokey(replacements)``).  ``RuntimeIOKey`` adds a third tier
    for keys whose concrete path depends on runtime state — for example, a
    subsession-scoped output key whose subsession ID is only known once the
    graph is running.

    Unlike ``IOKey``, ``RuntimeIOKey`` does not carry a ``target`` field.  The
    concrete ``IOKey`` (including its ``target``) is produced at runtime by the
    ``factory`` callable.  The ``alias`` field (inherited from ``BaseIOKey``,
    required here) serves as the statically-declared Jinja2 template variable
    name, enabling prompt-input validation without graph execution.

    Construction::

        RuntimeIOKey(
            alias="final_answer",
            factory=lambda state: some_key_template.to_iokey({...state...}),
        )

    Args:
        alias: Required. The static template variable name (e.g. ``"goal"``).
        factory: Callable ``(state) -> IOKey`` that resolves the concrete key
            at runtime.
    """

    factory: IOKeyFactory

    @model_validator(mode="after")
    def validate_alias(self) -> Self:
        """Ensure ``alias`` is provided.

        ``alias`` is the only build-time requirement for ``RuntimeIOKey`` — it
        is used by prompt-input validators to reference the template variable
        name without executing the graph.
        """
        if not self.alias or self.alias.strip() == "":
            raise ValueError("Field 'alias' is required for RuntimeIOKey")
        return self

    @property
    def template_variable_name(self) -> str:
        """Return the statically-declared template variable name.

        This allows prompt-input validators to check that every template variable has a corresponding input key without
        executing the graph.
        """
        return self.alias  # type: ignore[return-value]

    def to_iokey(self, state: FlowState) -> IOKey:
        """Resolve the concrete ``IOKey`` for the given runtime state.

        Args:
            state: Current flow state used to determine the concrete key.

        Returns:
            The resolved ``IOKey`` instance.
        """
        return self.factory(state)

    def to_nested_dict(self, value: Any, state: FlowState) -> dict[str, Any]:
        """Resolve the concrete ``IOKey`` at runtime and delegate ``to_nested_dict``.

        Requires ``state`` so that the concrete key (including its ``target``
        and ``subkeys``) can be resolved before building the nested dictionary.

        Args:
            value: The value to be placed at the nested location.
            state: Current flow state used to resolve the concrete ``IOKey``.

        Returns:
            A nested dictionary with the structure matching the resolved key's
            ``target`` and ``subkeys``.

        Examples:
            key = RuntimeIOKey(
                alias="status",
                factory=lambda _: IOKey(target="status"),
            )
            key.to_nested_dict("active", state)
            # Returns: {"status": "active"}
        """
        return self.to_iokey(state).to_nested_dict(value)

    def value_from_state(self, state: FlowState) -> Any:
        """Resolve the concrete IOKey at runtime and read its value from state."""
        return self.factory(state).value_from_state(state)

    def template_variable_from_state(self, state: FlowState) -> dict[str, Any]:
        """Resolve the concrete IOKey at runtime and return its template variable."""
        return self.factory(state).template_variable_from_state(state)


def get_vars_from_state(inputs: list[IOKey], state: FlowState) -> dict[str, Any]:
    variables: dict[str, Any] = {}

    for inp in inputs:
        variables = merge_nested_dict(
            variables, inp.template_variable_from_state(state)
        )

    return variables
