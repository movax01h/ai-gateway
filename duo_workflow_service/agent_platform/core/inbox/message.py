from contextlib import suppress
from dataclasses import dataclass
from typing import Annotated, ClassVar, Iterable, Literal, Mapping, Optional, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
)

from duo_workflow_service.agent_platform.core.inbox.part import (
    INBOUND_CONFIG,
    InboundPart,
    Part,
    build_part,
)
from duo_workflow_service.entities.state import ApprovalSource
from duo_workflow_service.slash_commands.goal_parser import is_slash_command
from duo_workflow_service.slash_commands.goal_parser import parse as parse_slash_command

__all__ = [
    "Answer",
    "Component",
    "Decision",
    "Directive",
    "InboundMessage",
    "InvalidInput",
    "Message",
    "Sender",
    "User",
    "inbound_message",
]


# A consumer ignores fields it doesn't know (ADR-002 §8), so a message saved by a newer
# gateway still loads.
_MESSAGE_CONFIG = INBOUND_CONFIG | ConfigDict(extra="ignore")


class User(BaseModel):
    """The user who owns the session, as the sender of what came through their own channel.

    It says whose authority the input carries, not that a human typed it: an automated input on the user's channel,
    such as `notify_me_when`, is the user's too.
    """

    model_config = _MESSAGE_CONFIG

    type: Literal["user"] = "user"


class Component(BaseModel):
    """A component of the flow that writes a message, such as a supervisor delegating to its subagent."""

    model_config = _MESSAGE_CONFIG

    type: Literal["component"] = "component"
    name: str

    @field_validator("name")
    @classmethod
    def _has_a_name(cls, name: str) -> str:
        if not name.strip():
            raise ValueError("A component sender needs a name.")
        return name


# Who wrote a message. Only the user's input carries the user's authority.
Sender = Annotated[User | Component, Field(discriminator="type")]


class _Inbound(BaseModel):
    """What every inbound message holds: an ID, and fields that can't be reassigned once built.

    Strict, so a field is the type it's annotated with. `InvalidInput.from_error` turns what a model refuses into the
    reason the client gets.
    """

    model_config = _MESSAGE_CONFIG

    id: str

    @field_validator("id")
    @classmethod
    def _has_an_id(cls, id: str) -> str:
        # Unique per input, so an empty one can't tell inputs apart.
        if not id:
            raise ValueError("An input needs an ID.")
        return id

    # A `Message` and a `Directive` have a recipient, and an `Answer` has none.
    @field_validator("recipient", check_fields=False)
    @classmethod
    def _names_a_recipient(cls, recipient: Optional[str]) -> Optional[str]:
        # A blank recipient would read as unaddressed to a check for truthiness.
        if recipient is not None and not recipient.strip():
            raise ValueError(
                "A recipient needs a name. Leave it out to address the component the"
                " turn is at."
            )
        return recipient


class _WithParts(_Inbound):
    """An input made of parts, which is a `Message` or an `Answer`.

    Parts are built ones, each as its type's class.
    """

    content: tuple[Part, ...] = ()

    @field_validator("content", mode="before")
    @classmethod
    def _parts_as_a_tuple(cls, content: object) -> object:
        # A checkpoint gives the parts back as a list.
        return tuple(content) if isinstance(content, list) else content

    @property
    def text(self) -> str:
        """The text parts joined with newlines."""
        return "\n".join(part.text for part in self.content)


class Decision(BaseModel):
    """A user's decision on an approval, with the attribution the gate records for each tool call.

    A gate can't guess what an unknown choice means for consent, so only approve and reject are one. `policy_ref` is
    flat strings, as the request's `PolicyRef` carries today.
    """

    model_config = _MESSAGE_CONFIG

    choice: Literal["approve", "reject"]
    approval_source: Optional[str] = None
    policy_ref: Optional[Mapping[str, str]] = None

    # Each wraps Pydantic's own check of the field, and replaces only its message.
    @field_validator("choice", mode="wrap")
    @classmethod
    def _is_a_known_choice(
        cls, choice: object, handler: ValidatorFunctionWrapHandler
    ) -> object:
        try:
            return handler(choice)
        except ValidationError as error:
            raise ValueError(
                f"A decision is approve or reject, not {choice!r}."
            ) from error

    @field_validator("policy_ref", mode="wrap")
    @classmethod
    def _maps_strings_to_strings(
        cls, policy_ref: object, handler: ValidatorFunctionWrapHandler
    ) -> object:
        try:
            return handler(policy_ref)
        except ValidationError as error:
            raise ValueError("A `policy_ref` maps strings to strings.") from error

    @field_validator("approval_source")
    @classmethod
    def _not_a_server_source(cls, approval_source: Optional[str]) -> Optional[str]:
        # Only the server grants a session approval or an auto mode skip; a writer's
        # claim of either is recorded as the user's own decision instead.
        if approval_source in (
            ApprovalSource.SESSION_APPROVAL,
            ApprovalSource.AUTO_MODE,
        ):
            return ApprovalSource.USER_EXPLICIT.value
        return approval_source

    @classmethod
    def optional(
        cls,
        choice: Optional[str],
        approval_source: Optional[str],
        policy_ref: Optional[Mapping[str, str]],
    ) -> Optional[Self]:
        """The decision a writer's fields make, or None when they hold none.

        Raises:
            ValueError: If the fields have attribution but no choice, or aren't a decision.
        """
        if choice is None:
            if approval_source is None and policy_ref is None:
                return None
            raise ValueError("A decision needs a `choice`: approve or reject.")
        return cls.model_validate(
            {
                "choice": choice,
                "approval_source": approval_source,
                "policy_ref": policy_ref,
            }
        )

    @property
    def takes_feedback(self) -> bool:
        """Whether text can go with it: a rejection says why, and an approval carries nothing."""
        return self.choice == "reject"


@dataclass(frozen=True, slots=True)
class InvalidInput:
    """Why an input can't become an `InboundMessage`, as the error the client gets."""

    reason: str

    @classmethod
    def from_error(cls, error: ValueError) -> Self:
        """The reason a model or a builder refused an input, from the first error it reports.

        A validator raises its reason as a `ValueError`. Any other error gets Pydantic's message with the field's
        location.
        """
        if not isinstance(error, ValidationError):
            return cls(str(error))
        # Pydantic reports errors in the order the fields are declared.
        first = error.errors()[0]
        if first["type"] == "value_error":
            return cls(str(first["ctx"]["error"]))
        location = ".".join(str(step) for step in first["loc"])
        return cls(f"`{location}`: {first['msg']}")


class Message(_WithParts):
    """A message at the turn boundary, a steer mid-run, or a supervisor's delegation."""

    type: Literal["message"] = "message"
    sender: Sender
    recipient: Optional[str] = None

    @model_validator(mode="after")
    def _is_a_message(self) -> Self:
        if not self.content:
            raise ValueError(
                "An input needs at least one part, unless it's a decision."
            )
        return self

    # Every kind has `type`, `text`, `content` and `decision`, so a tail renders any of
    # them alike (ADR-004 §6). A message carries no decision.
    @property
    def decision(self) -> None:
        return None


class Directive(_Inbound):
    """A platform directive from the user, which the component that claims it runs instead of rendering.

    Its `sender` is always the user, because only the user runs one. `command` and `argument` are parsed from `text`,
    so they can't disagree with it.
    """

    _COMMANDS: ClassVar[frozenset[str]] = frozenset({"compact"})

    type: Literal["directive"] = "directive"
    # Fixed, so a saved directive from anyone else fails to load instead of reading as the user's.
    sender: User = User()
    text: str
    recipient: Optional[str] = None

    @classmethod
    def from_parts(
        cls,
        id: str,
        content: tuple[Part, ...],
        recipient: Optional[str] = None,
    ) -> Self:
        """The directive a user's parts make.

        Raises:
            ValueError: If the text isn't a platform directive.
        """
        # Every part is text today. When a part type without text joins `Part`, mypy
        # flags this join, and ADR-002's rule comes back: as in legacy chat, a directive
        # is text only, so parts with a file attached make a message.
        return cls(
            id=id, text="\n".join(part.text for part in content), recipient=recipient
        )

    @model_validator(mode="after")
    def _is_a_directive(self) -> Self:
        self._parsed()
        return self

    # Every kind has `type`, `text`, `content` and `decision`, so a tail renders any of
    # them alike (ADR-004 §6). A directive keeps only its text, and carries no decision.
    @property
    def content(self) -> tuple[Part, ...]:
        return ()

    @property
    def decision(self) -> None:
        return None

    @property
    def command(self) -> str:
        return self._parsed()[0]

    @property
    def argument(self) -> Optional[str]:
        return self._parsed()[1]

    def _parsed(self) -> tuple[str, Optional[str]]:
        if is_slash_command(self.text):
            command, argument = parse_slash_command(self.text)
            if command is not None and command in self._COMMANDS:
                return command, argument
        raise ValueError(f"{self.text!r} isn't a platform directive.")


class Answer(_WithParts):
    """An input gate's answer, or an approval decision with or without feedback.

    It has no `recipient`, because a reply is found by the ID in `in_reply_to`, never by address. Its `sender` is always
    the user, because only the user answers.
    """

    type: Literal["answer"] = "answer"
    # Fixed, so a saved answer from anyone else fails to load instead of reading as the user's.
    sender: User = User()
    in_reply_to: str
    decision: Optional[Decision] = None

    @field_validator("in_reply_to")
    @classmethod
    def _replies_to_an_id(cls, in_reply_to: str) -> str:
        if not in_reply_to:
            raise ValueError(
                "An answer needs the ID of what it answers in `in_reply_to`."
            )
        return in_reply_to

    @model_validator(mode="after")
    def _is_an_answer(self) -> Self:
        if self.decision is None:
            if not self.content:
                raise ValueError(
                    "An input needs at least one part, unless it's a decision."
                )
        elif self.content and not self.decision.takes_feedback:
            raise ValueError("An approval carries nothing but the decision.")
        return self


# Saved with its `type`, so a saved message loads back as its own class.
InboundMessage = Annotated[Message | Directive | Answer, Field(discriminator="type")]


def inbound_message(
    id: str,
    content: Iterable[InboundPart],
    *,
    sender: Sender,
    recipient: Optional[str] = None,
    in_reply_to: Optional[str] = None,
    choice: Optional[str] = None,
    approval_source: Optional[str] = None,
    policy_ref: Optional[Mapping[str, str]] = None,
) -> InboundMessage | InvalidInput:
    """Build the message for one input, typed from its fields.

    Args:
        id: Unique per input, set by its writer.
        content: The input's parts.
        sender: Who wrote it: the `User` for what came through the user's own channel, or the `Component` that wrote it.
        recipient: The component it's addressed to. A reply drops it, because it's found by `in_reply_to`.
        in_reply_to: The ID of the approval or question it answers.
        choice: A decision's choice, approve or reject.
        approval_source: Who approved, for a decision.
        policy_ref: The policy that approved, for a decision, as strings mapped to strings.

    Returns:
        The message, or why the input is invalid. Parts are checked first, then each model checks its own fields, and
        a field of the wrong type is invalid too.
    """
    try:
        parts = tuple(build_part(part, id) for part in content)
        decision = Decision.optional(choice, approval_source, policy_ref)
        if in_reply_to is not None:
            if not isinstance(sender, User):
                raise ValueError("Only the user answers, so an answer has no sender.")
            return Answer(
                id=id, content=parts, in_reply_to=in_reply_to, decision=decision
            )
        if decision is not None:
            raise ValueError(
                "A decision needs the ID of the approval it answers in `in_reply_to`."
            )
        # Only the user runs a platform directive. An input that isn't one is a message,
        # and `Message` reports what's wrong with it, the directive's fields included.
        if isinstance(sender, User):
            with suppress(ValueError):
                return Directive.from_parts(id, parts, recipient)
        return Message(id=id, content=parts, sender=sender, recipient=recipient)
    except ValueError as error:
        return InvalidInput.from_error(error)
