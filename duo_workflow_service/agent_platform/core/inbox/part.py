from collections.abc import Mapping
from typing import Literal, Self

import structlog
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ModelWrapValidatorHandler,
    ValidationError,
    model_validator,
)

__all__ = [
    "INBOUND_CONFIG",
    "InboundPart",
    "Part",
    "TextPart",
    "build_part",
]

log = structlog.stdlib.get_logger(__name__)


# JSON has no NaN or infinity, and providers reject them. Every inbound model sets it,
# because a part's `JsonValue` takes it from the outermost model that builds it.
INBOUND_CONFIG = ConfigDict(
    frozen=True, strict=True, extra="forbid", allow_inf_nan=False
)


class _Part(BaseModel):
    """What every part holds, which maps one to one to a LangChain content block: a `type`, and only JSON values.

    It's built only as a concrete part type, and bytes never belong in one.
    """

    model_config = INBOUND_CONFIG | ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)

    type: str

    @model_validator(mode="wrap")
    @classmethod
    def _holds_only_json(
        cls, part: object, handler: ModelWrapValidatorHandler[Self]
    ) -> Self:
        # Wraps every other check, so a part type's own reasons pass through, and what's
        # left is a value Pydantic refused as JSON.
        try:
            return handler(part)
        except ValidationError as error:
            if error.errors()[0]["type"] == "value_error":
                raise
            raise ValueError("A part holds only JSON values.") from error


class TextPart(_Part):
    """Text, the one part type the gateway reads today."""

    type: Literal["text"]
    text: str

    @model_validator(mode="before")
    @classmethod
    def _has_text(cls, part: object) -> object:
        if not isinstance(part, Mapping) or part.get("type") != "text":
            raise ValueError("A `TextPart` is a part of type `text`.")
        # Providers reject blank text blocks.
        text = part.get("text")
        if not (isinstance(text, str) and text.strip()):
            raise ValueError("A text part needs text that isn't blank.")
        return part


# A built part, as one of the part types this gateway reads. A new part type joins it.
Part = TextPart

# What a writer passes for a part: a mapping with a `type`, or a part already built.
InboundPart = Mapping[str, object] | Part


def build_part(part: InboundPart, input_id: str) -> Part:
    """Build a writer's part as the class of its type.

    A type this gateway doesn't read is refused, because nothing here can render it: a newer client sent it, and the
    gateway needs an upgrade.

    Args:
        part: A mapping with a `type`, or a part already built.
        input_id: The ID of the input the part belongs to, which a refusal logs.

    Returns:
        The built part.

    Raises:
        ValueError: If the part has no `type`, has one this gateway doesn't read, or isn't valid for its type.
    """
    if isinstance(part, TextPart):
        return part
    kind = part.get("type") if isinstance(part, Mapping) else None
    if not isinstance(kind, str):
        raise ValueError("Every part needs a `type`.")
    if kind != "text":
        log.warning(
            "Refused an unsupported part type", part_type=kind, input_id=input_id
        )
        raise ValueError(
            f"This AI Gateway doesn't support `{kind}` parts. Upgrade it, or send the"
            " input without them."
        )
    return TextPart.model_validate(part)
