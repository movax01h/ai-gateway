"""What a prompt's `message` block reads when it renders the messages a component claims together.

A template never sees a message object. Every kind of message reads through the same four fields (ADR-004 §6), so a
template that doesn't branch on `type` works for all of them, a typo in a template can't reach a field the contract
doesn't name, and a field the message model gains later reaches templates only when it is added here on purpose.
"""

from collections.abc import Iterable
from typing import Literal, Optional, Self

from pydantic import BaseModel, ConfigDict

from duo_workflow_service.agent_platform.core.inbox.message import (
    Decision,
    InboundMessage,
)
from duo_workflow_service.agent_platform.core.inbox.part import Part

__all__ = [
    "InboundMessagesView",
    "MessageView",
    "inbound_messages_view",
]


class MessageView(BaseModel):
    """One message as a template reads it: the same fields whatever its kind.

    Renaming or removing a field breaks shipped templates; adding one is additive.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["message", "directive", "answer"]
    text: str
    # The parts as they arrived. A content block is built from them later, when attachments join the rendered message.
    content: tuple[Part, ...] = ()
    # An answer's decision; a message and a directive carry none.
    decision: Optional[Decision] = None

    @classmethod
    def of(cls, message: InboundMessage) -> Self:
        """The view of one message.

        Every kind exposes these four, so no kind is special-cased here.
        """
        return cls(
            type=message.type,
            text=message.text,
            content=message.content,
            decision=message.decision,
        )


class InboundMessagesView(list[MessageView]):
    """The messages a component renders together, as a template reads them.

    A list, so `{% for message in inbound_messages %}` works, with `text` for the common case of rendering them as
    one block.
    """

    @property
    def text(self) -> str:
        """The messages' text, joined with blank lines; a message without text adds none."""
        return "\n\n".join(message.text for message in self if message.text)


def inbound_messages_view(messages: Iterable[InboundMessage]) -> InboundMessagesView:
    """Build what a prompt's `message` block reads for the messages rendered together.

    Args:
        messages: The messages, as the inbox delivers them. Whoever reads them from a checkpoint, where they are
            plain mappings, loads them back as messages first.

    Returns:
        The view a template renders.
    """
    return InboundMessagesView(MessageView.of(message) for message in messages)
