"""What a prompt's `message` block reads, for every kind of message."""

from typing import Any

import pytest

from duo_workflow_service.agent_platform.core.inbox import User, inbound_message
from duo_workflow_service.agent_platform.core.inbox.message import (
    Decision,
    InvalidInput,
)
from duo_workflow_service.agent_platform.core.inbox.part import TextPart
from duo_workflow_service.agent_platform.core.inbox.view import (
    MessageView,
    inbound_messages_view,
)

USER = User()


def _text(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def _message(**kwargs: Any):
    built = inbound_message(id="m1", sender=USER, **kwargs)

    assert not isinstance(built, InvalidInput), built

    return built


_A_MESSAGE = _message(content=[_text("fix the pipeline")])
_A_DIRECTIVE = _message(content=[_text("/compact")])
_AN_ANSWER = _message(content=[], in_reply_to="call-1", choice="approve")


class TestOneMessage:
    def test_the_fields_a_template_may_read(self):
        # The template contract (ADR-004 §6): renaming or removing one breaks shipped templates.
        assert tuple(MessageView.model_fields) == (
            "type",
            "text",
            "content",
            "decision",
        )

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            (
                _A_MESSAGE,
                MessageView(
                    type="message",
                    text="fix the pipeline",
                    content=(TextPart(type="text", text="fix the pipeline"),),
                ),
            ),
            # A directive keeps its text only: the component that claims it runs it.
            (_A_DIRECTIVE, MessageView(type="directive", text="/compact")),
            # An approval carries its decision and no text.
            (
                _AN_ANSWER,
                MessageView(
                    type="answer", text="", decision=Decision(choice="approve")
                ),
            ),
        ],
        ids=["message", "directive", "answer"],
    )
    def test_what_a_template_reads(self, message, expected):
        assert MessageView.of(message) == expected


class TestTheMessagesRenderedTogether:
    def test_it_reads_as_a_list_and_as_joined_text(self):
        view = inbound_messages_view(
            [_message(content=[_text("first")]), _message(content=[_text("second")])]
        )

        assert [message.text for message in view] == ["first", "second"]
        assert view.text == "first\n\nsecond"

    def test_a_message_without_text_leaves_no_blank_line(self):
        view = inbound_messages_view([_AN_ANSWER, _A_MESSAGE])

        assert view.text == "fix the pipeline"
