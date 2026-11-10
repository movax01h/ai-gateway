import pytest
from pydantic import TypeAdapter, ValidationError

from duo_workflow_service.agent_platform.core.inbox import (
    Answer,
    Component,
    Decision,
    Directive,
    InboundMessage,
    InvalidInput,
    Message,
    TextPart,
    User,
    inbound_message,
)
from duo_workflow_service.checkpointer.utils.serializer import CheckpointSerializer

USER = User()
SUPERVISOR = Component(name="supervisor")


def _text(text: str) -> dict[str, str]:
    # What a writer passes for text.
    return {"type": "text", "text": text}


def _text_part(text: str) -> TextPart:
    return TextPart(type="text", text=text)


APPROVE = Decision(choice="approve", approval_source="user")
POLICY_APPROVAL = Decision(
    choice="approve", approval_source="policy", policy_ref={"name": "allow-reads"}
)


class TestInboundMessageTypeDerivation:
    @pytest.mark.parametrize(
        ("content", "sender", "in_reply_to", "choice", "expected"),
        [
            ([_text("Fix the build")], USER, None, None, Message),
            ([_text("/compact")], USER, None, None, Directive),
            ([_text("/compact keep the plan")], USER, None, None, Directive),
            ([_text("/explain this")], USER, None, None, Message),
            ([_text("/home/me/notes.md")], USER, None, None, Message),
            ([_text("/compact")], SUPERVISOR, None, None, Message),
            ([_text("Use main")], USER, "question-1", None, Answer),
            ([], USER, "msg-1", "approve", Answer),
            ([], USER, "msg-1", "reject", Answer),
            ([_text("Not that file")], USER, "msg-1", "reject", Answer),
            ([_text("/compact")], USER, "msg-1", "reject", Answer),
            ([_text("/compact")], USER, "question-1", None, Answer),
        ],
        ids=[
            "plain text is a message",
            "a platform directive",
            "a platform directive with an instruction",
            "a slash command the platform doesn't run is a message",
            "a path is a message",
            "a directive's text from a component is a message",
            "text in reply to a question is an answer",
            "an approval is an answer",
            "a rejection without feedback is an answer",
            "a rejection with feedback is an answer",
            "a directive's text in feedback stays an answer",
            "a directive's text answering a question is an answer",
        ],
    )
    def test_type_follows_from_the_fields(
        self, content, sender, in_reply_to, choice, expected
    ):
        message = inbound_message(
            "row-1",
            content,
            sender=sender,
            in_reply_to=in_reply_to,
            choice=choice,
        )

        assert isinstance(message, expected)

    @pytest.mark.parametrize(
        ("content", "fields", "expected"),
        [
            (
                [_text("Not that file")],
                {"sender": SUPERVISOR, "recipient": "reviewer"},
                Message(
                    id="row-1",
                    content=(_text_part("Not that file"),),
                    sender=SUPERVISOR,
                    recipient="reviewer",
                ),
            ),
            (
                [],
                {
                    "sender": USER,
                    "recipient": "developer",
                    "in_reply_to": "msg-1",
                    "choice": "approve",
                    "approval_source": "policy",
                    "policy_ref": {"name": "allow-reads"},
                },
                Answer(id="row-1", in_reply_to="msg-1", decision=POLICY_APPROVAL),
            ),
            (
                [],
                {
                    "sender": USER,
                    "in_reply_to": "msg-1",
                    "choice": "approve",
                    "approval_source": "session_approval",
                },
                Answer(
                    id="row-1",
                    in_reply_to="msg-1",
                    decision=Decision(
                        choice="approve", approval_source="user_explicit"
                    ),
                ),
            ),
        ],
        ids=[
            "a message",
            "an approval keeps its attribution and drops its recipient",
            "a writer's session approval is the user's own",
        ],
    )
    def test_fields_carry_over(self, content, fields, expected):
        assert inbound_message("row-1", content, **fields) == expected

    @pytest.mark.parametrize("claimed", ["session_approval", "auto_mode"])
    def test_a_writer_cannot_claim_a_server_only_source(self, claimed):
        decision = Decision.optional("approve", claimed, None)

        assert decision is not None
        assert decision.approval_source == "user_explicit"

    @pytest.mark.parametrize(
        ("content", "text", "argument"),
        [
            (
                [_text("/compact keep the plan")],
                "/compact keep the plan",
                "keep the plan",
            ),
            ([_text("/ compact")], "/ compact", None),
            (
                [_text("/compact"), _text("keep the plan")],
                "/compact\nkeep the plan",
                "keep the plan",
            ),
        ],
        ids=[
            "with an instruction",
            "with a space after the slash, as in legacy chat",
            "split across text parts",
        ],
    )
    def test_a_directive_carries_its_command(self, content, text, argument):
        directive = inbound_message(
            "row-1", content, sender=USER, recipient="developer"
        )

        assert directive == Directive(id="row-1", text=text, recipient="developer")
        assert isinstance(directive, Directive)
        assert (directive.command, directive.argument) == ("compact", argument)


class TestInboundMessageValidation:
    @pytest.mark.parametrize(
        ("id", "content", "in_reply_to", "choice", "reason"),
        [
            ("", [_text("Hi")], None, None, "An input needs an ID."),
            (
                "row-1",
                [],
                None,
                None,
                "An input needs at least one part, unless it's a decision.",
            ),
            ("row-1", [], "question-1", None, "unless it's a decision"),
            ("row-1", [_text("Hi")], "", None, "An answer needs the ID"),
            ("row-1", [_text("  \n")], None, None, "isn't blank"),
            ("row-1", [], None, "reject", "A decision needs the ID"),
            (
                "row-1",
                [_text("Sure")],
                "msg-1",
                "approve",
                "carries nothing but the decision",
            ),
        ],
        ids=[
            "no id",
            "no parts",
            "an answer without parts",
            "a blank in_reply_to",
            "an invalid part",
            "a decision answering nothing",
            "an approval with text",
        ],
    )
    def test_invalid_input_says_what_to_fix(
        self, id, content, in_reply_to, choice, reason
    ):
        result = inbound_message(
            id, content, sender=USER, in_reply_to=in_reply_to, choice=choice
        )

        assert isinstance(result, InvalidInput)
        assert reason in result.reason

    @pytest.mark.parametrize(
        ("fields", "reason"),
        [
            ({"sender": USER, "recipient": "  "}, "A recipient needs a name"),
            (
                {"sender": SUPERVISOR, "in_reply_to": "question-1"},
                "Only the user answers",
            ),
            (
                {"sender": SUPERVISOR, "in_reply_to": "msg-1", "choice": "reject"},
                "Only the user answers",
            ),
            (
                {"sender": USER, "in_reply_to": "msg-1", "choice": "maybe"},
                "approve or reject, not 'maybe'",
            ),
            (
                {"sender": USER, "in_reply_to": "msg-1", "approval_source": "user"},
                "A decision needs a `choice`",
            ),
            (
                {
                    "sender": USER,
                    "in_reply_to": "msg-1",
                    "choice": "reject",
                    "policy_ref": {"origin": {"name": "reads"}},
                },
                "A `policy_ref` maps strings to strings.",
            ),
            (
                {
                    "sender": USER,
                    "in_reply_to": "msg-1",
                    "choice": "reject",
                    "approval_source": 1,
                },
                "`approval_source`: Input should be a valid string",
            ),
        ],
        ids=[
            "a blank recipient",
            "a component's answer",
            "a component's decision",
            "an unknown choice",
            "attribution without a choice",
            "a policy_ref with a nested value",
            "a field of the wrong type",
        ],
    )
    def test_invalid_fields_say_what_to_fix(self, fields, reason):
        result = inbound_message("row-1", [_text("No")], **fields)

        assert isinstance(result, InvalidInput)
        assert reason in result.reason

    def test_a_component_needs_a_name(self):
        with pytest.raises(ValueError, match="A component sender needs a name."):
            Component(name=" ")

    @pytest.mark.parametrize(
        ("content", "reason"),
        [
            ([{"type": "text"}], "A text part needs text that isn't blank."),
            ([_text("Hi")], "An input needs an ID."),
        ],
        ids=["parts before fields", "fields in the order they're declared"],
    )
    def test_the_first_reason_follows_the_order_of_the_checks(self, content, reason):
        result = inbound_message("", content, sender=USER, recipient=" ")

        assert result == InvalidInput(reason)

    def test_a_generator_of_parts_is_read_once(self):
        message = inbound_message(
            "row-1", (part for part in [_text("Hi")]), sender=USER
        )

        assert isinstance(message, Message)
        assert message.text == "Hi"


class TestInboundMessageShape:
    @pytest.mark.parametrize(
        ("content", "fields", "keys"),
        [
            (
                [_text("Fix the build")],
                {},
                ("message", "Fix the build", (_text_part("Fix the build"),), None),
            ),
            ([_text("/compact")], {}, ("directive", "/compact", (), None)),
            (
                [],
                {
                    "in_reply_to": "msg-1",
                    "choice": "approve",
                    "approval_source": "user",
                },
                ("answer", "", (), APPROVE),
            ),
        ],
        ids=["a message", "a directive", "an answer"],
    )
    def test_every_kind_has_the_same_keys(self, content, fields, keys):
        message = inbound_message("row-1", content, sender=USER, **fields)

        assert (message.type, message.text, message.content, message.decision) == keys

    @pytest.mark.parametrize(
        "fields", [{}, {"in_reply_to": "q-1"}], ids=["a message", "an answer"]
    )
    def test_text_joins_the_text_parts_with_newlines(self, fields):
        message = inbound_message(
            "row-1", [_text("First"), _text("Second")], sender=USER, **fields
        )

        assert message.text == "First\nSecond"

    @pytest.mark.parametrize(
        "message",
        [
            Message(id="row-1", content=(_text_part("Hi"),), sender=SUPERVISOR),
            Directive(id="row-2", text="/compact", recipient="developer"),
            Answer(
                id="row-3",
                in_reply_to="msg-1",
                content=(_text_part("Not that file"),),
                decision=Decision(choice="reject", policy_ref={"name": "reads"}),
            ),
        ],
        ids=["a message", "a directive", "an answer"],
    )
    def test_a_saved_message_loads_back_as_its_kind(self, message):
        serializer = CheckpointSerializer()
        saved = serializer.loads_typed(serializer.dumps_typed(message.model_dump()))

        # A newer gateway may have saved a field this one doesn't know.
        loaded = TypeAdapter(InboundMessage).validate_python(
            {**saved, "priority": "high"}
        )

        assert loaded == message

    @pytest.mark.parametrize(
        "saved",
        [
            {"type": "directive", "id": "row-1", "text": "/compact"},
            {
                "type": "answer",
                "id": "row-1",
                "in_reply_to": "msg-1",
                "decision": {"choice": "approve"},
            },
        ],
        ids=["a directive", "an answer"],
    )
    def test_a_saved_input_from_a_component_never_loads_as_the_users(self, saved):
        # As a newer gateway that lets a component send one would save it.
        with pytest.raises(ValidationError, match="sender.type"):
            TypeAdapter(InboundMessage).validate_python(
                {**saved, "sender": {"type": "component", "name": "lead"}}
            )
