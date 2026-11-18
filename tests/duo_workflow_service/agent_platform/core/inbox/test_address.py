from typing import Optional

import pytest

from duo_workflow_service.agent_platform.core.inbox import (
    Address,
    Claimant,
    Message,
    RunInbox,
    TaskId,
    TextPart,
    User,
)


def _message(id: str, recipient: Optional[str]) -> Message:
    return Message(
        id=id,
        content=(TextPart(type="text", text=f"Text of {id}"),),
        sender=User(),
        recipient=recipient,
    )


class TestAddress:
    @pytest.mark.parametrize(
        ("address", "expected"),
        [
            (Address.of("developer"), "developer"),
            (
                Address.of("supervisor").child("reviewer", "call-7"),
                "supervisor/reviewer#call-7",
            ),
            (
                Address.of("supervisor")
                .child("reviewer", "call-7")
                .child("helper", "call-9"),
                "supervisor/reviewer#call-7/helper#call-9",
            ),
            (Address.of("dev#2"), "dev%232"),
            (Address.of("a/b"), "a%2Fb"),
            (Address.of("100%"), "100%25"),
            (Address.of("dev%232"), "dev%25232"),
            (
                Address.of("team/a").child("dev#2", "toolu_01/x#y"),
                "team%2Fa/dev%232#toolu_01%2Fx%23y",
            ),
        ],
        ids=[
            "a top-level component",
            "one run of a dispatched component",
            "a run below a run",
            "a name with the instance separator",
            "a name with the place separator",
            "a name with the escape character",
            "a name that looks escaped already",
            "separators in every part",
        ],
    )
    def test_an_address_is_its_escaped_segments(self, address: Address, expected: str):
        assert str(address) == expected

    @pytest.mark.parametrize(
        ("component_name", "instance"), [("", "call-7"), ("reviewer", "")]
    )
    def test_a_dispatched_run_needs_a_name_and_an_id(
        self, component_name: str, instance: str
    ):
        with pytest.raises(ValueError, match="needs a component name"):
            Address.of("supervisor").child(component_name, instance)

    def test_a_top_level_component_needs_a_name(self):
        with pytest.raises(ValueError, match="needs a component name"):
            Address.of("")


class TestEscapedNamesClaimApart:
    """A component named with a separator takes only its own messages."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("claimant", "expected"),
        [
            (Address.of("dev"), ("to-dev",)),
            (Address.of("dev#2"), ("to-dev-hash-2",)),
            (Address.of("dev").child("dev", "2"), ("to-run-2-of-dev-below-dev",)),
            (Address.of("dev/dev#2"), ("to-dev-slash-dev-hash-2",)),
        ],
        ids=[
            "dev",
            "the component named dev#2",
            "run 2 of a dev that dev dispatched",
            "the component named dev/dev#2",
        ],
    )
    async def test_a_name_with_a_separator_never_reads_as_another_address(
        self, claimant: Address, expected: tuple[str, ...]
    ):
        inbox = RunInbox()
        for id, address in (
            ("to-dev", Address.of("dev")),
            ("to-dev-hash-2", Address.of("dev#2")),
            ("to-run-2-of-dev-below-dev", Address.of("dev").child("dev", "2")),
            ("to-dev-slash-dev-hash-2", Address.of("dev/dev#2")),
        ):
            inbox.put(_message(id, str(address)))

        claimed = inbox.claim(TaskId("task-1"), Claimant(recipient=claimant))

        assert tuple(message.id for message in claimed) == expected
