import hashlib
import json

from google.protobuf.struct_pb2 import Struct

from duo_workflow_service.flow_config_digest import (
    definition_digest,
    flow_config_digest,
)

DEFINITION = {
    "version": "v1",
    "environment": "ambient",
    "prompts": [{"prompt_id": "p", "params": {"timeout": 180, "temperature": 0.5}}],
    "components": [{"name": "agent", "max_retries": 3, "toolset": []}],
}


def _struct(value: dict) -> Struct:
    struct = Struct()
    struct.update(value)
    return struct


def test_a_definition_matches_the_digest_of_the_struct_rails_sent():
    """GenerateToken digests the Struct it received.

    The fetched JSON must give the same digest.
    """
    assert definition_digest(DEFINITION) == flow_config_digest(_struct(DEFINITION))


def test_ints_and_floats_give_the_same_digest():
    """A Struct keeps numbers as doubles, so 180 and 180.0 are the same value."""
    as_floats = json.loads(json.dumps(DEFINITION).replace("180", "180.0"))

    assert definition_digest(DEFINITION) == definition_digest(as_floats)


def test_the_ints_are_not_digested_as_ints():
    plain = hashlib.sha256(
        json.dumps(DEFINITION, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    assert definition_digest(DEFINITION) != plain


def test_key_order_does_not_change_the_digest():
    reordered = dict(reversed(list(DEFINITION.items())))

    assert definition_digest(reordered) == definition_digest(DEFINITION)


def test_a_changed_definition_has_another_digest():
    changed = {**DEFINITION, "environment": "chat"}

    assert definition_digest(changed) != definition_digest(DEFINITION)


def test_the_definition_is_not_changed():
    definition = json.loads(json.dumps(DEFINITION))

    definition_digest(definition)

    assert definition == DEFINITION
