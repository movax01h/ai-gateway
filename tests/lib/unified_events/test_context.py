import uuid
from contextvars import ContextVar

import pytest

from lib.unified_events.context import (
    EventConstructContext,
    current_event_construct,
)

VALID_EVENT_ID = str(uuid.uuid4())
VALID_ORG_UUID = str(uuid.uuid4())


def test_event_construct_default_values():
    context = EventConstructContext(event_id=VALID_EVENT_ID)
    assert context.event_id == VALID_EVENT_ID
    assert context.environment == "development"
    assert context.source == "ai-gateway-python"
    assert context.metadata == {}
    # Every non-required field defaults to None.
    assert context.event_type is None
    assert context.subject is None
    assert context.is_billable is None
    assert context.unit_of_measure is None
    assert context.assignments is None


def test_event_construct_custom_values():
    context = EventConstructContext(
        event_id=VALID_EVENT_ID,
        event_type="ai_request",
        environment="production",
        source="gitlab-rails",
        subject="42",
        subject_type="user",
        user_id="hashed-user",
        deployment_type=".com",
        realm="saas",
        organization_id=7,
        organization_uuid=VALID_ORG_UUID,
        root_namespace_id=3,
        namespace_id=2,
        project_id=1,
        plan="ultimate",
        is_billable=True,
        unit_of_measure="token",
        quantity=1234,
        assignments=["Duo Pro", "Duo Enterprise"],
        cost_bearer="customer",
        estimated_cost=0.42,
        metadata={"key": "value"},
    )
    assert context.event_type == "ai_request"
    assert context.environment == "production"
    assert context.deployment_type == ".com"
    assert context.realm == "saas"
    assert context.organization_uuid == VALID_ORG_UUID
    assert context.is_billable is True
    assert context.unit_of_measure == "token"
    assert context.quantity == 1234
    assert context.assignments == ["Duo Pro", "Duo Enterprise"]
    assert context.estimated_cost == 0.42
    assert context.metadata == {"key": "value"}


def test_event_id_is_required():
    with pytest.raises(ValueError):
        EventConstructContext()


def test_event_id_must_be_uuid_v4():
    with pytest.raises(ValueError):
        EventConstructContext(event_id="not-a-uuid")


def test_organization_uuid_must_be_uuid():
    with pytest.raises(ValueError):
        EventConstructContext(event_id=VALID_EVENT_ID, organization_uuid="nope")


@pytest.mark.parametrize(
    "field,value",
    [
        ("deployment_type", "saas"),  # not a valid deployment_type
        ("realm", "SaaS"),  # wrong casing
        ("unit_of_measure", "tokens"),  # plural is invalid
    ],
)
def test_enum_fields_reject_invalid_values(field, value):
    with pytest.raises(ValueError):
        EventConstructContext(event_id=VALID_EVENT_ID, **{field: value})


def test_assignments_reject_unknown_value():
    with pytest.raises(ValueError):
        EventConstructContext(event_id=VALID_EVENT_ID, assignments=["Duo Basic"])


def test_assignments_must_be_unique():
    with pytest.raises(ValueError):
        EventConstructContext(
            event_id=VALID_EVENT_ID, assignments=["Duo Pro", "Duo Pro"]
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("organization_id", -1),
        ("organization_id", 2147483648),
        ("quantity", -1),
        ("estimated_cost", -0.01),
    ],
)
def test_numeric_bounds_are_enforced(field, value):
    with pytest.raises(ValueError):
        EventConstructContext(event_id=VALID_EVENT_ID, **{field: value})


@pytest.mark.parametrize(
    "field,max_length",
    [
        ("event_type", 255),
        ("source", 32),
        ("correlation_id", 64),
        ("plan", 32),
    ],
)
def test_string_max_length_is_enforced(field, max_length):
    # A value exactly at the limit is accepted; one char over is rejected.
    EventConstructContext(event_id=VALID_EVENT_ID, **{field: "x" * max_length})
    with pytest.raises(ValueError):
        EventConstructContext(
            event_id=VALID_EVENT_ID, **{field: "x" * (max_length + 1)}
        )


def test_unknown_fields_are_forbidden():
    # additionalProperties: false -> unknown keys must be rejected, not dropped.
    with pytest.raises(ValueError):
        EventConstructContext(event_id=VALID_EVENT_ID, bogus_field="x")


def test_current_event_construct_default_is_none():
    assert isinstance(current_event_construct, ContextVar)
    assert current_event_construct.get() is None


def test_current_event_construct_set_and_reset():
    original = current_event_construct.get()

    new_context = EventConstructContext(event_id=VALID_EVENT_ID, environment="staging")
    token = current_event_construct.set(new_context)

    try:
        assert current_event_construct.get().environment == "staging"
    finally:
        current_event_construct.reset(token)

    assert current_event_construct.get() == original
