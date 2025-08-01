import pytest

from lib.billing_events.context import BillingEventContext


def test_billing_event_context_required_fields():
    context = BillingEventContext(
        event_id="test-event-123",
        event_type="ai_completion",
        unit_of_measure="tokens",
        quantity=150.0,
        timestamp="2023-12-01T10:00:00Z",
    )
    assert context.event_id == "test-event-123"
    assert context.event_type == "ai_completion"
    assert context.unit_of_measure == "tokens"
    assert context.quantity == 150.0
    assert context.timestamp == "2023-12-01T10:00:00Z"
    assert context.realm is None
    assert context.instance_id is None
    assert context.unique_instance_id is None
    assert context.host_name is None
    assert context.project_id is None
    assert context.namespace_id is None
    assert context.subject is None
    assert context.root_namespace_id is None
    assert context.correlation_id is None
    assert context.seat_ids is None
    assert context.metadata == {}


def test_billing_event_context_all_fields():
    context = BillingEventContext(
        event_id="test-event-456",
        event_type="code_suggestions",
        unit_of_measure="requests",
        quantity=5.0,
        timestamp="2023-12-01T11:00:00Z",
        realm="user",
        instance_id="instance-123",
        unique_instance_id="unique-instance-456",
        host_name="gitlab.example.com",
        project_id=789,
        namespace_id=101,
        subject="user:123",
        root_namespace_id=101,
        correlation_id="corr-123",
        seat_ids=["seat-1", "seat-2"],
        metadata={"model": "claude-3", "feature": "completion"},
    )
    assert context.event_id == "test-event-456"
    assert context.event_type == "code_suggestions"
    assert context.unit_of_measure == "requests"
    assert context.quantity == 5.0
    assert context.timestamp == "2023-12-01T11:00:00Z"
    assert context.realm == "user"
    assert context.instance_id == "instance-123"
    assert context.unique_instance_id == "unique-instance-456"
    assert context.host_name == "gitlab.example.com"
    assert context.project_id == 789
    assert context.namespace_id == 101
    assert context.subject == "user:123"
    assert context.root_namespace_id == 101
    assert context.correlation_id == "corr-123"
    assert context.seat_ids == ["seat-1", "seat-2"]
    assert context.metadata == {"model": "claude-3", "feature": "completion"}


def test_billing_event_context_model_validation():
    with pytest.raises(ValueError):
        BillingEventContext(
            event_id=123,
            event_type="ai_completion",
            unit_of_measure="tokens",
            quantity=150.0,
            timestamp="2023-12-01T10:00:00Z",
        )

    with pytest.raises(ValueError):
        BillingEventContext(
            event_id="test-event-123",
            event_type="ai_completion",
            unit_of_measure="tokens",
            quantity="invalid",
            timestamp="2023-12-01T10:00:00Z",
        )


def test_billing_event_context_missing_required_fields():
    with pytest.raises(ValueError):
        BillingEventContext()

    with pytest.raises(ValueError):
        BillingEventContext(event_id="test-event-123")


def test_billing_event_context_optional_metadata():
    context = BillingEventContext(
        event_id="test-event-789",
        event_type="duo_chat",
        unit_of_measure="messages",
        quantity=1.0,
        timestamp="2023-12-01T12:00:00Z",
        metadata={"session_id": "session-123", "user_agent": "GitLab/16.0"},
    )
    assert context.metadata["session_id"] == "session-123"
    assert context.metadata["user_agent"] == "GitLab/16.0"


def test_billing_event_context_seat_ids_list():
    context = BillingEventContext(
        event_id="test-event-999",
        event_type="ai_completion",
        unit_of_measure="tokens",
        quantity=200.0,
        timestamp="2023-12-01T13:00:00Z",
        seat_ids=["seat-1", "seat-2", "seat-3"],
    )
    assert context.seat_ids is not None
    assert len(context.seat_ids) == 3
