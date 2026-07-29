import hashlib
from typing import Any

import pytest

from lib.billing_events.context import BillingEventContext, UsageQuotaEventContext
from lib.internal_events.context import EventContext


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
    assert context.global_user_id is None
    assert context.root_namespace_id is None
    assert context.correlation_id is None
    assert context.seat_ids is None
    assert context.metadata == {}
    assert context.deployment_type is None
    assert context.subject_type is None


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
        global_user_id="user-123",
        root_namespace_id=101,
        correlation_id="corr-123",
        seat_ids=["seat-1", "seat-2"],
        metadata={"model": "claude-3", "feature": "completion"},
        deployment_type="self-managed",
        subject_type="human",
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
    assert context.global_user_id == "user-123"
    assert context.root_namespace_id == 101
    assert context.correlation_id == "corr-123"
    assert context.seat_ids == ["seat-1", "seat-2"]
    assert context.metadata == {"model": "claude-3", "feature": "completion"}
    assert context.deployment_type == "self-managed"
    assert context.subject_type == "human"


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


class TestUsageQuotaContext:
    def test_from_internal_event(self):
        event = EventContext(
            environment="prod",
            project_id=1234,
            ultimate_parent_namespace_id=1,
            namespace_id=2,
        )
        context = UsageQuotaEventContext.from_internal_event(event)
        assert context.environment == "prod"
        assert context.project_id == 1234
        assert context.namespace_id == 2
        assert context.root_namespace_id == 1

    def test_to_cache_key_all_field_present(self):
        context = UsageQuotaEventContext(
            environment="production",
            realm="saas",
            deployment_type="saas",
            instance_id="00000000-1111-2222-3333-000000000000",
            unique_instance_id="00000000-1111-2222-3333-000000000000",
            feature_enablement_type="duo_pro",
            feature_qualified_name="sast_fp_detection/v1",
            ultimate_parent_namespace_id=123,
            namespace_id=456,
            user_id="user_123",
            global_user_id="gid://gitlab/User/123",
            correlation_id="correlation_id",
            event_type="ai_gateway_proxy_use",
            model_id="claude-opus-4-8",
        )
        # sha256 of the JSON-encoded CACHE_KEY_FIELDS payload (field order + values).
        expected_payload = (
            '["production","saas","user_123","gid://gitlab/User/123",123,'
            '"00000000-1111-2222-3333-000000000000","duo_pro","sast_fp_detection/v1",'
            '"ai_gateway_proxy_use","claude-opus-4-8"]'
        )
        expected_key = hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()
        assert context.to_cache_key() == expected_key

    def test_to_cache_key_encodes_none_fields_as_json_null(self) -> None:
        context = UsageQuotaEventContext(
            environment="prod",
            user_id="123",
            feature_enablement_type="beta",
        )

        # Every CACHE_KEY_FIELDS entry keeps its slot; None -> JSON null.
        expected_payload = '["prod",null,"123",null,null,null,"beta",null,null,null]'
        expected_key = hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()
        assert context.to_cache_key() == expected_key

    def test_to_cache_key_no_collision_across_field_boundaries(self) -> None:
        """A None field must not let an adjacent field's value shift into its slot.

        ``event_type=None, model_id="X"`` and ``event_type="X", model_id=None``
        must produce distinct keys; otherwise the model/event-type scoping is void.
        """
        base_kwargs: dict[str, Any] = {
            "environment": "production",
            "realm": "saas",
            "user_id": "user_123",
            "feature_qualified_name": "ai_gateway_proxy_use",
        }
        context_a = UsageQuotaEventContext(
            **base_kwargs, event_type=None, model_id="ai_gateway_proxy_use"
        )
        context_b = UsageQuotaEventContext(
            **base_kwargs, event_type="ai_gateway_proxy_use", model_id=None
        )

        assert context_a.to_cache_key() != context_b.to_cache_key()

    @pytest.mark.parametrize(
        "model_id_a,model_id_b",
        [
            ("text-embedding-005", "claude-opus-4-8"),
            ("claude-opus-4-8", "claude-3-5-sonnet-20241022"),
            ("text-embedding-005", None),
        ],
    )
    def test_to_cache_key_differs_when_model_id_changes(
        self, model_id_a: str | None, model_id_b: str | None
    ):
        """Cache keys must differ when model_id differs to prevent cross-model cache hits."""
        base_kwargs: dict[str, Any] = {
            "environment": "production",
            "realm": "saas",
            "user_id": "user_123",
            "feature_qualified_name": "ai_gateway_proxy_use",
            "event_type": "ai_gateway_proxy_use",
        }
        context_a = UsageQuotaEventContext(**base_kwargs, model_id=model_id_a)
        context_b = UsageQuotaEventContext(**base_kwargs, model_id=model_id_b)

        assert context_a.to_cache_key() != context_b.to_cache_key()

    @pytest.mark.parametrize(
        "event_type_a,event_type_b",
        [
            ("ai_gateway_proxy_use", "code_suggestions"),
            ("ai_gateway_proxy_use", None),
            ("code_suggestions", "duo_chat"),
        ],
    )
    def test_to_cache_key_differs_when_event_type_changes(
        self, event_type_a: str | None, event_type_b: str | None
    ):
        """Cache keys must differ when event_type differs to prevent cross-event cache hits."""
        base_kwargs: dict[str, Any] = {
            "environment": "production",
            "realm": "saas",
            "user_id": "user_123",
            "feature_qualified_name": "ai_gateway_proxy_use",
            "model_id": "text-embedding-005",
        }
        context_a = UsageQuotaEventContext(**base_kwargs, event_type=event_type_a)
        context_b = UsageQuotaEventContext(**base_kwargs, event_type=event_type_b)

        assert context_a.to_cache_key() != context_b.to_cache_key()

    def test_to_cache_key_identical_for_same_event_and_model(self):
        """Identical event/model contexts must produce the same cache key (cache reuse)."""
        kwargs: dict[str, Any] = {
            "environment": "production",
            "realm": "saas",
            "user_id": "user_123",
            "feature_qualified_name": "ai_gateway_proxy_use",
            "event_type": "ai_gateway_proxy_use",
            "model_id": "claude-opus-4-8",
        }
        context_a = UsageQuotaEventContext(**kwargs)
        context_b = UsageQuotaEventContext(**kwargs)

        assert context_a.to_cache_key() == context_b.to_cache_key()
