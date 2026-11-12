from unittest.mock import MagicMock, patch

import pydantic
import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims

from lib.billing_events import (
    BillingEvent,
    BillingEventDropReason,
    BillingEventsClient,
)
from lib.billing_events.service import (
    BILL_ONCE_PER_WORKFLOW_FEATURES,
    BillingEventService,
    ExecutionEnvironment,
    LLMOperation,
    LLMOperationType,
    SelfHostedLLMOperations,
)
from lib.events import GLReportingEventContext
from lib.internal_events.client import InternalEventsClient
from lib.internal_events.context import EventContext, current_event_context


def test_bill_once_per_workflow_features_matches_expected_prefixes():
    """Lock in the feature prefixes that are billed at most once per workflow.

    These are matched against the prefix of `featureQualifiedName` (everything
    before the first '/'), so adding or removing an entry here changes
    customer-visible billing behaviour.
    """
    assert BILL_ONCE_PER_WORKFLOW_FEATURES == frozenset(
        {
            "code_review",
            "sast_fp_detection",
            "secrets_fp_detection",
            "resolve_sast_vulnerability",
        }
    )


@pytest.fixture(name="gl_context")
def mock_gl_context():
    """Fixture for GLReportingEventContext."""
    return GLReportingEventContext.from_workflow_definition(
        "software_development", is_ai_catalog_item=True
    )


@pytest.fixture(name="llm_operation")
def mock_llm_operation():
    """Fixture for LLMOperation."""
    return LLMOperation(
        model_id="claude-3-5-sonnet",
        model_engine="anthropic",
        model_provider="anthropic",
        model_vendor="anthropic",
        token_count=150,
        prompt_tokens=100,
        completion_tokens=50,
    )


def get_call_metadata(mock_client):
    """Helper to extract metadata from billing client call."""
    return mock_client.send_billing_event.call_args[1]["metadata"]


class TestBillingEventService:
    """Test suite for BillingEventService class."""

    def test_track_billing_with_explicit_operations(
        self,
        billing_event_service,
        billing_event_client,
        user,
        gl_context,
        llm_operation,
    ):
        """Test billing event with explicit LLM operations and custom parameters."""
        billing_event_service.track_billing(
            user,
            gl_context,
            workflow_id="workflow-123",
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
            unit_of_measure="execution",
            quantity=1,
            llm_ops=[llm_operation],
        )

        billing_event_client.send_billing_event.assert_called_once_with(
            user,
            BillingEvent.DAP_FLOW_ON_COMPLETION,
            "test_category",
            unit_of_measure="execution",
            quantity=1,
            metadata={
                "feature_qualified_name": "software_development",
                "feature_ai_catalog_item": True,
                "execution_environment": ExecutionEnvironment.DAP.value,
                "llm_operations": [
                    {
                        "model_id": "claude-3-5-sonnet",
                        "model_engine": "anthropic",
                        "model_provider": "anthropic",
                        "model_vendor": "anthropic",
                        "token_count": 150,
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "agent_name": None,
                        "cache_read_tokens": 0,
                        "cache_write_tokens": 0,
                        "operation_type": "standard",
                    }
                ],
                "tool_names": [],
                "orbit_called": False,
                "workflow_id": "workflow-123",
            },
        )

    @pytest.mark.parametrize("source", ["merge_request_code_conflict", None])
    def test_track_billing_with_source(
        self,
        billing_event_service,
        billing_event_client,
        user,
        gl_context,
        llm_operation,
        source,
    ):
        """Test that source is added to metadata only when provided."""
        billing_event_service.track_billing(
            user,
            gl_context,
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
            llm_ops=[llm_operation],
            source=source,
        )

        metadata = billing_event_client.send_billing_event.call_args.kwargs["metadata"]
        assert metadata.get("source") == source

    def test_track_billing_with_multiple_operations(
        self,
        billing_event_service,
        billing_event_client,
        user,
        gl_context,
        llm_operation,
    ):
        """Test billing event with multiple LLM operations."""
        second_op = LLMOperation(
            model_id="claude-3-5-sonnet",
            model_engine="anthropic",
            model_provider="anthropic",
            token_count=200,
            prompt_tokens=120,
            completion_tokens=80,
        )

        billing_event_service.track_billing(
            user,
            gl_context,
            workflow_id="workflow-456",
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
            llm_ops=[llm_operation, second_op],
        )

        metadata = get_call_metadata(billing_event_client)
        assert len(metadata["llm_operations"]) == 2
        assert metadata["llm_operations"][0]["token_count"] == 150
        assert metadata["llm_operations"][1]["token_count"] == 200

    def test_track_billing_defaults_and_gl_context_variations(
        self, billing_event_service, billing_event_client, user, llm_operation
    ):
        """Test default parameters and different GL context configurations."""
        gl_context = GLReportingEventContext.from_workflow_definition(
            "chat", is_ai_catalog_item=False
        )

        billing_event_service.track_billing(
            user,
            gl_context,
            workflow_id="workflow-test",
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
            llm_ops=[llm_operation],
        )

        call_args = billing_event_client.send_billing_event.call_args
        assert call_args[1]["unit_of_measure"] == "request"
        assert call_args[1]["quantity"] == 1

        metadata = call_args[1]["metadata"]
        assert metadata["feature_qualified_name"] == "chat"
        assert metadata["feature_ai_catalog_item"] is False

    def test_track_billing_exception_handling(
        self,
        billing_event_service,
        billing_event_client,
        user,
        gl_context,
        llm_operation,
    ):
        """Test that exceptions during billing propagate."""
        billing_event_client.send_billing_event.side_effect = Exception("Billing error")

        with pytest.raises(Exception, match="Billing error"):
            billing_event_service.track_billing(
                user,
                gl_context,
                workflow_id="workflow-error",
                event=BillingEvent.DAP_FLOW_ON_COMPLETION,
                execution_env=ExecutionEnvironment.DAP,
                category="test_category",
                llm_ops=[llm_operation],
            )

        billing_event_client.send_billing_event.assert_called_once()

    @patch("lib.billing_events.service.get_llm_operations")
    def test_track_billing_with_context_operations(
        self,
        mock_get_llm_operations,
        billing_event_service,
        billing_event_client,
        user,
        gl_context,
    ):
        """Test billing event retrieves LLM operations from context when not explicitly provided."""
        context_ops = [
            {
                "model_id": "gpt-4",
                "model_engine": "openai",
                "model_provider": "openai",
                "token_count": 300,
                "prompt_tokens": 200,
                "completion_tokens": 100,
            }
        ]
        mock_get_llm_operations.return_value = context_ops

        billing_event_service.track_billing(
            user,
            gl_context,
            workflow_id="workflow-context",
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
        )

        mock_get_llm_operations.assert_called_once()
        metadata = get_call_metadata(billing_event_client)
        assert len(metadata["llm_operations"]) == 1
        assert metadata["llm_operations"][0]["model_id"] == "gpt-4"
        assert metadata["llm_operations"][0]["token_count"] == 300

    @patch("lib.billing_events.service.get_llm_operations")
    def test_track_billing_raises_error_when_no_operations_available(
        self, mock_get_llm_operations, billing_event_service, user, gl_context
    ):
        """Test that ValueError is raised when no LLM operations are available."""
        mock_get_llm_operations.return_value = None

        with pytest.raises(
            ValueError, match="No LLM operations available for billing tracking"
        ):
            billing_event_service.track_billing(
                user,
                gl_context,
                workflow_id="workflow-no-ops",
                event=BillingEvent.DAP_FLOW_ON_COMPLETION,
                execution_env=ExecutionEnvironment.DAP,
                category="test_category",
            )

    @patch("lib.billing_events.service.get_llm_operations")
    @pytest.mark.parametrize(
        ("explicit_ops_provided", "expected_model"),
        [
            (False, "context-model"),  # Context ops used when no explicit ops provided
            (True, "claude-3-5-sonnet"),  # Explicit ops take priority over context
        ],
    )
    def test_track_billing_operation_priority(
        self,
        mock_get_llm_operations,
        billing_event_service,
        billing_event_client,
        user,
        gl_context,
        llm_operation,
        explicit_ops_provided,
        expected_model,
    ):
        """Test operation priority: explicit > context."""
        mock_get_llm_operations.return_value = [
            {
                "model_id": "context-model",
                "model_engine": "openai",
                "model_provider": "openai",
                "token_count": 999,
                "prompt_tokens": 500,
                "completion_tokens": 499,
            }
        ]

        billing_event_service.track_billing(
            user,
            gl_context,
            workflow_id="workflow-priority",
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
            llm_ops=[llm_operation] if explicit_ops_provided else None,
        )

        metadata = get_call_metadata(billing_event_client)
        assert metadata["llm_operations"][0]["model_id"] == expected_model

    @patch("lib.billing_events.service.get_llm_operations")
    def test_track_billing_explicit_operations_skip_context(
        self,
        mock_get_llm_operations,
        billing_event_service,
        billing_event_client,
        user,
        gl_context,
        llm_operation,
    ):
        """Test that explicit operations skip context retrieval."""
        billing_event_service.track_billing(
            user,
            gl_context,
            workflow_id="workflow-explicit",
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
            llm_ops=[llm_operation],
        )

        mock_get_llm_operations.assert_not_called()
        metadata = get_call_metadata(billing_event_client)
        assert metadata["llm_operations"][0]["model_id"] == "claude-3-5-sonnet"

    def test_track_billing_with_cache_tokens(
        self, billing_event_service, billing_event_client, user, gl_context
    ):
        """Test cache token fields are included in billing metadata."""
        op = LLMOperation(
            model_id="claude-3-5-sonnet",
            model_engine="anthropic",
            model_provider="anthropic",
            token_count=150,
            prompt_tokens=100,
            completion_tokens=50,
            cache_read_tokens=30,
            cache_write_tokens=20,
        )

        billing_event_service.track_billing(
            user,
            gl_context,
            workflow_id="workflow-cache",
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
            llm_ops=[op],
        )

        metadata = get_call_metadata(billing_event_client)
        assert metadata["llm_operations"][0]["cache_read_tokens"] == 30
        assert metadata["llm_operations"][0]["cache_write_tokens"] == 20

    def test_track_billing_includes_tool_names(
        self,
        billing_event_service,
        billing_event_client,
        user,
        gl_context,
        llm_operation,
    ):
        billing_event_service.track_billing(
            user,
            gl_context,
            workflow_id="workflow-tools",
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
            llm_ops=[llm_operation],
            tool_execs=["read_file", "write_file", "read_file"],
        )
        metadata = get_call_metadata(billing_event_client)
        assert metadata["tool_names"] == ["read_file", "write_file", "read_file"]

    def test_track_billing_with_orbit_called(
        self,
        billing_event_service,
        billing_event_client,
        user,
        gl_context,
        llm_operation,
    ):
        """Test that orbit_called flag is included in billing metadata."""
        billing_event_service.track_billing(
            user,
            gl_context,
            workflow_id="workflow-123",
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
            llm_ops=[llm_operation],
            orbit_called=True,
        )

        metadata = get_call_metadata(billing_event_client)
        assert metadata["orbit_called"] is True

    def test_track_billing_without_workflow_id(
        self,
        billing_event_service,
        billing_event_client,
        user,
        gl_context,
        llm_operation,
    ):
        """Test billing event without workflow_id for stateless operations like code suggestions."""
        billing_event_service.track_billing(
            user,
            gl_context,
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
            llm_ops=[llm_operation],
        )

        billing_event_client.send_billing_event.assert_called_once()
        metadata = get_call_metadata(billing_event_client)
        assert "workflow_id" not in metadata
        assert metadata["feature_qualified_name"] == "software_development"
        assert metadata["execution_environment"] == ExecutionEnvironment.DAP.value
        assert len(metadata["llm_operations"]) == 1

    @pytest.mark.parametrize("client_result", [True, False])
    def test_track_billing_returns_client_result(
        self,
        billing_event_service,
        billing_event_client,
        user,
        gl_context,
        llm_operation,
        client_result,
    ):
        """Test that track_billing reports whether the client actually tracked the event."""
        billing_event_client.send_billing_event.return_value = client_result

        tracked = billing_event_service.track_billing(
            user,
            gl_context,
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
            llm_ops=[llm_operation],
        )

        assert tracked is client_result

    def test_track_billing_emits_event_with_compaction_ops(
        self,
        billing_event_service,
        billing_event_client,
        user,
        gl_context,
    ):
        """Test that billing events are emitted for compaction ops.

        CustomersDot uses operation_type to decide billing treatment.
        """
        compaction_op = LLMOperation(
            model_id="claude-3-5-sonnet",
            model_engine="anthropic",
            model_provider="anthropic",
            token_count=150,
            prompt_tokens=100,
            completion_tokens=50,
            operation_type="compaction_auto",
        )

        billing_event_service.track_billing(
            user,
            gl_context,
            workflow_id="workflow-compaction",
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
            llm_ops=[compaction_op],
        )

        billing_event_client.send_billing_event.assert_called_once()
        metadata = get_call_metadata(billing_event_client)
        assert metadata["llm_operations"][0]["operation_type"] == "compaction_auto"

    def test_track_billing_with_mixed_operation_types(
        self,
        billing_event_service,
        billing_event_client,
        user,
        gl_context,
    ):
        """Test that billing proceeds when there are both compaction and standard ops."""
        compaction_op = LLMOperation(
            model_id="claude-3-5-sonnet",
            model_engine="anthropic",
            model_provider="anthropic",
            token_count=150,
            prompt_tokens=100,
            completion_tokens=50,
            operation_type="compaction_auto",
        )
        standard_op = LLMOperation(
            model_id="claude-3-5-sonnet",
            model_engine="anthropic",
            model_provider="anthropic",
            token_count=200,
            prompt_tokens=150,
            completion_tokens=50,
            operation_type="standard",
        )

        billing_event_service.track_billing(
            user,
            gl_context,
            workflow_id="workflow-mixed",
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
            llm_ops=[compaction_op, standard_op],
        )

        billing_event_client.send_billing_event.assert_called_once()
        metadata = get_call_metadata(billing_event_client)
        assert len(metadata["llm_operations"]) == 2
        assert metadata["llm_operations"][0]["operation_type"] == "compaction_auto"
        assert metadata["llm_operations"][1]["operation_type"] == "standard"

    @patch("lib.billing_events.service.get_llm_operations")
    def test_track_billing_with_decision_operation(
        self,
        mock_get_llm_operations,
        billing_event_service,
        billing_event_client,
        user,
        gl_context,
    ):
        """Decision calls reach the billing event as their own operation type."""
        mock_get_llm_operations.return_value = [
            {
                "model_id": "gpt-6-luna",
                "model_engine": "openai_decisions",
                "model_provider": "openai_decisions",
                "model_vendor": "openai",
                "token_count": 120,
                "prompt_tokens": 120,
                "completion_tokens": 0,
                "operation_type": "decision",
            }
        ]

        billing_event_service.track_billing(
            user,
            gl_context,
            workflow_id="workflow-decision",
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
        )

        metadata = get_call_metadata(billing_event_client)
        assert metadata["llm_operations"][0]["operation_type"] == "decision"

    def test_llm_operation_defaults_for_operation_type(self):
        """Test that LLMOperation defaults operation_type to 'standard'."""
        op = LLMOperation(
            model_id="claude-3-5-sonnet",
            model_engine="anthropic",
            model_provider="anthropic",
            token_count=150,
            prompt_tokens=100,
            completion_tokens=50,
        )
        assert op.operation_type == "standard"

    def test_llm_operation_model_validate_fills_defaults(self):
        """Test that model_validate fills defaults for missing operation_type."""
        raw_dict = {
            "model_id": "gpt-4",
            "model_engine": "openai",
            "model_provider": "openai",
            "token_count": 300,
            "prompt_tokens": 200,
            "completion_tokens": 100,
        }
        op = LLMOperation.model_validate(raw_dict)
        assert op.operation_type == "standard"


class TestLLMOperationType:
    """Test suite for the LLMOperationType enum."""

    def test_string_values_become_enum_members(self):
        """Prompt YAML and raw billing dicts pass strings; they validate to the enum member."""
        op = LLMOperation(
            model_id="gpt-6-luna",
            model_engine="openai",
            model_provider="openai",
            token_count=120,
            prompt_tokens=120,
            completion_tokens=0,
            operation_type="decision",
        )
        assert op.operation_type is LLMOperationType.DECISION

    def test_llm_operation_accepts_standard(self):
        """Test that LLMOperation accepts 'standard' as operation_type."""
        op = LLMOperation(
            model_id="claude-3-5-sonnet",
            model_engine="anthropic",
            model_provider="anthropic",
            token_count=100,
            prompt_tokens=80,
            completion_tokens=20,
            operation_type="standard",
        )
        assert op.operation_type == "standard"

    def test_llm_operation_accepts_compaction_auto(self):
        """Test that LLMOperation accepts 'compaction_auto' as operation_type."""
        op = LLMOperation(
            model_id="claude-3-5-sonnet",
            model_engine="anthropic",
            model_provider="anthropic",
            token_count=100,
            prompt_tokens=80,
            completion_tokens=20,
            operation_type="compaction_auto",
        )
        assert op.operation_type == "compaction_auto"

    def test_llm_operation_accepts_web_search_agentcore(self):
        """Test that LLMOperation accepts 'web_search_agentcore' as operation_type."""
        op = LLMOperation(
            model_id="claude-3-5-sonnet",
            model_engine="anthropic",
            model_provider="anthropic",
            token_count=100,
            prompt_tokens=80,
            completion_tokens=20,
            operation_type="web_search_agentcore",
        )
        assert op.operation_type == "web_search_agentcore"

    def test_llm_operation_type_is_exported(self):
        """Test that LLMOperationType is importable from lib.billing_events.service."""
        # If this import works, the type is properly exported
        assert LLMOperationType is not None

    def test_llm_operation_rejects_invalid_operation_type(self):
        """Test that LLMOperation rejects unknown operation_type values via Pydantic validation."""
        with pytest.raises(pydantic.ValidationError):
            LLMOperation(
                model_id="claude-3-5-sonnet",
                model_engine="anthropic",
                model_provider="anthropic",
                token_count=100,
                prompt_tokens=80,
                completion_tokens=20,
                operation_type="unknown_type",
            )


class TestSelfHostedLLMOperations:
    """Test suite for SelfHostedLLMOperations class."""

    def test_get_operations_defaults_to_standard(self):
        """Test that get_operations defaults operation_type to 'standard'."""
        ops = SelfHostedLLMOperations.get_operations()
        assert len(ops) == 1
        assert ops[0].operation_type == "standard"

    def test_get_operations_returns_placeholder_metadata(self):
        """Test that get_operations returns expected placeholder values."""
        ops = SelfHostedLLMOperations.get_operations()
        assert len(ops) == 1
        op = ops[0]
        assert op.model_id == "self-hosted-model"
        assert op.model_engine == "litellm"
        assert op.model_provider == "litellm"
        assert op.token_count == 1
        assert op.prompt_tokens == 1
        assert op.completion_tokens == 1


@pytest.fixture(name="metered_service")
def metered_service_fixture(metered_client):
    return BillingEventService(client=metered_client)


@pytest.fixture(name="cloud_connector_user")
def cloud_connector_user_fixture():
    return CloudConnectorUser(
        authenticated=True, claims=UserClaims(gitlab_instance_uid="abc")
    )


def _sample(registry, name, **labels):
    return registry.get_sample_value(name, labels)


class TestBillingEventServiceMetrics:
    """The service owns the emitted count, so failures before the client is reached still lower the ratio."""

    EVENT = BillingEvent.DAP_FLOW_ON_COMPLETION

    @patch("lib.billing_events.service.get_llm_operations")
    @pytest.mark.parametrize(
        "context_ops",
        [
            pytest.param(None, id="no_llm_operations"),
            pytest.param(
                [{"model_id": "missing-token-counts"}], id="invalid_llm_operation"
            ),
        ],
    )
    def test_failure_before_client_counts_emitted_and_missing_field(
        self,
        mock_get_llm_operations,
        metered_service,
        cloud_connector_user,
        gl_context,
        registry,
        context_ops,
    ):
        mock_get_llm_operations.return_value = context_ops

        with (
            patch.object(
                metered_service.client, "send_billing_event"
            ) as mock_send_billing_event,
            pytest.raises(ValueError),
        ):
            metered_service.track_billing(
                cloud_connector_user,
                gl_context,
                event=self.EVENT,
                execution_env=ExecutionEnvironment.DAP,
                category="test_category",
            )

        mock_send_billing_event.assert_not_called()
        labels = {"event_type": self.EVENT.value}
        assert _sample(registry, "billing_events_emitted_total", **labels) == 1
        assert (
            _sample(
                registry,
                "billing_events_dropped_total",
                reason=BillingEventDropReason.MISSING_FIELD.value,
                **labels,
            )
            == 1
        )
        # delivered / emitted is 0 / 1, so the reconciliation alert can fire.
        assert _sample(registry, "billing_events_delivered_total", **labels) == 0

    def test_successful_attempt_counts_emitted_once(
        self, metered_service, cloud_connector_user, gl_context, llm_operation, registry
    ):
        current_event_context.set(EventContext())

        with patch("snowplow_tracker.Tracker.track") as mock_track:
            tracked = metered_service.track_billing(
                cloud_connector_user,
                gl_context,
                event=self.EVENT,
                execution_env=ExecutionEnvironment.DAP,
                category="test_category",
                llm_ops=[llm_operation],
            )

        assert tracked is True
        mock_track.assert_called_once()
        labels = {"event_type": self.EVENT.value}
        assert _sample(registry, "billing_events_emitted_total", **labels) == 1
        assert all(
            _sample(
                registry,
                "billing_events_dropped_total",
                reason=reason.value,
                **labels,
            )
            == 0
            for reason in BillingEventDropReason
        )

    @patch("lib.billing_events.service.get_llm_operations", return_value=None)
    def test_disabled_client_counts_nothing(
        self,
        _mock_get_llm_operations,
        metrics,
        registry,
        cloud_connector_user,
        gl_context,
    ):
        service = BillingEventService(
            client=BillingEventsClient(
                enabled=False,
                endpoint="https://billing.local",
                app_id="gitlab_ai_gateway-billing",
                namespace="gl",
                batch_size=1,
                thread_count=1,
                internal_event_client=MagicMock(spec=InternalEventsClient),
                metrics=metrics,
            )
        )

        with pytest.raises(ValueError):
            service.track_billing(
                cloud_connector_user,
                gl_context,
                event=self.EVENT,
                execution_env=ExecutionEnvironment.DAP,
                category="test_category",
            )

        labels = {"event_type": self.EVENT.value}
        assert _sample(registry, "billing_events_emitted_total", **labels) == 0
        assert (
            _sample(
                registry,
                "billing_events_dropped_total",
                reason=BillingEventDropReason.MISSING_FIELD.value,
                **labels,
            )
            == 0
        )
