from unittest.mock import Mock, patch

import pytest

from lib.billing_events import BillingEvent, BillingEventsClient
from lib.billing_events.service import (
    BillingEventService,
    ExecutionEnvironment,
    LLMOperation,
)
from lib.events import GLReportingEventContext


@pytest.fixture(name="billing_client")
def mock_billing_client():
    """Fixture for a mocked BillingEventsClient."""
    return Mock(spec=BillingEventsClient)


@pytest.fixture(name="billing_service")
def mock_billing_service(billing_client):
    """Fixture for BillingEventService with mocked client."""
    return BillingEventService(client=billing_client)


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
        token_count=150,
        prompt_tokens=100,
        completion_tokens=50,
    )


def get_call_metadata(mock_client):
    """Helper to extract metadata from billing client call."""
    return mock_client.track_billing_event.call_args[1]["metadata"]


class TestBillingEventService:
    """Test suite for BillingEventService class."""

    def test_track_billing_with_explicit_operations(
        self, billing_service, billing_client, user, gl_context, llm_operation
    ):
        """Test billing event with explicit LLM operations and custom parameters."""
        billing_service.track_billing(
            workflow_id="workflow-123",
            user=user,
            gl_context=gl_context,
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
            unit_of_measure="execution",
            quantity=1,
            llm_ops=[llm_operation],
        )

        billing_client.track_billing_event.assert_called_once_with(
            user,
            BillingEvent.DAP_FLOW_ON_COMPLETION,
            "test_category",
            unit_of_measure="execution",
            quantity=1,
            metadata={
                "workflow_id": "workflow-123",
                "feature_qualified_name": "software_development",
                "feature_ai_catalog_item": True,
                "execution_environment": ExecutionEnvironment.DAP.value,
                "llm_operations": [
                    {
                        "model_id": "claude-3-5-sonnet",
                        "model_engine": "anthropic",
                        "model_provider": "anthropic",
                        "token_count": 150,
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                    }
                ],
            },
        )

    def test_track_billing_with_multiple_operations(
        self, billing_service, billing_client, user, gl_context, llm_operation
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

        billing_service.track_billing(
            workflow_id="workflow-456",
            user=user,
            gl_context=gl_context,
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
            llm_ops=[llm_operation, second_op],
        )

        metadata = get_call_metadata(billing_client)
        assert len(metadata["llm_operations"]) == 2
        assert metadata["llm_operations"][0]["token_count"] == 150
        assert metadata["llm_operations"][1]["token_count"] == 200

    def test_track_billing_defaults_and_gl_context_variations(
        self, billing_service, billing_client, user, llm_operation
    ):
        """Test default parameters and different GL context configurations."""
        gl_context = GLReportingEventContext.from_workflow_definition(
            "chat", is_ai_catalog_item=False
        )

        billing_service.track_billing(
            workflow_id="workflow-test",
            user=user,
            gl_context=gl_context,
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
            llm_ops=[llm_operation],
        )

        call_args = billing_client.track_billing_event.call_args
        assert call_args[1]["unit_of_measure"] == "request"
        assert call_args[1]["quantity"] == 1

        metadata = call_args[1]["metadata"]
        assert metadata["feature_qualified_name"] == "chat"
        assert metadata["feature_ai_catalog_item"] is False

    def test_track_billing_exception_handling(
        self, billing_service, billing_client, user, gl_context, llm_operation
    ):
        """Test that exceptions during billing propagate."""
        billing_client.track_billing_event.side_effect = Exception("Billing error")

        with pytest.raises(Exception, match="Billing error"):
            billing_service.track_billing(
                workflow_id="workflow-error",
                user=user,
                gl_context=gl_context,
                event=BillingEvent.DAP_FLOW_ON_COMPLETION,
                execution_env=ExecutionEnvironment.DAP,
                category="test_category",
                llm_ops=[llm_operation],
            )

        billing_client.track_billing_event.assert_called_once()

    @patch("lib.billing_events.service.get_llm_operations")
    def test_track_billing_with_context_operations(
        self,
        mock_get_llm_operations,
        billing_service,
        billing_client,
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

        billing_service.track_billing(
            workflow_id="workflow-context",
            user=user,
            gl_context=gl_context,
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
        )

        mock_get_llm_operations.assert_called_once()
        metadata = get_call_metadata(billing_client)
        assert len(metadata["llm_operations"]) == 1
        assert metadata["llm_operations"][0]["model_id"] == "gpt-4"
        assert metadata["llm_operations"][0]["token_count"] == 300

    @patch("lib.billing_events.service.get_llm_operations")
    def test_track_billing_raises_error_when_no_operations_available(
        self, mock_get_llm_operations, billing_service, user, gl_context
    ):
        """Test that ValueError is raised when no LLM operations are available."""
        mock_get_llm_operations.return_value = None

        with pytest.raises(
            ValueError, match="No LLM operations available for billing tracking"
        ):
            billing_service.track_billing(
                workflow_id="workflow-no-ops",
                user=user,
                gl_context=gl_context,
                event=BillingEvent.DAP_FLOW_ON_COMPLETION,
                execution_env=ExecutionEnvironment.DAP,
                category="test_category",
            )

    @patch("lib.billing_events.service.self_hosted_dap_billing_enabled")
    @patch("lib.billing_events.service.get_llm_operations")
    @pytest.mark.parametrize(
        ("explicit_ops_provided", "expected_model"),
        [
            (False, "self-hosted-model"),  # Self-hosted takes priority over context
            (True, "self-hosted-model"),  # Self-hosted takes priority over explicit
        ],
    )
    def test_track_billing_operation_priority(
        self,
        mock_get_llm_operations,
        mock_self_hosted_enabled,
        billing_service,
        billing_client,
        user,
        gl_context,
        llm_operation,
        explicit_ops_provided,
        expected_model,
    ):
        """Test operation priority: self-hosted > explicit > context."""
        mock_self_hosted_enabled.get.return_value = True
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

        billing_service.track_billing(
            workflow_id="workflow-priority",
            user=user,
            gl_context=gl_context,
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
            llm_ops=[llm_operation] if explicit_ops_provided else None,
        )

        mock_get_llm_operations.assert_not_called()
        metadata = get_call_metadata(billing_client)
        assert metadata["llm_operations"][0]["model_id"] == expected_model

    @patch("lib.billing_events.service.get_llm_operations")
    def test_track_billing_explicit_operations_skip_context(
        self,
        mock_get_llm_operations,
        billing_service,
        billing_client,
        user,
        gl_context,
        llm_operation,
    ):
        """Test that explicit operations skip context retrieval."""
        billing_service.track_billing(
            workflow_id="workflow-explicit",
            user=user,
            gl_context=gl_context,
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
            llm_ops=[llm_operation],
        )

        mock_get_llm_operations.assert_not_called()
        metadata = get_call_metadata(billing_client)
        assert metadata["llm_operations"][0]["model_id"] == "claude-3-5-sonnet"
