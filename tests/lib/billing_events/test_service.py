from unittest.mock import Mock

import pytest

from lib.billing_events import BillingEvent, BillingEventsClient
from lib.billing_events.service import (
    AIModelMetadata,
    BillingEventService,
    BillingOperationsTracker,
    ExecutionEnvironment,
    LLMTokenUsage,
)
from lib.events import GLReportingEventContext


@pytest.fixture(name="mock_billing_client")
def mock_billing_client_fixture():
    """Fixture for a mocked BillingEventsClient."""
    return Mock(spec=BillingEventsClient)


@pytest.fixture(name="billing_service")
def billing_service_fixture(mock_billing_client):
    """Fixture for BillingEventService with mocked client."""
    return BillingEventService(client=mock_billing_client)


@pytest.fixture(name="gl_context")
def gl_context_fixture():
    """Fixture for GLReportingEventContext."""
    return GLReportingEventContext.from_workflow_definition(
        "software_development", is_ai_catalog_item=True
    )


@pytest.fixture(name="ai_model_metadata")
def ai_model_metadata_fixture():
    """Fixture for AIModelMetadata."""
    return AIModelMetadata(
        identifier="claude-3-5-sonnet",
        engine="anthropic",
        provider="anthropic",
    )


@pytest.fixture(name="llm_token_usage")
def llm_token_usage_fixture():
    """Fixture for LLMTokenUsage."""
    return LLMTokenUsage(
        token_count=150,
        prompt_tokens=100,
        completion_tokens=50,
    )


class TestBillingOperationsTracker:
    """Test suite for BillingOperationsTracker class."""

    def test_track_single_operation(self, ai_model_metadata, llm_token_usage):
        """Test tracking a single operation for a workflow."""
        tracker = BillingOperationsTracker()
        workflow_id = "workflow-123"

        tracker(
            workflow_id,
            ai_model_metadata=ai_model_metadata,
            llm_token_usage=llm_token_usage,
        )

        accumulated = tracker.accumulated()
        assert workflow_id in accumulated
        assert len(accumulated[workflow_id]) == 1

        operation = accumulated[workflow_id][0]
        assert operation["model_id"] == "claude-3-5-sonnet"
        assert operation["model_engine"] == "anthropic"
        assert operation["model_provider"] == "anthropic"
        assert operation["token_count"] == 150
        assert operation["prompt_tokens"] == 100
        assert operation["completion_tokens"] == 50

    def test_track_multiple_operations_same_workflow(
        self, ai_model_metadata, llm_token_usage
    ):
        """Test tracking multiple operations for the same workflow."""
        tracker = BillingOperationsTracker()
        workflow_id = "workflow-456"

        # First operation
        tracker(
            workflow_id,
            ai_model_metadata=ai_model_metadata,
            llm_token_usage=llm_token_usage,
        )

        # Second operation with different usage
        second_usage = LLMTokenUsage(
            token_count=200,
            prompt_tokens=120,
            completion_tokens=80,
        )
        tracker(
            workflow_id,
            ai_model_metadata=ai_model_metadata,
            llm_token_usage=second_usage,
        )

        accumulated = tracker.accumulated()
        assert len(accumulated[workflow_id]) == 2
        assert accumulated[workflow_id][0]["token_count"] == 150
        assert accumulated[workflow_id][1]["token_count"] == 200

    def test_track_multiple_workflows(self, ai_model_metadata, llm_token_usage):
        """Test tracking operations for multiple workflows."""
        tracker = BillingOperationsTracker()
        workflow_id_1 = "workflow-1"
        workflow_id_2 = "workflow-2"

        tracker(
            workflow_id_1,
            ai_model_metadata=ai_model_metadata,
            llm_token_usage=llm_token_usage,
        )

        second_metadata = AIModelMetadata(
            identifier="gpt-4",
            engine="openai",
            provider="openai",
        )
        tracker(
            workflow_id_2,
            ai_model_metadata=second_metadata,
            llm_token_usage=llm_token_usage,
        )

        accumulated = tracker.accumulated()
        assert len(accumulated) == 2
        assert workflow_id_1 in accumulated
        assert workflow_id_2 in accumulated
        assert accumulated[workflow_id_1][0]["model_id"] == "claude-3-5-sonnet"
        assert accumulated[workflow_id_2][0]["model_id"] == "gpt-4"


class TestBillingEventService:
    """Test suite for BillingEventService class."""

    def test_start_billing_context_manager_yields_tracker(
        self, billing_service, user, gl_context
    ):
        """Test that start_billing yields a BillingOperationsTracker."""
        with billing_service.start_billing(
            user=user,
            gl_context=gl_context,
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
        ) as tracker:
            assert isinstance(tracker, BillingOperationsTracker)

    def test_start_billing_no_operations_tracked(
        self, billing_service, mock_billing_client, user, gl_context
    ):
        """Test that no billing events are sent when no operations are tracked."""
        with billing_service.start_billing(
            user=user,
            gl_context=gl_context,
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
        ):
            pass  # No operations tracked

        mock_billing_client.track_billing_event.assert_not_called()

    def test_start_billing_single_workflow_single_operation(
        self,
        billing_service,
        mock_billing_client,
        user,
        gl_context,
        ai_model_metadata,
        llm_token_usage,
    ):
        """Test billing event sent for single workflow with single operation."""
        workflow_id = "workflow-123"

        with billing_service.start_billing(
            user=user,
            gl_context=gl_context,
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
            unit_of_measure="execution",
            quantity=1,
        ) as tracker:
            tracker(
                workflow_id,
                ai_model_metadata=ai_model_metadata,
                llm_token_usage=llm_token_usage,
            )

        mock_billing_client.track_billing_event.assert_called_once()
        call_args = mock_billing_client.track_billing_event.call_args

        assert call_args[0][0] == user
        assert call_args[0][1] == BillingEvent.DAP_FLOW_ON_COMPLETION
        assert call_args[0][2] == "test_category"
        assert call_args[1]["unit_of_measure"] == "execution"
        assert call_args[1]["quantity"] == 1

        metadata = call_args[1]["metadata"]
        assert metadata["workflow_id"] == workflow_id
        assert metadata["feature_qualified_name"] == "software_development"
        assert metadata["feature_ai_catalog_item"] is True
        assert metadata["execution_environment"] == ExecutionEnvironment.DAP.value
        assert len(metadata["llm_operations"]) == 1
        assert metadata["llm_operations"][0]["model_id"] == "claude-3-5-sonnet"
        assert metadata["llm_operations"][0]["token_count"] == 150

    def test_start_billing_single_workflow_multiple_operations(
        self,
        billing_service,
        mock_billing_client,
        user,
        gl_context,
        ai_model_metadata,
        llm_token_usage,
    ):
        """Test billing event sent for single workflow with multiple operations."""
        workflow_id = "workflow-456"

        with billing_service.start_billing(
            user=user,
            gl_context=gl_context,
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            execution_env=ExecutionEnvironment.DAP,
            category="test_category",
        ) as tracker:
            # First operation
            tracker(
                workflow_id,
                ai_model_metadata=ai_model_metadata,
                llm_token_usage=llm_token_usage,
            )

            # Second operation
            second_usage = LLMTokenUsage(
                token_count=200,
                prompt_tokens=120,
                completion_tokens=80,
            )
            tracker(
                workflow_id,
                ai_model_metadata=ai_model_metadata,
                llm_token_usage=second_usage,
            )

        mock_billing_client.track_billing_event.assert_called_once()
        call_args = mock_billing_client.track_billing_event.call_args

        metadata = call_args[1]["metadata"]
        assert len(metadata["llm_operations"]) == 2
        assert metadata["llm_operations"][0]["token_count"] == 150
        assert metadata["llm_operations"][1]["token_count"] == 200

    def test_start_billing_exception_handling(
        self,
        billing_service,
        mock_billing_client,
        user,
        gl_context,
        ai_model_metadata,
        llm_token_usage,
    ):
        """Test that exceptions during billing propagate from context manager."""
        mock_billing_client.track_billing_event.side_effect = Exception("Billing error")

        # Exception should propagate from the context manager
        with pytest.raises(Exception, match="Billing error"):
            with billing_service.start_billing(
                user=user,
                gl_context=gl_context,
                event=BillingEvent.DAP_FLOW_ON_COMPLETION,
                execution_env=ExecutionEnvironment.DAP,
                category="test_category",
            ) as tracker:
                tracker(
                    "workflow-error",
                    ai_model_metadata=ai_model_metadata,
                    llm_token_usage=llm_token_usage,
                )

        # Verify the call was attempted
        mock_billing_client.track_billing_event.assert_called_once()
