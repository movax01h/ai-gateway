from enum import StrEnum

from gitlab_cloud_connector import CloudConnectorUser
from pydantic import BaseModel

from lib.billing_events.client import BillingEvent, BillingEventsClient
from lib.context.llm_operations import get_llm_operations
from lib.events.base import GLReportingEventContext
from lib.events.contextvar import self_hosted_dap_billing_enabled

__all__ = [
    "ExecutionEnvironment",
    "LLMOperation",
    "BillingEventService",
]


class ExecutionEnvironment(StrEnum):
    """Execution environment identifiers for billing categorization."""

    DAP = "duo_agent_platform"


class LLMOperation(BaseModel):
    """Represents a single LLM operation with token usage and model metadata.

    Attributes:
        token_count: Total tokens used (prompt + completion)
        model_id: Identifier of the LLM model used
        model_engine: Engine that executed the model (e.g., 'litellm', 'anthropic')
        model_provider: Provider of the model (e.g., 'anthropic', 'openai')
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
    """

    token_count: int
    model_id: str
    model_engine: str
    model_provider: str
    prompt_tokens: int
    completion_tokens: int


class SelfHostedLLMOperations:
    """Provides standardized LLM operations for self-hosted billing tracking."""

    @staticmethod
    def get_operations() -> list[LLMOperation]:
        """Get self-hosted LLM operations with standardized metadata.

        Returns:
            List containing a single LLMOperation with placeholder values for self-hosted billing.
        """
        return [
            LLMOperation(
                model_id="self-hosted-model",
                model_engine="litellm",
                model_provider="litellm",
                token_count=1,
                prompt_tokens=1,
                completion_tokens=1,
            )
        ]


class BillingEventService:
    """Service for tracking billable LLM operations and workflow events.

    This service handles billing event tracking across different deployment types (SaaS, self-hosted) and execution
    environments, ensuring proper LLM operation metadata is captured for billing purposes.
    """

    def __init__(self, client: BillingEventsClient):
        self.client = client

    def track_billing(
        self,
        workflow_id: str,
        user: CloudConnectorUser,
        gl_context: GLReportingEventContext,
        *,
        event: BillingEvent,
        execution_env: ExecutionEnvironment,
        category: str,
        unit_of_measure: str = "request",
        quantity: int = 1,
        llm_ops: list[LLMOperation] | None = None,
    ) -> None:
        """Track billing for a workflow execution with LLM operation metadata.

        LLM operations are retrieved in priority order:
        1. Self-hosted standardized operations (if self_hosted_dap_billing_enabled)
        2. Explicitly provided operations (llm_ops parameter)
        3. Operations from request context (via get_llm_operations)

        Args:
            workflow_id: Unique identifier for the workflow being billed
            user: CloudConnectorUser containing user claims and authentication info
            gl_context: GitLab reporting event context with feature metadata
            event: Type of billable event being tracked
            execution_env: Execution environment where the workflow ran (e.g., DAP)
            category: Location or category where the billing event occurred
            unit_of_measure: Base unit for measurement (default: "request")
            quantity: Number of units consumed (default: 1)
            llm_ops: Optional explicit LLM operations to track. Use this for legacy code paths
                that don't work well with get_llm_operations(). Prefer migrating to get_llm_operations()
                for new implementations.

        Raises:
            ValueError: If no LLM operations are available from any source

        Note:
            For self-hosted deployments, standardized placeholder operations are used since
            actual token counts and model details are not available from self-hosted providers.
        """
        # Retrieve LLM operations based on deployment type and availability
        if self_hosted_dap_billing_enabled.get():
            llm_operations = [
                ops.model_dump() for ops in SelfHostedLLMOperations.get_operations()
            ]
        elif llm_ops:
            llm_operations = [ops.model_dump() for ops in llm_ops]
        elif raw_ops := get_llm_operations():
            # Context-based: validate raw dicts from request context and convert to serializable format
            llm_operations = [
                LLMOperation.model_validate(ops).model_dump() for ops in raw_ops
            ]
        else:
            raise ValueError("No LLM operations available for billing tracking")

        metadata = {
            "workflow_id": workflow_id,
            "feature_qualified_name": gl_context.feature_qualified_name,
            "feature_ai_catalog_item": gl_context.feature_ai_catalog_item,
            "execution_environment": execution_env.value,
            "llm_operations": llm_operations,
        }

        self.client.track_billing_event(
            user,
            event,
            category,
            unit_of_measure=unit_of_measure,
            quantity=quantity,
            metadata=metadata,
        )
