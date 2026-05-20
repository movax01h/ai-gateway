from enum import StrEnum
from typing import Any

from gitlab_cloud_connector import CloudConnectorUser
from pydantic import BaseModel

from lib.billing_events.client import BillingEvent, BillingEventsClient
from lib.context.llm_operations import get_llm_operations
from lib.context.tool_executions import ToolExecutions
from lib.events.base import GLReportingEventContext

__all__ = [
    "SelfHostedLLMOperations",
    "ExecutionEnvironment",
    "LLMOperation",
    "BillingEventService",
    "BILL_ONCE_PER_WORKFLOW_FEATURES",
]


BILL_ONCE_PER_WORKFLOW_FEATURES: frozenset[str] = frozenset(
    {
        "code_review",
        "sast_fp_detection",
        "secrets_fp_detection",
        "resolve_sast_vulnerability",
    }
)


class ExecutionEnvironment(StrEnum):
    """Execution environment identifiers for billing categorization."""

    DAP = "duo_agent_platform"
    CODE_COMPLETIONS = "code_completions"
    CODE_GENERATIONS = "code_generations"


class LLMOperation(BaseModel):
    """Represents a single LLM operation with token usage and model metadata.

    Attributes:
        token_count: Total tokens used (prompt + completion)
        model_id: Identifier of the LLM model used
        model_engine: Engine that executed the model (e.g., 'litellm', 'anthropic')
        model_provider: Provider of the model (e.g., 'anthropic', 'openai')
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        agent_name: Name of the agent that triggered this operation, if any
        cache_read_tokens: Number of (read) tokens in the cache
        cache_write_tokens: Number of (write) tokens in the cache
        operation_type: Categorizes the LLM call (e.g., 'standard', 'compaction_auto')
    """

    token_count: int
    model_id: str
    model_engine: str
    model_provider: str
    prompt_tokens: int
    completion_tokens: int
    agent_name: str | None = None
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    operation_type: str = "standard"


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
        user: CloudConnectorUser,
        gl_context: GLReportingEventContext,
        *,
        workflow_id: str | None = None,
        event: BillingEvent,
        execution_env: ExecutionEnvironment,
        category: str,
        unit_of_measure: str = "request",
        quantity: int = 1,
        llm_ops: list[LLMOperation] | None = None,
        tool_execs: ToolExecutions | None = None,
        orbit_called: bool = False,
    ) -> None:
        """Track billing for a workflow execution with LLM operation metadata.

        LLM operations are retrieved in priority order:
        1. Explicitly provided operations (llm_ops parameter)
        2. Operations from request context (via get_llm_operations)

        Args:
            user: CloudConnectorUser containing user claims and authentication info
            gl_context: GitLab reporting event context with feature metadata
            workflow_id: Optional unique identifier for the workflow being billed.
                Not applicable for stateless operations like code suggestions.
            event: Type of billable event being tracked
            execution_env: Execution environment where the workflow ran (e.g., DAP)
            category: Location or category where the billing event occurred
            unit_of_measure: Base unit for measurement (default: "request")
            quantity: Number of units consumed (default: 1)
            llm_ops: Optional explicit LLM operations to track. Use this for legacy code paths
                that don't work well with get_llm_operations(). Prefer migrating to get_llm_operations()
                for new implementations.
            tool_execs: Optional explicit tool names to track.
            orbit_called: Whether any Orbit tools were called during the workflow session.

        Raises:
            ValueError: If no LLM operations are available from any source

        Note:
            For self-hosted deployments, standardized placeholder operations are used since
            actual token counts and model details are not available from self-hosted providers.
        """
        if llm_ops:
            llm_operations = [ops.model_dump() for ops in llm_ops]
        elif raw_ops := get_llm_operations():
            # Context-based: validate raw dicts from request context and convert to serializable format
            llm_operations = [
                LLMOperation.model_validate(ops).model_dump() for ops in raw_ops
            ]
        else:
            raise ValueError("No LLM operations available for billing tracking")

        metadata: dict[str, Any] = {
            "feature_qualified_name": gl_context.feature_qualified_name,
            "feature_ai_catalog_item": gl_context.feature_ai_catalog_item,
            "execution_environment": execution_env.value,
            "llm_operations": llm_operations,
            "tool_names": tool_execs or [],
            "orbit_called": orbit_called,
        }
        if workflow_id:
            metadata["workflow_id"] = workflow_id

        self.client.track_billing_event(
            user,
            event,
            category,
            unit_of_measure=unit_of_measure,
            quantity=quantity,
            metadata=metadata,
        )
