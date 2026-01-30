from collections import defaultdict
from contextlib import contextmanager
from enum import StrEnum
from typing import Any, Iterator

from gitlab_cloud_connector import CloudConnectorUser
from pydantic import BaseModel, Field

from lib.billing_events.client import BillingEvent, BillingEventsClient
from lib.events.base import GLReportingEventContext

__all__ = [
    "ExecutionEnvironment",
    "BillingEventService",
    "LLMTokenUsage",
    "BillingOperationsTracker",
    "SelfHostedBilling",
]


class ExecutionEnvironment(StrEnum):
    DAP = "duo_agent_platform"


class AIModelMetadata(BaseModel):
    identifier: str = Field(serialization_alias="model_id")
    engine: str = Field(serialization_alias="model_engine")
    provider: str = Field(serialization_alias="model_provider")


class LLMTokenUsage(BaseModel):
    token_count: int
    prompt_tokens: int
    completion_tokens: int


class BillingOperationsTracker:
    def __init__(self):
        self._accumulated = defaultdict(list)

    def __call__(
        self,
        workflow_id: str,
        *,
        ai_model_metadata: AIModelMetadata,
        llm_token_usage: LLMTokenUsage,
    ):
        self._accumulated[workflow_id].append(
            {
                **ai_model_metadata.model_dump(by_alias=True),
                **llm_token_usage.model_dump(),
            }
        )

    def accumulated(self) -> dict[str, list[dict[str, Any]]]:
        return self._accumulated


class BillingEventService:
    def __init__(self, client: BillingEventsClient):
        self.client = client

    @contextmanager
    def start_billing(
        self,
        user: CloudConnectorUser,
        gl_context: GLReportingEventContext,
        *,
        event: BillingEvent,
        execution_env: ExecutionEnvironment,
        category: str,
        unit_of_measure: str = "request",
        quantity: int = 1,
    ) -> Iterator[BillingOperationsTracker]:
        tracker = BillingOperationsTracker()

        yield tracker

        metadata = [
            {
                "workflow_id": flow_id,
                "feature_qualified_name": gl_context.feature_qualified_name,
                "feature_ai_catalog_item": gl_context.feature_ai_catalog_item,
                "execution_environment": execution_env.value,
                "llm_operations": llm_operations,
            }
            for flow_id, llm_operations in tracker.accumulated().items()
        ]

        for m in metadata:
            self.client.track_billing_event(
                user,
                event,
                category,
                unit_of_measure=unit_of_measure,
                quantity=quantity,
                metadata=m,
            )


##
# Special use cases below
##


class SelfHostedBilling:
    @staticmethod
    def ai_model_metadata() -> AIModelMetadata:
        return AIModelMetadata(
            identifier="self-hosted-model", engine="litellm", provider="litellm"
        )

    @staticmethod
    def llm_token_usage() -> LLMTokenUsage:
        return LLMTokenUsage(token_count=1, prompt_tokens=1, completion_tokens=1)
