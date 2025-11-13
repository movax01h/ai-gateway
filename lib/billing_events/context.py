from typing import Any, ClassVar, Dict, List, Optional, Self

from pydantic import BaseModel, ConfigDict, Field

from lib.internal_events.context import EventContext

__all__ = [
    "BillingEventContext",
]


class BillingEventContext(BaseModel):
    """This model class represents the available attributes in the AI Gateway for the GitLab billable usage context.

    See https://gitlab.com/gitlab-org/iglu/-/tree/master/public/schemas/com.gitlab/billable_usage?ref_type=heads
    about the spec of the GitLab billable usage context.
    """

    event_id: str
    event_type: str
    unit_of_measure: str
    quantity: float
    realm: Optional[str] = None
    timestamp: str
    instance_id: Optional[str] = None
    unique_instance_id: Optional[str] = None
    host_name: Optional[str] = None
    project_id: Optional[int] = None
    namespace_id: Optional[int] = None
    subject: Optional[str] = None
    global_user_id: Optional[str] = None
    root_namespace_id: Optional[int] = None
    correlation_id: Optional[str] = None
    seat_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    deployment_type: Optional[str] = None
    assignments: Optional[List[str]] = None


class UsageQuotaEventContext(BaseModel):
    """Represents contextual metadata for usage quota events based on the GitLab billable usage context."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    environment: Optional[str] = None
    source: Optional[str] = None
    instance_id: Optional[str] = None
    unique_instance_id: Optional[str] = None
    instance_version: Optional[str] = None
    host_name: Optional[str] = None
    project_id: Optional[int] = None
    namespace_id: Optional[int] = None
    root_namespace_id: Optional[int] = Field(alias="ultimate_parent_namespace_id")
    user_id: Optional[str] = None
    global_user_id: Optional[str] = None
    realm: Optional[str] = None
    deployment_type: Optional[str] = None
    feature_enablement_type: Optional[str] = None

    @classmethod
    def from_internal_event(cls, internal_event_context: EventContext) -> Self:
        """Factory method to convert an internal `EventContext` object into a `UsageQuotaEventContext`.

        Args:
            internal_event_context (EventContext): An instance of the internal event context.

        Returns:
            UsageQuotaEventContext: A new instance populated with data from `internal_event_context`.
        """
        return cls.model_validate(internal_event_context.model_dump(), by_alias=True)
