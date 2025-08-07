from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

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
