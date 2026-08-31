from contextvars import ContextVar
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "EventConstructContext",
    "current_event_construct",
]

# UUID v4 for `event_id` (idempotency + Snowflake<->DIP join key).
_UUID_V4_PATTERN = (
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}"
    r"-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)
# Generic UUID for `organization_uuid` (organizations.uuid).
_UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"

DeploymentType = Literal["self-managed", ".com", "dedicated"]
Realm = Literal["self-managed", "saas", "dedicated"]
UnitOfMeasure = Literal[
    "byte",
    "megabyte",
    "gigabyte",
    "second",
    "minute",
    "hour",
    "token",
    "request",
    "execution",
    "scan",
    "suggestion",
    "workflow",
    "secret",
]
Assignment = Literal["Duo Pro", "Duo Enterprise"]


class EventConstructContext(BaseModel):
    """The unified event construct: a single self-describing context carrying
    product-usage + billing information.

    This model mirrors the ``com.gitlab/event_construct/jsonschema/1-0-0`` iglu
    schema field-for-field. Fields are grouped in the same order as the schema:
    ENVELOPE, WHO, WHERE, WHAT, BILLING, METADATA.

    See https://gitlab.com/gitlab-org/iglu/-/tree/master/public/schemas/com.gitlab/event_construct
    about the spec of the GitLab unified event construct.
    """

    # `additionalProperties: false` on the schema -> reject unknown keys so drift
    # is caught at the Python boundary rather than silently dropped on the wire.
    # Opt out of the protected "model_" namespace for future-proofing.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # -- ENVELOPE --------------------------------------------------------------
    event_id: str = Field(
        pattern=_UUID_V4_PATTERN,
        max_length=36,
        description="Unique ID of the specific event. UUID v4.",
    )
    event_type: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Type of the event. Maps 1:1 to billable_usage.event_type for DIP.",
    )
    timestamp: Optional[str] = Field(
        default=None,
        max_length=32,
        description="ISO8601 event time. Required by DIP and enforced by the gate for billable events.",
    )
    environment: str = Field(
        default="development",
        max_length=32,
        description="Name of the source environment, such as `production` or `staging`.",
    )
    source: Optional[str] = Field(
        default="ai-gateway-python",
        max_length=32,
        description="Name of the source application, such as `gitlab-rails` or `gitlab-javascript`.",
    )
    correlation_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Unique request id for each request.",
    )

    # -- WHO -------------------------------------------------------------------
    subject: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Raw (unhashed) ID of the subject performing the action. Used for billing attribution.",
    )
    subject_type: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Type of subject: user, service_account, ci_pipeline, agent, runner.",
    )
    global_user_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Globally unique user ID.",
    )
    user_id: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Hashed ID of the associated user (analytics/privacy).",
    )
    is_gitlab_team_member: Optional[bool] = Field(
        default=None,
        description="Indicates if triggered by a GitLab team member.",
    )

    # -- WHERE -----------------------------------------------------------------
    deployment_type: Optional[DeploymentType] = Field(
        default=None,
        description="self-managed, .com or dedicated. Created to replace 'realm' in the future.",
    )
    realm: Optional[Realm] = Field(
        default=None,
        description="Self-Managed, SaaS or Dedicated.",
    )
    unique_instance_id: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Unique ID of the GitLab instance where the request comes from.",
    )
    host_name: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Host name of the GitLab instance where the request comes from.",
    )
    instance_version: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Version of the GitLab instance where the request comes from.",
    )
    organization_id: Optional[int] = Field(
        default=None,
        ge=0,
        le=2147483647,
        description="ID of the associated organization.",
    )
    organization_uuid: Optional[str] = Field(
        default=None,
        pattern=_UUID_PATTERN,
        max_length=36,
        description="UUID of the associated organization (organizations.uuid).",
    )
    root_namespace_id: Optional[int] = Field(
        default=None,
        ge=0,
        le=2147483647,
        description="ID of the associated ultimate parent (root) namespace. Canonical billing tenant.",
    )
    namespace_id: Optional[int] = Field(
        default=None,
        ge=0,
        le=2147483647,
        description="ID of the associated namespace.",
    )
    project_id: Optional[int] = Field(
        default=None,
        ge=0,
        le=2147483647,
        description="ID of the associated project.",
    )

    # -- WHAT ------------------------------------------------------------------
    feature_enablement_type: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Enablement type for the namespace that allows the user to use the tracked feature.",
    )
    plan: Optional[str] = Field(
        default=None,
        max_length=32,
        description="Name of the plan, such as `free`, `premium` or `ultimate`.",
    )
    google_analytics_id: Optional[str] = Field(
        default=None,
        max_length=32,
        description="Google Analytics ID from the marketing site.",
    )

    # -- BILLING ---------------------------------------------------------------
    is_billable: Optional[bool] = Field(
        default=None,
        description="Whether the event is billable. Resolved from the catalog, not set ad hoc by feature code.",
    )
    unit_of_measure: Optional[UnitOfMeasure] = Field(
        default=None,
        description="Base billing unit (singular). Required-when-billable via the gate.",
    )
    quantity: Optional[float] = Field(
        default=None,
        ge=0,
        le=2147483647,
        description="Usage quantity in unit_of_measure. Required-when-billable via the gate.",
    )
    assignments: Optional[List[Assignment]] = Field(
        default=None,
        description="Product assignments/licenses associated with the subject at event time.",
    )
    cost_bearer: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Party bearing the cost of the event, such as `customer` or `gitlab`.",
    )
    estimated_cost: Optional[float] = Field(
        default=None,
        ge=0,
        le=2147483647,
        description="Estimated cost of the event, expressed in USD.",
    )

    # -- METADATA --------------------------------------------------------------
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Free-form key-value pairs for adding additional context.",
    )

    @field_validator("assignments")
    @classmethod
    def _assignments_unique(
        cls, value: Optional[List[Assignment]]
    ) -> Optional[List[Assignment]]:
        # Schema declares `uniqueItems: true`; pydantic does not enforce it on a list.
        if value is not None and len(value) != len(set(value)):
            raise ValueError("assignments must not contain duplicate items")
        return value


# Per-request unified event construct. Default is None (not a shared instance):
# `event_id` is a required UUID, so there is no meaningful empty default and a
# module-level instance would leak one event_id across every request.
current_event_construct: ContextVar[Optional[EventConstructContext]] = ContextVar(
    "current_event_construct", default=None
)
