"""AI Context for Snowplow events."""

from dataclasses import dataclass
from typing import Optional

__all__ = ["AIContext"]


@dataclass
class AIContext:
    """AI Context for Snowplow events.

    This context follows the com.gitlab/ai_context/jsonschema/1-0-2 schema.
    Every property is optional and nullable; the schema sets
    ``additionalProperties: false``, so a field not declared there must not be
    added here.
    """

    # pylint: disable=too-many-instance-attributes

    session_id: Optional[str] = None
    workflow_id: Optional[str] = None
    flow_type: Optional[str] = None
    agent_name: Optional[str] = None
    # Resolved flow-registry identity (DWS only; None for AI Gateway events).
    flow_name: Optional[str] = None
    item_version: Optional[str] = None
    item_schema_version: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    ephemeral_5m_input_tokens: Optional[int] = None
    ephemeral_1h_input_tokens: Optional[int] = None
    cache_read: Optional[int] = None
    cache_creation: Optional[int] = None
