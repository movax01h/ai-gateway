"""AI Context for Snowplow events."""

from dataclasses import dataclass
from typing import Optional

__all__ = ["AIContext"]


@dataclass
class AIContext:
    """AI Context for Snowplow events.

    This context follows the com.gitlab/ai_context/jsonschema/1-0-0 schema.
    """

    session_id: Optional[str] = None
    workflow_id: Optional[str] = None
    flow_type: Optional[str] = None
    agent_name: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    ephemeral_5m_input_tokens: Optional[int] = None
    ephemeral_1h_input_tokens: Optional[int] = None
    cache_read: Optional[int] = None
