"""Orbit tool usage tracking context variables.

These context variables track Orbit tool calls within a workflow session for telemetry (Snowplow events) and billing
(orbit_called flag).
"""

from contextvars import ContextVar
from typing import Optional

# MCP server name identifier for Orbit tools. Workhorse prefixes MCP tool names
# with "<server-name>_" (e.g., "orbit_query_graph"), so detection uses a strict
# prefix match to avoid false positives on tool names that merely contain
# "orbit" as a substring (which would otherwise affect billing).
ORBIT_TOOL_IDENTIFIER = "orbit"
_ORBIT_TOOL_NAME_PREFIX = f"{ORBIT_TOOL_IDENTIFIER}_"


def is_orbit_tool(tool_name: str) -> bool:
    """Return True if ``tool_name`` follows Workhorse's Orbit MCP naming convention."""
    return tool_name.startswith(_ORBIT_TOOL_NAME_PREFIX)


orbit_tool_call_count: ContextVar[int] = ContextVar("orbit_tool_call_count", default=0)
total_tool_call_count: ContextVar[int] = ContextVar("total_tool_call_count", default=0)


def init_orbit_counters() -> None:
    """Reset Orbit tool call counters at the start of a workflow session.

    Defensive against ContextVar leakage if workflows ever run sequentially in
    the same context (currently isolated by ``asyncio.create_task``'s
    ``copy_context()``, but we don't want to rely on that).
    """
    orbit_tool_call_count.set(0)
    total_tool_call_count.set(0)


def build_orbit_session_summary_extras(
    workflow_id: str, workflow_type: str
) -> Optional[dict]:
    """Build kwargs for ORBIT_DAP_SESSION_SUMMARY, or None when no orbit tools were called."""
    orbit_count = orbit_tool_call_count.get()
    if orbit_count == 0:
        return None
    total_count = total_tool_call_count.get()
    return {
        "value": workflow_id,
        "workflow_type": workflow_type,
        "orbit_calls_count": orbit_count,
        "non_orbit_tool_calls": total_count - orbit_count,
        "total_tool_calls": total_count,
    }
