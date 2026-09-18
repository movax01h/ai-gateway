"""Feature-owned tools for the analytics_agent flow (GLQL)."""

from langchain.tools import BaseTool

from .get_glql_schema import GetGlqlSchema
from .run_glql_query import RunGLQLQuery

__all__ = ["FEATURE_TOOLS", "GetGlqlSchema", "RunGLQLQuery"]

# Tools this feature owns, grouped by the agent privilege that gates them.
# The tools registry merges these into its privilege map at discovery time.
FEATURE_TOOLS: dict[str, list[type[BaseTool]]] = {
    "read_only_gitlab": [GetGlqlSchema, RunGLQLQuery],
}
