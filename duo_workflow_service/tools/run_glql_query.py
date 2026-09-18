"""Deprecated import location. Moved into the analytics_agent feature package.

Import from ``ai.features.insights.analytics_agent.components`` instead.
"""

import warnings

from ai.features.insights.analytics_agent.components.run_glql_query import (
    GLQLQueryInput,
    RunGLQLQuery,
)

warnings.warn(
    "duo_workflow_service.tools.run_glql_query has moved to "
    "ai.features.insights.analytics_agent.components.run_glql_query",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["GLQLQueryInput", "RunGLQLQuery"]
