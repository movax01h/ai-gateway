"""Deprecated import location. Moved into the analytics_agent feature package.

Import from ``ai.features.insights.analytics_agent.components`` instead.
"""

import warnings

from ai.features.insights.analytics_agent.components.get_glql_schema import (
    GetGlqlSchema,
    GetGlqlSchemaInput,
)

warnings.warn(
    "duo_workflow_service.tools.get_glql_schema has moved to "
    "ai.features.insights.analytics_agent.components.get_glql_schema",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["GetGlqlSchema", "GetGlqlSchemaInput"]
