# flake8: noqa

from typing import Any

from .ascp import *
from .audit_events import *
from .branch import *
from .code_review import *
from .ci_linter import *
from .clarification_question import *
from .command import *
from .commit import *
from .documentation_search import *
from .duo_base_tool import format_tool_display_message
from .epic import *
from .fetch_glql_schema import *
from .filesystem import *
from .findings import *
from .git import *
from .handover import *
from .issue import *
from .job import *
from .mcp_tools import *
from .merge_request import *
from .merge_request_notes import *
from .notify_me_when import *
from .pipeline import *
from .planner import *
from .project import *
from .render_ui import *
from .repository_files import *
from .request_user_clarification import *
from .risk_classification import *
from .search import *
from .search_system import *
from .security import *
from .session_context import *
from .start_flow import *
from .testing import *
from .todo_write import *
from .toolset import *
from .user import *
from .vulnerabilities import *
from .web_search import *
from .wiki import *
from .work_item import *

# Tools that live in their feature package under ai/features/ (see
# docs/module_boundaries.md). Resolve the old attribute access lazily to
# avoid a circular import between this package and the moved modules.
_MOVED_TOOLS = {
    "GetGlqlSchema": "ai.features.insights.analytics_agent.components.get_glql_schema",
    "GetGlqlSchemaInput": "ai.features.insights.analytics_agent.components.get_glql_schema",
    "RunGLQLQuery": "ai.features.insights.analytics_agent.components.run_glql_query",
    "GLQLQueryInput": "ai.features.insights.analytics_agent.components.run_glql_query",
}


def __getattr__(name: str) -> Any:
    module_path = _MOVED_TOOLS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib  # pylint: disable=import-outside-toplevel
    import warnings  # pylint: disable=import-outside-toplevel

    warnings.warn(
        f"duo_workflow_service.tools.{name} moved to {module_path}",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    # Keep the moved names visible to dir() and enumeration-based tooling.
    return sorted(set(globals()) | set(_MOVED_TOOLS))
