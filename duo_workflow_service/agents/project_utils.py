from typing import Optional

from langchain_core.messages.tool import ToolCall

from duo_workflow_service.gitlab.gitlab_api import Project


def resolve_project_name_for_tool(
    project: Optional[Project], tool_call: ToolCall
) -> Optional[str]:
    """Return the current project name only when the tool targets that project.

    Compares tool_call's project_id argument against the current project ID using string coercion to handle both int and
    str values from LLM responses.
    """
    if not project:
        return None
    tool_project_id = (tool_call.get("args") or {}).get("project_id")
    if tool_project_id is not None and str(tool_project_id) == str(project.get("id")):
        return project.get("name")
    return None
