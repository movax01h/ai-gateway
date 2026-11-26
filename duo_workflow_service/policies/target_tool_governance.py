"""Checks a GitLab write tool call against the tool rules of the project or group it targets.

A session's claim carries the rules of the top-level group the session runs in. A tool call can name a project in
another top-level group, so before such a call runs, GitLab resolves the rules of every project and group the call
names. Only a DENY is acted on: the verdict can refuse a call, never pre-approve one.
"""

import json
from functools import cache
from typing import TYPE_CHECKING, Any

import structlog
from langchain_core.tools import BaseTool, ToolException

from duo_workflow_service.gitlab.gitlab_api import workflow_global_id
from lib.feature_flags import FeatureFlag, is_feature_enabled

if TYPE_CHECKING:
    from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

log = structlog.stdlib.get_logger("target_tool_governance")

# A write waits on this check, so it gets one short attempt instead of the client's retries.
_CHECK_TIMEOUT_SECONDS = 5.0

# The argument names GitLab's resolver reads, shared with the GitLab MCP server's tools.
_TARGET_ARGUMENTS = ("url", "project_id", "group_id")

_PERMISSIONS = ("ALLOW", "ASK", "DENY")

_TOOL_CALL_GOVERNANCE_QUERY = """
query($workflowId: AiDuoWorkflowsWorkflowID!, $toolName: String!, $toolCallArgs: String!) {
    duoWorkflowWorkflows(workflowId: $workflowId) {
        nodes {
            toolCallGovernance(toolName: $toolName, toolCallArgs: $toolCallArgs) {
                permission
                message
            }
        }
    }
}
"""


class ToolCallDeniedByTargetRules(ToolException):
    """The tool rules of a project or group the call names deny the call."""


class TargetToolGovernanceUnavailable(ToolException):
    """The check could not get a verdict, so the call was refused without one."""


@cache
def _gitlab_write_tool_classes() -> frozenset[type[BaseTool]]:
    # Imported here: the registry imports every tool, and every tool imports DuoBaseTool.
    from duo_workflow_service.components.tools_registry import _AGENT_PRIVILEGES

    return frozenset(_AGENT_PRIVILEGES["read_write_gitlab"]) - frozenset(
        _AGENT_PRIVILEGES["read_only_gitlab"]
    )


def governs(tool: BaseTool) -> bool:
    """Whether the tool is a GitLab write tool that names its target in an argument GitLab can resolve."""
    if type(tool) not in _gitlab_write_tool_classes():
        return False

    schema = tool.args_schema
    fields = getattr(schema, "model_fields", {})
    return any(name in fields for name in _TARGET_ARGUMENTS)


async def enforce_target_tool_governance(
    tool: "DuoBaseTool", kwargs: dict[str, Any]
) -> None:
    """Raise ToolCallDeniedByTargetRules when the rules of a project or group the call targets deny it.

    GitLab pushes ``duo_workflow_target_tool_governance`` only when it serves the field, so the check is
    skipped without the flag. With the flag, any failure to get a verdict refuses the call.
    """
    if not is_feature_enabled(FeatureFlag.DUO_WORKFLOW_TARGET_TOOL_GOVERNANCE):
        return

    if not governs(tool):
        return

    target_args = {
        name: kwargs[name] for name in _TARGET_ARGUMENTS if kwargs.get(name) is not None
    }
    # A call that names no target has nothing in another group to write to.
    if not target_args:
        return

    workflow_id = tool.workflow_id

    try:
        response = await tool.gitlab_client.graphql(
            _TOOL_CALL_GOVERNANCE_QUERY,
            {
                "workflowId": workflow_global_id(workflow_id),
                "toolName": tool.name,
                "toolCallArgs": json.dumps(target_args, separators=(",", ":")),
            },
            timeout=_CHECK_TIMEOUT_SECONDS,
            retry_transient_errors=False,
        )
        governance = response["duoWorkflowWorkflows"]["nodes"][0]["toolCallGovernance"]
        permission = governance["permission"]
        if permission not in _PERMISSIONS:
            raise ValueError(
                f"Unexpected tool call governance permission: {permission!r}"
            )
    # graphql() raises bare Exceptions for timeouts, decode and query errors.
    except Exception as exc:
        log.warning(
            "Target tool governance check failed, refusing the tool call",
            reason="check_unavailable",
            tool_name=tool.name,
            workflow_id=workflow_id,
            target_args=target_args,
            exc_info=exc,
        )
        raise TargetToolGovernanceUnavailable(
            f"{tool.name} was not run because GitLab could not check the AI tool rules of its target. "
            "Try again later."
        ) from exc

    if permission != "DENY":
        return

    log.info(
        "Tool call denied by the AI tool rules of its target",
        reason="denied_by_target_rules",
        tool_name=tool.name,
        workflow_id=workflow_id,
        target_args=target_args,
    )
    raise ToolCallDeniedByTargetRules(
        governance.get("message") or f"{tool.name} is denied by AI tool rules."
    )
