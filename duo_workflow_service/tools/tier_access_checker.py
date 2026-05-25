import json
import urllib.parse
from typing import Any, Optional

import structlog
from langchain_core.tools import ToolException

from duo_workflow_service.errors.typing import TierAccessDeniedException
from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.gitlab.http_client import GitlabHttpClient, GitLabHttpResponse
from duo_workflow_service.gitlab.resource_resolver import resolve_identifier_to_path
from duo_workflow_service.tools.version_compatibility import (
    supports_licensed_feature_availability,
)
from duo_workflow_service.tracking.errors import log_exception

log = structlog.stdlib.get_logger("workflow")

# Licensed feature constants for tier access checks.
# Must match value in GitLab's GitlabSubscriptions::LicensedFeatureEnum GraphQL enum.
LICENSED_FEATURE_SECURITY_DASHBOARD = "SECURITY_DASHBOARD"

_TIER_CHECK_QUERY_TEMPLATE = """
query($fullPath: ID!, $feature: LicensedFeature!) {
    %s(fullPath: $fullPath) {
        licensedFeatureAvailability(feature: $feature) { available requiredPlan }
    }
}
"""


class TierAccessChecker:
    """Encapsulates tier access check logic for DuoBaseTool subclasses.

    This class is responsible for determining whether a tool's empty or error response is caused by a missing GitLab
    subscription tier, and raising TierAccessDeniedException when appropriate.
    """

    def __init__(
        self,
        tool_name: str,
        gitlab_client: GitlabHttpClient,
        project: Optional[Project] = None,
    ) -> None:
        self._tool_name = tool_name
        self._gitlab_client = gitlab_client
        self._project = project

    async def check_tier_access(
        self,
        feature: Optional[str],
        tool_result: Any,
        saved_kwargs: dict[str, Any],
    ) -> None:
        """Run the tier access check if the feature is set and the response looks empty/error."""
        if not feature or not self._is_empty_or_error_response(tool_result):
            return

        if not supports_licensed_feature_availability():
            log.debug(
                "Skipping tier access check: GitLab version too old",
                feature=feature,
            )
            return

        await self._verify_feature_availability(feature, saved_kwargs)

    @staticmethod
    def _is_empty_or_error_response(result: Any) -> bool:
        """Check if a tool result looks like an empty or error response."""
        if not isinstance(result, str):
            return False
        try:
            parsed = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return False
        if isinstance(parsed, dict):
            if parsed.get("error"):
                return True
            return any(
                isinstance(v, (list, dict)) and len(v) == 0 for v in parsed.values()
            )
        return isinstance(parsed, list) and not parsed

    async def _verify_feature_availability(
        self,
        feature: str,
        saved_kwargs: dict[str, Any],
    ) -> None:
        """Raise TierAccessDeniedException if the feature is unavailable."""
        log.info(
            "Running tier access check",
            tool=self._tool_name,
            feature=feature,
        )

        result = await self._get_resource_path(saved_kwargs)
        if not result:
            return

        resource_path, scope = result
        root_key = "namespace" if scope != "project" else "project"
        query = _TIER_CHECK_QUERY_TEMPLATE % root_key

        try:
            response = await self._gitlab_client.apost(
                path="/api/graphql",
                body=json.dumps(
                    {
                        "query": query,
                        "variables": {"fullPath": resource_path, "feature": feature},
                    }
                ),
            )
            body = (
                response.body if isinstance(response, GitLabHttpResponse) else response
            )
            if isinstance(body, str):
                body = json.loads(body)

            check = (body or {}).get("data", {}).get(root_key, {})
            check = check.get("licensedFeatureAvailability", {}) if check else {}

            if check and check.get("available") is False:
                raise TierAccessDeniedException(
                    required_plan=check.get("requiredPlan"),
                    feature=feature,
                )
        except TierAccessDeniedException:
            raise
        except Exception as e:
            log_exception(
                e,
                extra={
                    "context": "Tier access check failed",
                    "feature": feature,
                },
            )

    async def _get_resource_path(
        self, kwargs: dict[str, Any]
    ) -> Optional[tuple[str, str]]:
        """Return (full_path, scope) for the resource to tier-check."""
        project_full_path = kwargs.get("project_full_path")
        if project_full_path:
            return str(project_full_path), "project"
        for key, scope in (("group_id", "namespace"), ("project_id", "project")):
            value = kwargs.get(key)
            if value:
                try:
                    resolved = await resolve_identifier_to_path(
                        self._gitlab_client, str(value), scope
                    )
                except ToolException:
                    return None
                return resolved, scope
        if self._project:
            web_url = self._project.get("web_url", "")
            if web_url:
                path = urllib.parse.urlparse(web_url).path.lstrip("/")
                if path:
                    return path, "project"
        return None
