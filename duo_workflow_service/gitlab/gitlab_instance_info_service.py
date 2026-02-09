import re
from typing import Optional
from urllib.parse import urlparse

import structlog
from pydantic import BaseModel

from duo_workflow_service.gitlab.gitlab_api import Namespace, Project
from duo_workflow_service.tracking.errors import log_exception
from lib.context import gitlab_version

log = structlog.stdlib.get_logger(__name__)


class GitLabInstanceInfo(BaseModel):
    """GitLab instance information."""

    instance_type: str
    instance_url: str
    instance_version: str


class GitLabInstanceInfoService:
    """Service for creating GitLab instance information from prefetched data."""

    def create_from_project(self, project: Optional[Project]) -> GitLabInstanceInfo:
        """Create GitLab instance info from project data.

        Args:
            project: Project data from GitLab API

        Returns:
            GitLabInstanceInfo with instance type, URL, and version
        """
        if not project:
            return self._create_fallback_info()

        web_url = project.get("web_url", "")
        return self._create_info_from_web_url(web_url)

    def create_from_namespace(
        self, namespace: Optional[Namespace]
    ) -> GitLabInstanceInfo:
        """Create GitLab instance info from namespace data.

        Args:
            namespace: Namespace data from GitLab API

        Returns:
            GitLabInstanceInfo with instance type, URL, and version
        """
        if not namespace:
            return self._create_fallback_info()

        web_url = namespace.get("web_url", "")
        return self._create_info_from_web_url(web_url)

    def create_from_project_and_namespace(
        self, project: Optional[Project], namespace: Optional[Namespace]
    ) -> GitLabInstanceInfo:
        """Create GitLab instance info from project and namespace data.

        Project takes priority over namespace if both are provided.

        Args:
            project: Project data from GitLab API
            namespace: Namespace data from GitLab API

        Returns:
            GitLabInstanceInfo with instance type, URL, and version
        """
        if project:
            return self.create_from_project(project)
        elif namespace:
            return self.create_from_namespace(namespace)
        else:
            return self._create_fallback_info()

    def _create_info_from_web_url(self, web_url: str) -> GitLabInstanceInfo:
        """Create GitLab instance info from a web URL.

        Args:
            web_url: Web URL from project or namespace

        Returns:
            GitLabInstanceInfo with instance type, URL, and version
        """
        instance_type = self._determine_instance_type_from_url(web_url)
        instance_url = self._extract_base_url_from_web_url(web_url)
        instance_version = self._get_gitlab_version()

        return GitLabInstanceInfo(
            instance_type=instance_type,
            instance_url=instance_url,
            instance_version=instance_version,
        )

    def _determine_instance_type_from_url(self, web_url: str) -> str:
        """Determine GitLab instance type based on the URL.

        Args:
            web_url: Web URL to analyze

        Returns:
            Instance type string
        """
        if not web_url or web_url == "Unknown":
            return "Unknown"

        web_url_lower = web_url.lower()

        # GitLab Dedicated uses dedicated-*.gitlab.com domains
        # Use regex to match the hostname pattern specifically
        if re.search(r"://dedicated-[^/]*\.gitlab\.com", web_url_lower):
            return "GitLab Dedicated"

        # GitLab.com (SaaS)
        if "gitlab.com" in web_url_lower:
            return "GitLab.com (SaaS)"

        # Everything else is considered self-managed
        return "Self-Managed"

    def _extract_base_url_from_web_url(self, web_url: str) -> str:
        """Extract base URL from a web URL.

        Args:
            web_url: Full web URL

        Returns:
            Base URL (scheme + netloc)
        """
        if not web_url or web_url == "Unknown":
            return "Unknown"

        try:
            parsed = urlparse(web_url)
            if parsed.scheme and parsed.netloc:
                return f"{parsed.scheme}://{parsed.netloc}"
            else:
                return "Unknown"
        except Exception as e:
            log_exception(e, extra={"context": "Failed to parse web URL"})
            return "Unknown"

    def _get_gitlab_version(self) -> str:
        """Get GitLab version from the version interceptor.

        Returns:
            GitLab version string or "Unknown" if not available
        """
        try:
            version = gitlab_version.get()
            return version if version else "Unknown"
        except Exception as e:
            log_exception(e, extra={"context": "Failed to get GitLab version"})
            return "Unknown"

    def _create_fallback_info(self) -> GitLabInstanceInfo:
        """Create fallback GitLab instance info when no data is available.

        Returns:
            GitLabInstanceInfo with "Unknown" values
        """
        return GitLabInstanceInfo(
            instance_type="Unknown",
            instance_url="Unknown",
            instance_version="Unknown",
        )
