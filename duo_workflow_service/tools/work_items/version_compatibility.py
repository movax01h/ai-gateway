"""Version compatibility utilities for work item GraphQL queries."""

import structlog
from packaging.version import InvalidVersion, Version

from duo_workflow_service.tracking.errors import log_exception
from lib.context import gitlab_version

log = structlog.stdlib.get_logger(__name__)

# Version thresholds for feature availability
HIERARCHY_WIDGET_VERSION = Version("18.7.0")
DEFAULT_FALLBACK_VERSION = Version("18.6.0")


def get_gitlab_version() -> Version:
    """Get the current GitLab version from context.

    Returns:
        Version object representing the GitLab version.
        Falls back to DEFAULT_FALLBACK_VERSION if version cannot be determined.
    """
    try:
        version_str = gitlab_version.get()
        if version_str:
            return Version(str(version_str))
    except (InvalidVersion, TypeError) as ex:
        log_exception(ex, extra={"context": "Failed to parse GitLab version"})

    log.warning(
        "GitLab version not available, using fallback",
        fallback_version=str(DEFAULT_FALLBACK_VERSION),
    )
    return DEFAULT_FALLBACK_VERSION


def supports_hierarchy_widget() -> bool:
    """Check if the current GitLab version supports hierarchy widget fields.

    Returns:
        True if hierarchy widget is supported, False otherwise.
    """
    return get_gitlab_version() >= HIERARCHY_WIDGET_VERSION


def get_query_variables_for_version() -> dict:
    """Get GraphQL query variables based on GitLab version.

    Returns:
        Dictionary with version-specific variables for GraphQL queries.
    """
    return {
        "includeHierarchyWidget": supports_hierarchy_widget(),
    }
