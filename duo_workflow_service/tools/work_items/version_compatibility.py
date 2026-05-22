"""Version compatibility utilities for work item GraphQL queries."""

import structlog
from packaging.version import InvalidVersion, Version

from duo_workflow_service.tracking.errors import log_exception
from lib.context import gitlab_version

log = structlog.stdlib.get_logger(__name__)

# Version thresholds for feature availability
HIERARCHY_WIDGET_VERSION = Version("18.7.0")
NOTE_RESOLVABLE_AND_RESOLVED_FIELDS_VERSION = Version("18.9.0")
BASE_DISCUSSION_ID_FIELD_VERSION = Version("18.9.0")
DEVELOPMENT_WIDGET_VERSION = Version("18.9.0")
LICENSED_FEATURE_AVAILABILITY_VERSION = Version("18.11.0")
AGENT_PLAN_WIDGET_VERSION = Version("19.0.0")
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


def supports_note_resolved_and_resolvable_fields() -> bool:
    """Check if the current GitLab version supports note resolved and resolvable fields.

    Returns:
        True if note resolved and resolvable are supported, False otherwise.
    """
    return get_gitlab_version() >= NOTE_RESOLVABLE_AND_RESOLVED_FIELDS_VERSION


def supports_discussion_id_field() -> bool:
    """Check if the current GitLab version supports base discussion ID field.

    Returns:
        True if note discussion ID is supported, False otherwise.
    """
    return get_gitlab_version() >= BASE_DISCUSSION_ID_FIELD_VERSION


def supports_development_widget() -> bool:
    """Check if the current GitLab version supports development widget fields.

    Returns:
        True if development widget is supported, False otherwise.
    """
    return get_gitlab_version() >= DEVELOPMENT_WIDGET_VERSION


def supports_agent_plan_widget() -> bool:
    """Check if the current GitLab version supports the agent plan widget.

    Returns:
        True if the agent plan widget is supported (GitLab >= 19.0), False otherwise.
    """
    return get_gitlab_version() >= AGENT_PLAN_WIDGET_VERSION


def get_query_variables_for_version(*requested_keys: str) -> dict:
    """Get GraphQL query variables based on GitLab version.

    Args:
        *requested_keys: Optional positional keys to return. If none provided returns all keys.

    Returns:
        Dictionary with version-specific variables for GraphQL queries.
    """
    all_variables = {
        "includeHierarchyWidget": supports_hierarchy_widget(),
        "includeNoteResolvedAndResolvableFields": supports_note_resolved_and_resolvable_fields(),
        "includeDiscussionIdField": supports_discussion_id_field(),
        "includeDevelopmentWidget": supports_development_widget(),
    }

    if not requested_keys:
        return all_variables

    filtered_variables = {}
    for key in requested_keys:
        if key in all_variables:
            filtered_variables[key] = all_variables[key]

    return filtered_variables


_AGENT_PLAN_WIDGET_PLACEHOLDER = "# AGENT_PLAN_WIDGET_PLACEHOLDER"
_AGENT_PLAN_WIDGET_FRAGMENT = (
    "... on WorkItemWidgetAgentPlan {\n                    content\n                }"
)


def get_query_with_agent_plan_widget(base_query: str) -> str:
    """Return the query with the WorkItemWidgetAgentPlan fragment injected when supported.

    On GitLab < 19.0 the type does not exist and including it in the query causes a schema
    validation error even when guarded with ``@include(if: false)``.  Each affected
    ``.graphql`` file contains a ``# AGENT_PLAN_WIDGET_PLACEHOLDER`` comment that this
    function replaces with the fragment on GitLab >= 19.0, or removes on older versions.

    Args:
        base_query: The base GraphQL query/mutation string containing the placeholder comment.

    Returns:
        The query string with the placeholder replaced by the fragment or removed.
    """
    if supports_agent_plan_widget():
        replacement = _AGENT_PLAN_WIDGET_FRAGMENT
    else:
        replacement = ""
    return base_query.replace(_AGENT_PLAN_WIDGET_PLACEHOLDER, replacement)


def supports_licensed_feature_availability() -> bool:
    """Check if the GitLab instance exposes the licensedFeatureAvailability GraphQL field."""
    return get_gitlab_version() >= LICENSED_FEATURE_AVAILABILITY_VERSION
