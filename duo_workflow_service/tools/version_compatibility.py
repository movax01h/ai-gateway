"""Version compatibility utilities for work item GraphQL queries."""

import structlog
from packaging.version import InvalidVersion, Version

from duo_workflow_service.tracking.errors import log_exception
from lib.context import gitlab_version

log = structlog.stdlib.get_logger(__name__)

# Version thresholds for feature availability
DEFAULT_FALLBACK_VERSION = Version("18.6.0")
HIERARCHY_WIDGET_VERSION = Version("18.7.0")
NOTE_RESOLVABLE_AND_RESOLVED_FIELDS_VERSION = Version("18.9.0")
BASE_DISCUSSION_ID_FIELD_VERSION = Version("18.9.0")
DEVELOPMENT_WIDGET_VERSION = Version("18.9.0")
LICENSED_FEATURE_AVAILABILITY_VERSION = Version("18.11.0")
AGENT_PLAN_WIDGET_VERSION = Version("19.0.0")
GROUP_LEVEL_CUSTOM_INSTRUCTIONS_VERSION = Version("19.0.0")


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


def supports_licensed_feature_availability() -> bool:
    """Check if the GitLab instance exposes the licensedFeatureAvailability GraphQL field."""
    return get_gitlab_version() >= LICENSED_FEATURE_AVAILABILITY_VERSION


def supports_group_level_custom_instructions() -> bool:
    """Check if the GitLab instance supports group level custom instructions.

    Returns:
        True if group level custom instructions are supported, False otherwise.
    """
    return get_gitlab_version() >= GROUP_LEVEL_CUSTOM_INSTRUCTIONS_VERSION
