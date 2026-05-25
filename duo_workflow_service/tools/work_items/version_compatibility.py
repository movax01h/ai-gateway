"""Version compatibility utilities for work item GraphQL queries."""

from duo_workflow_service.tools.version_compatibility import (
    supports_agent_plan_widget,
    supports_development_widget,
    supports_discussion_id_field,
    supports_hierarchy_widget,
    supports_note_resolved_and_resolvable_fields,
)


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
