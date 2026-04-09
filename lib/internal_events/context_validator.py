import structlog

from lib.internal_events.context import EventContext

__all__ = [
    "validate_event_context",
    "ALWAYS_EXPECTED_FIELDS",
    "CONTEXTUAL_FIELDS",
]

log = structlog.stdlib.get_logger("internal_events_context_validator")

_DEPLOYMENT_TYPE_DOTCOM = ".com"

# Fields expected on every request, any missing values indicate a bug in the caller.
ALWAYS_EXPECTED_FIELDS = [
    "realm",
    "instance_id",
    "unique_instance_id",
    "host_name",
    "global_user_id",
    "deployment_type",
]

# Fields expected only in certain contexts (GitLab.com).
CONTEXTUAL_FIELDS = [
    "project_id",
    "namespace_id",
    "feature_enabled_by_namespace_ids",
    "is_gitlab_team_member",
    "ultimate_parent_namespace_id",
]


def validate_event_context(context: EventContext, **location) -> None:
    """Validate event context and log one line per missing field.

    Args:
        context: The EventContext to validate.
        **location: Extra kwargs added to each log line
            (e.g. ``endpoint="/v1/chat"``,
            ``grpc_method="/duo_workflow.v1.DuoWorkflow/Execute"``).
    """
    for field in ALWAYS_EXPECTED_FIELDS:
        if getattr(context, field, None) is None:
            log.warning(
                "Internal event context missing required field",
                missing_field=field,
                field_type="required",
                correlation_id=context.correlation_id,
                **location,
            )

    if context.deployment_type == _DEPLOYMENT_TYPE_DOTCOM:
        for field in CONTEXTUAL_FIELDS:
            if getattr(context, field, None) is None:
                log.info(
                    "Internal event context missing contextual field",
                    missing_field=field,
                    field_type="contextual",
                    correlation_id=context.correlation_id,
                    **location,
                )
