import os

import sentry_sdk
import structlog
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sentry_sdk.integrations.grpc import GRPCIntegration

log = structlog.stdlib.get_logger("error_tracking")


def setup_error_tracking():
    if sentry_tracking_available():
        sentry_sdk.init(
            dsn=os.environ.get("SENTRY_DSN"),
            environment=os.environ.get("DUO_WORKFLOW_SERVICE_ENVIRONMENT"),
            traces_sample_rate=1.0,
            before_send=remove_private_info_fields,
            profiles_sample_rate=1.0,
            integrations=[GRPCIntegration(), AsyncioIntegration()],
            max_value_length=30 * 1024,
        )


def sentry_tracking_available():
    if os.environ.get("SENTRY_ERROR_TRACKING_ENABLED") == "true":
        if os.environ.get("SENTRY_DSN"):
            log.debug("Using Sentry for error tracking...")
            return True
        log.debug("Could not find Sentry DSN for error tracking setup...")
    else:
        log.debug("Sentry error tracking disabled...")
    return False


def remove_private_info_fields(event, hint):
    # Remove sensitive information from event data
    updated_event = event

    if "server_name" in updated_event:
        updated_event["server_name"] = None
    return updated_event
