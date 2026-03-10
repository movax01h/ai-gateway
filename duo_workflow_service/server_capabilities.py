"""Server capability negotiation for token generation."""

from structlog import get_logger

log = get_logger(__name__)


def get_dws_capabilities() -> list[str]:
    """Returns DWS capabilities for negotiation with GitLab Rails.

    Returns:
        List of capability strings (e.g., ["tool_call_approval"])
    """
    capabilities = [
        "tool_call_approval",
    ]

    log.info(
        "Returning DWS server capabilities for negotiation",
        capabilities=capabilities,
    )

    return capabilities
