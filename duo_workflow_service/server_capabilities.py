"""Server capability negotiation for token generation and capability listing."""

from typing import NamedTuple

from structlog import get_logger

log = get_logger(__name__)


class CapabilityInfo(NamedTuple):
    """A single server capability and its associated metadata."""

    name: str
    metadata: str


# The single source of truth for DWS capabilities advertised for negotiation
# with GitLab Rails. `metadata` is currently unused and reserved for future use
# (e.g. per-capability version information).
_DWS_CAPABILITIES: list[CapabilityInfo] = [
    CapabilityInfo(name="tool_call_approval", metadata=""),
    CapabilityInfo(name="tool_call_pattern_approval", metadata=""),
    CapabilityInfo(name="flow_semantic_versioning", metadata=""),
]


def get_dws_capabilities() -> list[str]:
    """Returns DWS capability names for negotiation with GitLab Rails.

    Returns:
        List of capability strings (e.g., ["tool_call_approval"])
    """
    capabilities = [capability.name for capability in _DWS_CAPABILITIES]

    log.info(
        "Returning DWS server capabilities for negotiation",
        capabilities=capabilities,
    )

    return capabilities


def get_dws_capabilities_with_metadata() -> list[CapabilityInfo]:
    """Returns DWS capabilities with their metadata for the ListCapabilities RPC.

    Returns:
        List of CapabilityInfo(name, metadata) tuples.
    """
    log.info(
        "Returning DWS server capabilities with metadata",
        capabilities=_DWS_CAPABILITIES,
    )

    return _DWS_CAPABILITIES
