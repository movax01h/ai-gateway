from packaging.version import InvalidVersion, Version

from ai_gateway.instrumentators.model_requests import (
    client_capabilities,
    gitlab_version,
)


def is_client_capable(capability: str) -> bool:
    # Starting gitlab version 18.7, workhorse intercepts clientCapabilities payload for backwards compatibility
    # This version check can be removed when we no longer support version 18.7
    try:
        gl_version = Version(str(gitlab_version.get() or ""))
    except (InvalidVersion, TypeError):
        return False

    if gl_version < Version("18.7.0"):
        return False

    return capability in client_capabilities.get()
