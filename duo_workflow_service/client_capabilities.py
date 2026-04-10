from packaging.version import InvalidVersion, Version

from lib.context import client_capabilities, gitlab_version


def is_client_capable(capabilities: str | frozenset[str]) -> bool:
    """Check if the client supports all of the given capabilities.

    Args:
        capabilities: A single capability string or a frozenset of capability strings
            that must all be present.

    Returns:
        True if the client declares all required capabilities, False otherwise.
    """
    # Starting gitlab version 18.7, workhorse intercepts clientCapabilities payload for backwards compatibility
    # This version check can be removed when we no longer support version 18.7
    try:
        gl_version = Version(str(gitlab_version.get() or ""))
    except (InvalidVersion, TypeError):
        return False

    if gl_version < Version("18.7.0"):
        return False

    declared = client_capabilities.get()
    if isinstance(capabilities, str):
        return capabilities in declared
    return all(cap in declared for cap in capabilities)
