from ai.apps.aigw import container as aigw_container
from ai.apps.dws import container as dws_container

__all__ = [
    "ContainerApplication",
]


class ContainerApplication(
    aigw_container.ContainerApplication, dws_container.ContainerApplication
):
    """Union of the AIGW and DWS containers, which both entrypoints still use.

    Issue #2501 moves each entrypoint to its own app container and removes this class.
    """
