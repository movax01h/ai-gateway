"""The flows that the Duo Workflow Service serves over gRPC."""

from pathlib import Path
from typing import Optional

from duo_workflow_service.agent_platform.serving_surface import (
    ServingSurface,
    collect_serving_surfaces,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    discover_feature_flow_configs,
)
from lib.feature_roots import FOUNDATIONAL_AGENTS_DOMAIN

__all__ = ["GRPC_DWS", "register_served_flows"]

GRPC_DWS = ServingSurface(transport="grpc", deployable="dws")


def register_served_flows(features_dir: Optional[Path] = None) -> None:
    """Register the flow config of each feature whose ``serving.py`` declares ``GRPC_DWS``.

    A flow fetched from the AI Catalog into ``ai/features/foundational_agents/``
    has no ``serving.py`` and is always registered.

    Args:
        features_dir: Root directory to scan. Defaults to the repo's
            ``ai/features`` tree when not provided.

    Raises:
        ValueError: If a feature with a ``config/`` dir does not declare
            ``GRPC_DWS``, for example because its ``serving.py`` is missing or
            fails to import, or if two features use the same feature id.
    """
    served = {
        feature
        for feature, surfaces in collect_serving_surfaces(features_dir).items()
        if GRPC_DWS in surfaces
    }

    def _require_served(feature_dir: Path) -> bool:
        if (
            feature_dir.parent.name == FOUNDATIONAL_AGENTS_DOMAIN
            or feature_dir.name in served
        ):
            return True
        raise ValueError(
            f"Flow feature {feature_dir} has a config/ dir, but its serving.py does "
            "not declare grpc/dws. Declare it, or fix the serving.py import error "
            "logged above."
        )

    discover_feature_flow_configs(features_dir, include=_require_served)
