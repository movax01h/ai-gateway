"""Serving surface for the ensure_clean_git_state feature.

ensure_clean_git_state is a prompt of the resolve_sast_vulnerability flow, so it is served over gRPC on the Duo Workflow
Service. This reproduces its current serving surface (baseline).
"""

from duo_workflow_service.agent_platform.serving_surface import ServingSurface

SERVING_SURFACE = [ServingSurface(transport="grpc", deployable="dws")]
