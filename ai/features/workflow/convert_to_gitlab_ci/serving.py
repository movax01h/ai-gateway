"""Serving surface for the convert_to_gitlab_ci feature.

workflow/convert_to_gitlab_ci is a prompt of the legacy convert_to_gitlab_ci workflow, so it is served over gRPC on the
Duo Workflow Service. This reproduces its current serving surface (baseline).
"""

from duo_workflow_service.agent_platform.serving_surface import ServingSurface

SERVING_SURFACE = [ServingSurface(transport="grpc", deployable="dws")]
