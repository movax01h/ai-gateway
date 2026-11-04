"""Serving surface for the convert_ci_push_changes feature.

convert_ci_push_changes is a prompt of the convert_to_gl_ci flow, so it is served over gRPC on the Duo Workflow Service.
This reproduces its current serving surface (baseline).
"""

from duo_workflow_service.agent_platform.serving_surface import ServingSurface

SERVING_SURFACE = [ServingSurface(transport="grpc", deployable="dws")]
