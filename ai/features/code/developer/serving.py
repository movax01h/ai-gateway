"""Serving surface for the developer feature.

developer is a Duo Workflow Service flow, served over gRPC. This reproduces its current serving surface (baseline).
"""

from duo_workflow_service.agent_platform.serving_surface import ServingSurface

SERVING_SURFACE = [ServingSurface(transport="grpc", deployable="dws")]
