"""Serving surface for the skillguard_scan feature.

skillguard_scan is a flow served over gRPC on the Duo Workflow Service. This reproduces its current serving surface
(baseline).
"""

from duo_workflow_service.agent_platform.serving_surface import ServingSurface

SERVING_SURFACE = [ServingSurface(transport="grpc", deployable="dws")]
