"""Serving surface for the local_code_review feature.

local_code_review is a prompt run as a graph node in Duo Workflow Service flows, served over gRPC. This reproduces its
current serving surface (baseline).
"""

from duo_workflow_service.agent_platform.serving_surface import ServingSurface

SERVING_SURFACE = [ServingSurface(transport="grpc", deployable="dws")]
