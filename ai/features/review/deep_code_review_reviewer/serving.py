"""Serving surface for the deep_code_review_reviewer feature.

deep_code_review_reviewer is the reviewer prompt of the deep_code_review flow, so it is served over gRPC on the Duo
Workflow Service. This reproduces its current serving surface (baseline).
"""

from duo_workflow_service.agent_platform.serving_surface import ServingSurface

SERVING_SURFACE = [ServingSurface(transport="grpc", deployable="dws")]
