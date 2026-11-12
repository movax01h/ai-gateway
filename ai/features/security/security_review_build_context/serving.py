"""Serving surface for the security_review_build_context feature.

security_review_build_context is a prompt of the security_review flow, so it is served over gRPC on the Duo Workflow
Service. This reproduces its current serving surface (baseline).
"""

from duo_workflow_service.agent_platform.serving_surface import ServingSurface

SERVING_SURFACE = [ServingSurface(transport="grpc", deployable="dws")]
