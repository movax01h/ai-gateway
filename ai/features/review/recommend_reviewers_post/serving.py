"""Serving surface for the recommend_reviewers_post feature.

recommend_reviewers_post is a prompt of the recommend_reviewers flow, so it is served over gRPC on the Duo Workflow
Service. This reproduces its current serving surface (baseline).
"""

from duo_workflow_service.agent_platform.serving_surface import ServingSurface

SERVING_SURFACE = [ServingSurface(transport="grpc", deployable="dws")]
