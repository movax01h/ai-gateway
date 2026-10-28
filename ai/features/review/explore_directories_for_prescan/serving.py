"""Serving surface for the explore_directories_for_prescan feature.

explore_directories_for_prescan is a prompt of the code_review flow, so it is served over gRPC on the Duo Workflow
Service. This reproduces its current serving surface (baseline).
"""

from duo_workflow_service.agent_platform.serving_surface import ServingSurface

SERVING_SURFACE = [ServingSurface(transport="grpc", deployable="dws")]
