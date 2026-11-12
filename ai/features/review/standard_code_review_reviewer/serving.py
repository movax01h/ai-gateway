"""Serving surface for the standard_code_review_reviewer feature.

standard_code_review_reviewer is the reviewer prompt of the standard_code_review flow, so it is served over gRPC on the
Duo Workflow Service.
"""

from duo_workflow_service.agent_platform.serving_surface import ServingSurface

SERVING_SURFACE = [ServingSurface(transport="grpc", deployable="dws")]
