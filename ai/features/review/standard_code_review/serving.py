"""Serving surface for the standard_code_review feature.

standard_code_review is a flow served over gRPC on the Duo Workflow Service.
"""

from duo_workflow_service.agent_platform.serving_surface import ServingSurface

SERVING_SURFACE = [ServingSurface(transport="grpc", deployable="dws")]
