"""Serving surface for the risk_assessment feature.

risk_assessment assesses each area a merge request touches against that area's criteria. It is a flow served over gRPC
on the Duo Workflow Service.
"""

from duo_workflow_service.agent_platform.serving_surface import ServingSurface

SERVING_SURFACE = [ServingSurface(transport="grpc", deployable="dws")]
