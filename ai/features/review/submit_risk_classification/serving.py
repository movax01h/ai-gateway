"""Serving surface for the submit_risk_classification feature.

submit_risk_classification is a prompt of the risk_classification flow, so it is served over gRPC on the Duo Workflow
Service.
"""

from duo_workflow_service.agent_platform.serving_surface import ServingSurface

SERVING_SURFACE = [ServingSurface(transport="grpc", deployable="dws")]
