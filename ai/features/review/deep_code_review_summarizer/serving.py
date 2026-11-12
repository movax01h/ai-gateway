"""Serving surface for the deep_code_review_summarizer feature.

deep_code_review_summarizer writes the summary of the deep_code_review flow, so it is served over gRPC on the Duo
Workflow Service.
"""

from duo_workflow_service.agent_platform.serving_surface import ServingSurface

SERVING_SURFACE = [ServingSurface(transport="grpc", deployable="dws")]
