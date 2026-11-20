"""Serving surface for the tool_approval_review feature.

tool_approval_review is the prompt of the auto mode tool-approval classifier's LLM reviewer. Duo Workflow Service runs
it as a graph node, so it's served over gRPC.
"""

from duo_workflow_service.agent_platform.serving_surface import ServingSurface

SERVING_SURFACE = [ServingSurface(transport="grpc", deployable="dws")]
