"""Serving surface for the secret_post_results_to_gitlab_agent_prompt feature.

secret_post_results_to_gitlab_agent_prompt is a prompt of the secrets_fp_detection flow, so it is served over gRPC on
the Duo Workflow Service. This reproduces its current serving surface (baseline).
"""

from duo_workflow_service.agent_platform.serving_surface import ServingSurface

SERVING_SURFACE = [ServingSurface(transport="grpc", deployable="dws")]
