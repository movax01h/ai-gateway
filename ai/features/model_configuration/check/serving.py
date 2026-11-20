"""Serving surface for the model_configuration/check feature.

model_configuration/check is the self-hosted model connection test prompt that GitLab Rails calls over REST on the AI
Gateway.
"""

from duo_workflow_service.agent_platform.serving_surface import ServingSurface

SERVING_SURFACE = [ServingSurface(transport="rest", deployable="aigw")]
