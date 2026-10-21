"""Serving surface for the summarize_review feature.

summarize_review is a prompt served over REST on the AI Gateway. This reproduces its current serving surface (baseline).
"""

from duo_workflow_service.agent_platform.serving_surface import ServingSurface

SERVING_SURFACE = [ServingSurface(transport="rest", deployable="aigw")]
