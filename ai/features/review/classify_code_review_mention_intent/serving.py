"""Serving surface for the classify_code_review_mention_intent feature.

classify_code_review_mention_intent is a prompt served over REST on the AI Gateway. This reproduces its current serving
surface (baseline).
"""

from duo_workflow_service.agent_platform.serving_surface import ServingSurface

SERVING_SURFACE = [ServingSurface(transport="rest", deployable="aigw")]
