"""Serving surface for the update_description_from_decision feature.

update_description_from_decision is a prompt served over REST on the AI Gateway. GitLab Rails calls it when a Duo
decision on a work item is resolved, to fold the answer into the work item description.
"""

from duo_workflow_service.agent_platform.serving_surface import ServingSurface

SERVING_SURFACE = [ServingSurface(transport="rest", deployable="aigw")]
