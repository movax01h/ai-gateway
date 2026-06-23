"""Parsing of runtime LangGraph node names back to design-time components.

The inverse of the ``{component}{NODE_ROLE_SEPARATOR}{role}`` naming convention
that v1 components compile to (see ``NODE_ROLE_SEPARATOR``).
"""

from typing import Optional

from duo_workflow_service.agent_platform.constants import NODE_ROLE_SEPARATOR


def component_name_from_node(node_name: Optional[str]) -> Optional[str]:
    """Recover the design-time component name from a runtime LangGraph node name.

    v1 components compile to nodes named ``{component}{NODE_ROLE_SEPARATOR}{role}`` (e.g. ``"researcher#agent"``); the
    segment before the separator is the component name. Names without it (e.g. legacy workflow nodes) are returned
    unchanged.
    """
    if not node_name:
        return None
    return node_name.partition(NODE_ROLE_SEPARATOR)[0]
