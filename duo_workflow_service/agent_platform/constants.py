# Separates a component's name from a node's role in a compiled graph node name,
# e.g. ``"researcher#agent"``. Component names never contain it, so the segment
# before it recovers the design-time component name.
NODE_ROLE_SEPARATOR = "#"

# Maximum number of steps LangGraph is allowed to execute in a single workflow run
# before raising a GraphRecursionError.
RECURSION_LIMIT = 300
