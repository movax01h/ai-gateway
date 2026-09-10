# Separates a component's name from a node's role in a compiled graph node name,
# e.g. ``"researcher#agent"``. Component names never contain it, so the segment
# before it recovers the design-time component name.
NODE_ROLE_SEPARATOR = "#"

# Maximum number of steps LangGraph is allowed to execute in a single workflow run
# before raising a GraphRecursionError. Raised from 300 to 600 in
# gitlab-org/modelops/applied-ml/code-suggestions/ai-assist#2590: 300 react
# iterations was proving too restrictive for long-running/interactive flows
# (e.g. developer). See AgentComponentBase.max_cycles for the separate, lower
# soft per-component limit this is not automatically tied to.
RECURSION_LIMIT = 600

# Graph node name of v1's built-in AbortComponent. Components that need to route a
# failure straight to the abort node reference this rather than the literal string,
# so renaming the entry hook cannot silently break their routing. Lives here rather
# than in v1/components/base.py so experimental/ can adopt the same name and keep
# parity with v1; experimental has no abort component of its own yet.
ABORT_FLOW_ENTRY_HOOK = "abort_flow"
