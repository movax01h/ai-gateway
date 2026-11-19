# Separates a component's name from a node's role in a compiled graph node name,
# e.g. ``"researcher#agent"``. Component names never contain it, so the segment
# before it recovers the design-time component name.
NODE_ROLE_SEPARATOR = "#"

# Number of steps LangGraph is allowed to execute in a single workflow run before raising
# a GraphRecursionError. Raised from 300 to 600 in
# gitlab-org/modelops/applied-ml/code-suggestions/ai-assist#2590: 300 react
# iterations was proving too restrictive for long-running/interactive flows
# (e.g. developer). Flow Registry v1 flows derive their own limit from the graph that
# ``FlowGraphBuilder`` builds (see ``FlowGraphBuilder.derived_recursion_limit``). This
# value stays for every workflow that does not, for a ``Flow`` before it compiles, and as
# the lowest limit for a v1 flow whose routers loop without a ``HumanInputComponent``.
RECURSION_LIMIT = 600

# Delegations that the derived limit assumes for a supervisor that sets no ``max_delegations``.
# This is an estimate for the limit only. Nothing stops the supervisor at this count, so one that
# delegates more can reach the hard limit. Set ``max_delegations`` to replace the estimate. A
# supervisor's own ``max_cycles`` caps the count too, because it delegates at most once per cycle.
DEFAULT_ASSUMED_DELEGATIONS = 10

# Steps added to the derived limit: 1 for the flow end or abort node, which no component declares,
# and 1 because LangGraph raises GraphRecursionError when a run needs as many steps as the limit.
END_NODE_STEPS = 2

# The environment whose flows run one root ``AgentComponent`` as a chat
# assistant. The registry forks on it to pick the chat executor, and
# ``AgentComponent`` fills the chat-surface defaults under it.
CHAT_PARTIAL_ENVIRONMENT = "chat-partial"

INTERACTIVE_ENVIRONMENTS = frozenset(("chat", CHAT_PARTIAL_ENVIRONMENT))

# Registry flows that expose only the metadata of their runs: which steps and tools
# ran, their status and timing. The live stream, audit events and tracing carry no
# model or user text for them. Rails restricts its session read APIs for the same
# flows through its own setting.
METADATA_ONLY_FLOWS: frozenset[str] = frozenset({"bl_security"})

# Registry flows that save each step's checkpoint before the next step starts, so a
# checkpoint GitLab refuses stops the run at the next step. Other flows keep
# LangGraph's default asynchronous saves, where a refused save surfaces only when
# the graph exits.
SYNC_CHECKPOINT_FLOWS: frozenset[str] = frozenset({"bl_security"})
