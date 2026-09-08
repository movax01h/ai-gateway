---
name: agent-flows
description: >
  Shape and critique Duo agent flows — foundational flows in this repo (configs
  under duo_workflow_service/agent_platform), custom flows and agents for the AI
  Catalog. Load before proposing or creating a flow, before adding a component or
  node to an existing one, before growing a flow prompt, and when asked whether a
  flow could be simpler. Also load when someone asks for new Duo automation in a
  repo or project without mentioning flows ("can Duo do X automatically?") — an
  existing flow often covers it.
---

# Duo agent flows

Two kinds. A **custom flow** belongs to someone's own project: self-contained YAML with
local prompts, pasted into the AI Catalog, `environment: ambient`, and it lives wherever
suits that repo — never in this repo's registry layout. A **foundational flow** ships
with the platform and lives here. Ask which one is wanted if it is not obvious; the rest
of this file is about foundational flows, though the shape advice applies to both.

Configs: `duo_workflow_service/agent_platform/v1/flows/configs/<flow>/<version>.yml`.
The parallel `experimental/` tree is the unstable framework version — build in `v1`.
Reference: `docs/flow_registry/v1.md` for component types and examples, `index.md` for
the input/output system, `contribution_guidelines.md` for versioning and review.

## Reuse before you build

A new flow is a config, prompts, versions to maintain, and a pattern other people copy.
Before writing any YAML, list `configs/` and tell the human which existing flow is
closest and one of two things:

- it already covers this with a different goal — then stop and let them decide, or
- why it does not fit.

`developer` is a general-purpose coding agent: one node, the file and shell toolset, and
a goal. Most requests phrased as "make Duo do X in a repo" are that flow with a different
goal, so it is usually the flow to rule out first.

## Build the reusable one

That `developer` absorbs so many requests is not luck: its prompt describes a coding
agent, not a task, so the task arrives in the goal. Between two designs that are
otherwise equal, take the one that survives a different goal — a reviewer told to review
"a change" also reviews a branch, a commit or an MR, while one told to review "the
developer's working copy" reviews only that. Reuse cuts the other way too: a capability
several flows will want can be worth its own flow precisely so anything can trigger it.

## One agent node is the baseline

The default shape is a single `AgentComponent` with a good prompt and the right toolset.
More structure needs a named reason — any one of these is enough, and the human naming
one settles it:

- a context-compression boundary
- a genuinely different toolset per stage
- a human decision point (`HumanInputComponent`)
- a durable pause and resume
- a hard cost or determinism target

When someone asks for a split, ask what it buys before you build it. A reason has to be
a fact you can point at in the flow as it stands: the tool the second node needs and the
first does not, the transcript the second node must not see, the approval the flow has to
wait for. "Cleaner separation", "each node does one thing" and symmetry with an existing
multi-node flow are labels, not reasons, and a toolset difference you would create by
splitting does not count — the reason has to exist before the change. Do not pick one off
the list to justify a split that is already decided.

## Put agency where judgement is needed, nowhere else

Take the least agency that does the job. Each rung buys adaptability and pays in tokens,
latency and unpredictability. Parameters and syntax are in `docs/flow_registry/v1.md`
under Component Types; what matters here is when to reach for which.

- **`DeterministicStepComponent`** — the action must always happen and its arguments are
  known. No model call, cannot decline, cannot adapt. Brittle if the inputs vary.
- **`OneOffComponent`** — the tool calls need judgement but not a conversation: one round,
  then out. Cheaper and far more predictable than a loop, and it cannot chase what the
  tools returned. It needs a toolset.
- **`AgentComponent`** — the work depends on what it finds. You pay a model turn plus the
  prompt on every pass, and it can decide not to act. A step that only reasons and returns
  a structured answer is this rung with `toolset: []`: nothing to loop over, so that is not
  extra agency.
- **`subagents:` on an `AgentComponent`** — one stage needs its own persona, toolset or a
  clean context, but not its own place in the graph. The subagent reasons in a fresh
  conversation, which is the benefit and the cost: isolation, but the supervisor pays for
  any context it forgets to hand over. Delegation is decided at runtime rather than wired
  as an edge, so it flexes where a graph cannot, and it is correspondingly harder to route
  or feed into a deterministic step downstream. Reach for it when the isolation is the
  point, not to tidy a linear pipeline.

Prefer a general tool such as `run_command` over a bespoke tool that wraps one command.

When trimming a flow, removing agency usually beats removing stages.

## Prompts are principles, not procedures

Give the persona, what good looks like, and priorities. Not step-by-step scripts,
enumerated edge cases, or post-mortems of past bad runs. "Reading one file per call is
what made a run loop" is a war story; "read in bulk and reuse what you have fetched" is
the same rule, shorter, and it ages better.

The model already knows the trade. A prompt that reads `1. run git status 2. run git
diff` or `use run_command for git operations` is scaffolding around capability the agent
has; the toolset already told it which tools exist. A numbered `## Process` section is
the tell: if the prompt you just wrote has one, you have written a script.

This is not an argument for empty prompts. A line earns its place once you have watched
the flow fail without it. Ship lean, then add what real runs show you need.

### Before you add a rule to a prompt

"Always do X" in a prompt is a request, not a guarantee, and the next rule makes the ones
above it weaker. So when a flow misbehaves, or when someone asks for another rule, say
what the structural fix would be before you edit anything:

- an action that must always happen — a deterministic step
- something that must hold before the next node — a router condition
- the agent used a tool wrongly — the tool description or the toolset

Then add the instruction if none of those fits, or if the human still wants it.

## Ship the config and validate it

The deliverable is one `<version>.yml` plus its prompts, and no flow carries a README or
a usage page. Keep the prompt in the config unless more than one flow or version uses it;
only then is a file-based prompt in `ai_gateway/prompts/definitions/` worth the extra
files.

Validate before you claim it works:

```python
poetry run pytest tests/duo_workflow_service/agent_platform/test_configs.py
```

It loads every config, checks prompt inputs against the templates, and picks up a new
file with no registration step.

## Read the corpus selectively

Configs range from one component to well over a dozen. The long ones are the exception,
not the target. Two worth opening, latest version in each directory:

| Flow | Read it as |
|---|---|
| `developer/` | The baseline: one `AgentComponent`, one router to `end`, a prompt that extends a shared partial and describes a coding agent rather than a task. |
| `code_review/` | Earned structure. It runs on every merge request, so the target was a flow cheap enough to sell cheaply, accepting some quality for cost. Hence hard-wired tool calls, staged context, one-off steps instead of agent loops. A cost target that tight can justify a specialised flow; without one, this shape is over-built. |
