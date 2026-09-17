# Duo Developer design principles

This document is for anyone changing the Duo Developer flow: its configs, its shared
system prompt, its toolset, or its versions. It records the design principles of the
[Agent Developer team](https://handbook.gitlab.com/handbook/engineering/ai/agent-foundations/agent-developer/), so that contributors can align with them without carrying the
team's history in their heads.

Where today's code has not fully caught up with a principle, the principle states the
direction — follow the principle, and treat the mismatch as debt, not as a pattern to
copy.

In short:

1. Duo Developer is a reusable building block, not a [flow](https://docs.gitlab.com/development/ai_features/glossary/#flow) that is optimized for a single specific task
1. The client owns the goal; the agent stays generic.
1. The system prompt states principles, not procedures.
1. Keep the stable prompt stable — dynamic context goes below the cache breakpoint.
1. Versions exist for compatibility, not for use cases.
1. Changes are based on evidence: traces first, no summary-only claims.
1. Bigger changes need a baseline-vs-variant eval run, judged on a trade-off network.
1. The stable prompt is model-agnostic, not optimized for one family.

For flow-building mechanics (component types, versioning rules), see the
[Flow Registry contribution guidelines](/docs/flow_registry/contribution_guidelines.md)
and the [agent-flows skill](/.agents/skills/agent-flows/SKILL.md). This document does
not repeat them; it states how they apply to Duo Developer specifically.

## 1. Duo Developer is a reusable building block

Duo Developer is one general-purpose coding agent, used as a shared platform primitive
by many surfaces: mentions and assignments in issues and merge requests, the Duo CLI,
Slack, and CI-driven runs. It is not a collection of task-specific flows.

The test for every change: does it keep the building block reusable? A new use case is
served by a new goal (section 2), not by a new flow, a new version, or a specialized
system prompt. Before adding anything, name the smallest change that keeps `developer`
generic — and rule out "the existing flow with a different goal" before building
something new.

## 2. The client owns the goal

The task — what triggered the run, which resource it concerns, what "done" looks like —
belongs to the client that starts the session. It arrives in the goal, built from the
goal templates in the GitLab repository
([`ee/app/models/ai/catalog/goal_templates/developer/`](https://gitlab.com/gitlab-org/gitlab/-/tree/master/ee/app/models/ai/catalog/goal_templates/developer)).
A mention in an MR comment, an assignment, and a Slack message are different goals into
the same agent.

Everything that changes from one client to the next travels in the goal. The flow config
and the system prompt stay client-agnostic. Some older code paths still encode client
specifics elsewhere; the direction is to move them into goals, not to add more.

## 3. The system prompt states principles, not procedures

The shared system prompt
([`ai_gateway/prompts/definitions/common/developer/`](/ai_gateway/prompts/definitions/common/developer/))
describes what the agent is, what good looks like, and what it must never do. It is not
a runbook: no step-by-step scripts, no enumerated edge cases, no instructions added in
reaction to one bad session.

Resist overfitting to individual failure cases. Every instruction is paid for on every
turn, weakens the instructions around it, and ages poorly. When the agent misbehaves,
first find the layer that owns the failure — the goal, the tool contract, the toolset,
the environment setup, missing runtime context — and fix it there. Add a prompt
instruction only when the failure is universal and no other layer owns it. The evidence
bar in section 6 applies.

Extend the prompt through its existing Jinja blocks instead of duplicating it, and keep
dynamic conditionals out of it (section 4).

### A mental model: the training distribution

Language models follow a behavioral distribution from their training. With no
instructions at all, the agent simply does whatever its default trained behavior is.
Two consequences fall out of that:

**Match the environment to the distribution first.** A coding model is trained on full
repository checkouts, familiar shell tools, and common project layouts. If the
environment is off-distribution — a shallow clone, an unfamiliar tool shape — the model
copes, but pays for it in extra LLM calls, latency, and worse behavior. Fixing the
environment beats instructing around it: this is why unshallowing the clone is a
deterministic step in the flow, not a rule in the prompt.

**Prompt instructions are nudges away from the default.** Use them when the default is
not what we want — for example, steering the agent toward a better but unfamiliar tool,
such as a knowledge graph with graph queries when it would default to the grep patterns
it knows. Each nudge has a cost, and too many nudges push the model into territory where
it performs poorly. So shape behavior at the level of principles and priorities — and
expect a nudge to need revisiting as models evolve.

## 4. Keep the stable prompt stable

The system prompt is the cached prefix of every session. Changing it invalidates the
prompt cache and makes sessions more expensive; changing it mid-session is worse.

Therefore: facts that are stable belong in the system prompt. Facts that change at run
or turn boundaries — execution environment, interaction mode, plan/build mode — belong
in the message stream or run context, below the cache breakpoint.

Interactive and ambient runs should render the same system prompt. When a session moves
between runtimes — for example from a local interactive session to a remote executor —
announce the environment switch as a message in the history, not by swapping the system
prompt.

## 5. Versions exist for compatibility, not for use cases

Flow versions exist primarily for backwards compatibility: clients pin a version
constraint (the GitLab AI Catalog pins `^3.0.0` today), and a version must keep working
for the instances that still pin it. `1.0.0`, for example, still carries a `git_setup`
step for GitLab instances that predate runner-level Git setup.

Compatibility is not the only legitimate reason. Minor versions carry incremental
updates, and a temporary version behind a feature flag is a good way to test a larger
change safely. What versions are not for: use cases. A new use case arrives as a new
goal (section 2), never as a fork of the flow.

Old versions stay until the clients pinning them are gone.

## 6. Changes are based on evidence

Every change to the flow, the prompt, or the toolset starts from observed behavior, not
from an idea about what might go wrong.

The minimum evidence is a link to a real session — a LangSmith trace — pointing at the
exact step where things went wrong: the wrong tool call, the bad model output, quoted or
shown directly. A summary of a session is a lead, not evidence. This holds double for
AI-generated session analyses, which have repeatedly pointed at the wrong root cause.
Verify every claim against the raw trace before proposing a fix.

Common misreads to check against the trace:

- Assuming the agent was never told something the goal or the prompt already says.
- Assuming which tool produced an action, when the work happened through a general tool
  such as `run_command`.
- Judging the agent's reasoning from UI-visible output alone, when the trace shows the
  agent already recognized and reported the problem — in which case the fix belongs to
  the environment or the UI, not the prompt.

Then fix the narrowest owning layer with the smallest change (section 3).

Not every change fixes an observed failure. A new feature or a genuine improvement idea
is a valid change — treat it as a hypothesis: state the reasoning, discuss the approach
with the team before building (is this change the right solution, or does the problem
belong to another layer?), and say up front how we will know whether it worked.

## 7. Bigger changes need an eval run

Small, tightly-scoped changes can ship on trace evidence alone when the regression risk
is clearly low. Bigger changes — system-prompt rewrites, toolset changes, new
structure — need a baseline-versus-variant evaluation before they merge:

- **Dataset**: SWE-bench (Verified sample; Pro for the harder set). Same dataset for
  baseline and variant.
- **Report the standard table**, with a LangSmith link to the traces: runs
  total/success, MRs created, **resolution rate** (MR created, the change resolves the
  issue, tests pass), **median agent latency** (flow start to finish, excluding executor
  spin-up), **median tokens per task**, and **median/total LLM calls**. Ideally, exclude
  timed-out runs from latency and token statistics to make the runs more comparable.
- **Note the caveats** of each run (setup differences, known bugs) next to the numbers.

Judge the evaluation result on all metrics together, not just on problem resolved rate. Tokens, latency, and LLM
call count all carry real cost — LLM calls currently feed directly into customer
billing. A resolution win that doubles token use is not a free win; say so in the MR and
let reviewers weigh it.

To run an evaluation, use the CEF service — see the
[quick start guide](https://gitlab.com/gitlab-org/modelops/ai-model-validation-and-research/ai-evaluation/prompt-library/-/blob/main/doc/server/quick_start.md)
and the
[Duo Developer scenario](https://gitlab.com/gitlab-org/modelops/ai-model-validation-and-research/ai-evaluation/prompt-library/-/blob/main/doc/server/eval_scenarios/duo_developer.md)
in the prompt-library repository. Note that the `sanity-tests` CI job
([docs](/docs/tests.md)) is a one-instance smoke test, not a regression gate — it does
not replace an eval run.

## 8. Keeping stable prompt model agnostic

The system prompt is written to work well across all supported model families, not
optimized for any one of them. This is a deliberate trade-off: a model-specific prompt
may perform better on that model, but it multiplies maintenance cost: every prompt
change must be validated against each family, and diverging prompts drift apart over
time.

If a significant and sustained performance gap is observed across model families,
this trade-off may be revisited, with maintenance cost weighed explicitly against
the gain.

## Ownership and questions

The Agent Developer team co-owns the Duo Developer surface (see `CODEOWNERS`): the flow
configs in this directory, `ai_gateway/prompts/definitions/common/developer/`, and
`ai_gateway/prompts/definitions/developer_*/`. Expect a review from the team on changes
to these paths.

Questions: see the
[team page](https://handbook.gitlab.com/handbook/engineering/ai/agent-foundations/agent-developer/)
for the current communication channels.
