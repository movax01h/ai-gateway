# Flow Creator: Maintenance guide

This page is the maintenance guide for the Flow Creator agent (config identifier:
`flow_creator`), the chat agent that helps developers author Flow Registry v1 YAML.

> **Config identifier**: the flow config directory and benchmark suite use the identifier
> `flow_creator`. Older issues and merge requests may refer to this agent as
> "Flow Registry Assistant", "Flow Registry Chat Helper", or `flow_registry_chat_helper`; all
> names refer to the same agent.

[[_TOC_]]

## Related resources

- Flow config: `duo_workflow_service/agent_platform/v1/flows/configs/flow_creator/`
  (added in [!6473](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/6473))
- Benchmark suite: [`agent_tests/flow_creator/`](../../../../../../agent_tests/flow_creator/README.md)
  (added in [!6474](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/6474))
- GitLab chat entry:
  [`FoundationalChatAgentsDefinitions.rb`](https://gitlab.com/gitlab-org/gitlab/blob/master/ee/lib/ai/foundational_chat_agents_definitions.rb)
  (added in [GitLab!248128](https://gitlab.com/gitlab-org/gitlab/-/merge_requests/248128))
- Generic update process for foundational agents:
  [Update a foundational agent](https://docs.gitlab.com/development/ai_features/foundational_chat_agents/)
  in the GitLab development documentation
- Tracking issue for this process:
  [GitLab#604712](https://gitlab.com/gitlab-org/gitlab/-/work_items/604712)

## Ownership

`group::ai catalog` (surfacing in chat) and `group::agent developer` (framework knowledge) are
deciding ownership of the agent prompt in
[GitLab#604712](https://gitlab.com/gitlab-org/gitlab/-/work_items/604712). Until the decision is
recorded here and in `.gitlab/CODEOWNERS`, request review on prompt changes from both groups.

## Maintenance triggers

The agent's prompt instructs it to read the live framework documentation before every response,
but its hard rules and pre-output checklist encode failure patterns that must be updated
manually. Update the agent when:

- **A new component type or parameter ships.** Update the framework reference (for example,
  [`v1.md`](../../../../../../docs/flow_registry/v1.md)) in the same merge request as the framework change. See
  [Keeping framework documentation in sync](../../../../../../docs/flow_registry/contribution_guidelines.md#keeping-framework-documentation-in-sync).
- **A new failure pattern is discovered through a real session.** Add a hard rule using the
  process below. The `sends_response_to` constraint (must point to an already-run component) is
  an example of a rule discovered through production failures.
- **A context envelope changes.** Breaking changes to envelopes the agent documents (for
  example, the `agent_platform_standard_context` `service_account_name` change in June 2026)
  require a prompt update and a flow version bump per the
  [envelope versioning policy](../../../../../../docs/flow_registry/contribution_guidelines.md#envelope-versioning).

## Translating session failures into hard rules

Target: a failure pattern discovered in a session becomes a hard rule on a timeline set by its
severity, following the GitLab
[bug triage severity](https://handbook.gitlab.com/handbook/product-development/how-we-work/issue-triage/#severity)
and [severity SLO targets](https://handbook.gitlab.com/handbook/product-development/how-we-work/issue-triage/#severity-slos).
The Flow Creator was released directly to generally available, so failures are triaged at full
severity: a failure that recurs every session (a broken feature with no workaround) is roughly a
~"severity::1" to be fixed within the same milestone, with longer SLO targets for lower
severities. Track the timeline in iterations and milestones/releases rather than sprints, so the
language matches the SLO targets.

1. **Capture.** Save the broken session transcript or generated YAML, and identify the single
   rule the output violated.
1. **Add the hard rule.** Update the system prompt in a new flow version. Follow the
   [prompt authoring constraints](../../../../../../docs/flow_registry/contribution_guidelines.md#prompt-authoring-constraints) and
   the [version guidelines](../../../../../../docs/flow_registry/contribution_guidelines.md#version-guidelines).
1. **Extend the benchmark.** Add the corresponding case (in `test_smoke.py` or
   `test_debugging.py`) and rule check (in `test_hard_rules.py`) to the benchmark suite in
   `agent_tests/flow_creator/`, so future prompt versions are scored against the new rule. Prefer
   a deterministic check on the parsed YAML; fall back to the LLM judge only where the rule cannot
   be checked deterministically. See the [suite README](../../../../../../agent_tests/flow_creator/README.md)
   for the case and rule structure.
1. **Score.** Run the suite with `make test-agents AGENT_TEST_DIR=flow_creator/` (requires
   `ANTHROPIC_API_KEY`; calls a real model and costs tokens). The run writes a per-file pass
   rate to `.test-reports/agent_tests/flow_creator-summary.md` (retries count once; skips are
   excluded from the denominator). In CI, the `tests:agents:flow-creator` job runs the same
   command and attaches `.test-reports/agent_tests/` as job artifacts, which expire after one
   week.
1. **Record the result.** Nothing stores benchmark results automatically: the summary file is
   the only output, and CI artifacts expire. Copy the new pass rate into the merge request
   description alongside the previous version's rate. The "previous baseline" is the pass rate
   recorded in the merge request that last changed the prompt; if it was never recorded,
   re-run the suite against the prior flow version to regenerate it. The pass rate is not
   deterministic: because the suite runs against a live model, the same prompt can score
   differently between runs (a spread of roughly 85-93% has been observed). Treat a single-point
   difference between two versions as a signal, not proof, and re-run if a change looks marginal.
   See [the recorded baseline](../../../../../../agent_tests/flow_creator/README.md#the-recorded-baseline)
   in the suite README for the sampled range and which rules fail most often.
1. **Validate.** Exercise the change in at least one full chat session before merging. Use the
   [evaluation prompts](https://gitlab.com/gitlab-org/gitlab/-/work_items/604710#example-prompts)
   (six end-to-end flow requests, from single-agent through supervisor) as the session scripts.

## Review requirements for prompt changes

Every change to the agent prompt must include, in the merge request:

- The new version's benchmark pass rate (from
  `.test-reports/agent_tests/flow_creator-summary.md`) alongside the previous version's, so the
  two are directly comparable. Note that the rate varies between runs on an unchanged prompt, so
  small differences may reflect model non-determinism rather than the prompt change; compare
  [which rules fail](../../../../../../agent_tests/flow_creator/README.md#the-recorded-baseline),
  not only the headline number.
- Evidence of at least one full session validation.
- A flow version bump consistent with the
  [version guidelines](../../../../../../docs/flow_registry/contribution_guidelines.md#version-guidelines).

Release, `flow_version` pinning in GitLab, user-facing documentation, and communication follow
the generic
[update process for foundational agents](https://docs.gitlab.com/development/ai_features/foundational_chat_agents/)
in the GitLab development documentation.
