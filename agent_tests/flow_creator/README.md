# Flow Creator: benchmark baseline

These tests are the **benchmark baseline** for the Flow Creation foundational agent
([`flow_creator`](../../duo_workflow_service/agent_platform/v1/flows/configs/flow_creator/1.0.0.yml)),
the single-agent flow that helps developers author Flow Registry v1 YAML.

They exist to answer two questions:

1. Does the agent work? Does it produce complete, valid, runnable flow YAML?
1. Is a change to the agent an improvement? Any future variant - a multi-agent
   version, a reworked prompt, a different model - **must be scored against this
   same suite** so the numbers are comparable.

Parent epic: [Flow Creation foundational agent](https://gitlab.com/groups/gitlab-org/-/work_items/22644).

## What is scored

| File | What it checks | Source |
|---|---|---|
| `test_smoke.py` | Each of six representative flow requests yields exactly one complete YAML document that parses, has every required section, and elides nothing. A second test parses that YAML with the framework's own `FlowConfig` schema. | [#604710](https://gitlab.com/gitlab-org/gitlab/-/work_items/604710) |
| `test_hard_rules.py` | The seven hard rules, scored per case: `project_id` threading, `flow.inputs` for branch context, HITL gate wiring, stopping instructions, `unit_primitives`, `placeholder: history`, and alias-to-placeholder matching. | [#604709](https://gitlab.com/gitlab-org/gitlab/-/work_items/604709) |
| `test_debugging.py` | Given a flow that is broken in exactly one way, the agent names the violated rule by number. | [#604709](https://gitlab.com/gitlab-org/gitlab/-/work_items/604709) |

The six cases live in `cases.py` with their prompts copied verbatim from the
#604710 matrix. Every rule check lives in `helpers.py` as a `check_rule_N`
function that returns a list of violations, so the checks can be exercised
against real flow configs without an LLM.

Because this suite only runs as a manual CI job, the checks themselves are unit
tested in
[`tests/agent_tests/flow_creator/test_helpers.py`](../../tests/agent_tests/flow_creator/test_helpers.py),
which runs on every pipeline and needs no API key. Change a `check_rule_N`
function and update those tests with it, otherwise a scoring regression will pass
unnoticed and every future benchmark number will shift.

Only Rule 4 (an explicit stopping instruction) uses the LLM judge - whether a
closing sentence is a genuine termination condition is a judgement about natural
language. Everything else is a deterministic check on the parsed YAML.

## Running the suite

The suite calls a real Anthropic model, so it is a manual job in CI and needs a
key locally:

```shell
export ANTHROPIC_API_KEY=<your-key>
AGENT_TEST_DIR=flow_creator/ make test-agents
```

To score a different execution model:

```shell
AGENT_TEST_DIR=flow_creator/ EXECUTION_MODEL=claude-opus-4-7 make test-agents
```

Each case is generated once per run and cached, so the smoke tests and the
hard-rule tests score the same YAML. Generated YAML and full conversation
transcripts - including every tool call, its arguments, and its result - are
written to
`.test-reports/agent_tests/flow_creator/<execution-model>/`, which
is where to look first when a rule fails. Pass `--refresh-cache` to regenerate
instead of reusing the cache; it bypasses the on-disk cache but still generates
each case only once per run.

## Reading the score

The pass rate is printed at the end of the run and written to
`.test-reports/agent_tests/flow_creator-summary.md` (and `.json`)
along with the models used and a timestamp:

- Rates are reported **per test file**, so a regression in rule compliance is
  not hidden by passing smoke tests, and **per case**, because which cases
  regressed is more stable across runs than the headline number. Compare the
  per-case table first when scoring a variant.
- **Skipped tests are excluded from the denominator.** A rule that does not apply
  to a case (Rule 3 where the flow has no approval gate) is skipped, not counted
  as a pass. Where the matrix says a case must exercise a construct, its absence
  is a failure rather than a skip.
- Retries from `--reruns` are ignored; each test contributes its final outcome
  once.

## The recorded baseline

Three runs at commit `1af48b46` over the 61-test suite, `claude-sonnet-4-6`
executing and `claude-haiku-4-5` judging, each regenerating every case with
`--refresh-cache`:

| Run | Passed | Failed | Skipped | Denominator | Pass rate |
|---|---|---|---|---|---|
| 1 | 41 | 7 | 13 | 48 | 85% |
| 2 | 39 | 8 | 14 | 47 | 83% |
| 3 | 40 | 8 | 13 | 48 | 83% |

**The baseline is 83-85%, with 7 to 8 failures.** Quote it as a range. A single
run is one sample, and a variant that scores 85% against a baseline recorded as
"83%" has not necessarily improved - it may not have moved at all.

Raw counts are given alongside the percentage because the denominator moves
between runs (47 or 48 here). Skips are excluded from it, and how many tests skip
depends on what the agent generated: a flow with no approval gate skips Rule 3.
So two runs can report different percentages *of different totals*.

### By case

Pooled over the three runs. Which cases regress is more stable than the headline
number, so compare this table first when scoring a variant:

| Case | Pass rate |
|---|---|
| `two_agent_pipeline` | 54% (13/24) |
| `gitlab_api_tools` | 81% (17/21) |
| `branch_creation` | 85% (22/26) |
| `supervisor_two_subagents` | 92% (22/24) |
| `single_agent_summarizer` | 95% (20/21) |
| `hitl_approval_gate` | 96% (23/24) |

`two_agent_pipeline` is the weakest case by a wide margin and the least stable:
it scored 3/8, 7/8 and 3/8 across the three runs.

### By rule

Where the failures fall is more stable than how many there are. Counting every
failure across all three runs:

- **Rule 1 (`project_id` threading) failed 8 times, in all three runs** and
  across five different cases. This is the agent's most reliable weakness and
  the first thing to fix.
- **Rule 6 (`placeholder: history`) failed 5 times, in all three runs**, across
  four cases.
- The smoke test on complete YAML failed 3 times: the agent still occasionally
  elides part of the config.
- Rules 2, 4 and 7 failed twice each, and Rule 5 once.

The structural check on `expects_multiple_agents` and `expects_supervisor`
passed in all three runs. It scores two cases and skips the other four, so for
now it acts as a regression guard rather than a source of failures.

When comparing a variant, compare *which rules* it fails as well as the headline
number. A variant that fixes Rule 1 is an improvement even if its overall rate
lands inside this range.

The spread is sampling noise. The suite sets no `temperature`, so it generates at
the same provider default the agent uses in production, and the same request
yields different YAML on each run. Tightening the estimate means more runs per
case, which costs a multi-turn conversation each time.

## Deliberate decisions worth knowing before reading a failure

**Rule 7 requires Jinja2 `{{alias}}` and nothing else.** The framework renders
prompt templates with Jinja2 ([`ai_gateway/prompts/base.py`](../../ai_gateway/prompts/base.py)),
so `<<alias>>` is inert literal text: it substitutes nothing and raises no error.
The agent's system prompt teaches the correct `{{alias}}` form, escaped with
`{% raw %}` so that documenting a placeholder does not substitute it when the
prompt itself is loaded.

**The agent asks before designing.** Rule 8 of its prompt tells it to ask how the
flow will be triggered, so the first turn is usually clarifying questions. Each
case therefore replays up to two fixed follow-ups (see `FOLLOW_UPS` in
`cases.py`) that supply a trigger and a project ID, and stops as soon as YAML
appears. The number of turns a case needed is recorded in its transcript.

**Rule 6 is a convention, not a framework requirement.** The framework adds the
`history` placeholder automatically
([`lib/prompts/utilities.py`](../../lib/prompts/utilities.py)), and most existing
flow configs in this repository omit it. Rule 6 exists because the agent's prompt
mandates it explicitly, and the suite scores the agent against its own rules.

**The agent gets documentation tools, not its full toolset.** It is given mocked
read-only `read_file`, `get_repository_file`, and `gitlab_documentation_search`
tools that serve this repository's real `docs/flow_registry/` files. Its prompt
names a doc in `gitlab-org/gitlab` as the primary source, which is not available
here, so the baseline measures the prompt plus the secondary documentation rather
than the full production research loop. The toolset the flow config declares is
deliberately not used to build the agent under test: those tools reach live
GitLab and filesystem resources unavailable in CI, and none of them change
whether the emitted YAML is correct.

**Output tokens follow the flow config.** The shared `real_llm` fixture caps
output at 4096 tokens, which truncates a multi-component flow mid-YAML. This
suite uses the `max_tokens` its own config declares, so a failure is about the
agent's design choices rather than a truncation artifact. A response cut off by
the token limit is reported as such.
