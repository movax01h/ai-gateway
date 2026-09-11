---
name: dws-session-logs
description: Pull Duo Workflow Service (DWS) logs from GCP via gcloud to debug a failed or stuck Duo Agent Platform / Duo Workflow session — by workflow_id, a CI flow-job log, or any Duo Workflow error.
license: MIT
compatibility: opencode duo
metadata:
  slash-command: enabled
---
## Prerequisites

1. `gcloud` binary — if not installed let the user know `gcloud` is required for effective debugging
2. Run `gcloud projects list | grep gitlab-runway-` and confirm the output contains the expected project name and project ID.
3. If not logged in yet, prompt the user to interactively log in using `gcloud auth login` before proceeding.

If no environment was specified ask the user:

> Which environment would you like to debug in?
> 1. Production (gitlab-runway-production)
> 2. Staging (gitlab-runway-staging)

If no specific goal was given prompt the user:

> Which session IDs should I summarize? (workflow_id)

Afterwards prompt the user for a goal or proceed with their request:

> [if not provided] What details do you want for these sessions?
> Default: Pull the Duo Workflow Service logs for the above session IDs (`workflow_id`) and summarise what went wrong — any errors, plus general session statistics.

## Where the logs live

- GCP project: `gitlab-runway-production` (staging: `gitlab-runway-staging`)
- Resource: Cloud Run service `duo-workflow-svc`
- Filter on `jsonPayload.workflow_id` (it is a **string**) plus `resource.labels.service_name="duo-workflow-svc"`
- Retention is ~30 days; older sessions return nothing.

## Query

Always bound by time (unbounded queries time out). Take the timestamps from the CI job log or the issue and add a few minutes on each side. If neither exists (e.g. an IDE-run session), anchor on the session start — `GET /api/v4/ai/duo_workflows/workflows/<ID>` or the first checkpoint timestamp — and widen the window by ~15 minutes on each side. Timestamps in the filter are RFC 3339 / ISO 8601 UTC (e.g. `2026-09-07T13:30:00Z`), not epoch seconds. Write the output to a file, then inspect it with a script rather than reading raw JSON.

If your shell does not enforce a command timeout natively, wrap the command in `timeout` and pick a duration that fits the window size.

```sh
gcloud logging read \
  'resource.labels.service_name="duo-workflow-svc" AND jsonPayload.workflow_id="<ID>" AND timestamp>="<START>" AND timestamp<="<END>"' \
  --project <PROJECT> --limit 3000 --format json --order asc > /tmp/dws-<ID>.json
```

If the query returns exactly `--limit` (3000) entries it was truncated — narrow the time window and re-run.

Then:

1. Count entries by `jsonPayload.level`; print all `error`/`warning` entries first.
2. For errors, print `jsonPayload.exception` in full (Python traceback) — this is usually the root cause. Note: warning entries often carry their text in `jsonPayload.event` with `message` unset — print `event` as a fallback.
3. Print the flow timeline: events matching `startRequest`, `Flow route decision`, `step:`, `Request to LLM`, `Request to LLM complete`, `Finished ExecuteWorkflow RPC`.
4. Note `duration_s`, `servicer_context_code`, `servicer_context_details` (the gRPC status detail, often the error/abort message) and `workflow_definition` from the `Finished ExecuteWorkflow RPC` entry.
5. Note the `jsonPayload.correlation_id` — use it to trace the same request in Rails/Workhorse logs (Kibana) when the failure is not in DWS.

## Gotchas

- **IDE / locally-executed sessions** (`environment: ide`): the executor runs client-side, so DWS logs may only cover a short bootstrap window (start request, plan context) with no `Finished ExecuteWorkflow RPC` — even for a session that later shows `failed`. In that case the failure is client-side: get the CLI/LSP logs from the user.
- **Self-hosted DWS customers**: if entries only show `TrackSelfHostedExecuteWorkflow` / "Received self-hosted client event" with `workflow_id: "undefined"` and `gitlab_realm: self-managed`, the customer runs their own DWS. GitLab's hosted DWS has no execution logs for them — ask the customer for their DWS and AI Gateway logs for that `workflow_id`. Filter by `jsonPayload.gitlab_host_name="<their host>"` to confirm.
- `workflowStatus` in the CLI's `Received new checkpoint` lines is LangGraph state, not the Rails DB status. `CREATED` for a whole flow-registry v1 run is normal; only `FAILED`/`FINISHED` are meaningful.
- If the gcloud query returns `[]`, check: time window, project, retention, and whether the ID is quoted as a string.

## Output

Report per session: timeline, the first error with its traceback, the likely root cause, and whether the failure is client-side (CLI/job log), Rails, or DWS. State clearly when logs were unavailable and why.
