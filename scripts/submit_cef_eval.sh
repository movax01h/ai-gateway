#!/usr/bin/env sh
#
# Submits a SWE Bench evaluation to the CEF service, polls the experiment
# status, and fails on error.
#
# CEF pulls its own (latest) Docker image server-side, so no CEF version is
# pinned here.
#
# Usage:
#   submit_cef_eval.sh <aigw_commit> <aigw_project_path> <notes> [gl_commit]
#
# Positional arguments:
#   aigw_commit        AI Gateway commit SHA to evaluate.
#   aigw_project_path  AI Gateway project path (e.g. "group/project").
#   notes              Free-form notes attached to the experiment.
#   gl_commit          Optional GitLab commit/ref to evaluate against. Omit to
#                      let CEF pick its own pre-tested GitLab commit.
#
# Required environment variables:
#   CEF_SERVICE_URL          Base URL of the CEF service.
#   CEF_SERVICE_ACCOUNT_PAT  GitLab PAT (see docs/tests.md for rotation steps).
#   CEF_POLL_INTERVAL        Seconds between status polls.
#   CEF_POLL_TIMEOUT         Overall timeout (seconds) before giving up.
#
# Optional environment variables:
#   CEF_STATE_FILE           Path to write the raw terminal state JSON to.
#                            Defaults to cef_eval_state.json in the working
#                            directory.
#
# Depends on: curl, jq, coreutils (GNU date). On Alpine these are installed
# via `apk add --no-cache curl jq coreutils` in the job's before_script.

set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: $0 <aigw_commit> <aigw_project_path> <notes> [gl_commit]" >&2
  exit 2
fi

AIGW_COMMIT=$1
AIGW_PROJECT_PATH=$2
NOTES=$3
GL_COMMIT=${4:-}
STATE_FILE=${CEF_STATE_FILE:-cef_eval_state.json}

PAYLOAD=$(jq -n \
  --arg gl_commit "$GL_COMMIT" \
  --arg aigw_commit "$AIGW_COMMIT" \
  --arg aigw_project_path "$AIGW_PROJECT_PATH" \
  --arg notes "$NOTES" \
  '{
    aigw_commit: $aigw_commit,
    aigw_project_path: $aigw_project_path,
    model_selection: {
      feature_setting: "duo_agent_platform",
      default_models: ["claude_haiku_4_5_20251001_vertex"]
    },
    evaluate_config: {
      langsmith: {
        "dataset": "swe.swebench-verified.validation-stratified-b06f4db4-p30",
        "split": "base",
        "limit": 1
      },
      flow: { flow_config_id: "duo_developer" },
      inference: { max_concurrency: 1 },
      assessment: {
        evaluators: [ {name: "mr_created"}, {name: "issue_to_mr_resolved"} ],
        max_concurrency: 1
      },
      timeout: 1200
    },
    notes: $notes
  }
  | if $gl_commit == "" then . else .gl_commit = $gl_commit end')

if ! RESPONSE=$(curl --fail-with-body -sS -X POST "$CEF_SERVICE_URL/v1/experiments/register" \
  -H "Authorization: Bearer $CEF_SERVICE_ACCOUNT_PAT" \
  -H "Content-Type: application/json" \
  --data-binary "$PAYLOAD"); then
  echo "ERROR: CEF registration request failed. Response: $RESPONSE" >&2
  exit 1
fi
REQUEST_ID=$(printf '%s' "$RESPONSE" | jq -r '.request_id')
if [ -z "$REQUEST_ID" ] || [ "$REQUEST_ID" = "null" ]; then
  echo "ERROR: CEF response did not contain a valid request_id. Response: $RESPONSE" >&2
  exit 1
fi
echo "CEF experiment registered. request_id=${REQUEST_ID}"

# Drops the credential-adjacent fields the state endpoint returns: this project
# is public, and so are its job logs and artifacts. Non-JSON input (a proxy
# error page, say) is passed through unchanged so diagnostics aren't swallowed.
redact() {
  if REDACTED=$(printf '%s' "$1" | jq 'del(.callback_token_hash, .env_secret_name)' 2>/dev/null); then
    printf '%s' "$REDACTED"
  else
    printf '%s' "$1"
  fi
}

# Prints a terminal experiment state as labelled lines instead of raw JSON, so
# multi-line failure reasons are readable in the job log, and saves the raw
# JSON to STATE_FILE for triage (kept as a CI artifact).
print_state() {
  printf '%s' "$1" | jq . > "$STATE_FILE"
  echo "Raw state written to $STATE_FILE"

  # printf, not echo: some shells interpret the backslash escapes in
  # error_message and hand jq unparsable JSON.
  printf '%s' "$1" | jq -r --arg rid "$REQUEST_ID" '
    "request_id:  \($rid)",
    "status:      \(.status)",
    (if .langsmith_experiment_url then "langsmith:   \(.langsmith_experiment_url)" else empty end),
    (if .traces_url then "traces:      \(.traces_url)" else empty end),
    (if .instance_name then "instance:    \(.instance_name) (\(.instance_zone // "unknown zone"))" else empty end),
    "modified_at: \(.modified_at)"
  '

  REASON=$(printf '%s' "$1" | jq -r '.error_message // empty')
  if [ -n "$REASON" ]; then
    echo "failure reason:"
    printf '%s\n' "$REASON" | sed 's/^/  /'
  elif [ "$(printf '%s' "$1" | jq -r '.status')" = "fail" ]; then
    echo "failure reason: none reported"
  fi
}

deadline=$(( $(date +%s) + CEF_POLL_TIMEOUT ))
while true; do
  if ! STATE=$(curl --fail-with-body -sS "$CEF_SERVICE_URL/v1/experiments/state?request_id=$REQUEST_ID" \
    -H "Authorization: Bearer $CEF_SERVICE_ACCOUNT_PAT"); then
    echo "ERROR: CEF status request failed. Response: $(redact "$STATE")" >&2
    exit 1
  fi
  STATE=$(redact "$STATE")
  STATUS=$(printf '%s' "$STATE" | jq -r '.status')
  echo "$(date -u +%H:%M:%S) status: $STATUS"

  case "$STATUS" in
    complete)
      print_state "$STATE"
      echo "CEF experiment completed successfully."
      break
      ;;
    fail)
      print_state "$STATE"
      echo "ERROR: CEF experiment failed."
      exit 1
      ;;
  esac

  if [ "$(date +%s)" -ge "$deadline" ]; then
    print_state "$STATE"
    echo "ERROR: Timed out after ${CEF_POLL_TIMEOUT}s waiting for CEF experiment to reach a terminal state (last status: $STATUS)."
    exit 1
  fi
  sleep "$CEF_POLL_INTERVAL"
done
