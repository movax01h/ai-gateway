#!/usr/bin/env bash

if [ -n "$WEB_CONCURRENCY" ] && [ "$WEB_CONCURRENCY" -gt 1 ]; then
  METRICS_DIR=$(mktemp -d -t ai_gateway.XXXXXX)
  echo "Storing multiprocess metrics in $METRICS_DIR..."
  export PROMETHEUS_MULTIPROC_DIR=$METRICS_DIR
fi

# Run both AI Gateway and Duo Workflow Service by default.
# Optionally, you can disable a service by setting `DISABLE_AI_GATEWAY` or `DISABLE_DUO_WORKFLOW_SERVICE` variable.
commands=()

if [[ -z "${DISABLE_AI_GATEWAY}" ]]; then
  commands+=("poetry run ai_gateway")
fi

if [[ -z "${DISABLE_DUO_WORKFLOW_SERVICE}" ]]; then
  commands+=("poetry run duo-workflow-service")
fi

if [ ${#commands[@]} -gt 0 ]; then
  printf "%s\n" "${commands[@]}" | parallel --will-cite --line-buffer --halt never
fi
