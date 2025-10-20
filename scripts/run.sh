#!/usr/bin/env bash

if [ -n "$WEB_CONCURRENCY" ] && [ "$WEB_CONCURRENCY" -gt 1 ]; then
  METRICS_DIR=$(mktemp -d -t ai_gateway.XXXXXX)
  echo "Storing multiprocess metrics in $METRICS_DIR..."
  export PROMETHEUS_MULTIPROC_DIR=$METRICS_DIR
fi

# Run AI Gateway by default.
# Optionally, you can run Duo Workflow Service by setting `ENABLE_DUO_WORKFLOW_SERVICE=true` variable.
# Optionally, you can disable AI Gateway by setting `DISABLE_AI_GATEWAY=true` variable.
commands=()

if [[ -z "${DISABLE_AI_GATEWAY}" ]]; then
  commands+=("poetry run ai_gateway")
fi

if [[ -n "${ENABLE_DUO_WORKFLOW_SERVICE}" ]]; then
  commands+=("poetry run duo-workflow-service")
fi

if [ ${#commands[@]} -eq 1 ]; then
    echo "Running single service with exec..."
    exec ${commands[0]}
elif [ ${#commands[@]} -gt 1 ]; then
    echo "Running multiple services with parallel..."
    printf "%s\n" "${commands[@]}" | parallel --will-cite --line-buffer --halt never
else
    echo "No services configured to run. Exiting."
    exit 1
fi
