#!/usr/bin/env bash
# duo-workflow-service crashes if it can't reach GitLab's OIDC endpoint at
# startup, so wait for GitLab before launching it (give up after ~60s).

set -euo pipefail

# Per-stack GitLab host from caproni.local.yaml; `|| true` keeps the fallback usable if the file is missing.
host="${CAPRONI_PRIMARY_HOSTNAME:-gitlab.caproni.test}"
url="http://${host}/oauth/discovery/keys"

for _ in $(seq 1 30); do
  curl -sf "$url" >/dev/null 2>&1 && exit 0
  echo "waiting for GitLab at $url ..."
  sleep 2
done

echo "Timed out waiting for GitLab OIDC endpoint ($url)" >&2
exit 1
