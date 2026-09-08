#!/usr/bin/env bash
# duo-workflow-service crashes if it can't reach GitLab's OIDC endpoint at
# startup, so wait for GitLab before launching it (give up after ~60s).

set -euo pipefail

# Per-stack GitLab host from caproni.local.yaml; `|| true` keeps the fallback usable if the file is missing.
host="${CAPRONI_PRIMARY_HOSTNAME:-gitlab.caproni.test}"

# GitLab answers on port 80 unless the k3d loadbalancer is published elsewhere
# (rootless podman cannot bind <1024, so those rigs remap to e.g. 8080:80).
# Discover the host port from the resolved caproni config so the poll follows
# cluster.k3d.port_mappings; CAPRONI_GITLAB_HTTP_PORT overrides it. Empty
# means portless, which keeps the default (Colima, port 80) path unchanged.
# yq is pinned in gitlab-caproni's .tool-versions, so its absence is a broken
# rig rather than a reason to guess.
port="${CAPRONI_GITLAB_HTTP_PORT:-}"
if [[ -z "$port" ]]; then
  command -v yq >/dev/null 2>&1 || { echo "yq not found; it is pinned in gitlab-caproni's .tool-versions (mise install)" >&2; exit 1; }
  port="$(caproni config print 2>/dev/null | yq -r '
    .cluster.k3d.port_mappings[]?
    | select(test("^([0-9.]+:)?[0-9]+:80(/tcp)?@loadbalancer$"))
    | sub("^([0-9.]+:)?([0-9]+):.*$"; "${2}")' | head -1)"
fi
url="http://${host}${port:+:$port}/oauth/discovery/keys"

for _ in $(seq 1 30); do
  curl -sf "$url" >/dev/null 2>&1 && exit 0
  echo "waiting for GitLab at $url ..."
  sleep 2
done

echo "Timed out waiting for GitLab OIDC endpoint ($url)" >&2
exit 1
