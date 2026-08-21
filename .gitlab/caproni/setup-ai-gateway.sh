#!/usr/bin/env bash

# setup-ai-gateway.sh - Installs dependencies and configures GitLab to use the
# local AI Gateway.
#
# Runs as the `edit_mode_start` lifecycle hook of the ai-gateway repository
# (see caproni-project.yaml at the repo root). Ported from the deprecated
# caproni-demo project (monolith-edit-mode/scripts/setup.sh).
#
# Prerequisites:
#   - `caproni up` has completed
#   - ANTHROPIC_API_KEY is set (only needed on first run)
#
# Usage:
#   export ANTHROPIC_API_KEY=<your-key>
#   ./scripts/setup-ai-gateway.sh

set -euo pipefail

# ANSI color codes for output formatting.
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

error()   { echo >&2 -e "${RED}ERROR: ${1-}${NC}"; }
warn()    { echo >&2 -e "${YELLOW}WARNING: ${1-}${NC}"; }
success() { echo >&2 -e "${GREEN}${1-}${NC}"; }

# Namespace where GitLab is deployed. Override with GITLAB_NS=<ns> if needed.
GITLAB_NS="${GITLAB_NS:-gitlab}"

# In-cluster AI Gateway URL. gitlab:duo:setup requires AI_GATEWAY_URL and
# writes it to ApplicationSetting#ai_gateway_url; the Rails processes reach
# it via cluster DNS (locally through mirrord's outgoing network).
AI_GATEWAY_URL="${AI_GATEWAY_URL:-http://ai-gateway.ai-gateway.svc.cluster.local}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------------------
# Install dependencies.
# ---------------------------------------------------------------------------

echo ""
echo "==> Installing dependencies..."

REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

if [ ! -f .env ]; then
  cp example.env .env
fi

override_env() {
  local key="$1" value="$2"
  awk -v k="$key" -v v="$value" 'BEGIN{FS=OFS="="} $1==k {$0=k"="v} 1' .env > .env.tmp && mv .env.tmp .env
}

gitlab_base_url="http://${CAPRONI_PRIMARY_HOSTNAME:-"gitlab.caproni.test"}"

override_env AIGW_GITLAB_URL                    "$gitlab_base_url/"
override_env AIGW_GITLAB_API_URL                "${gitlab_base_url}/api/v4/"
override_env DUO_WORKFLOW_AUTH__ENABLED         true
override_env DUO_WORKFLOW_AUTH__OIDC_GITLAB_URL "$gitlab_base_url"

if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  override_env ANTHROPIC_API_KEY "$ANTHROPIC_API_KEY"
fi

echo "BASE URL: $gitlab_base_url"

if ! command -v mise >/dev/null 2>&1; then
  error "mise is not installed. Please install mise first: https://mise.jdx.dev/getting-started.html"
  exit 1
fi

mise install
mise exec -- poetry install

success "  ✓ Dependencies installed."

# ---------------------------------------------------------------------------
# Most features default to Vertex/Fireworks/Bedrock models, but we're using
# Anthropic here, so we need to remap them. Edit .env if needed.
# ---------------------------------------------------------------------------

if grep -qE "^#* *AIGW_MODEL_SELECTION__DEFAULT_MODELS='\{\}'" .env; then
  echo ""
  echo "==> Switching default models to Anthropic..."

  default_models=$(mise exec -- poetry run python "$SCRIPT_DIR/models-override.py")
  if [[ -n "$default_models" && "$default_models" != "{}" ]]; then
    sed "s|^#* *AIGW_MODEL_SELECTION__DEFAULT_MODELS='{}'|AIGW_MODEL_SELECTION__DEFAULT_MODELS='${default_models}'|" \
      .env >.env.tmp && mv .env.tmp .env
    success "  ✓ Default models set to Anthropic."
  fi
fi

# ---------------------------------------------------------------------------
# Create the Anthropic API key secret if it doesn't already exist.
# ---------------------------------------------------------------------------

echo ""
echo "==> Checking Anthropic API key secret..."

caproni kubectl create namespace ai-gateway 2>/dev/null || true

if caproni kubectl get secret anthropic-api-key --namespace ai-gateway &>/dev/null; then
  success "  ✓ Secret anthropic-api-key already exists – skipping."
else
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    error "ANTHROPIC_API_KEY is not set. Export it before running this script."
    exit 1
  fi
  caproni kubectl create secret generic anthropic-api-key \
    --namespace ai-gateway \
    --from-literal=ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
  success "  ✓ Created secret anthropic-api-key."
fi

# ---------------------------------------------------------------------------
# Skip the (slow) GitLab Duo seeding when it has already completed.
#
# Everything below boots Rails several times via mirrord (~5+ minutes), which
# makes every `caproni run` restart painful. The steps are idempotent, so a
# marker file is enough. Delete it or set CAPRONI_DUO_SETUP_FORCE=1 to re-run
# (needed after recreating the cluster/database).
# ---------------------------------------------------------------------------

DUO_SETUP_MARKER="$REPO_DIR/.caproni-duo-setup-complete"

if [[ -f "$DUO_SETUP_MARKER" && "${CAPRONI_DUO_SETUP_FORCE:-0}" != "1" ]]; then
  echo ""
  success "==> GitLab Duo already seeded – skipping (marker: $DUO_SETUP_MARKER)."
  success "    Set CAPRONI_DUO_SETUP_FORCE=1 (or delete the marker) to re-run, e.g. after recreating the cluster."
  exit 0
fi

# ---------------------------------------------------------------------------
# Locate a running toolbox pod.
# ---------------------------------------------------------------------------

echo ""
echo "==> Waiting for GitLab toolbox to be ready..."

caproni kubectl wait pod -n "$GITLAB_NS" -l app=toolbox \
  --for=condition=ready --timeout=240s

TOOLBOX_POD=$(caproni kubectl get pod -n "$GITLAB_NS" -l app=toolbox \
  -o jsonpath='{.items[0].metadata.name}')

success "  ✓ Found pod: $TOOLBOX_POD"

GITLAB_DIR="$(cd "$SCRIPT_DIR/../../repos/gitlab" && pwd)"

# ---------------------------------------------------------------------------
# Ensure the GitLab repo's cluster-config sync has run first.
#
# caproni processes repository hooks in alphabetical order, so this
# (ai-gateway) hook fires BEFORE gitlab's own edit_mode_start hooks. The
# Rails steps below need config/database.yml etc., which gitlab's
# .gitlab/caproni/setup.sh copies from the cluster. Run it now if the config
# is missing (the script is idempotent — it re-runs harmlessly when gitlab's
# own hook fires later in the same `caproni run`).
# ---------------------------------------------------------------------------

if [[ ! -f "$GITLAB_DIR/config/database.yml" ]]; then
  echo ""
  echo "==> GitLab config not synced yet – running gitlab's .gitlab/caproni/setup.sh first..."
  "$GITLAB_DIR/.gitlab/caproni/setup.sh"
fi

# The Rails steps below also need the local checkout's migrations applied:
# the development database is cloned from the chart's stable release, while
# the local checkout is usually newer (e.g. ApplicationSetting validators
# referencing columns added by pending migrations raise NoMethodError).
# db_migrate.sh is idempotent and skips quickly when nothing is pending.

echo ""
echo "==> Ensuring database migrations are up to date..."
"$GITLAB_DIR/.gitlab/caproni/db_migrate.sh"

run_rake() {
  (cd "$GITLAB_DIR" && mise trust --quiet && RAILS_ENV=development AI_GATEWAY_URL="$AI_GATEWAY_URL" mise exec -- \
    mirrord exec \
      --target "pod/$TOOLBOX_POD/container/toolbox" \
      --target-namespace "$GITLAB_NS" \
      --config-file ".gitlab/caproni/.mirrord/exec.json" \
      -- bundle exec rake "$1")
}

rails_runner() {
  (cd "$GITLAB_DIR" && mise trust --quiet && RAILS_ENV=development mise exec -- \
    mirrord exec \
      --target "pod/$TOOLBOX_POD/container/toolbox" \
      --target-namespace "$GITLAB_NS" \
      --config-file ".gitlab/caproni/.mirrord/exec.json" \
      -- bundle exec rails runner -)
}

# ---------------------------------------------------------------------------
# Configure GitLab Duo.
# ---------------------------------------------------------------------------

echo ""
echo "==> Configuring GitLab Duo..."

rails_runner <<RUBY
::Ai::Setting.instance.update!(
  self_hosted_duo_agent_platform_service_secure: false,
  duo_agent_platform_service_url: 'ai-gateway.ai-gateway.svc.cluster.local:50052'
)
::ApplicationSetting.current_without_cache.update!(
  instance_level_ai_beta_features_enabled: true,
  duo_features_enabled: true,
  outbound_local_requests_allowlist_raw: 'ai-gateway.ai-gateway.svc.cluster.local'
)
RUBY

success "  ✓ GitLab Duo configured."

# ---------------------------------------------------------------------------
# Activate the EE license, which gitlab:duo:setup needs.
# ---------------------------------------------------------------------------

echo ""
echo "==> Activating EE license..."

# activate-license.sh never exits non-zero (skips are deliberate no-ops so it
# can't block `caproni run`), so detect success from its output markers and
# fall back to checking License.current directly. gitlab:duo:setup below
# hard-fails with an opaque 'No license found' otherwise, so fail fast here
# with actionable instructions instead.
license_output=$(CAPRONI_ACTIVATE_LICENSE=1 "$GITLAB_DIR/.gitlab/caproni/activate-license.sh" 2>&1) || true
echo "$license_output"

if ! echo "$license_output" | grep -qE "ALREADY_ACTIVE|SUCCESS"; then
  echo ""
  echo "==> License activation skipped or failed – verifying a license is active..."
  if ! rails_runner <<'RUBY' | grep -q "LICENSE_PRESENT"
puts(License.current ? "LICENSE_PRESENT" : "LICENSE_MISSING")
RUBY
  then
    error "No active EE license found — gitlab:duo:setup requires an EE Ultimate license."
    error "Run 'op signin' and re-run 'caproni run', or activate manually at:"
    error "  ${gitlab_base_url}/admin/subscription"
    exit 1
  fi
  success "  ✓ License already active."
fi

# ---------------------------------------------------------------------------
# Run gitlab:duo:setup.
# ---------------------------------------------------------------------------

echo ""
echo "==> Running gitlab:duo:setup..."
run_rake "gitlab:duo:setup"

success "  ✓ gitlab-duo/test created, feature flags enabled."

# ---------------------------------------------------------------------------
# Run gitlab:duo:onboard_dap.
# ---------------------------------------------------------------------------

echo ""
echo "==> Running gitlab:duo:onboard_dap..."
run_rake "gitlab:duo:onboard_dap"

success "  ✓ Duo Agent Platform onboarded."

# ---------------------------------------------------------------------------
# Disable use_mock_dot_api_for_usage_quota so credit checks use the real
# CustomerDot instead of the mock server (localhost:4567) we don't run.
# ---------------------------------------------------------------------------

echo ""
echo "==> Disabling use_mock_dot_api_for_usage_quota..."
rails_runner <<'RUBY'
Feature.disable(:use_mock_dot_api_for_usage_quota)
puts "use_mock_dot_api_for_usage_quota enabled? #{Feature.enabled?(:use_mock_dot_api_for_usage_quota, :instance)}"
RUBY
success "  ✓ Usage-quota mock disabled (uses real CustomerDot)."

# ---------------------------------------------------------------------------
# Seed AI catalog items and ItemConsumers.
# ---------------------------------------------------------------------------

echo ""
echo "==> Seeding AI catalog items and ItemConsumers..."

rails_runner <<'RUBY'
org = Organizations::Organization.find_by_id(1)
::Ai::Catalog::Flows::SeedFoundationalFlowsService.new(organization: org).execute if ::Ai::Catalog::Item.none?
group = Group.find_by!(path: 'gitlab-duo')
group.namespace_settings.update!(duo_foundational_flows_enabled: true)
ids = ::Ai::Catalog::Item.pluck(:id)
user = User.admins.first
[group, Project.find_by_full_path('gitlab-duo/test')].each do |c|
  c.sync_enabled_foundational_flows!(ids)
  ::Ai::Catalog::Flows::SyncFoundationalFlowsService.new(c, current_user: user).execute
end
RUBY

success "  ✓ AI catalog configured."

# ---------------------------------------------------------------------------
# Tag the instance runner with `gitlab--duo` so it picks up Duo flow runs.
# ---------------------------------------------------------------------------

echo ""
echo "==> Tagging instance runner..."

rails_runner <<'RUBY'
r = Ci::Runner.instance_type.online.first
if r.nil?
  warn "No online instance runner found to tag. You may need to register a runner first."
else
  r.update!(tag_list: (r.tag_list + [::Ai::DuoWorkflows::Workflow::WORKLOAD_TAG]).uniq)
end
RUBY

success "  ✓ Runner tagged."

touch "$DUO_SETUP_MARKER"

echo ""
success "==> Setup complete."
