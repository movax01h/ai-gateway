#!/bin/bash
#
# Deploy DWS with Pyroscope profiling to Cloud Run
#
# Usage: ./deploy_with_pyroscope_to_cloudrun.sh [PROJECT_ID] [REGION] [IMAGE_TAG]
#
# Arguments:
#   PROJECT_ID: GCP project ID (default: dev-ai-research-0e2f8974)
#   REGION: GCP region (default: us-central1)
#   IMAGE_TAG: model-gateway image tag (default: latest)
#

set -e

PROJECT_ID="${1:-dev-ai-research-0e2f8974}"
REGION="${2:-us-central1}"
SOURCE_IMAGE_TAG="${3:-latest}"
SERVICE_NAME="dws-loadtest"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SOURCE_IMAGE="registry.gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/model-gateway:$SOURCE_IMAGE_TAG"
IMAGE_TAG="$(date +%Y%m%d-%H%M%S)"
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/ai-gateway/ai-gateway:dws-loadtest-$IMAGE_TAG"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

if ! command -v gcloud &> /dev/null; then
    log_error "gcloud CLI is not installed"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    log_error "docker is not installed"
    exit 1
fi

ENV_FILE="$REPO_ROOT/.env"
if [ ! -f "$ENV_FILE" ]; then
    log_error ".env file not found at $ENV_FILE"
    exit 1
fi

if ! grep -q "PYROSCOPE_SERVER_URL" "$ENV_FILE"; then
    log_error "PYROSCOPE_SERVER_URL not found in .env"
    echo ""
    echo "First deploy the Pyroscope server:"
    echo "  ./scripts/dws_load_test/deploy_pyroscope_server.sh"
    echo ""
    echo "Then add the URL to .env:"
    echo "  PYROSCOPE_SERVER_URL=https://pyroscope-server-xxxxx.a.run.app"
    exit 1
fi

log_info "Deploying DWS with Pyroscope profiling to Cloud Run"
log_info "Project: $PROJECT_ID"
log_info "Region: $REGION"
log_info "Image: $IMAGE"
echo ""

log_info "Configuring Docker authentication..."
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet

log_info "Pulling source image: $SOURCE_IMAGE"
docker pull "$SOURCE_IMAGE"

log_info "Retagging as: $IMAGE"
docker tag "$SOURCE_IMAGE" "$IMAGE"

log_info "Ensuring Artifact Registry repository exists..."
if ! gcloud artifacts repositories describe ai-gateway \
        --location="$REGION" \
        --project="$PROJECT_ID" &> /dev/null; then
    gcloud artifacts repositories create ai-gateway \
        --repository-format=docker \
        --location="$REGION" \
        --project="$PROJECT_ID" \
        --description="AI Gateway Docker images"
fi

log_info "Pushing image to GCP Artifact Registry..."
docker push "$IMAGE"

log_success "Image pushed successfully"

# Allowlist of env var keys to pass to Cloud Run as plain env vars
ALLOWLISTED_VARS=(
    AIGW_BILLING_EVENT__ENABLED
    AIGW_SNOWPLOW__ENABLED
    AIGW_CUSTOM_MODELS__ENABLED
    AIGW_GOOGLE_CLOUD_PLATFORM__PROJECT
    AIGW_INSTRUMENTATOR__THREAD_MONITORING_ENABLED
    AIGW_INTERNAL_EVENT__ENABLED
    AIGW_VERTEX_SEARCH__FALLBACK_DATASTORE_VERSION
    ANTHROPIC_API_KEY
    DUO_WORKFLOW_AUTH__ENABLED
    DUO_WORKFLOW_AUTH__OIDC_CUSTOMER_PORTAL_URL
    DUO_WORKFLOW_AUTH__OIDC_GITLAB_URL
    DUO_WORKFLOW_GOOGLE_CLOUD_PROFILER__ENABLED
    DUO_WORKFLOW_SERVICE_ENVIRONMENT
    DUO_WORKFLOW_LOGGING__LEVEL
    DUO_WORKFLOW_LOGGING__JSON_FORMAT
    DUO_WORKFLOW_USE_CACHING_PROXY
    DUO_WORKFLOW_CACHING_PROXY_URL
    DUO_WORKFLOW_CLOUD_CONNECTOR_SERVICE_NAME
    DUO_WORKFLOW_DIRECT_CONNECTION_BASE_URL
    DUO_WORKFLOW_GRPC_REFLECTION_ENABLED
    DUO_WORKFLOW__VERTEX_LOCATION
    TRANSFORMERS_NO_ADVISORY_WARNINGS
    LANGCHAIN_TRACING_V2
    SENTRY_ERROR_TRACKING_ENABLED
    SENTRY_DSN
    PYROSCOPE_SERVER_URL
    CUSTOMER_PORTAL_USAGE_QUOTA_API_USER
    CUSTOMER_PORTAL_USAGE_QUOTA_API_TOKEN
)

# Multiline values stored in Secret Manager
SECRET_VARS=(
    DUO_WORKFLOW_SELF_SIGNED_JWT__SIGNING_KEY
    DUO_WORKFLOW_SELF_SIGNED_JWT__VALIDATION_KEY
)

log_info "Reading allowlisted environment variables from .env"
ENV_VARS=""

for key in "${ALLOWLISTED_VARS[@]}"; do
    line=$(grep "^${key}=" "$ENV_FILE" | head -1)
    [ -z "$line" ] && continue
    # Strip only the first '=', preserving any '=' in the value
    value="${line#*=}"
    # Strip surrounding single or double quotes (common in .env files)
    value="${value#\"}"
    value="${value%\"}"
    value="${value#\'}"
    value="${value%\'}"
    ENV_VARS="${ENV_VARS:+$ENV_VARS,}$key=$value"
done

ENV_VARS="${ENV_VARS:+$ENV_VARS,}ENABLE_DUO_WORKFLOW_SERVICE=true,DISABLE_AI_GATEWAY=true,PYROSCOPE_ENABLED=true"
# ENV_VARS="${ENV_VARS:+$ENV_VARS,}ENABLE_DUO_WORKFLOW_SERVICE=true,DISABLE_AI_GATEWAY=true"

log_info "Syncing multiline secrets to Secret Manager..."
SET_SECRETS=""

for key in "${SECRET_VARS[@]}"; do
    # Extract multiline value between the first quote after KEY= and its closing quote
    value=$(python3 -c "
import re, sys
text = open('$ENV_FILE').read()
m = re.search(r'^${key}=\"(.*?)\"', text, re.DOTALL | re.MULTILINE)
if m:
    sys.stdout.write(m.group(1).replace('\\\\n', '\\n'))
")
    if [ -z "$value" ]; then
        log_warning "$key not found in .env, skipping"
        continue
    fi

    if gcloud secrets describe "$key" --project="$PROJECT_ID" &> /dev/null; then
        printf '%s' "$value" | gcloud secrets versions add "$key" \
            --data-file=- --project="$PROJECT_ID" --quiet
        log_info "Updated secret: $key"
    else
        printf '%s' "$value" | gcloud secrets create "$key" \
            --data-file=- --project="$PROJECT_ID" --quiet
        log_info "Created secret: $key"
    fi

    SET_SECRETS="${SET_SECRETS:+$SET_SECRETS,}$key=$key:latest"
done

log_info "Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
    --image "$IMAGE" \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --platform managed \
    --allow-unauthenticated \
    --port 8080 \
    --use-http2 \
    --cpu 2 \
    --memory 4Gi \
    --scaling 1 \
    --timeout 3600 \
    --set-env-vars "$ENV_VARS" \
    ${SET_SECRETS:+--set-secrets "$SET_SECRETS"} \
    --execution-environment gen2

log_success "Deployment successful!"
echo ""

SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --format 'value(status.url)')

PYROSCOPE_URL=$(grep "^PYROSCOPE_SERVER_URL=" "$ENV_FILE" | head -1)
PYROSCOPE_URL="${PYROSCOPE_URL#*=}"
PYROSCOPE_URL="${PYROSCOPE_URL#\"}"
PYROSCOPE_URL="${PYROSCOPE_URL%\"}"
PYROSCOPE_URL="${PYROSCOPE_URL#\'}"
PYROSCOPE_URL="${PYROSCOPE_URL%\'}"

log_success "DWS deployed with Pyroscope profiling!"
log_info "Service URL: ${SERVICE_URL#https://}"
log_info "Pyroscope UI: $PYROSCOPE_URL"
echo ""

echo -e "${RED}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!${NC}"
echo -e "${RED}!!  WARNING: This service is publicly accessible and        !!${NC}"
echo -e "${RED}!!  unauthenticated. Destroy it as soon as testing is done. !!${NC}"
echo -e "${RED}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!${NC}"
echo ""

log_info "Next steps:"
echo "1. Send requests to trigger profiling"
echo ""
echo "2. View profiles in Pyroscope UI:"
echo "   Open $PYROSCOPE_URL in your browser"
echo ""
echo "3. Run load tests to generate continuous profiling data"
echo "   See performance_tests/stress_tests/README.md"
echo ""
