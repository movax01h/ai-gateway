#!/bin/bash
#
# Deploy Pyroscope server to Cloud Run
#
# Usage: ./deploy_pyroscope_server.sh [PROJECT_ID] [REGION]
#
# Arguments:
#   PROJECT_ID: GCP project ID (default: dev-ai-research-0e2f8974)
#   REGION: GCP region (default: us-central1)
#

set -e

# Configuration
PROJECT_ID="${1:-dev-ai-research-0e2f8974}"
REGION="${2:-us-central1}"
SERVICE_NAME="pyroscope-server"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check for required tools
if ! command -v gcloud &> /dev/null; then
    log_error "gcloud CLI is not installed"
    exit 1
fi

log_info "Deploying Pyroscope server to Cloud Run"
log_info "Project: $PROJECT_ID"
log_info "Region: $REGION"
echo ""

# Deploy Pyroscope server using official image
# Note: Using filesystem storage (ephemeral) for simplicity.
log_info "Deploying Pyroscope server to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
    --image grafana/pyroscope:latest \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --platform managed \
    --allow-unauthenticated \
    --port 4040 \
    --cpu 4 \
    --memory 8Gi \
    --scaling 1 \
    --timeout 3600 \
    --execution-environment gen2

log_success "Pyroscope server deployed successfully!"

PYROSCOPE_SERVER_URL=$(gcloud run services describe "$SERVICE_NAME" \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --format 'value(status.url)')

echo ""
log_success "Pyroscope server is available at: $PYROSCOPE_SERVER_URL"
echo ""

echo -e "${RED}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!${NC}"
echo -e "${RED}!!  WARNING: This service is publicly accessible and        !!${NC}"
echo -e "${RED}!!  unauthenticated. Destroy it as soon as testing is done. !!${NC}"
echo -e "${RED}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!${NC}"
echo ""

log_info "Next steps:"
echo "1. Add this to your .env file:"
echo "   PYROSCOPE_SERVER_URL=$PYROSCOPE_SERVER_URL"
echo ""
echo "2. Deploy DWS with Pyroscope profiling:"
echo "   ./scripts/dws_load_test/deploy_with_pyroscope_to_cloudrun.sh"
echo ""
echo "3. View profiles in the Pyroscope UI:"
echo "   Open $PYROSCOPE_SERVER_URL in your browser"
echo ""
