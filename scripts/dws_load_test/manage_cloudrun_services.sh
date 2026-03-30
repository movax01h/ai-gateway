#!/bin/bash
#
# Manage Cloud Run services for load testing and profiling. Specifically:
#
#   dws-loadtest        Duo Workflow Service instance used as the load test target
#   llm-cache-proxy     LLM caching proxy that sits between DWS and the LLM backend
#   pyroscope-server    Pyroscope profiling server for capturing CPU profiles
#
# Usage: ./manage_cloudrun_services.sh <command> [OPTIONS]
#
# Commands:
#   enable              Enable all services (set to manual scaling with 1 instance)
#   disable             Disable all services (set to manual scaling with 0 instances)
#   delete              Delete all services
#   status              Show status of all services
#
# Options:
#   --project PROJECT_ID    GCP project ID (default: dev-ai-research-0e2f8974)
#   --region REGION         GCP region (default: us-central1)
#   --help                  Show this help message
#
# Examples:
#   ./manage_cloudrun_services.sh enable
#   ./manage_cloudrun_services.sh disable
#   ./manage_cloudrun_services.sh delete
#   ./manage_cloudrun_services.sh status
#

set -e

PROJECT_ID="dev-ai-research-0e2f8974"
REGION="us-central1"
COMMAND=""

SERVICE_NAMES=(
    "dws-loadtest"
    "llm-cache-proxy"
    "pyroscope-server"
)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_header() { echo -e "\n${CYAN}=== $1 ===${NC}\n"; }

while [[ $# -gt 0 ]]; do
    case $1 in
        --project)
            PROJECT_ID="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --help)
            grep '^#' "$0" | tail -n +2 | sed 's/^# //'
            exit 0
            ;;
        *)
            if [ -z "$COMMAND" ]; then
                COMMAND="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$COMMAND" ]; then
    log_error "No command specified"
    echo ""
    grep '^#' "$0" | tail -n +2 | sed 's/^# //'
    exit 1
fi

check_requirements() {
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed"
        exit 1
    fi
}

get_service_url() {
    local service_name=$1
    gcloud run services describe "$service_name" \
        --region "$REGION" \
        --project "$PROJECT_ID" \
        --format 'value(status.url)' 2>/dev/null || echo "N/A"
}

# Get configured min instances (handles both manual and auto scaling)
get_min_instances() {
    local service_name=$1
    local result
    result=$(gcloud run services describe "$service_name" \
        --region "$REGION" \
        --project "$PROJECT_ID" \
        --format json 2>/dev/null | \
        python3 -c "
import sys, json
data = json.load(sys.stdin)
annotations = data.get('metadata', {}).get('annotations', {})
scaling_mode = annotations.get('run.googleapis.com/scalingMode', 'auto')

if scaling_mode == 'manual':
    manual_count = annotations.get('run.googleapis.com/manualInstanceCount', '0')
    print(manual_count)
else:
    min_count = data.get('spec', {}).get('scaling', {}).get('minInstanceCount', 0)
    print(min_count)
" 2>/dev/null)
    echo "${result:-0}"
}

get_current_instances() {
    local service_name=$1

    # Check if service is ready (has active instances)
    local ready_condition
    ready_condition=$(gcloud run services describe "$service_name" \
        --region "$REGION" \
        --project "$PROJECT_ID" \
        --format json 2>/dev/null | \
        python3 -c "import sys, json; data = json.load(sys.stdin); conditions = data.get('status', {}).get('conditions', []); ready = next((c for c in conditions if c.get('type') == 'Ready'), {}); print(ready.get('status', 'Unknown'))" 2>/dev/null)

    local min_instances=$(get_min_instances "$service_name")

    if [ "$ready_condition" = "True" ] && [ "$min_instances" -gt 0 ]; then
        echo ">=${min_instances}"
    elif [ "$ready_condition" = "True" ]; then
        echo "ready"
    else
        echo "0"
    fi
}

get_service_status() {
    local service_name=$1
    gcloud run services describe "$service_name" \
        --region "$REGION" \
        --project "$PROJECT_ID" \
        --format 'value(spec.template.spec.containerConcurrency)' 2>/dev/null || echo "Not found"
}

service_exists() {
    local service_name=$1
    gcloud run services describe "$service_name" \
        --region "$REGION" \
        --project "$PROJECT_ID" \
        &>/dev/null
}

enable_services() {
    log_header "Enabling Services"
    log_info "Project: $PROJECT_ID"
    log_info "Region: $REGION"
    echo ""

    local failed=0

    for service_name in "${SERVICE_NAMES[@]}"; do
        if service_exists "$service_name"; then
            log_info "Enabling $service_name..."
            if gcloud run services update "$service_name" \
                --region "$REGION" \
                --project "$PROJECT_ID" \
                --scaling=1 \
                --quiet; then
                log_success "$service_name enabled (scaling: manual, instances: 1)"
            else
                log_error "Failed to enable $service_name"
                ((failed++))
            fi
        else
            log_warning "$service_name not found (skipping)"
        fi
    done

    echo ""
    if [ $failed -eq 0 ]; then
        log_success "All services enabled successfully!"
    else
        log_error "$failed service(s) failed to enable"
        exit 1
    fi
}

# Disable services (set to manual scaling with 0 instances)
disable_services() {
    log_header "Disabling Services"
    log_info "Project: $PROJECT_ID"
    log_info "Region: $REGION"
    echo ""

    local failed=0

    for service_name in "${SERVICE_NAMES[@]}"; do
        if service_exists "$service_name"; then
            log_info "Disabling $service_name..."
            if gcloud run services update "$service_name" \
                --region "$REGION" \
                --project "$PROJECT_ID" \
                --scaling=0 \
                --quiet; then
                log_success "$service_name disabled (scaling: manual, instances: 0)"
            else
                log_error "Failed to disable $service_name"
                ((failed++))
            fi
        else
            log_warning "$service_name not found (skipping)"
        fi
    done

    echo ""
    if [ $failed -eq 0 ]; then
        log_success "All services disabled successfully!"
    else
        log_error "$failed service(s) failed to disable"
        exit 1
    fi
}


get_service_info() {
    local service_name=$1
    local temp_file=$2

    if service_exists "$service_name"; then
        local url=$(get_service_url "$service_name")
        local min_instances=$(get_min_instances "$service_name")
        local current_instances=$(get_current_instances "$service_name")
        echo "$service_name|Active|$current_instances|$min_instances|$url" > "$temp_file"
    else
        echo "$service_name|Missing|N/A|N/A|N/A" > "$temp_file"
    fi
}

show_status() {
    log_header "Service Status"
    log_info "Project: $PROJECT_ID"
    log_info "Region: $REGION"
    echo ""

    local temp_dir=$(mktemp -d)
    local pids=()
    local temp_files=()

    # Fetch all service info in parallel
    for i in "${!SERVICE_NAMES[@]}"; do
        local temp_file="$temp_dir/${SERVICE_NAMES[$i]}.txt"
        temp_files+=("$temp_file")
        get_service_info "${SERVICE_NAMES[$i]}" "$temp_file" &
        pids+=($!)
    done

    # Wait for all fetches to complete
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    printf "%-25s %-10s %-10s %-10s %-50s\n" "Service" "Status" "Current" "Min" "URL"
    printf "%-25s %-10s %-10s %-10s %-50s\n" "-------" "------" "-------" "---" "---"

    for temp_file in "${temp_files[@]}"; do
        if [ -f "$temp_file" ]; then
            IFS='|' read -r service_name status current min url < "$temp_file"
            printf "%-25s %-10s %-10s %-10s %-50s\n" "$service_name" "$status" "$current" "$min" "$url"
        fi
    done

    rm -rf "$temp_dir"

    echo ""
    log_info "Current: Active instances (>=N = at least N running, ready = scaled to zero but ready, 0 = not ready)"
    log_info "Min: Minimum instances at Service level (0 = scales to zero, applies across all revisions)"
    echo ""
    log_info "To get detailed status of a service:"
    echo "  gcloud run services describe <service-name> --region $REGION --project $PROJECT_ID"
}


delete_services() {
    log_header "Deleting Services"
    log_info "Project: $PROJECT_ID"
    log_info "Region: $REGION"
    echo ""

    log_warning "This will delete all services. This action cannot be undone."
    read -p "Are you sure you want to delete all services? (yes/no): " -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log_info "Deletion cancelled"
        exit 0
    fi

    local failed=0

    for service_name in "${SERVICE_NAMES[@]}"; do
        if service_exists "$service_name"; then
            log_info "Deleting $service_name..."
            if gcloud run services delete "$service_name" \
                --region "$REGION" \
                --project "$PROJECT_ID" \
                --quiet; then
                log_success "$service_name deleted"
            else
                log_error "Failed to delete $service_name"
                ((failed++))
            fi
        else
            log_warning "$service_name not found (skipping)"
        fi
    done

    echo ""
    if [ $failed -eq 0 ]; then
        log_success "All services deleted successfully!"
    else
        log_error "$failed service(s) failed to delete"
        exit 1
    fi
}

check_requirements

case "$COMMAND" in
    enable)
        enable_services
        ;;
    disable)
        disable_services
        ;;
    delete)
        delete_services
        ;;
    status)
        show_status
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        echo ""
        grep '^#' "$0" | tail -n +2 | sed 's/^# //'
        exit 1
        ;;
esac
