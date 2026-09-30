#!/usr/bin/env bash

set -euo pipefail

# The repository archive endpoint is rate limited on GitLab.com (5 requests per
# minute for unauthenticated requests), so failures are expected occasionally.
# Wait a full minute between attempts so each retry lands in a new rate-limit window.
DOWNLOAD_RETRIES="${GITLAB_DOCS_DOWNLOAD_RETRIES:-5}"
DOWNLOAD_RETRY_DELAY="${GITLAB_DOCS_DOWNLOAD_RETRY_DELAY:-60}"

rm -Rf "${GITLAB_DOCS_CLONE_DIR}"
mkdir -p "${GITLAB_DOCS_CLONE_DIR}"

PROTOCOL=$(echo "${GITLAB_DOCS_REPO}" | sed 's|://.*||')
HOST=$(echo "${GITLAB_DOCS_REPO}" | sed 's|.*://\([^/]*\).*|\1|')
PROJECT_PATH=$(echo "${GITLAB_DOCS_REPO}" | sed "s|.*://${HOST}/||" | sed 's|\.git$||')
PROJECT_ID=$(printf '%s' "${PROJECT_PATH}" | sed 's|/|%2F|g')

ARCHIVE_URL="${PROTOCOL}://${HOST}/api/v4/projects/${PROJECT_ID}/repository/archive.tar.gz?sha=${GITLAB_DOCS_REPO_REF}"

ARCHIVE_FILE=$(mktemp)
trap 'rm -f "${ARCHIVE_FILE}"' EXIT

# Download to a file first so an HTTP error fails here instead of being piped into tar.
# `--retry` covers timeouts and HTTP 408/429/5xx, and honors a `Retry-After` header when longer than the delay.
if ! curl --fail --location \
  --retry "${DOWNLOAD_RETRIES}" \
  --retry-delay "${DOWNLOAD_RETRY_DELAY}" \
  --output "${ARCHIVE_FILE}" \
  "${ARCHIVE_URL}"; then
  echo "Failed to download ${ARCHIVE_URL}" >&2
  exit 1
fi

tar -xzf "${ARCHIVE_FILE}" -C "${GITLAB_DOCS_CLONE_DIR}" --strip-components=1
