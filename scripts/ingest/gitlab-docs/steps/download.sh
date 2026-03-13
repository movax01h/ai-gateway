#!/usr/bin/env bash

set -eu

rm -Rf "${GITLAB_DOCS_CLONE_DIR}"
mkdir -p "${GITLAB_DOCS_CLONE_DIR}"

PROTOCOL=$(echo "${GITLAB_DOCS_REPO}" | sed 's|://.*||')
HOST=$(echo "${GITLAB_DOCS_REPO}" | sed 's|.*://\([^/]*\).*|\1|')
PROJECT_PATH=$(echo "${GITLAB_DOCS_REPO}" | sed "s|.*://${HOST}/||" | sed 's|\.git$||')
PROJECT_ID=$(printf '%s' "${PROJECT_PATH}" | sed 's|/|%2F|g')

ARCHIVE_URL="${PROTOCOL}://${HOST}/api/v4/projects/${PROJECT_ID}/repository/archive.tar.gz?sha=${GITLAB_DOCS_REPO_REF}"

curl -L "${ARCHIVE_URL}" | tar -xz -C "${GITLAB_DOCS_CLONE_DIR}" --strip-components=1
