#!/usr/bin/env bash

set -eu

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
STEPS_DIR="${SCRIPT_DIR}/steps"
echo "SCRIPT_DIR: ${SCRIPT_DIR}"

TEST_TAG=v17.0.1-ee
TEST_FILE=docs-${TEST_TAG}.sha256
TEST_CLONE=/tmp/gitlab-docs-${TEST_TAG}

GITLAB_DOCS_CLONE_DIR=${TEST_CLONE}
GITLAB_DOCS_JSONL_EXPORT_PATH=${TEST_CLONE}/docs-${TEST_TAG}.jsonl

echo "------------------------------------------------------- Downloading Docs -------------------------------------------------------"
GITLAB_DOCS_CLONE_DIR="${GITLAB_DOCS_CLONE_DIR}" GITLAB_DOCS_REPO_REF="${TEST_TAG}" "${STEPS_DIR}"/download.sh

echo "------------------------------------------------------- Validating -------------------------------------------------------"
"${STEPS_DIR}"/validate.sh
echo "------------------------------------------------------- Exporting Variables -------------------------------------------------------"
source "${STEPS_DIR}"/export_variables.sh
echo "------------------------------------------------------- Parsing -------------------------------------------------------"
poetry run python "${STEPS_DIR}"/parse.py

echo "------------------------------------------------------- Comparing Results -------------------------------------------------------"
cp "${SCRIPT_DIR}/testdata/${TEST_FILE}" "${TEST_CLONE}/"
cd ${TEST_CLONE}
sha256sum -c ${TEST_FILE}
cd -
