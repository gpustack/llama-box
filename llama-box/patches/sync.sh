#!/bin/bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"

function info() {
    echo "[INFO] $1"
}

function fatal() {
    echo "[FATA] $1"
    exit 1
}

# shellcheck disable=SC2010
ls -l "${ROOT_DIR}"/llama-box/patches | grep "^d" | awk '{print $NF}' | while read -r VENDOR; do
    info "Syncing ${VENDOR}"
    if [[ "${VENDOR}" == "ggml" ]]; then
        pushd "${ROOT_DIR}/llama.cpp/ggml" 1>/dev/null 2>&1 || true
    else
        pushd "${ROOT_DIR}/${VENDOR}" 1>/dev/null 2>&1 || true
    fi
    find "${ROOT_DIR}"/llama-box/patches/"${VENDOR}" -type f -name "*.patch" | while read -r FILE; do
        info "  Syncing ${FILE}"
        git apply --whitespace=nowarn "${FILE}" 1>/dev/null || fatal "Failed to apply patch ${FILE}"
        git diff > "${FILE}" || fatal "Failed to update patch ${FILE}"
        git reset --hard 1>/dev/null || fatal "Failed to reset patch ${FILE}"
    done
    popd 1>/dev/null 2>&1 || true
    info "Synced ${VENDOR}"
done