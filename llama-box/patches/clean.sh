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
    if [[ "${VENDOR}" == "ggml" ]]; then
        pushd "${ROOT_DIR}/llama.cpp/ggml" 1>/dev/null 2>&1 || true
    else
        pushd "${ROOT_DIR}/${VENDOR}" 1>/dev/null 2>&1 || true
    fi
    git reset --hard 1>/dev/null || fatal "Failed to reset ${VENDOR}"
    git clean -df 1>/dev/null || fatal "Failed to clean ${VENDOR}"
    popd 1>/dev/null 2>&1 || true
    info "Cleaned ${VENDOR}"
done
