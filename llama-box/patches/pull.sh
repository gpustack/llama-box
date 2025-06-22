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
    info "Pulling ${VENDOR}"
    if [[ "${VENDOR}" == "ggml" ]]; then
        continue # Skip ggml as it is handled separately
    fi
    git submodule update --remote "${VENDOR}"
    git submodule update --init "${VENDOR}"
    info "Pulled ${VENDOR}"
done