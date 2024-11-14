#!/bin/bash

set -e

#
# MIT license
# Copyright (c) 2024 llama-box authors
# SPDX-License-Identifier: MIT
#

LOG_FILE=${LOG_FILE:-/dev/null}

API_URL="${API_URL:-http://127.0.0.1:8080}"

trim() {
    shopt -s extglob
    set -- "${1##+([[:space:]])}"
    printf "%s" "${1%%+([[:space:]])}"
}

trim_trailing() {
    shopt -s extglob
    printf "%s" "${1%%+([[:space:]])}"
}

N="${N:-"1"}"
RESPONSE_FORMAT="b64_json"
SIZE="${SIZE:-"512x512"}"
QUALITY="${QUALITY:-"standard"}"
STYLE="${STYLE:-"null"}"
SAMPLER="${SAMPLER:-"null"}"
SEED="${SEED:-"null"}"
CFG_SCALE="${CFG_SCALE:-"9"}"
SAMPLE_STEPS="${SAMPLE_STEPS:-"20"}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-""}"

image_generate() {
    PROMPT="$(trim_trailing "$1")"
    if [[ "${PROMPT:0:1}" == "@" ]] && [[ -f "${PROMPT:1}" ]]; then
      DATA="$(cat "${PROMPT:1}")"
    else
      DATA="{\"prompt\":\"${PROMPT}\"}"
    fi
    if [[ "${SAMPLER}" != "null" ]]; then
      DATA="$(echo -n "${DATA}" | jq \
                --argjson n "${N}" \
                --argjson response_format "\"${RESPONSE_FORMAT}\"" \
                --argjson size "\"${SIZE}\"" \
                --argjson sampler "\"${SAMPLER}\"" \
                --argjson seed "${SEED}" \
                --argjson cfg_scale "${CFG_SCALE}" \
                --argjson sample_steps "${SAMPLE_STEPS}" \
                --argjson negative_prompt "\"${NEGATIVE_PROMPT}\"" \
                '{
                  n: $n,
                  response_format: $response_format,
                  size: $size,
                  sampler: $sampler,
                  seed: $seed,
                  cfg_scale: $cfg_scale,
                  sample_steps: $sample_steps,
                  negative_prompt: $negative_prompt
                } * .')"
    elif [[ "${STYLE}" != "null" ]]; then
      DATA="$(echo -n "${DATA}" | jq \
                --argjson n "${N}" \
                --argjson response_format "\"${RESPONSE_FORMAT}\"" \
                --argjson size "\"${SIZE}\"" \
                --argjson quality "\"${QUALITY}\"" \
                --argjson style "\"${STYLE}\"" \
                '{
                  n: $n,
                  response_format: $response_format,
                  size: $size,
                  quality: $quality,
                  style: $style
                } * .')"
    else
      DATA="$(echo -n "${DATA}" | jq \
                --argjson n "${N}" \
                --argjson response_format "\"${RESPONSE_FORMAT}\"" \
                --argjson size "\"${SIZE}\"" \
                --argjson quality "\"${QUALITY}\"" \
                '{
                  n: $n,
                  response_format: $response_format,
                  size: $size,
                  quality: $quality
                } * .')"
    fi
    echo "Q: ${DATA}" >> "${LOG_FILE}"

    START_TIME=$(date +%s)
    ANSWER="$(curl \
      --silent \
      --no-buffer \
      --request POST \
      --url "${API_URL}/v1/images/generations" \
      --header "Content-Type: application/json" \
      --data-raw "${DATA}")"
    printf "%s" "A: ${ANSWER}" >> "${LOG_FILE}"

    CONTENT="$(echo "${ANSWER}" | jq -c -r '.data')"
    if [[ "${CONTENT}" == "null" ]]; then
        echo "Error: ${ANSWER}"
        return
    fi
    printf "%s" "${CONTENT}" > /tmp/image_generate_result.json
    for i in $(seq 0 $(("${N}" - 1))); do
        TIME=$(date +%s)
        jq -c -r ".[${i}] | .b64_json" /tmp/image_generate_result.json | base64 -d 2>/dev/null > "/tmp/image_generate_${TIME}.png"
        if [[ "$(uname -s)" == "Darwin" ]]; then
            if command -v feh > /dev/null; then
                feh "/tmp/image_generate_${TIME}.png"
            elif command -v open > /dev/null; then
                open "/tmp/image_generate_${TIME}.png"
            else
                echo "Generated image: /tmp/image_generate_${TIME}.png"
            fi
        else
            echo "Generated image: /tmp/image_generate_${TIME}.png"
        fi
        sleep 1
    done

    printf "\n------------------------"
    ELAPSED=$(($(date +%s) - START_TIME))
    printf "\n- TC   : %10.2fs   -" "${ELAPSED}"
    printf "\n------------------------"

    printf "\n"
}

echo "====================================================="
echo "LOG_FILE          : ${LOG_FILE}"
echo "API_URL           : ${API_URL}"
echo "N                 : ${N}"
echo "RESPONSE_FORMAT   : ${RESPONSE_FORMAT}"
echo "SIZE              : ${SIZE}"
echo "QUALITY           : ${QUALITY}"
echo "STYLE             : ${STYLE}"
echo "SAMPLER           : ${SAMPLER} // OVERRIDE \"QUALITY\" and \"STYLE\" IF NOT NULL, ONE OF [euler_a, euler, heun, dpm2, dpm++2s_a, dpm++2mv2, ipndm, ipndm_v, lcm]"
echo "SEED              : ${SEED} // AVAILABLE FOR SAMPLER"
echo "CFG_SCALE         : ${CFG_SCALE} // AVAILABLE FOR SAMPLER"
echo "SAMPLE_STEPS      : ${SAMPLE_STEPS} // AVAILABLE FOR SAMPLER"
echo "NEGATIVE_PROMPT   : ${NEGATIVE_PROMPT} // AVAILABLE FOR SAMPLER"
printf "=====================================================\n\n"

if [[ -f "${LOG_FILE}" ]]; then
    : > "${LOG_FILE}"
fi
if [[ ! -f "${LOG_FILE}" ]]; then
    touch "${LOG_FILE}"
fi

if [[ "${#@}" -ge 1 ]]; then
    echo "> ${*}"
    image_generate "${*}"
else
    while true; do
        read -r -e -p "> " QUESTION
        image_generate "${QUESTION}"
    done
fi
