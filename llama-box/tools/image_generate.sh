#!/bin/bash

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
SEED="${SEED:-"null"}"
QUALITY="${QUALITY:-"standard"}"
RESPONSE_FORMAT="b64_json"
SIZE="${SIZE:-"512x512"}"
STYLE="${STYLE:-"null"}"
SAMPLER="${SAMPLER:-"null"}"
CFG_SCALE="${CFG_SCALE:-"9"}"
SAMPLE_STEPS="${SAMPLE_STEPS:-"20"}"

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
                --argjson seed "${SEED}" \
                --argjson response_format "\"${RESPONSE_FORMAT}\"" \
                --argjson size "\"${SIZE}\"" \
                --argjson sampler "\"${SAMPLER}\"" \
                --argjson cfg_scale "${CFG_SCALE}" \
                --argjson sample_steps "${SAMPLE_STEPS}" \
                '{
                  n: $n,
                  seed: $seed,
                  response_format: $response_format,
                  size: $size,
                  sampler: $sampler,
                  cfg_scale: $cfg_scale,
                  sample_steps: $sample_steps
                } * .')"
    elif [[ "${STYLE}" != "null" ]]; then
      DATA="$(echo -n "${DATA}" | jq \
                --argjson n "${N}" \
                --argjson seed "${SEED}" \
                --argjson response_format "\"${RESPONSE_FORMAT}\"" \
                --argjson size "\"${SIZE}\"" \
                --argjson quality "\"${QUALITY}\"" \
                --argjson style "\"${STYLE}\"" \
                '{
                  n: $n,
                  seed: $seed,
                  response_format: $response_format,
                  size: $size,
                  quality: $quality,
                  style: $style
                } * .')"
    else
      DATA="$(echo -n "${DATA}" | jq \
                --argjson n "${N}" \
                --argjson seed "${SEED}" \
                --argjson response_format "\"${RESPONSE_FORMAT}\"" \
                --argjson size "\"${SIZE}\"" \
                --argjson quality "\"${QUALITY}\"" \
                '{
                  n: $n,
                  seed: $seed,
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
    echo "A: ${ANSWER}" >> "${LOG_FILE}"

    CONTENT="$(echo "${ANSWER}" | jq -r '.data')"
    if [[ "${CONTENT}" == "null" ]]; then
        CONTENT="[]"
    fi
    while IFS= read -r IMAGE; do
        TIME=$(date +%s)
        echo -n "${IMAGE}" | base64 -d 2>/dev/null > "/tmp/image_generate_${TIME}.png"
        if [[ -f "/tmp/image_generate_${TIME}.png" ]]; then
          if command -v feh > /dev/null; then
              feh "/tmp/image_generate_${TIME}.png"
          elif command -v open > /dev/null; then
              open "/tmp/image_generate_${TIME}.png"
          else
              echo "Generated image: /tmp/image_generate_${TIME}.png"
          fi
        else
            echo "Failed to generate image" && break
        fi
        sleep 1
    done < <(echo "${CONTENT}" | jq -r '.[] | .b64_json')

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
echo "SEED              : ${SEED}"
echo "RESPONSE_FORMAT   : ${RESPONSE_FORMAT}"
echo "SIZE              : ${SIZE}"
echo "QUALITY           : ${QUALITY}"
echo "STYLE             : ${STYLE}"
echo "SAMPLER           : ${SAMPLER} // OVERRIDE \"QUALITY\" and \"STYLE\" IF NOT NULL, ONE OF [euler_a, euler, heun, dpm2, dpm++2s_a, dpm++2mv2, ipndm, ipndm_v, lcm]"
echo "CFG_SCALE         : ${CFG_SCALE} // AVAILABLE FOR SAMPLER"
echo "SAMPLE_STEPS      : ${SAMPLE_STEPS} // AVAILABLE FOR SAMPLER"
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
