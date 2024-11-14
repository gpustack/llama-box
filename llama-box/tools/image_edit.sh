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
RESPONSE_FORMAT="b64_json"
SIZE="${SIZE:-"512x512"}"
QUALITY="${QUALITY:-"standard"}"
IMAGE="${IMAGE:-""}"
MASK="${MASK:-""}"
SAMPLER="${SAMPLER:-"null"}"
SEED="${SEED:-"null"}"
CFG_SCALE="${CFG_SCALE:-"9"}"
SAMPLE_STEPS="${SAMPLE_STEPS:-"20"}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-""}"

image_edit() {
    PROMPT="$(trim_trailing "$1")"
    if [[ "${IMAGE:0:1}" == "@" ]]; then
        IMAGE="${IMAGE:1}"
    fi
    if [[ ! -f "${IMAGE}" ]]; then
        echo "Image not found: ${IMAGE}" && exit 1
    fi
    if [[ -n "${MASK}" ]]; then
      if [[ "${MASK:0:1}" == "@" ]]; then
              MASK="${MASK:1}"
          fi
      if [[ ! -f "${MASK}" ]]; then
          echo "Mask not found: ${MASK}" && exit 1
      fi
    fi
    DATA="{\"prompt\":\"${PROMPT}\"}"
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
                --argjson image "\"${IMAGE}\"" \
                --argjson mask "\"${MASK}\"" \
                '{
                  n: $n,
                  response_format: $response_format,
                  size: $size,
                  sampler: $sampler,
                  seed: $seed,
                  cfg_scale: $cfg_scale,
                  sample_steps: $sample_steps,
                  negative_prompt: $negative_prompt,
                  image: $image,
                  mask: $mask
                } * .')"
    else
      DATA="$(echo -n "${DATA}" | jq \
                --argjson n "${N}" \
                --argjson response_format "\"${RESPONSE_FORMAT}\"" \
                --argjson size "\"${SIZE}\"" \
                --argjson quality "\"${QUALITY}\"" \
                --argjson image "\"${IMAGE}\"" \
                --argjson mask "\"${MASK}\"" \
                '{
                  n: $n,
                  response_format: $response_format,
                  size: $size,
                  quality: $quality,
                  image: $image,
                  mask: $mask
                } * .')"
    fi
    echo "Q: ${DATA}" >> "${LOG_FILE}"

    START_TIME=$(date +%s)
    if [[ "${SAMPLER}" != "null" ]]; then
      if [[ -n "${MASK}" ]]; then
        ANSWER="$(curl \
          --silent \
          --no-buffer \
          --request POST \
          --url "${API_URL}/v1/images/edits" \
          --form "prompt=${PROMPT}" \
          --form "n=${N}" \
          --form "response_format=${RESPONSE_FORMAT}" \
          --form "size=${SIZE}" \
          --form "sampler=${SAMPLER}" \
          --form "seed=${SEED}" \
          --form "cfg_scale=${CFG_SCALE}" \
          --form "sample_steps=${SAMPLE_STEPS}" \
          --form "negative_prompt=${NEGATIVE_PROMPT}" \
          --form "image=@${IMAGE}" \
          --form "mask=@${MASK}")"
      else
        ANSWER="$(curl \
          --silent \
          --no-buffer \
          --request POST \
          --url "${API_URL}/v1/images/edits" \
          --form "prompt=${PROMPT}" \
          --form "n=${N}" \
          --form "response_format=${RESPONSE_FORMAT}" \
          --form "size=${SIZE}" \
          --form "sampler=${SAMPLER}" \
          --form "seed=${SEED}" \
          --form "cfg_scale=${CFG_SCALE}" \
          --form "sample_steps=${SAMPLE_STEPS}" \
          --form "negative_prompt=${NEGATIVE_PROMPT}" \
          --form "image=@${IMAGE}")"
      fi
    elif [[ -n "${MASK}" ]]; then
      ANSWER="$(curl \
        --silent \
        --no-buffer \
        --request POST \
        --url "${API_URL}/v1/images/edits" \
        --form "prompt=${PROMPT}" \
        --form "n=${N}" \
        --form "response_format=${RESPONSE_FORMAT}" \
        --form "size=${SIZE}" \
        --form "quality=${QUALITY}" \
        --form "image=@${IMAGE}" \
        --form "mask=@${MASK}")"
    else
      ANSWER="$(curl \
        --silent \
        --no-buffer \
        --request POST \
        --url "${API_URL}/v1/images/edits" \
        --form "prompt=${PROMPT}" \
        --form "n=${N}" \
        --form "response_format=${RESPONSE_FORMAT}" \
        --form "size=${SIZE}" \
        --form "quality=${QUALITY}" \
        --form "image=@${IMAGE}")"
    fi
    echo "A: ${ANSWER}" >> "${LOG_FILE}"

    CONTENT="$(echo "${ANSWER}" | jq -r '.data')"
    if [[ "${CONTENT}" == "null" ]]; then
        CONTENT="[]"
    fi
    while IFS= read -r IMAGE; do
        TIME=$(date +%s)
        echo -n "${IMAGE}" | base64 -d 2>/dev/null > "/tmp/image_generate_${TIME}.png"
        if [[ -f "/tmp/image_generate_${TIME}.png" ]]; then
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
echo "RESPONSE_FORMAT   : ${RESPONSE_FORMAT}"
echo "SIZE              : ${SIZE}"
echo "QUALITY           : ${QUALITY}"
echo "IMAGE             : ${IMAGE}"
echo "MASK              : ${MASK}"
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
    image_edit "${*}"
else
    while true; do
        read -r -e -p "> " QUESTION
        image_edit "${QUESTION}"
    done
fi
