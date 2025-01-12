#!/bin/bash

#
# MIT license
# Copyright (c) 2024 llama-box authors
# SPDX-License-Identifier: MIT
#

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

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
PREVIEW="${PREVIEW:-"true"}"
PREVIEW_FASTER="${PREVIEW_FASTER:-"true"}"
SAMPLE_METHOD="${SAMPLE_METHOD:-"null"}"
SAMPLING_STEPS="${SAMPLING_STEPS:-"10"}"
SCHEDULE_METHOD="${SCHEDULE_METHOD:-"default"}"
SEED="${SEED:-"null"}"
GUIDANCE="${GUIDANCE:-"3.5"}"
STRENGTH="${STRENGTH:-"0.75"}"
CFG_SCALE="${CFG_SCALE:-"4.5"}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-""}"

parse() {
    TIME="${1:-$(date +%s)}"
    echo "A: ${LINE}" >>"${LOG_FILE}"
    if [[ ! "${LINE}" = data:* ]]; then
        if [[ "${LINE}" =~ error:.* ]]; then
            LINE="${LINE:7}"
            echo "Error: ${LINE}"
        fi
        return 0
    fi
    if [[ "${LINE}" =~ data:\ \[DONE\].* ]]; then
        return 0
    fi
    LINE="${LINE:5}"
    CONTENT="$(echo "${LINE}" | jq -cr '.data')"
    if [[ "${CONTENT}" == "null" ]]; then
        echo "Error: ${LINE}"
        return 1
    fi
    RESULT_JSON="/tmp/image_edit_${TIME}.json"
    printf "%s" "${LINE}" >"${RESULT_JSON}"
    printf "%i: %3.2f%%...\r" "$(jq -cr ".data[0] | .index" "${RESULT_JSON}")" "$(jq -cr ".data[0] | .progress" "${RESULT_JSON}")"
    if [[ "$(jq -cr ".data[0] | .b64_json" "${RESULT_JSON}")" == "null" ]]; then
        return 0
    fi
    RESULT_PNG_IDX="$(jq -cr ".data[0] | .index" "${RESULT_JSON}")"
    RESULT_PNG_PROGRESSED_STEPS="$(jq -cr ".data[0] | .progressed_steps" "${RESULT_JSON}")"
    RESULT_PNG_B64="/tmp/image_generate_${TIME}_${RESULT_PNG_IDX}_${RESULT_PNG_PROGRESSED_STEPS}.png.b64"
    if [[ ! -f "${RESULT_PNG_B64}" ]]; then
        touch "${RESULT_PNG_B64}"
        jq -cr ".data[0] | .b64_json" "${RESULT_JSON}" >"${RESULT_PNG_B64}"
    else
        jq -cr ".data[0] | .b64_json" "${RESULT_JSON}" >>"${RESULT_PNG_B64}"
    fi
    set +e
    RESULT_PNG="/tmp/image_edit_${TIME}_${RESULT_PNG_IDX}.png"
    if command -v gbase64 >/dev/null; then
        gbase64 -d "${RESULT_PNG_B64}" >"${RESULT_PNG}"
    else
        base64 -d "${RESULT_PNG_B64}" >"${RESULT_PNG}"
    fi
    if [[ "$(uname -s)" =~ Darwin ]]; then
        osascript "${ROOT_DIR}/image_view.scpt" "image_edit_${TIME}_${RESULT_PNG_IDX}.png" "file://${RESULT_PNG}" >/dev/null 2>&1
    fi
    if [[ "$(jq -cr ".data[0] | .finish_reason" "${RESULT_JSON}")" != "stop" ]]; then
        return 0
    fi
    printf "\n"
    echo "Generated image: ${RESULT_PNG}"
    set -e
    USAGE="$(jq -cr '.usage' "${RESULT_JSON}")"
    if [[ "${USAGE}" != "null" ]]; then
        printf "\n------------------------"
        printf "\n- TTP  : %10.2fms  -" "$(echo "${USAGE}" | jq -cr '.time_to_process_ms')"
        printf "\n- TPG  : %10.2fms  -" "$(echo "${USAGE}" | jq -cr '.time_per_generation_ms')"
        printf "\n- GPS  : %10.2f    -" "$(echo "${USAGE}" | jq -cr '.generation_per_second')"
        ELAPSED=$(($(date +%s) - START_TIME))
        printf "\n- TC   : %10.2fs   -" "${ELAPSED}"
        printf "\n------------------------\n"
    fi
    return 0
}

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
    if [[ "${SAMPLE_METHOD}" != "null" ]]; then
        DATA="$(echo -n "${DATA}" | jq -cr \
            --argjson n "${N}" \
            --argjson response_format "\"${RESPONSE_FORMAT}\"" \
            --argjson size "\"${SIZE}\"" \
            --argjson sample_method "\"${SAMPLE_METHOD}\"" \
            --argjson sampling_steps "${SAMPLING_STEPS}" \
            --argjson schedule_method "\"${SCHEDULE_METHOD}\"" \
            --argjson seed "${SEED}" \
            --argjson guidance "${GUIDANCE}" \
            --argjson strength "${STRENGTH}" \
            --argjson cfg_scale "${CFG_SCALE}" \
            --argjson negative_prompt "\"${NEGATIVE_PROMPT}\"" \
            --argjson image "\"${IMAGE}\"" \
            --argjson mask "\"${MASK}\"" \
            --argjson preview "${PREVIEW}" \
            --argjson preview_faster "${PREVIEW_FASTER}" \
            '{
                  n: $n,
                  response_format: $response_format,
                  size: $size,
                  sample_method: $sample_method,
                  sampling_steps: $sampling_steps,
                  schedule_method: $schedule_method,
                  seed: $seed,
                  guidance: $guidance,
                  strength: $strength,
                  cfg_scale: $cfg_scale,
                  negative_prompt: $negative_prompt,
                  image: $image,
                  mask: $mask,
                  stream: true,
                  stream_options_preview: $preview,
                  stream_options_preview_faster: $preview_faster,
                  stream_options_chunk_result: true,
                  stream_options_chunk_size: 1048576
                } * .')"
    else
        DATA="$(echo -n "${DATA}" | jq -cr \
            --argjson n "${N}" \
            --argjson response_format "\"${RESPONSE_FORMAT}\"" \
            --argjson size "\"${SIZE}\"" \
            --argjson quality "\"${QUALITY}\"" \
            --argjson image "\"${IMAGE}\"" \
            --argjson mask "\"${MASK}\"" \
            --argjson preview "${PREVIEW}" \
            --argjson preview_faster "${PREVIEW_FASTER}" \
            '{
                  n: $n,
                  response_format: $response_format,
                  size: $size,
                  quality: $quality,
                  image: $image,
                  mask: $mask,
                  stream: true,
                  stream_options_preview: $preview,
                  stream_options_preview_faster: $preview_faster,
                  stream_options_chunk_result: true,
                  stream_options_chunk_size: 1048576
                } * .')"
    fi
    echo "Q: ${DATA}" >>"${LOG_FILE}"

    START_TIME=$(date +%s)

    TIME=$(date +%s)
    set -e
    if [[ "${SAMPLE_METHOD}" != "null" ]]; then
        if [[ -n "${MASK}" ]]; then
            while IFS= read -r LINE; do
                if ! parse "${TIME}"; then
                    break
                fi
            done < <(curl \
                --silent \
                --no-buffer \
                --request POST \
                --url "${API_URL}/v1/images/edits" \
                --form "prompt=${PROMPT}" \
                --form "n=${N}" \
                --form "response_format=${RESPONSE_FORMAT}" \
                --form "size=${SIZE}" \
                --form "sample_method=${SAMPLE_METHOD}" \
                --form "sampling_steps=${SAMPLING_STEPS}" \
                --form "schedule_method=${SCHEDULE_METHOD}" \
                --form "seed=${SEED}" \
                --form "guidance=${GUIDANCE}" \
                --form "strength=${STRENGTH}" \
                --form "cfg_scale=${CFG_SCALE}" \
                --form "negative_prompt=${NEGATIVE_PROMPT}" \
                --form "image=@${IMAGE}" \
                --form "mask=@${MASK}" \
                --form "stream=true" \
                --form "stream_options_preview=${PREVIEW}" \
                --form "stream_options_preview_faster=${PREVIEW_FASTER}" \
                --form "stream_options_chunk_result=true" \
                --form "stream_options_chunk_size=1048576")
        else
            while IFS= read -r LINE; do
                if ! parse "${TIME}"; then
                    break
                fi
            done < <(curl \
                --silent \
                --no-buffer \
                --request POST \
                --url "${API_URL}/v1/images/edits" \
                --form "prompt=${PROMPT}" \
                --form "n=${N}" \
                --form "response_format=${RESPONSE_FORMAT}" \
                --form "size=${SIZE}" \
                --form "sample_method=${SAMPLE_METHOD}" \
                --form "sampling_steps=${SAMPLING_STEPS}" \
                --form "schedule_method=${SCHEDULE_METHOD}" \
                --form "seed=${SEED}" \
                --form "guidance=${GUIDANCE}" \
                --form "strength=${STRENGTH}" \
                --form "cfg_scale=${CFG_SCALE}" \
                --form "negative_prompt=${NEGATIVE_PROMPT}" \
                --form "image=@${IMAGE}" \
                --form "stream=true" \
                --form "stream_options_preview=${PREVIEW}" \
                --form "stream_options_preview_faster=${PREVIEW_FASTER}" \
                --form "stream_options_chunk_result=true" \
                --form "stream_options_chunk_size=1048576")
        fi
    elif [[ -n "${MASK}" ]]; then
        while IFS= read -r LINE; do
            if ! parse "${TIME}"; then
                break
            fi
        done < <(curl \
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
            --form "mask=@${MASK}" \
            --form "stream=true" \
            --form "stream_options_preview=${PREVIEW}" \
            --form "stream_options_preview_faster=${PREVIEW_FASTER}" \
            --form "stream_options_chunk_result=true" \
            --form "stream_options_chunk_size=1048576")
    else
        while IFS= read -r LINE; do
            if ! parse "${TIME}"; then
                break
            fi
        done < <(curl \
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
            --form "stream=true" \
            --form "stream_options_preview=${PREVIEW}" \
            --form "stream_options_preview_faster=${PREVIEW_FASTER}" \
            --form "stream_options_chunk_result=true" \
            --form "stream_options_chunk_size=1048576")
    fi
    set +e

    rm -f /tmp/image_edit_*.json
    printf "\n"
}

echo "====================================================="
echo "LOG_FILE          : ${LOG_FILE}"
echo "API_URL           : ${API_URL}"
echo "N                 : ${N}"
echo "RESPONSE_FORMAT   : ${RESPONSE_FORMAT}"
echo "SIZE              : ${SIZE}"
echo "QUALITY           : ${QUALITY} // ONE OF [standard, hd]"
echo "IMAGE             : ${IMAGE}"
echo "MASK              : ${MASK}"
echo "PREVIEW           : ${PREVIEW}"
echo "PREVIEW_FASTER    : ${PREVIEW_FASTER}"
echo "SAMPLE_METHOD     : ${SAMPLE_METHOD} // OVERRIDE \"QUALITY\" and \"STYLE\" IF NOT NULL, ONE OF [euler_a, euler, heun, dpm2, dpm++2s_a, dpm++2mv2, ipndm, ipndm_v, lcm, ddim_trailing]"
echo "SAMPLING_STEPS    : ${SAMPLING_STEPS} // AVAILABLE FOR SAMPLE_METHOD"
echo "SCHEDULE_METHOD   : ${SCHEDULE_METHOD} // AVAILABLE FOR SAMPLE_METHOD, ONE OF [default, discrete, karras, exponential, ays, gits]"
echo "SEED              : ${SEED} // AVAILABLE FOR SAMPLE_METHOD"
echo "GUIDANCE          : ${GUIDANCE} // AVAILABLE FOR SAMPLE_METHOD"
echo "STRENGTH          : ${STRENGTH} // AVAILABLE FOR SAMPLE_METHOD"
echo "CFG_SCALE         : ${CFG_SCALE} // AVAILABLE FOR SAMPLE_METHOD"
echo "NEGATIVE_PROMPT   : ${NEGATIVE_PROMPT} // AVAILABLE FOR SAMPLE_METHOD"
printf "=====================================================\n\n"

if [[ -f "${LOG_FILE}" ]]; then
    : >"${LOG_FILE}"
fi
if [[ ! -f "${LOG_FILE}" ]]; then
    touch "${LOG_FILE}"
fi

if [[ "${#@}" -ge 1 ]]; then
    echo "> ${*}"
    image_edit "${*}"
else
    while true; do
        read -r -e -p "> " PROMPT
        image_edit "${PROMPT}"
    done
fi
