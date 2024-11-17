#!/bin/bash

#
# MIT license
# Copyright (c) 2024 llama-box authors
# SPDX-License-Identifier: MIT
#

#
# MIT license
# Copyright (c) 2023-2024 The ggml authors
# SPDX-License-Identifier: MIT
#

LOG_FILE=${LOG_FILE:-/dev/null}

API_URL="${API_URL:-http://127.0.0.1:8080}"

CHAT=(
    "Hello, Assistant."
    "Hello. How may I help you today?"
    "Please tell me the largest city in Europe."
    "Sure. The largest city in Europe is Moscow, the capital of Russia."
)

INSTRUCTION="### System: You are a helpful assistant."

trim() {
    shopt -s extglob
    set -- "${1##+([[:space:]])}"
    printf "%s" "${1%%+([[:space:]])}"
}

trim_trailing() {
    shopt -s extglob
    printf "%s" "${1%%+([[:space:]])}"
}

format_prompt() {
    echo -n "${INSTRUCTION}"
    printf "\n### Human: %s\n### Assistant: %s" "${CHAT[@]}" "$1"
}

tokenize() {
    curl \
        --silent \
        --request POST \
        --url "${API_URL}/tokenize" \
        --header "Content-Type: application/json" \
        --data-raw "$(jq -ns --arg content "$1" '{content:$content}')" |
        jq '.tokens[]'
}

N_PREDICT="${N_PREDICT:-"-1"}"
SEED="${SEED:-"-1"}"
STOP="${STOP:-"[\"\\n### Human:\"]"}"
TEMP="${TEMP:-"0.8"}"
TOP_P="${TOP_P:-"0.9"}"
TOP_K="${TOP_K:-"40"}"
N_KEEP=$(tokenize "${INSTRUCTION}" | wc -l)

completion() {
    PROMPT="$(trim_trailing "$(format_prompt "$1")")"
    DATA="$(echo -n "${PROMPT}" | jq -crs \
        --argjson n_predict "${N_PREDICT}" \
        --argjson seed "${SEED}" \
        --argjson stop "${STOP}" \
        --argjson temp "${TEMP}" \
        --argjson top_p "${TOP_P}" \
        --argjson top_k "${TOP_K}" \
        --argjson n_keep "${N_KEEP}" \
        '{
        prompt: .,
        n_predict: $n_predict,
        seed: $seed,
        stop: $stop,
        temperature: $temp,
        top_p: $top_p,
        top_k: $top_k,
        n_keep: $n_keep,
        cache_prompt: false,
        stream: true
      }')"
    echo "Q: ${DATA}" >>"${LOG_FILE}"

    ANSWER=''
    PRE_CONTENT=''
    START_TIME=$(date +%s)

    while IFS= read -r LINE; do
        echo "A: ${LINE}" >>"${LOG_FILE}"
        if [[ ! "${LINE}" = data:* ]]; then
            if [[ "${LINE}" =~ error:.* ]]; then
                LINE="${LINE:7}"
                echo "Error: ${LINE}"
            fi
            continue
        fi
        if [[ "${LINE}" =~ data:\ \[DONE\].* ]]; then
            break
        fi
        LINE="${LINE:5}"
        CONTENT="$(echo "${LINE}" | jq -cr '.content')"
        if [[ "${CONTENT}" == "null" ]]; then
            CONTENT=""
        fi
        if [[ "${PRE_CONTENT: -1}" == "\\" ]] && [[ "${CONTENT}" =~ ^b|n|r|t|\\|\'|\"$ ]]; then
            printf "\b "
            case "${CONTENT}" in
            b) printf "\b\b" ;;
            n) printf "\b\n" ;;
            r) printf "\b\r" ;;
            t) printf "\b\t" ;;
            \\) printf "\b\\" ;;
            \') printf "\b'" ;;
            \") printf "\b\"" ;;
            esac
            CONTENT=""
        fi
        PRE_CONTENT="${CONTENT}"
        printf "%s" "${CONTENT}"
        ANSWER+="${CONTENT}"
        TIMINGS="$(echo "${LINE}" | jq -cr '.timings')"
        if [[ "${TIMINGS}" != "null" ]]; then
            printf "\n------------------------"
            printf "\n- TTFT : %10.2fms  -" "$(echo "${TIMINGS}" | jq -cr '.prompt_ms')"
            printf "\n- TBT  : %10.2fms  -" "$(echo "${TIMINGS}" | jq -cr '.predicted_per_token_ms')"
            printf "\n- TPS  : %10.2f    -" "$(echo "${TIMINGS}" | jq -cr '.predicted_per_second')"
            DRAFTED_N="$(echo "${TIMINGS}" | jq -cr '.drafted_n')"
            if [[ "${DRAFTED_N}" != "null" ]]; then
                printf "\n- DT   : %10d    -" "${DRAFTED_N}"
                printf "\n- DTA  : %10.2f%%   -" "$(echo "${TIMINGS}" | jq -cr '.drafted_accepted_p*100')"
            fi
            ELAPSED=$(($(date +%s) - START_TIME))
            printf "\n- TC   : %10.2fs   -" "${ELAPSED}"
            printf "\n------------------------"
            break
        fi
    done < <(curl \
        --silent \
        --no-buffer \
        --request POST \
        --url "${API_URL}/completion" \
        --header "Content-Type: application/json" \
        --data-raw "${DATA}")

    printf "\n"

    CHAT+=("$1" "$(trim "$ANSWER")")
}

echo "====================================================="
echo "LOG_FILE  : ${LOG_FILE}"
echo "API_URL   : ${API_URL}"
echo "N_PREDICT : ${N_PREDICT}"
echo "SEED      : ${SEED}"
echo "STOP      : ${STOP}"
echo "TEMP      : ${TEMP}"
echo "TOP_P     : ${TOP_P}"
echo "TOP_K     : ${TOP_K}"
printf "=====================================================\n\n"

if [[ -f "${LOG_FILE}" ]]; then
    : >"${LOG_FILE}"
fi
if [[ ! -f "${LOG_FILE}" ]]; then
    touch "${LOG_FILE}"
fi

if [[ "${#@}" -ge 1 ]]; then
    echo "> ${*}"
    completion "${*}"
else
    while true; do
        read -r -e -p "> " QUESTION
        completion "${QUESTION}"
    done
fi
