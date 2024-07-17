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
        --data-raw "$(jq -ns --arg content "$1" '{content:$content}')" \
    | jq '.tokens[]'
}

N_PREDICT="${N_PREDICT:-"256"}"
TEMPERATURE="${TEMPERATURE:-"0.2"}"
TOP_P="${TOP_P:-"0.9"}"
TOP_K="${TOP_K:-"40"}"
STOP="${STOP:-"[\"\\n### Human:\"]"}"
SEED="${SEED:-"-1"}"
N_KEEP=$(tokenize "${INSTRUCTION}" | wc -l)

completion() {
    PROMPT="$(trim_trailing "$(format_prompt "$1")")"
    DATA="$(echo -n "$PROMPT" | jq -Rs \
      --argjson n_predict "${N_PREDICT}" \
      --argjson temperature "${TEMPERATURE}" \
      --argjson top_p "${TOP_P}" \
      --argjson top_k "${TOP_K}" \
      --argjson stop "${STOP}" \
      --argjson seed "${SEED}" \
      --argjson n_keep "${N_KEEP}" \
      '{
        prompt: .,
        temperature: $temperature,
        top_k: $top_k,
        top_p: $top_p,
        n_predict: $n_predict,
        cache_prompt: false,
        n_keep: $n_keep,
        seed: $seed,
        stop: $stop,
        stream: true
    }')"
    echo "Q: ${DATA}" >> "${LOG_FILE}"

    ANSWER=''
    PRE_CONTENT=''

    while IFS= read -r LINE; do
        if [[ $LINE = data:* ]]; then
            echo "A: ${LINE}" >> "${LOG_FILE}"
            CONTENT="$(echo "${LINE:5}" | jq -r '.content')"
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
            TIMINGS="$(echo "${LINE:5}" | jq -r '.timings')"
            if [ "${TIMINGS}" != "null" ]; then
                printf "\n------------------------"
                printf "\n- TTFT : %10.2f ms -" "$(echo "${TIMINGS}" | jq -r '.prompt_ms')"
                printf "\n- TBT  : %10.2f ms -" "$(echo "${TIMINGS}" | jq -r '.predicted_per_token_ms')"
                printf "\n- TPS  : %10.2f    -" "$(echo "${TIMINGS}" | jq -r '.predicted_per_second')"
                printf "\n------------------------"
            fi
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
echo "LOG_FILE    : ${LOG_FILE}"
echo "API_URL     : ${API_URL}"
echo "TEMPERATURE : ${TEMPERATURE}"
echo "TOP_P       : ${TOP_P}"
echo "TOP_K       : ${TOP_K}"
echo "N_PREDICT   : ${N_PREDICT}"
echo "STOP        : ${STOP}"
echo "SEED        : ${SEED}"
printf "=====================================================\n\n"

if [[ -f "${LOG_FILE}" ]]; then
    rm -f "${LOG_FILE}"
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
