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

MESSAGES=(
  "{\"role\":\"system\",\"content\":\"You are a helpful assistant.\"}"
)

trim() {
    shopt -s extglob
    set -- "${1##+([[:space:]])}"
    printf "%s" "${1%%+([[:space:]])}"
}

trim_trailing() {
    shopt -s extglob
    printf "%s" "${1%%+([[:space:]])}"
}

format_messages() {
    printf "%s," "${MESSAGES[@]}"
}

FREQUENCY_PENALTY="${FREQUENCY_PENALTY:-"0.0"}"
LOGPROBS="${LOGPROBS:-"false"}"
TOP_LOGPROBS="${TOP_LOGPROBS:-"null"}"
MAX_TOKENS="${MAX_TOKENS:-"null"}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-"0.0"}"
RESPONSE_FORMAT="${RESPONSE_FORMAT:-"text"}"
SEED="${SEED:-"null"}"
STOP="${STOP:-"null"}"
TEMP="${TEMP:-"1"}"
TOP_P="${TOP_P:-"1"}"

chat_completion() {
    PROMPT="$(trim_trailing "$1")"
    if [[ "${PROMPT:0:1}" == "@" ]] && [[ -f "${PROMPT:1}" ]]; then
      DATA="$(cat "${PROMPT:1}")"
    else
      DATA="{\"messages\": [$(format_messages){\"role\":\"user\",\"content\":\"${PROMPT}\"}]}"
    fi
    DATA="$(echo -n "${DATA}" | jq \
      --argjson frequency_penalty "${FREQUENCY_PENALTY}" \
      --argjson logprobs "${LOGPROBS}" \
      --argjson top_logprobs "${TOP_LOGPROBS}" \
      --argjson max_tokens "${MAX_TOKENS}" \
      --argjson presence_penalty "${PRESENCE_PENALTY}" \
      --argjson response_format "{\"type\":\"${RESPONSE_FORMAT}\"}" \
      --argjson seed "${SEED}" \
      --argjson stop "${STOP}" \
      --argjson temp "${TEMP}" \
      --argjson top_p "${TOP_P}" \
      '{
        frequency_penalty: $frequency_penalty,
        logprobs: $logprobs,
        top_logprobs: $top_logprobs,
        max_tokens: $max_tokens,
        n: 1,
        presence_penalty: $presence_penalty,
        response_format: $response_format,
        seed: $seed,
        stop: $stop,
        stream: true,
        stream_options: {include_usage: true},
        temperature: $temp,
        top_p: $top_p
      } * .')"
    echo "Q: ${DATA}" >> "${LOG_FILE}"

    ANSWER=''
    PRE_CONTENT=''
    START_TIME=$(date +%s)

    while IFS= read -r LINE; do
        if [[ "${LINE}" = data:* ]]; then
            echo "A: ${LINE}" >> "${LOG_FILE}"
            if [[ "${LINE}" == "data: [DONE]" ]]; then
                break
            fi
            LINE="${LINE:5}"
            CONTENT="$(echo "${LINE}" | jq -r '.choices[0].delta.content')"
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
            USAGE="$(echo "${LINE}" | jq -r '.usage')"
            if [[ "${USAGE}" != "null" ]]; then
                printf "\n------------------------"
                printf "\n- TTFT : %10.2fms  -" "$(echo "${USAGE}" | jq -r '.time_to_first_token_ms')"
                printf "\n- TBT  : %10.2fms  -" "$(echo "${USAGE}" | jq -r '.time_per_output_token_ms')"
                printf "\n- TPS  : %10.2f    -" "$(echo "${USAGE}" | jq -r '.tokens_per_second')"
                DRAFTED_N="$(echo "${USAGE}" | jq -r '.draft_tokens')"
                if [[ "${DRAFTED_N}" != "null" ]]; then
                    printf "\n- DT   : %10d    -" "${DRAFTED_N}"
                    printf "\n- DTA  : %10.2f%%   -" "$(echo "${USAGE}" | jq -r '.draft_tokens_acceptance*100')"
                fi
                ELAPSED=$(($(date +%s) - START_TIME))
                printf "\n- TC   : %10.2fs   -" "${ELAPSED}"
                printf "\n------------------------"
                break
            fi
        fi
    done < <(curl \
        --silent \
        --no-buffer \
        --request POST \
        --url "${API_URL}/v1/chat/completions" \
        --header "Content-Type: application/json" \
        --data-raw "${DATA}")

    printf "\n"

    MESSAGES+=(
        "{\"role\":\"user\",\"content\":\"$PROMPT\"}"
        "{\"role\":\"assistant\",\"content\":\"$ANSWER\"}")
}

echo "====================================================="
echo "LOG_FILE          : ${LOG_FILE}"
echo "API_URL           : ${API_URL}"
echo "FREQUENCY_PENALTY : ${FREQUENCY_PENALTY}"
echo "LOGPROBS          : ${LOGPROBS}"
echo "TOP_LOGPROBS      : ${TOP_LOGPROBS}"
echo "MAX_TOKENS        : ${MAX_TOKENS}"
echo "PRESENCE_PENALTY  : ${PRESENCE_PENALTY}"
echo "RESPONSE_FORMAT   : ${RESPONSE_FORMAT}"
echo "SEED              : ${SEED}"
echo "STOP              : ${STOP}"
echo "TEMP              : ${TEMP}"
echo "TOP_P             : ${TOP_P}"
printf "=====================================================\n\n"

if [[ -f "${LOG_FILE}" ]]; then
    : > "${LOG_FILE}"
fi
if [[ ! -f "${LOG_FILE}" ]]; then
    touch "${LOG_FILE}"
fi

if [[ "${#@}" -ge 1 ]]; then
    echo "> ${*}"
    chat_completion "${*}"
else
    while true; do
        read -r -e -p "> " QUESTION
        chat_completion "${QUESTION}"
    done
fi
