#!/bin/bash

#
# MIT license
# Copyright (c) 2025 llama-box authors
# SPDX-License-Identifier: MIT
#

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

LOG_FILE=${LOG_FILE:-/dev/null}

API_URL="${API_URL:-http://127.0.0.1:8080}"

MESSAGES=(
    "{\"role\":\"system\",\"content\":\"Today is $(date +"%Y-%m-%d").\nYou are a helpful assistant.\"}"
)

TOOLNAMES=()
TOOLS=()
for file in "${ROOT_DIR}/"*; do
    if [[ -f "${file}" ]] && [[ "${file}" =~ .*/chat_tool_.*\.sh ]]; then
        # shellcheck disable=SC1090
        source "${file}"
    fi
done

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
SEED="${SEED:-"$(date +%s)"}"
STOP="${STOP:-"null"}"
TEMP="${TEMP:-"1"}"
TOP_P="${TOP_P:-"0.95"}"
MAX_TOKENS_PER_SECOND="${MAX_TOKENS_PER_SECOND:-"0"}"

chat_completion() {
    PROMPT="$(trim_trailing "$1")"
    if [[ -z "${PROMPT}" ]]; then
        return
    fi
    if [[ "${PROMPT:0:1}" == "@" ]] && [[ -f "${PROMPT:1}" ]]; then
        DATA="$(cat "${PROMPT:1}")"
        while IFS= read -r LINE; do
            MESSAGES+=("${LINE}")
        done < <(echo "${DATA}" | jq -cr '.messages[]')
    else
        DATA="{\"messages\":[$(format_messages){\"role\":\"user\",\"content\":\"${PROMPT}\"}]}"
        MESSAGES+=("{\"role\":\"user\",\"content\":\"$PROMPT\"}")
    fi
    while true; do
        DATA="$(echo -n "${DATA}" | jq -cr \
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
            --argjson tools "$(printf '%s\n' "${TOOLS[@]}" | jq -cs .)" \
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
                top_p: $top_p,
                tools: $tools,
                parallel_tool_calls: false
              } * .')"
        echo "Q: ${DATA}" >>"${LOG_FILE}"
        echo "${DATA}" >/tmp/request.json

        TOOL_CALLS=''
        TOOL_RESULTS=()
        CONTENT=''
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
            TOOL_CALLS="$(echo "${LINE}" | jq -cr '.choices[0].delta.tool_calls')"
            if [[ "${TOOL_CALLS}" != "null" ]]; then
                while IFS= read -r TOOL_CALL; do
                    ID="$(echo "${TOOL_CALL}" | jq -cr '.id')"
                    FUNC_NAME="$(echo "${TOOL_CALL}" | jq -cr '.function.name')"
                    FUNC_ARGS="$(echo "${TOOL_CALL}" | jq -cr '.function.arguments')"
                    printf "\n🛠️: calling %s %s %s\r" "${FUNC_NAME}" "${FUNC_ARGS}" "${ID}"
                    RESULT=$("${FUNC_NAME}" "${FUNC_ARGS}" "${ID}")
                    printf "\n🛠️: %s\n" "${RESULT}"
                    TOOL_RESULTS+=("${RESULT}")
                done < <(jq -cr '.[]' <<<"${TOOL_CALLS}")
            else
                TOOL_CALLS=''
            fi
            CONTENT_SEG="$(
                echo "${LINE}" | jq -cr '.choices[0].delta.content'
                echo -n "#"
            )"
            CONTENT_SEG="${CONTENT_SEG:0:${#CONTENT_SEG}-2}"
            if [[ "${CONTENT_SEG}" != "null" ]]; then
                if [[ "${PRE_CONTENT: -1}" == "\\" ]] && [[ "${CONTENT_SEG}" =~ ^b|n|r|t|\\|\'|\"$ ]]; then
                    printf "\b "
                    case "${CONTENT_SEG}" in
                    b) printf "\b\b" ;;
                    n) printf "\b\n" ;;
                    r) printf "\b\r" ;;
                    t) printf "\b\t" ;;
                    \\) printf "\b\\" ;;
                    \') printf "\b'" ;;
                    \") printf "\b\"" ;;
                    esac
                    CONTENT_SEG=""
                else
                    PRE_CONTENT="${CONTENT_SEG}"
                    printf "%s" "${CONTENT_SEG}"
                fi
                CONTENT+="${CONTENT_SEG}"
            fi
            USAGE="$(echo "${LINE}" | jq -cr '.usage')"
            if [[ "${USAGE}" != "null" ]]; then
                printf "\n------------------------"
                printf "\n- TTFT : %10.2fms  -" "$(echo "${USAGE}" | jq -cr '.time_to_first_token_ms')"
                printf "\n- TBT  : %10.2fms  -" "$(echo "${USAGE}" | jq -cr '.time_per_output_token_ms')"
                printf "\n- TPS  : %10.2f    -" "$(echo "${USAGE}" | jq -cr '.tokens_per_second')"
                DRAFTED_N="$(echo "${USAGE}" | jq -cr '.draft_tokens')"
                if [[ "${DRAFTED_N}" != "null" ]]; then
                    printf "\n- DT   : %10d    -" "${DRAFTED_N}"
                    printf "\n- DTA  : %10.2f%%   -" "$(echo "${USAGE}" | jq -cr '.draft_tokens_acceptance*100')"
                fi
                ELAPSED=$(($(date +%s) - START_TIME))
                printf "\n- TC   : %10.2fs   -" "${ELAPSED}"
                printf "\n------------------------\n"
                break
            fi
        done < <(curl \
            --silent \
            --no-buffer \
            --request POST \
            --url "${API_URL}/v1/chat/completions" \
            --header "Content-Type: application/json" \
            --header "X-Request-Tokens-Per-Second: ${MAX_TOKENS_PER_SECOND}" \
            --data @/tmp/request.json)

        printf "\n"

        if [[ -n "${TOOL_CALLS}" ]]; then
            MESSAGES+=("{\"role\":\"assistant\",\"content\":null,\"tool_calls\":$TOOL_CALLS}")
        fi
        if [[ -n "${CONTENT}" ]]; then
            MESSAGES+=("{\"role\":\"assistant\",\"content\":$(jq -Rs . <<<"${CONTENT}")}")
        fi
        if [[ "${#TOOL_RESULTS[@]}" -gt 0 ]]; then
            MESSAGES+=("${TOOL_RESULTS[@]}")
            DATA="{\"messages\":$(printf '%s\n' "${MESSAGES[@]}" | jq -cs .)}"
        else
            break
        fi
    done
}

if [[ "${TOOLS_WITH:-"false"}" == "false" ]]; then
    TOOLNAMES=()
    TOOLS=()
fi

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
echo "TOOLS             : $(printf '%s\n' "${TOOLNAMES[@]}" | jq -R . | jq -cs .)"
printf "=====================================================\n\n"

if [[ -f "${LOG_FILE}" ]]; then
    : >"${LOG_FILE}"
fi
if [[ ! -f "${LOG_FILE}" ]]; then
    touch "${LOG_FILE}"
fi

if [[ "${#@}" -ge 1 ]]; then
    echo "> ${*}"
    chat_completion "${*}"
else
    while true; do
        read -r -e -p "> " PROMPT
        if [[ "${PROMPT}" == "exit" || "${PROMPT}" == "quit" ]]; then
            break
        fi
        chat_completion "${PROMPT}"
    done
fi
