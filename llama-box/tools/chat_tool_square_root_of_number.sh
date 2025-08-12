#!/bin/bash

#
# MIT license
# Copyright (c) 2025 llama-box authors
# SPDX-License-Identifier: MIT
#

function square_root_of_number() {
    ARGS="${1}"
    ID="${2}"

    INPUT_NUM=$(echo "${ARGS}" | jq -cr '.input_num')
    RESPONSE="$(echo "scale=4; sqrt(${INPUT_NUM})" | bc)"
    if [[ -z "${RESPONSE}" ]]; then
        MESSAGE="{\"role\":\"tool\",\"content\":\"{\\\"error\\\":\\\"Failed to calculate.\\\"}\",\"tool_call_id\":\"${ID}\"}"
    else
        MESSAGE="{\"role\":\"tool\",\"content\":\"{\\\"result\\\":\\\"${RESPONSE}\\\"}\",\"tool_call_id\":\"${ID}\"}"
    fi

    echo "${MESSAGE}"
}

function register_square_root_of_number() {
    TOOLNAMES+=("square_root_of_number")
    TOOLS+=("{\"type\":\"function\",\"function\":{\"name\":\"square_root_of_number\",\"description\":\"Output the square root of the number.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"input_num\":{\"type\":\"number\",\"description\":\"input_num is the radicand.\"}},\"required\":[\"input_num\"]}}}")
}

register_square_root_of_number
