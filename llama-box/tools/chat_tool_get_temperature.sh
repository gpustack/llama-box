#!/bin/bash

#
# MIT license
# Copyright (c) 2025 llama-box authors
# SPDX-License-Identifier: MIT
#

function get_temperature() {
    ARGS="${1}"
    ID="${2}"

    REQUEST=$(echo "${ARGS}" | jq -cr '.location' | cut -d',' -f1)
    RESPONSE="$(curl -s https://wttr.in/"${REQUEST}"?format="%t")"
    if [[ -z "${RESPONSE}" ]]; then
        MESSAGE="{\"role\":\"tool\",\"content\":\"{\\\"error\\\":\\\"Location not found.\\\"}\",\"tool_call_id\":\"${ID}\"}"
    else
        MESSAGE="{\"role\":\"tool\",\"content\":\"{\\\"temperature\\\":\\\"${RESPONSE%%°C}\\\"}\",\"tool_call_id\":\"${ID}\"}"
    fi

    echo "${MESSAGE}"
}

function register_tool_get_temperature() {
    TOOLNAMES+=("get_temperature")
    TOOLS+=("{\"type\":\"function\",\"function\":{\"name\":\"get_temperature\",\"description\":\"Get the degrees Celsius temperature value of the given location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"}},\"required\":[\"location\"]}}}")
}

register_tool_get_temperature
