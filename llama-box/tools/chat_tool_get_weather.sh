#!/bin/bash

#
# MIT license
# Copyright (c) 2025 llama-box authors
# SPDX-License-Identifier: MIT
#

function get_weather() {
    ARGS="${1}"
    ID="${2}"

    REQUEST=$(echo "${ARGS}" | jq -cr '.location' | cut -d',' -f1)
    RESPONSE="$(curl -s https://wttr.in/"${REQUEST}"?format="%C")"
    if [[ -z "${RESPONSE}" ]]; then
        MESSAGE="{\"role\":\"tool\",\"content\":\"{\\\"error\\\":\\\"Location not found.\\\"}\",\"tool_call_id\":\"${ID}\"}"
    else
        MESSAGE="{\"role\":\"tool\",\"content\":\"{\\\"weather\\\":\\\"${RESPONSE}\\\"}\",\"tool_call_id\":\"${ID}\"}"
    fi

    echo "${MESSAGE}"
}

function register_tool_get_weather() {
    TOOLNAMES+=("get_weather")
    TOOLS+=("{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get the weather of the given location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"}},\"required\":[\"location\"]}}}")
}

register_tool_get_weather
