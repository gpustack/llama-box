#!/bin/bash

#
# MIT license
# Copyright (c) 2024 llama-box authors
# SPDX-License-Identifier: MIT
#

function get_weather() {
    ARGS="${1}"
    ID="${2}"

    LOCATION=$(echo "${ARGS}" | jq -cr '.location')
    WEATHER="$(curl -s https://wttr.in/"${LOCATION}"?format="%C")"

    MESSAGE="{\"role\":\"tool\",\"content\":\"{\\\"weather\\\":\\\"${WEATHER}\\\"}\",\"tool_call_id\":\"${ID}\"}"
    echo "${MESSAGE}"
}

function register_tool_get_weather() {
    TOOLNAMES+=("get_weather")
    TOOLS+=("{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get the weather of the given location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"}},\"required\":[\"location\"]}}}")
}

register_tool_get_weather
