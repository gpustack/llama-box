#!/bin/bash

#
# MIT license
# Copyright (c) 2024 llama-box authors
# SPDX-License-Identifier: MIT
#

function get_temperature() {
    ARGS="${1}"
    ID="${2}"

    LOCATION=$(echo "${ARGS}" | jq -cr '.location')
    TEMPERATURE="$(curl -s https://wttr.in/"${LOCATION}"?format="%t")"

    MESSAGE="{\"role\":\"tool\",\"content\":\"{\\\"temperature\\\":\\\"${TEMPERATURE%%°C}\\\"}\",\"tool_call_id\":\"${ID}\"}"
    echo "${MESSAGE}"
}

function register_tool_get_temperature() {
    TOOLNAMES+=("get_temperature")
    TOOLS+=("{\"type\":\"function\",\"function\":{\"name\":\"get_temperature\",\"description\":\"Get the degrees Celsius temperature value of the given location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"}},\"required\":[\"location\"]}}}")
}

register_tool_get_temperature
