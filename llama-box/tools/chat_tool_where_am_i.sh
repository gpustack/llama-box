#!/bin/bash

#
# MIT license
# Copyright (c) 2025 llama-box authors
# SPDX-License-Identifier: MIT
#

function where_am_i() {
    ARGS="${1}"
    ID="${2}"

    RESPONSE="$(curl -s https://wttr.in/?format="%l")"
    if [[ -z "${RESPONSE}" ]]; then
        MESSAGE="{\"role\":\"tool\",\"content\":\"{\\\"error\\\":\\\"City not found.\\\"}\",\"tool_call_id\":\"${ID}\"}"
    else
        MESSAGE="{\"role\":\"tool\",\"content\":\"{\\\"city\\\":\\\"${RESPONSE}\\\"}\",\"tool_call_id\":\"${ID}\"}"
    fi

    echo "${MESSAGE}"
}

function register_tool_where_am_i() {
    TOOLNAMES+=("get_location")
    TOOLS+=("{\"type\":\"function\",\"function\":{\"name\":\"where_am_i\",\"description\":\"Get the city where I am living or working.\",\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}}")
}

register_tool_where_am_i
