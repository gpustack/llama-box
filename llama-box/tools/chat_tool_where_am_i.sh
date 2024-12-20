#!/bin/bash

#
# MIT license
# Copyright (c) 2024 llama-box authors
# SPDX-License-Identifier: MIT
#

function where_am_i() {
    ARGS="${1}"
    ID="${2}"

    LOCATION="$(curl -s https://wttr.in/?format="%l")"

    MESSAGE="{\"role\":\"tool\",\"content\":\"{\\\"city\\\":\\\"${LOCATION}\\\"}\",\"tool_call_id\":\"${ID}\"}"
    echo "${MESSAGE}"
}

function register_tool_where_am_i() {
    TOOLNAMES+=("get_location")
    TOOLS+=("{\"type\":\"function\",\"function\":{\"name\":\"where_am_i\",\"description\":\"Get the city where I work.\",\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}}")
}

register_tool_where_am_i
