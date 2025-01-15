#!/bin/bash

#
# MIT license
# Copyright (c) 2025 llama-box authors
# SPDX-License-Identifier: MIT
#

function current_date_time() {
    ARGS="${1}"
    ID="${2}"

    MESSAGE="{\"role\":\"tool\",\"content\":\"{\\\"value\\\":\\\"$(date +"%Y-%m-%dT%H:%M:%S%z")\\\"}\",\"tool_call_id\":\"${ID}\"}"

    echo "${MESSAGE}"
}

function register_tool_current_date_time() {
    TOOLNAMES+=("current_date_time")
    TOOLS+=("{\"type\":\"function\",\"function\":{\"name\":\"current_date_time\",\"description\":\"Obtain the current system date and time, following the RFC3339 standard. Example: 2024-12-30T16:44:58+08:00.\",\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}}")
}

register_tool_current_date_time
