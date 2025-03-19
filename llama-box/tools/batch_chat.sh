#!/bin/bash

set -e

user_contents=(
    "Introduce China in at least 500 words."
    "Hello, please introduce yourself in at least 100 words."
    "Write a simple number guessing game in Python."
    "How to make an apple juice? Please write down the steps in detail."
    "Long long ago, there was a girl and a boy ... Now, tell me a story about a cat and a dog in at least 300 words."
    "I want to travel to Hong Kong. Are there any recommended attractions? Also, I live in New York. How can I get there?"
    "I want to use thread pools in Java programming, what issues do I need to pay attention to?"
    "Detailed analysis of the three major attention mechanisms in the Transformer architecture used by large models."
)
api_url="${API_URL:-http://127.0.0.1:8080}"
temp=${TEMP:-1}
top_p=${TOP_P:-0.95}
min_p=${MIN_P:-0.05}
top_k=${TOP_K:-40}
max_tokens=${MAX_TOKENS:-1024}
seed=${SEED:-$(date +%s)}
stream=${STREAM:-false}

function request() {
    rm -rf /tmp/request_*.json
    rm -rf /tmp/response_*.json

    tr="tr"
    if command -v gtr >/dev/null 2>&1; then
        tr="gtr"
    fi

    cc=${1:-1}
    ct="${2:-}"
    if [[ -n "$ct" ]]; then
        if [[ "${ct:0:1}" == "@" ]] && [[ -f "${ct:1}" ]]; then
            ct=$(cat "${ct:1}")
        elif [[ ${ct} =~ RANDOM_.* ]]; then
            cn=$(echo "${ct}" | awk -F'_' '{print $2}')
            ct=""
            for ((i = 0; i < $((cn)); i++)); do
                printf "Generating random content %.2f%%\r" "$(echo "scale=2; $i * 100 / $cn" | bc)"
                case $((RANDOM % 3)) in
                0) ct="${ct}$("${tr}" -dc '0-9' </dev/urandom | head -c1)" ;;
                1) ct="${ct}$("${tr}" -dc 'a-zA-Z' </dev/urandom | head -c1)" ;;
                2) ct="${ct}$(echo -ne "\u$(printf '%04x' "$((0x4e00 + RANDOM % 0x9fcc))")")" ;;
                esac
            done
            ct="{\"messages\":[{\"role\":\"user\",\"content\":\"${ct}\"}]}"
        else
            ct="{\"messages\":[{\"role\":\"user\",\"content\":\"${ct}\"}]}"
        fi
    fi

    # start
    if command -v gdate >/dev/null 2>&1; then
        start_time=$(gdate +%s%N)
    else
        start_time=$(date +%s%N)
    fi

    # requesting
    for ((i = 0; i < cc; i++)); do
        if [[ -z "$ct" ]]; then
            idx=$(echo "$i % ${#user_contents[@]}" | bc)
            ct="{\"messages\":[{\"role\":\"user\",\"content\":\"${user_contents[$idx]}\"}]}"
        fi

        data="$(echo -n "$ct" | jq -cr \
            --argjson temperature "${temp}" \
            --argjson top_p "${top_p}" \
            --argjson min_p "${min_p}" \
            --argjson top_k "${top_k}" \
            --argjson max_tokens "${max_tokens}" \
            --argjson seed "${seed}" \
            --argjson stream "${stream}" \
            '{
               temperature: $temperature,
               top_p: $top_p,
               min_p: $min_p,
               top_k: $top_k,
               max_tokens: $max_tokens,
               seed: $seed,
               stream: $stream,
             } * .')"
        echo "${data}" >"/tmp/request_$i.json"
        # normal
        if [[ "${stream}" != "true" ]]; then
            curl -ks --request POST \
                --url "${api_url}"/v1/chat/completions \
                --header "Content-Type: application/json" \
                --data "@/tmp/request_$i.json" >"/tmp/response_$i.json" &
            continue
        fi
        # stream
        # read chunk data from response,
        # pass all data until seeing the ".usage" field
        curl -ks --no-buffer --request POST \
            --url "${api_url}/v1/chat/completions" \
            --header "Content-Type: application/json" \
            --data "@/tmp/request_$i.json" | while IFS= read -r LINE; do
            if [[ ! "${LINE}" = data:* ]]; then
                if [[ "${LINE}" =~ error:.* ]]; then
                    LINE="${LINE:7}"
                    echo "${LINE}" >"/tmp/response_$i.json"
                fi
                continue
            fi
            if [[ "${LINE}" =~ data:\ \[DONE\].* ]]; then
                break
            fi
            LINE="${LINE:5}"
            echo "${LINE}" >"/tmp/response_$i.json"
        done &
    done
    wait

    # end
    if command -v gdate >/dev/null 2>&1; then
        end_time=$(gdate +%s%N)
    else
        end_time=$(date +%s%N)
    fi
    tt=$(((end_time - start_time) / 1000000))

    # observe
    oks=$cc
    ppss=0
    dpss=0
    pts=0
    dts=0
    for ((i = 0; i < cc; i++)); do
        pps=$(jq '.usage.prompt_tokens_per_second' "/tmp/response_$i.json")
        dps=$(jq '.usage.tokens_per_second' "/tmp/response_$i.json")
        pt=$(jq '.usage.prompt_tokens' "/tmp/response_$i.json")
        ct=$(jq '.usage.completion_tokens' "/tmp/response_$i.json")
        if [[ -n "${pps}" ]] && [[ "${pps}" != "null" ]]; then
            ppss=$(echo "$ppss + $pps" | bc)
        elif [[ -n "${pps}" ]]; then
            oks=$((oks - 1))
        fi
        if [[ -n "${dps}" ]] && [[ "${dps}" != "null" ]]; then
            dpss=$(echo "$dpss + $dps" | bc)
        fi
        pts=$((pts + pt))
        dts=$((dts + ct))
    done
    tts=$((pts + dts))

    # result
    tps=$(echo "scale=2; $tts * 1000 / $tt" | bc 2>/dev/null)
    avg_pps=$(echo "scale=2; $pts * 1000 / $tt / $oks" | bc 2>/dev/null)
    avg_dps=$(echo "scale=2; $dts * 1000 / $tt / $oks" | bc 2>/dev/null)
    if [[ "${ppss}" != "0" ]]; then
        avg_pps=$(echo "scale=2; $ppss / $oks" | bc 2>/dev/null)
    fi
    if [[ "${dpss}" != "0" ]]; then
        avg_dps=$(echo "scale=2; $dpss / $oks" | bc 2>/dev/null)
    fi
    printf " %2d (%2d) |%8d ms |%7d (%7d, %7d) |%9.2f tps |%9.2f tps |%9.2f tps \n" "$cc" "$oks" $tt $tts $pts $dts "$tps" "$avg_pps" "$avg_dps"
}

echo "STREAM=${stream} API_URL=${api_url} TEMP=${temp} TOP_P=${top_p} MIN_P=${min_p} TOP_K=${top_k} MAX_TOKENS=${max_tokens} SEED=${seed}"
echo " cc (ok) |    cost    | tokens (prefill, decoded) |  throughput  | avg. prefill | avg. decoded  "
echo "---------|------------|---------------------------|--------------|--------------|-------------- "
if [[ -n "${1:-}" ]]; then
    request "${1}" "${2:-}"
else
    batches=(1 16 12 8 4 1)
    for ((j = 0; j < ${#batches[@]}; j++)); do
        if [[ $j == 0 ]]; then
            request "${batches[$j]}" "${2:-}" >/dev/null 2>&1
            continue
        fi
        request "${batches[$j]}" "${2:-}"
    done
fi
