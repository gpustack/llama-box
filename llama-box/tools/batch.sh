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
api_server="${API_SERVER:-http://127.0.0.1:8080}"
temp=${TEMP:-1}
top_p=${TOP_P:-0.95}
min_p=${MIN_P:-0.05}
top_k=${TOP_K:-40}
max_tokens=${MAX_TOKENS:-1024}
seed=${SEED:-$(date +%s)}

function request() {
  rm -rf /tmp/response_*.json
  
  cc=${1:-1}

  # start
  if command -v gdate >/dev/null 2>&1; then
    start_time=$(gdate +%s%N)
  else 
    start_time=$(date +%s%N)
  fi

  # requesting
  for ((i=0; i<cc; i++))
  do
    idx=$(echo "$i % ${#user_contents[@]}" | bc)
    curl -ks "${api_server}"/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"temperature\":${temp},\"top_p\":${top_p},\"min_p\":${min_p},\"top_k\":${top_k},\"max_tokens\":${max_tokens},\"seed\":${seed},\"messages\":[{\"role\":\"user\",\"content\":\"${user_contents[$idx]}\"}]}" > "/tmp/response_$i.json" &
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
  for ((i=0; i<cc; i++))
  do
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

echo "API_SERVER=${api_server} TEMP=${temp} TOP_P=${top_p} MIN_P=${min_p} TOP_K=${top_k} MAX_TOKENS=${max_tokens} SEED=${seed}"
echo " cc (ok) |    cost    | tokens (prefill, decoded) |  throughput  | avg. prefill | avg. decoded  "
echo "---------|------------|---------------------------|--------------|--------------|-------------- "
if [[ -n "${1:-}" ]]; then
  request "${1}"
else
  batchs=(1 16 12 8 6 5 4 3 2 1)
  for ((j=0; j<${#batchs[@]}; j++)); do
    if [[ $j == 0 ]]; then
      request "${batchs[$j]}" >/dev/null 2>&1
      continue;
    fi
    request "${batchs[$j]}"
  done
fi

