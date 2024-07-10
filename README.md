# llama-box

[![](https://img.shields.io/github/actions/workflow/status/thxcode/llama-box/ci.yml?label=ci)](https://github.com/thxcode/llama-box/actions)
[![](https://img.shields.io/github/license/thxcode/llama-box?label=license)](https://github.com/thxcode/llama-box#license)
[![](https://img.shields.io/github/downloads/thxcode/llama-box/total)](https://github.com/thxcode/llama-box/releases)

LLaMA box is a clean, pure API(without frontend assets) LLMs inference server rather
than [llama-server](https://github.com/ggerganov/llama.cpp/blob/master/examples/server).

## Usage

```shell
usage: llama-box [options]

general:

  -h,    --help, --usage          print usage and exit
         --version                show version and build info
  -m,    --model FILE             model path (default: models/7B/ggml-model-f16.gguf)
  -a,    --alias NAME             model name alias (default: unknown)
  -s,    --seed N                 RNG seed (default: -1, use random seed for < 0)
  -t,    --threads N              number of threads to use during generation (default: 8)
  -tb,   --threads-batch N        number of threads to use during batch and prompt processing (default: same as --threads)
  -lcs,  --lookup-cache-static FILE
                                  path to static lookup cache to use for lookup decoding (not updated by generation)
  -lcd,  --lookup-cache-dynamic FILE
                                  path to dynamic lookup cache to use for lookup decoding (updated by generation)
  -c,    --ctx-size N             size of the prompt context (default: 0, 0 = loaded from model)
  -n,    --predict N              number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled)
  -b,    --batch-size N           logical maximum batch size (default: 2048)
  -ub,   --ubatch-size N          physical maximum batch size (default: 512)
         --keep N                 number of tokens to keep from the initial prompt (default: 0, -1 = all)
         --chunks N               max number of chunks to process (default: -1, -1 = all)
  -fa,   --flash-attn             enable Flash Attention (default: disabled)
         --no-escape              do not process escape sequences
         --samplers SAMPLERS      samplers that will be used for generation in the order, separated by ';'
                                  (default: top_k;tfs_z;typical_p;top_p;min_p;temperature)
         --sampling-seq SEQUENCE  simplified sequence for samplers that will be used (default: kfypmt)
         --penalize-nl            penalize newline tokens (default: false)
         --temp N                 temperature (default: 0.8)
         --top-k N                top-k sampling (default: 40, 0 = disabled)
         --top-p N                top-p sampling (default: 0.9, 1.0 = disabled)
         --min-p N                min-p sampling (default: 0.1, 0.0 = disabled)
         --tfs N                  tail free sampling, parameter z (default: 1.0, 1.0 = disabled)
         --typical N              locally typical sampling, parameter p (default: 1.0, 1.0 = disabled)
         --repeat-last-n N        last n tokens to consider for penalize (default: 64, 0 = disabled, -1 = ctx_size)
         --repeat-penalty N       penalize repeat sequence of tokens (default: 1.0, 1.0 = disabled)
         --presence-penalty N     repeat alpha presence penalty (default: 0.0, 0.0 = disabled)
         --frequency-penalty N    repeat alpha frequency penalty (default: 0.0, 0.0 = disabled)
         --dynatemp-range N       dynamic temperature range (default: 0.0, 0.0 = disabled)
         --dynatemp-exp N         dynamic temperature exponent (default: 1.0)
         --mirostat N             use Mirostat sampling.
                                  Top K, Nucleus, Tail Free and Locally Typical samplers are ignored if used.
                                  (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)
         --mirostat-lr N          Mirostat learning rate, parameter eta (default: 0.1)
         --mirostat-ent N         Mirostat target entropy, parameter tau (default: 5.0)
  -l     --logit-bias TOKEN_ID(+/-)BIAS
                                  modifies the likelihood of token appearing in the completion,
                                  i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',
                                  or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'
         --grammar GRAMMAR        BNF-like grammar to constrain generations (see samples in grammars/ dir) (default: '')
         --grammar-file FILE      file to read grammar from
  -j,    --json-schema SCHEMA     JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any JSON object
                                  For schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead
         --rope-scaling {none,linear,yarn}
                                  RoPE frequency scaling method, defaults to linear unless specified by the model
         --rope-scale N           RoPE context scaling factor, expands context by a factor of N
         --rope-freq-base N       RoPE base frequency, used by NTK-aware scaling (default: loaded from model)
         --rope-freq-scale N      RoPE frequency scaling factor, expands context by a factor of 1/N
         --yarn-orig-ctx N        YaRN: original context size of model (default: 0 = model training context size)
         --yarn-ext-factor N      YaRN: extrapolation mix factor (default: -1.0, 0.0 = full interpolation)
         --yarn-attn-factor N     YaRN: scale sqrt(t) or attention magnitude (default: 1.0)
         --yarn-beta-fast N       YaRN: low correction dim or beta (default: 32.0)
         --yarn-beta-slow N       YaRN: high correction dim or alpha (default: 1.0)
  -gan,  --grp-attn-n N           group-attention factor (default: 1)
  -gaw,  --grp-attn-w N           group-attention width (default: 512.0)
  -nkvo, --no-kv-offload          disable KV offload
  -ctk,  --cache-type-k TYPE      KV cache data type for K (default: f16)
  -ctv,  --cache-type-v TYPE      KV cache data type for V (default: f16)
  -dt,   --defrag-thold N         KV cache defragmentation threshold (default: -1.0, < 0 - disabled)
  -np,   --parallel N             number of parallel sequences to decode (default: 1)
  -cb,   --cont-batching          enable continuous batching (a.k.a dynamic batching) (default: enabled)
         --mmproj FILE            path to a multimodal projector file for LLaVA. see examples/llava/README.md
         --mlock                  force system to keep model in RAM rather than swapping or compressing
         --no-mmap                do not memory-map model (slower load but may reduce pageouts if not using mlock)
         --numa TYPE              attempt optimizations that help on some NUMA systems
                                    - distribute: spread execution evenly over all nodes
                                    - isolate: only spawn threads on CPUs on the node that execution started on
                                    - numactl: use the CPU map provided by numactl
                                  if run without this previously, it is recommended to drop the system page cache before using this
                                  see https://github.com/ggerganov/llama.cpp/issues/1437
         --override-kv KEY=TYPE:VALUE
                                  advanced option to override model metadata by key. may be specified multiple times.
                                  types: int, float, bool, str. example: --override-kv tokenizer.ggml.add_bos_token=bool:false
         --lora FILE              apply LoRA adapter (implies --no-mmap)
         --lora-scaled FILE SCALE 
                                  apply LoRA adapter with user defined scaling S (implies --no-mmap)
         --lora-base FILE         optional model to use as a base for the layers modified by the LoRA adapter
         --control-vector FILE    add a control vector
         --control-vector-scaled FILE SCALE
                                  add a control vector with user defined scaling SCALE
         --control-vector-layer-range START END
                                  layer range to apply the control vector(s) to, start and end inclusive
  -ngl,  --gpu-layers N           number of layers to store in VRAM
  -sm,   --split-mode SPLIT_MODE  how to split the model across multiple GPUs, one of:
                                    - none: use one GPU only
                                    - layer (default): split layers and KV across GPUs
                                    - row: split rows across GPUs
  -ts,   --tensor-split SPLIT     fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1
  -mg,   --main-gpu N             the GPU to use for the model (with split-mode = none),
                                  or for intermediate results and KV (with split-mode = row) (default: 0)

server:

         --host HOST              ip address to listen (default: 127.0.0.1)
         --port PORT              port to listen (default: 8080)
  -to    --timeout N              server read/write timeout in seconds (default: 600)
         --threads-http N         number of threads used to process HTTP requests (default: -1)
         --system-prompt-file FILE
                                  set a file to load a system prompt (initial prompt of all slots), this is useful for chat applications
         --metrics                enable prometheus compatible metrics endpoint (default: disabled)
         --infill                 enable infill endpoint (default: disabled)
         --embeddings             enable embedding endpoint (default: disabled)
         --no-slots               disables slots monitoring endpoint (default: enabled)
         --slot-save-path PATH    path to save slot kv cache (default: disabled)
         --chat-template JINJA_TEMPLATE
                                  set custom jinja chat template (default: template taken from model's metadata)
                                  only commonly used templates are accepted:
                                  https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
         --chat-template-file FILE
                                  set a file to load a custom jinja chat template
  -sps,  --slot-prompt-similarity N
                                  how much the prompt of a request must match the prompt of a slot in order to use that slot (default: 0.50, 0.0 = disabled)
                                  
         --conn-idle N            server connection idle in seconds (default: 60)
         --conn-keepalive N       server connection keep-alive in seconds (default: 15)
  -tps   --tokens-per-second N    maximum number of tokens per second (default: 0, 0 = disabled, -1 = try to detect)

logging:

         --log-format {text,json} 
                                  log output format: json or text (default: json)
```

## API Endpoints

- **GET** `/health`: Returns the current state of the llama-box.
    + 503 -> `{"status": "loading model"}` if the model is still being loaded.
    + 500 -> `{"status": "error"}` if the model failed to load.
    + 200 -> `{"status": "ok", "slots_idle": 1, "slots_processing": 2 }` if the model is successfully loaded and the
      server is ready for further requests mentioned below.
    + 200 -> `{"status": "no slot available", "slots_idle": 0, "slots_processing": 32}` if no slots are currently
      available.
    + 503 -> `{"status": "no slot available", "slots_idle": 0, "slots_processing": 32}` if the query
      parameter `fail_on_no_slot` is provided and no slots are currently available.

- **GET** `/metrics`: Returns the Prometheus compatible metrics of the llama-box.
    + This endpoint is only available if the `--metrics` flag is enabled.
    + `llamacpp:prompt_tokens_total`: (Counter) Number of prompt tokens processed.
    + `llamacpp:prompt_seconds_total`: (Counter) Prompt process time.
    + `llamacpp:tokens_predicted_total`: (Counter) Number of generation tokens processed.
    + `llamacpp:tokens_predicted_seconds_total`: (Counter) Predict process time.
    + `llamacpp:prompt_tokens_seconds`: (Gauge) Average prompt throughput in tokens/s.
    + `llamacpp:predicted_tokens_seconds`: (Gauge) Average generation throughput in tokens/s.
    + `llamacpp:kv_cache_usage_ratio`: (Gauge) KV-cache usage. 1 means 100 percent usage.
    + `llamacpp:kv_cache_tokens`: (Gauge) KV-cache tokens.
    + `llamacpp:requests_processing`: (Gauge) Number of request processing.
    + `llamacpp:requests_deferred`: (Gauge) Number of request deferred.

- **GET** `/props`: Returns current server settings.

- **POST** `/infill`: Returns the completion of the given prompt.
    + This endpoint is only available if the `--infill` flag is enabled.

- **POST** `/tokenize`: Convert text to tokens.

- **POST** `/detokenize`: Convert tokens to text.

- **GET** `/slots`: Returns the current slots processing state.
    + This endpoint is only available if the `--no-slots` flag is no provided.

- **POST** `/slots/:id_slot?action={save|restore|erase}`: Operate specific slot via ID.
    + This endpoint is only available if the `--no-slots` flag is no provided and `--slot-save-path` is provided.

- **POST** `/completion`: Returns the completion of the given prompt.

- **GET** `/v1/models`: (OpenAI-compatible) Returns the list of available models,
  see https://platform.openai.com/docs/api-reference/models/list.

- **POST** `/v1/completions`: (OpenAI-compatible) Returns the completion of the given prompt,
  see https://platform.openai.com/docs/api-reference/completions/create.

- **POST** `/v1/chat/completions` (OpenAI-compatible) Returns the completion of the given prompt,
  see https://platform.openai.com/docs/api-reference/chat/create.
    + This endpoint is compatible with [OpenAI Chat Vision API](https://platform.openai.com/docs/guides/vision) when
      enabled `--mmproj` flag,
      see https://huggingface.co/xtuner/llava-phi-3-mini-gguf/tree/main. (Note: do not support link `url`, use base64
      encoded image
      instead.)

- **POST** `/v1/embeddings`: (OpenAI-compatible) Returns the embeddings of the given prompt,
  see https://platform.openai.com/docs/api-reference/embeddings/create.
    + This endpoint is only available if the `--embeddings` flag is enabled.

## License

MIT
