#pragma once

#include <cstdarg>
#include <utility>

#include "llama.cpp/common/common.h"
#include "llama.cpp/common/json-schema-to-grammar.h"
#define JSON_ASSERT GGML_ASSERT
#include "llama.cpp/common/json.hpp"
#include "llama.cpp/ggml/include/ggml.h"
#include "llama.cpp/include/llama.h"
#include "rpcserver.hpp"
#include "stablediffusion.hpp"

// version
extern const char *LLAMA_BOX_COMMIT;
extern int LLAMA_BOX_BUILD_NUMBER;
extern const char *LLAMA_BOX_BUILD_VERSION;
extern const char *LLAMA_BOX_BUILD_COMPILER;
extern const char *LLAMA_BOX_BUILD_TARGET;
extern const char *LLAMA_CPP_COMMIT;
extern int LLAMA_CPP_BUILD_NUMBER;
extern const char *STABLE_DIFFUSION_CPP_COMMIT;
extern int STABLE_DIFFUSION_CPP_BUILD_NUMBER;

using json = nlohmann::json;

struct llama_box_params {
    common_params gparams;
    rpcserver_params rparams;
    stablediffusion_params sdparams;

    bool cache_prompt        = true;
    bool endpoint_infill     = false;
    bool endpoint_images     = false;
    int32_t conn_idle        = 60; // connection idle in seconds
    int32_t conn_keepalive   = 15; // connection keep-alive in seconds
    int32_t n_tps            = 0;  // maximum number of tokens per seconds
    int32_t lookup_ngram_min = 0;  // minimum n-gram size for lookup cache
};

static int unknown(const char *flag) {
    fprintf(stderr, "Unknown argument: %s\n", flag);
    return 1;
}

static int missing(const char *flag) {
    throw std::invalid_argument("Missing argument: " + std::string(flag));
    return 1;
}

static int invalid(const char *flag) {
    throw std::invalid_argument("Invalid argument: " + std::string(flag));
    return 1;
}

#ifdef __GNUC__
#ifdef __MINGW32__
#define LLAMA_COMMON_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define LLAMA_COMMON_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define LLAMA_COMMON_ATTRIBUTE_FORMAT(...)
#endif

static void llama_box_params_print_usage(int, char **argv, const llama_box_params &bparams) {
    struct opt {
        LLAMA_COMMON_ATTRIBUTE_FORMAT(4, 5)

        opt(const std::string &tags, const char *args, const char *desc, ...)
            : tags(tags), args(args), desc(desc) {
            va_list args_list;
            va_start(args_list, desc);
            char buffer[1024];
            vsnprintf(buffer, sizeof(buffer), desc, args_list);
            va_end(args_list);
            this->desc = buffer;
        }

        opt(const std::string &grp)
            : grp(grp) {
        }

        std::string tags;
        std::string args;
        std::string desc;
        std::string grp;
    };

    const auto &params   = bparams.gparams;
    const auto &rparams  = bparams.rparams;
    const auto &sampling = params.sampling;
    const auto &sdparams = bparams.sdparams;

    std::string default_sampler_type_chars;
    std::string default_sampler_type_names;
    for (const auto &sampler : sampling.samplers) {
        default_sampler_type_chars += common_sampler_type_to_chr(sampler);
        default_sampler_type_names += common_sampler_type_to_str(sampler) + ";";
    }
    default_sampler_type_names.pop_back();

    std::string sd_sampler_type_names;
    for (int m = 0; m < N_SAMPLE_METHODS; m++) {
        sd_sampler_type_names += std::string(sd_sample_method_to_argument(sample_method_t(m))) + ";";
    }
    sd_sampler_type_names.pop_back();
    std::string sd_scheduler_names;
    for (int d = 0; d < N_SCHEDULES; d++) {
        sd_scheduler_names += std::string(sd_schedule_to_argument(schedule_t(d))) + ";";
    }
    sd_scheduler_names.pop_back();

    std::string default_dry_sequence_breaker_names;
    for (const auto &breaker : sampling.dry_sequence_breakers) {
        default_dry_sequence_breaker_names += breaker + ";";
    }
    default_dry_sequence_breaker_names.pop_back();

    // clang-format off
    std::vector<opt> opts;
    // general //
    opts.push_back({ "general" });
    opts.push_back({ "general",                            "-h,    --help, --usage",                        "print usage and exit" });
    opts.push_back({ "general",                            "       --version",                              "print version and exit" });
    opts.push_back({ "general",                            "       --system-info",                          "print system info and exit" });
    if (llama_supports_gpu_offload()) {
        opts.push_back({ "general",                        "       --list-devices",                         "print list of available devices and exit" });
    }
    opts.push_back({ "general",                            "-v,    --verbose, --log-verbose",               "set verbosity level to infinity (i.e. log all messages, useful for debugging)" });
    opts.push_back({ "general",                            "-lv,   --verbosity, --log-verbosity V",         "set the verbosity threshold, messages with a higher verbosity will be ignored" });
    opts.push_back({ "general",                            "       --log-colors",                           "enable colored logging" });
    // general //
    // server //
    opts.push_back({ "server" });
    opts.push_back({ "server",                             "       --host HOST",                            "ip address to listen (default: %s)", params.hostname.c_str() });
    opts.push_back({ "server",                             "       --port PORT",                            "port to listen (default: %d)", params.port });
    opts.push_back({ "server",                             "-to    --timeout N",                            "server read/write timeout in seconds (default: %d)", params.timeout_read });
    opts.push_back({ "server",                             "       --threads-http N",                       "number of threads used to process HTTP requests (default: %d)", params.n_threads_http });
    opts.push_back({ "server",                             "       --conn-idle N",                          "server connection idle in seconds (default: %d)", bparams.conn_idle });
    opts.push_back({ "server",                             "       --conn-keepalive N",                     "server connection keep-alive in seconds (default: %d)", bparams.conn_keepalive });
    opts.push_back({ "server",                             "-m,    --model FILE",                           "model path (default: %s)", DEFAULT_MODEL_PATH });
    opts.push_back({ "server",                             "-a,    --alias NAME",                           "model name alias (default: %s)", params.model_alias.c_str() });
    opts.push_back({ "server",                             "       --lora FILE",                            "apply LoRA adapter (implies --no-mmap)" });
    opts.push_back({ "server",                             "       --lora-scaled FILE SCALE",               "apply LoRA adapter with user defined scaling S (implies --no-mmap)" });
    opts.push_back({ "server",                             "       --lora-init-without-apply",              "load LoRA adapters without applying them (apply later via POST /lora-adapters) (default: %s)", params.lora_init_without_apply ? "enabled" : "disabled" });
    opts.push_back({ "server",                             "-s,    --seed N",                               "RNG seed (default: %d, use random seed for %d)", sampling.seed, LLAMA_DEFAULT_SEED });
    if (llama_supports_gpu_offload()) {
        opts.push_back({ "server",                         "-mg,   --main-gpu N",                           "the GPU to use for the model (default: %d)", params.main_gpu });
    }
    opts.push_back({ "server",                             "-fa,   --flash-attn",                           "enable Flash Attention (default: %s)", params.flash_attn ? "enabled" : "disabled" });
    opts.push_back({ "server",                             "       --metrics",                              "enable prometheus compatible metrics endpoint (default: %s)", params.endpoint_metrics ? "enabled" : "disabled" });
    opts.push_back({ "server",                             "       --infill",                               "enable infill endpoint (default: %s)", bparams.endpoint_infill? "enabled" : "disabled" });
    opts.push_back({ "server",                             "       --embeddings",                           "enable embedding endpoint (default: %s)", params.embedding ? "enabled" : "disabled" });
    opts.push_back({ "server",                             "       --images",                               "enable image endpoint (default: %s)", bparams.endpoint_images ? "enabled" : "disabled" });
    opts.push_back({ "server",                             "       --rerank",                               "enable reranking endpoint (default: %s)", params.reranking ? "enabled" : "disabled" });
    opts.push_back({ "server",                             "       --slots",                                "enable slots monitoring endpoint (default: %s)", params.endpoint_slots ? "enabled" : "disabled" });
    if (llama_supports_rpc()) {
        opts.push_back({ "server",                         "       --rpc SERVERS",                          "comma separated list of RPC servers" });
    }
    // server // completion //
    opts.push_back({ "server/completion" });
    if (llama_supports_gpu_offload()) {
        opts.push_back({ "server/completion",              "-dev,  --device <dev1,dev2,...>",               "comma-separated list of devices to use for offloading (none = don't offload)\n"
                                                                                                            "use --list-devices to see a list of available devices"});
        opts.push_back({ "server/completion",              "-ngl,  --gpu-layers,  --n-gpu-layers N",        "number of layers to store in VRAM" });
        opts.push_back({ "server/completion",              "-sm,   --split-mode SPLIT_MODE",                "how to split the model across multiple GPUs, one of:\n"
                                                                                                            "  - none: use one GPU only\n"
                                                                                                            "  - layer (default): split layers and KV across GPUs\n"
                                                                                                            "  - row: split rows across GPUs, store intermediate results and KV in --main-gpu" });
        opts.push_back({ "server/completion",              "-ts,   --tensor-split SPLIT",                   "fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1" });
    }
    opts.push_back({ "server/completion",                  "       --override-kv KEY=TYPE:VALUE",           "advanced option to override model metadata by key. may be specified multiple times.\n"
                                                                                                            "types: int, float, bool, str. example: --override-kv tokenizer.ggml.add_bos_token=bool:false" });
    opts.push_back({ "server/completion",                  "       --chat-template JINJA_TEMPLATE",         "set custom jinja chat template (default: template taken from model's metadata)\n"
                                                                                                           "only commonly used templates are accepted: https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template" });
    opts.push_back({ "server/completion",                  "       --chat-template-file FILE",              "set a file to load a custom jinja chat template (default: template taken from model's metadata)" });
    opts.push_back({ "server/completion",                  "       --slot-save-path PATH",                  "path to save slot kv cache (default: disabled)" });
    opts.push_back({ "server/completion",                  "-sps,  --slot-prompt-similarity N",             "how much the prompt of a request must match the prompt of a slot in order to use that slot (default: %.2f, 0.0 = disabled)\n", params.slot_prompt_similarity });
    opts.push_back({ "server/completion",                  "-tps   --tokens-per-second N",                  "maximum number of tokens per second (default: %d, 0 = disabled, -1 = try to detect)\n"
                                                                                                           "when enabled, limit the request within its X-Request-Tokens-Per-Second HTTP header", bparams.n_tps });
    opts.push_back({ "server/completion",                  "-t,    --threads N",                            "number of threads to use during generation (default: %d)", params.cpuparams.n_threads });
#ifndef GGML_USE_OPENMP
    opts.push_back({ "server/completion",                  "-C,    --cpu-mask M",                           "set CPU affinity mask: arbitrarily long hex. Complements cpu-range (default: \"\")"});
    opts.push_back({ "server/completion",                  "-Cr,   --cpu-range lo-hi",                      "range of CPUs for affinity. Complements --cpu-mask"});
    opts.push_back({ "server/completion",                  "       --cpu-strict <0|1>",                     "use strict CPU placement (default: %u)\n", (unsigned) params.cpuparams.strict_cpu});
    opts.push_back({ "server/completion",                  "       --prio N",                               "set process/thread priority (default: %d), one of:\n"
                                                                                                            "  - 0-normal\n"
                                                                                                            "  - 1-medium\n"
                                                                                                            "  - 2-high\n"
                                                                                                            "  - 3-realtime", params.cpuparams.priority});
    opts.push_back({ "server/completion",                  "       --poll <0...100>",                       "use polling level to wait for work (0 - no polling, default: %u)\n", (unsigned) params.cpuparams.poll});
#endif
    opts.push_back({ "server/completion",                  "-tb,   --threads-batch N",                      "number of threads to use during batch and prompt processing (default: same as --threads)" });
#ifndef GGML_USE_OPENMP
    opts.push_back({ "server/completion",                  "-Cb,   --cpu-mask-batch M",                     "set CPU affinity mask: arbitrarily long hex. Complements cpu-range-batch (default: same as --cpu-mask)"});
    opts.push_back({ "server/completion",                  "-Crb,  --cpu-range-batch lo-hi",                "ranges of CPUs for affinity. Complements --cpu-mask-batch"});
    opts.push_back({ "server/completion",                  "       --cpu-strict-batch <0|1>",               "use strict CPU placement (default: same as --cpu-strict)"});
    opts.push_back({ "server/completion",                  "       --prio-batch N",                         "set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: --priority)"});
    opts.push_back({ "server/completion",                  "       --poll-batch <0...100>",                 "use polling to wait for work (default: same as --poll"});
#endif
    opts.push_back({ "server/completion",                  "-c,    --ctx-size N",                           "size of the prompt context (default: %d, 0 = loaded from model)", params.n_ctx });
    opts.push_back({ "server/completion",                  "       --no-context-shift",                     "disables context shift on infinite text generation (default: %s)", params.ctx_shift ? "disabled" : "enabled" });
    opts.push_back({ "server/completion",                  "-n,    --predict N",                            "number of tokens to predict (default: %d, -1 = infinity, -2 = until context filled)", params.n_predict });
    opts.push_back({ "server/completion",                  "-b,    --batch-size N",                         "logical maximum batch size (default: %d)", params.n_batch });
    opts.push_back({ "server/completion",                  "-ub,   --ubatch-size N",                        "physical maximum batch size (default: %d)", params.n_ubatch });
    opts.push_back({ "server/completion",                  "       --keep N",                               "number of tokens to keep from the initial prompt (default: %d, -1 = all)", params.n_keep });
    opts.push_back({ "server/completion",                  "-e,    --escape",                               R"(process escapes sequences (\n, \r, \t, \', \", \\) (default: %s))", params.escape ? "true" : "false" });
    opts.push_back({ "server/completion",                  "       --no-escape",                            "do not process escape sequences" });
    opts.push_back({ "server/completion",                  "       --samplers SAMPLERS",                    "samplers that will be used for generation in the order, separated by ';' (default: %s)", default_sampler_type_names.c_str() });
    opts.push_back({ "server/completion",                  "       --sampling-seq SEQUENCE",                "simplified sequence for samplers that will be used (default: %s)", default_sampler_type_chars.c_str() });
    opts.push_back({ "server/completion",                  "       --penalize-nl",                          "penalize newline tokens (default: %s)", sampling.penalize_nl ? "true" : "false" });
    opts.push_back({ "server/completion",                  "       --temp T",                               "temperature (default: %.1f)", (double)sampling.temp });
    opts.push_back({ "server/completion",                  "       --top-k N",                              "top-k sampling (default: %d, 0 = disabled)", sampling.top_k });
    opts.push_back({ "server/completion",                  "       --top-p P",                              "top-p sampling (default: %.1f, 1.0 = disabled)", (double) sampling.top_p });
    opts.push_back({ "server/completion",                  "       --min-p P",                              "min-p sampling (default: %.1f, 0.0 = disabled)", (double)sampling.min_p });
    opts.push_back({ "server/completion",                  "       --xtc-probability N",                    "xtc probability (default: %.1f, 0.0 = disabled)", (double)sampling.xtc_probability });
    opts.push_back({ "server/completion",                  "       --xtc-threshold N",                      "xtc threshold (default: %.1f, 1.0 = disabled)", (double)sampling.xtc_threshold });
    opts.push_back({ "server/completion",                  "       --typical P",                            "locally typical sampling, parameter p (default: %.1f, 1.0 = disabled)", (double)sampling.typ_p });
    opts.push_back({ "server/completion",                  "       --repeat-last-n N",                      "last n tokens to consider for penalize (default: %d, 0 = disabled, -1 = ctx_size)", sampling.penalty_last_n });
    opts.push_back({ "server/completion",                  "       --repeat-penalty N",                     "penalize repeat sequence of tokens (default: %.1f, 1.0 = disabled)", (double)sampling.penalty_repeat });
    opts.push_back({ "server/completion",                  "       --presence-penalty N",                   "repeat alpha presence penalty (default: %.1f, 0.0 = disabled)", (double)sampling.penalty_present });
    opts.push_back({ "server/completion",                  "       --frequency-penalty N",                  "repeat alpha frequency penalty (default: %.1f, 0.0 = disabled)", (double)sampling.penalty_freq });
    opts.push_back({ "server/completion",                  "       --dry-multiplier N",                     "set DRY sampling multiplier (default: %.1f, 0.0 = disabled)", (double)params.sampling.dry_multiplier });
    opts.push_back({ "server/completion",                  "       --dry-base N",                           "set DRY sampling base value (default: %.2f)", (double)params.sampling.dry_base });
    opts.push_back({ "server/completion",                  "       --dry-allowed-length N",                 "set allowed length for DRY sampling (default: %d)", params.sampling.dry_allowed_length });
    opts.push_back({ "server/completion",                  "       --dry-penalty-last-n N",                 "set DRY penalty for the last n tokens (default: %d, 0 = disable, -1 = context size)", params.sampling.dry_penalty_last_n });
    opts.push_back({ "server/completion",                  "       --dry-sequence-breaker N",               "add sequence breaker for DRY sampling, clearing out default breakers (%s) in the process; use \"none\" to not use any sequence breakers", default_dry_sequence_breaker_names.c_str() });
    opts.push_back({ "server/completion",                  "       --dynatemp-range N",                     "dynamic temperature range (default: %.1f, 0.0 = disabled)", (double)sampling.dynatemp_range });
    opts.push_back({ "server/completion",                  "       --dynatemp-exp N",                       "dynamic temperature exponent (default: %.1f)", (double)sampling.dynatemp_exponent });
    opts.push_back({ "server/completion",                  "       --mirostat N",                           "use Mirostat sampling, Top K, Nucleus, Tail Free and Locally Typical samplers are ignored if used (default: %d, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)", sampling.mirostat });
    opts.push_back({ "server/completion",                  "       --mirostat-lr N",                        "Mirostat learning rate, parameter eta (default: %.1f)", (double)sampling.mirostat_eta });
    opts.push_back({ "server/completion",                  "       --mirostat-ent N",                       "Mirostat target entropy, parameter tau (default: %.1f)", (double)sampling.mirostat_tau });
    opts.push_back({ "server/completion",                  "-l     --logit-bias TOKEN_ID(+/-)BIAS",         R"(modifies the likelihood of token appearing in the completion, i.e. "--logit-bias 15043+1" to increase likelihood of token ' Hello', or "--logit-bias 15043-1" to decrease likelihood of token ' Hello')" });
    opts.push_back({ "server/completion",                  "       --grammar GRAMMAR",                      "BNF-like grammar to constrain generations (see samples in grammars/ dir) (default: '%s')", sampling.grammar.c_str() });
    opts.push_back({ "server/completion",                  "       --grammar-file FILE",                    "file to read grammar from" });
    opts.push_back({ "server/completion",                  "-j,    --json-schema SCHEMA",                   "JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any JSON object. For schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead" });
    opts.push_back({ "server/completion",                  "       --rope-scaling {none,linear,yarn}",      "RoPE frequency scaling method, defaults to linear unless specified by the model" });
    opts.push_back({ "server/completion",                  "       --rope-scale N",                         "RoPE context scaling factor, expands context by a factor of N" });
    opts.push_back({ "server/completion",                  "       --rope-freq-base N",                     "RoPE base frequency, used by NTK-aware scaling (default: loaded from model)" });
    opts.push_back({ "server/completion",                  "       --rope-freq-scale N",                    "RoPE frequency scaling factor, expands context by a factor of 1/N" });
    opts.push_back({ "server/completion",                  "       --yarn-orig-ctx N",                      "YaRN: original context size of model (default: %d = model training context size)", params.yarn_orig_ctx });
    opts.push_back({ "server/completion",                  "       --yarn-ext-factor N",                    "YaRN: extrapolation mix factor (default: %.1f, 0.0 = full interpolation)", (double)params.yarn_ext_factor });
    opts.push_back({ "server/completion",                  "       --yarn-attn-factor N",                   "YaRN: scale sqrt(t) or attention magnitude (default: %.1f)", (double)params.yarn_attn_factor });
    opts.push_back({ "server/completion",                  "       --yarn-beta-fast N",                     "YaRN: low correction dim or beta (default: %.1f)", (double)params.yarn_beta_fast });
    opts.push_back({ "server/completion",                  "       --yarn-beta-slow N",                     "YaRN: high correction dim or alpha (default: %.1f)", (double)params.yarn_beta_slow });
    if (llama_supports_gpu_offload()) {
        opts.push_back({ "server/completion",              "-nkvo, --no-kv-offload",                        "disable KV offload" });
    }
    opts.push_back({ "server/completion",                  "       --no-cache-prompt",                      "disable caching prompt" });
    opts.push_back({ "server/completion",                  "       --cache-reuse N",                        "min chunk size to attempt reusing from the cache via KV shifting (default: %d)", params.n_cache_reuse });
    opts.push_back({ "server/completion",                  "-ctk,  --cache-type-k TYPE",                    "KV cache data type for K (default: %s)", params.cache_type_k.c_str() });
    opts.push_back({ "server/completion",                  "-ctv,  --cache-type-v TYPE",                    "KV cache data type for V (default: %s)", params.cache_type_v.c_str() });
    opts.push_back({ "server/completion",                  "-dt,   --defrag-thold N",                       "KV cache defragmentation threshold (default: %.1f, < 0 - disabled)", (double)params.defrag_thold });
    opts.push_back({ "server/completion",                  "-np,   --parallel N",                           "number of parallel sequences to decode (default: %d)", params.n_parallel });
    opts.push_back({ "server/completion",                  "-nocb, --no-cont-batching",                     "disable continuous batching" });
    opts.push_back({ "server/completion",                  "       --mmproj FILE",                          "path to a multimodal projector file for LLaVA" });
    if (llama_supports_mlock()) {
        opts.push_back({ "server/completion",              "       --mlock",                                "force system to keep model in RAM rather than swapping or compressing" });
    }
    if (llama_supports_mmap()) {
        opts.push_back({ "server/completion",              "       --no-mmap",                              "do not memory-map model (slower load but may reduce pageouts if not using mlock)" });
    }
    opts.push_back({ "server/completion",                  "       --numa TYPE",                            "attempt optimizations that help on some NUMA systems\n"
                                                                                                            "  - distribute: spread execution evenly over all nodes\n"
                                                                                                            "  - isolate: only spawn threads on CPUs on the node that execution started on\n"
                                                                                                            "  - numactl: use the CPU map provided by numactl\n"
                                                                                                            "if run without this previously, it is recommended to drop the system page cache before using this, see https://github.com/ggerganov/llama.cpp/issues/1437" });
    opts.push_back({ "server/completion",                  "       --control-vector FILE",                  "add a control vector" });
    opts.push_back({ "server/completion",                  "       --control-vector-scaled FILE SCALE",     "add a control vector with user defined scaling SCALE" });
    opts.push_back({ "server/completion",                  "       --control-vector-layer-range START END", "layer range to apply the control vector(s) to, start and end inclusive" });
    opts.push_back({ "server/completion",                  "       --no-warmup",                            "skip warming up the model with an empty run" });
    opts.push_back({ "server/completion",                  "       --spm-infill",                           "use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as some models prefer this (default: %s)", params.spm_infill ? "enabled" : "disabled" });
    opts.push_back({ "server/completion",                  "-sp,   --special",                              "special tokens output enabled (default: %s)", params.special ? "true" : "false" });
    // server // completion //
    // server // completion // speculative //
    opts.push_back({ "server/completion/speculative" });
    opts.push_back({ "server/completion/speculative",      "       --draft-max, --draft, --draft-n N",      "number of tokens to draft for speculative decoding (default: %d)", params.speculative.n_max });
    opts.push_back({ "server/completion/speculative",      "       --draft-min, --draft-n-min N",           "minimum number of draft tokens to use for speculative decoding (default: %d)", params.speculative.n_min });
    opts.push_back({ "server/completion/speculative",      "       --draft-p-min P",                        "minimum speculative decoding probability (greedy) (default: %.1f)", params.speculative.p_min });
    opts.push_back({ "server/completion/speculative",      "-md,   --model-draft FNAME",                    "draft model for speculative decoding (default: unused)" });
    if (llama_supports_gpu_offload()) {
        opts.push_back({ "server/completion/speculative",  "-devd, --device-draft <dev1,dev2,...>",         "comma-separated list of devices to use for offloading the draft model (none = don't offload)\n"
                                                                                                            "use --list-devices to see a list of available devices" });
        opts.push_back({ "server/completion/speculative",  "-ngld, --gpu-layers-draft, --n-gpu-layers-draft N",
                                                                                                            "number of layers to store in VRAM for the draft model" });
    }
    opts.push_back({ "server/completion/speculative",      "       --lookup-ngram-min N",                   "minimum n-gram size for lookup cache (default: %d, 0 = disabled)", bparams.lookup_ngram_min });
    opts.push_back({ "server/completion/speculative",      "-lcs,  --lookup-cache-static FILE",             "path to static lookup cache to use for lookup decoding (not updated by generation)" });
    opts.push_back({ "server/completion/speculative",      "-lcd,  --lookup-cache-dynamic FILE",            "path to dynamic lookup cache to use for lookup decoding (updated by generation)" });
    // server // completion // speculative //
    // server // embedding //
    opts.push_back({ "server/embedding",                   "       --pooling",                              "pooling type for embeddings, use model default if unspecified" });
    // server // embedding //
    // server // images //
    opts.push_back({ "server/images" });
    opts.push_back({ "server/images",                      "       --image-max-batch N",                    "maximum batch count (default: %d)", sdparams.max_batch_count});
    opts.push_back({ "server/images",                      "       --image-max-height N",                   "image maximum height, in pixel space, must be larger than 256 and be multiples of 64 (default: %d)", sdparams.max_height});
    opts.push_back({ "server/images",                      "       --image-max-width N",                    "image maximum width, in pixel space, must be larger than 256 and be multiples of 64 (default: %d)", sdparams.max_width});
    opts.push_back({ "server/images",                      "       --image-guidance N",                     "the value of guidance during the computing phase (default: %f)", sdparams.guidance });
    opts.push_back({ "server/images",                      "       --image-strength N",                     "strength for noising, range of [0.0, 1.0] (default: %f)", sdparams.strength });
    opts.push_back({ "server/images",                      "       --image-sampler TYPE",                   "sampler that will be used for generation, automatically retrieve the default value according to --model, select from %s", sd_sampler_type_names.c_str() });
    opts.push_back({ "server/images",                      "       --image-sample-steps N",                 "number of sample steps, automatically retrieve the default value according to --model, and +2 when requesting high definition generation" });
    opts.push_back({ "server/images",                      "       --image-cfg-scale N",                    "the scale of classifier-free guidance(CFG), automatically retrieve the default value according to --model (1.0 = disabled)" });
    opts.push_back({ "server/images",                      "       --image-slg-scale N",                    "the scale of skip-layer guidance(SLG), only for DiT model, automatically retrieve the default value according to --model (0.0 = disabled)" });
    opts.push_back({ "server/images",                      "       --image-slg-skip-layer",                 "the layers to skip when processing SLG, may be specified multiple times. (default: 7;8;9)" });
    opts.push_back({ "server/images",                      "       --image-slg-start N",                    "the phase to enable SLG (default: %.2f)", sdparams.slg_start });
    opts.push_back({ "server/images",                      "       --image-slg-end N",                      "the phase to disable SLG (default: %.2f)\n"
                                                                                                            "SLG will be enabled at step int([STEP]*[--image-slg-start]) and disabled at int([STEP]*[--image-slg-end])", sdparams.slg_end });
    opts.push_back({ "server/images",                      "       --image-schedule TYPE",                  "denoiser sigma schedule, select from %s (default: %s)", sd_scheduler_names.c_str(), sd_schedule_to_argument(sdparams.schedule) });
    if (llama_supports_gpu_offload()) {
        opts.push_back({ "server/images",                  "       --image-no-text-encoder-model-offload",  "disable text-encoder(clip-l/clip-g/t5xxl) model offload" });
    }
    opts.push_back({ "server/images",                      "       --image-clip-l-model PATH",              "path to the CLIP Large (clip-l) text encoder, or use --model included" });
    opts.push_back({ "server/images",                      "       --image-clip-g-model PATH",              "path to the CLIP Generic (clip-g) text encoder, or use --model included" });
    opts.push_back({ "server/images",                      "       --image-t5xxl-model PATH",               "path to the Text-to-Text Transfer Transformer (t5xxl) text encoder, or use --model included" });
    if (llama_supports_gpu_offload()) {
        opts.push_back({ "server/images",                  "       --image-no-vae-model-offload",           "disable vae(taesd) model offload" });
    }
    opts.push_back({ "server/images",                      "       --image-vae-model PATH",                 "path to Variational AutoEncoder (vae), or use --model included" });
    opts.push_back({ "server/images",                      "       --image-vae-tiling",                     "indicate to process vae decoder in tiles to reduce memory usage (default: %s)", sdparams.vae_tiling ? "enabled" : "disabled" });
    opts.push_back({ "server/images",                      "       --image-taesd-model PATH",               "path to Tiny AutoEncoder For StableDiffusion (taesd), or use --model included" });
    opts.push_back({ "server/images",                      "       --image-upscale-model PATH",             "path to the upscale model, or use --model included" });
    opts.push_back({ "server/images",                      "       --image-upscale-repeats N",              "how many times to run upscaler (default: %d)", sdparams.upscale_repeats });
    if (llama_supports_gpu_offload()) {
        opts.push_back({ "server/images",                  "       --image-no-control-net-model-offload",   "disable control-net model offload" });
    }
    opts.push_back({ "server/images",                      "       --image-control-net-model PATH",         "path to the control net model, or use --model included" });
    opts.push_back({ "server/images",                      "       --image-control-strength N",             "how strength to apply the control net (default: %f)", sdparams.control_strength });
    opts.push_back({ "server/images",                      "       --image-control-canny",                  "indicate to apply canny preprocessor (default: %s)", sdparams.control_canny ? "enabled" : "disabled" });
    // server // images //
    // server //
    // rpc-server //
    opts.push_back({ "rpc-server" });
    opts.push_back({ "rpc-server",                         "       --rpc-server-host HOST",                 "ip address to rpc server listen (default: %s)", rparams.hostname.c_str() });
    opts.push_back({ "rpc-server",                         "       --rpc-server-port PORT",                 "port to rpc server listen (default: %d, 0 = disabled)", rparams.port });
    if (llama_supports_gpu_offload()) {
        opts.push_back({ "rpc-server",                     "       --rpc-server-main-gpu N",                "the GPU VRAM to use for the rpc server (default: %d, -1 = disabled, use RAM)", rparams.main_gpu });
    }
    opts.push_back({ "rpc-server",                         "       --rpc-server-reserve-memory MEM",        "reserve memory in MiB (default: %zu)", rparams.reserve_memory });
    // rpc-server //

    // clang-format on

    printf("usage: %s [options]\n", argv[0]);

    for (const auto &o : opts) {
        if (!o.grp.empty()) {
            printf("\n%s:\n\n", o.grp.c_str());
            continue;
        }
        printf("  %-32s", o.args.c_str());
        if (o.args.length() > 30) {
            printf("\n%34s", "");
        }

        const auto desc = o.desc;
        size_t start    = 0;
        size_t end      = desc.find('\n');
        while (end != std::string::npos) {
            printf("%s\n%34s", desc.substr(start, end - start).c_str(), "");
            start = end + 1;
            end   = desc.find('\n', start);
        }

        printf("%s\n", desc.substr(start).c_str());
    }
    printf("\n");
}

//
// Utils
//

static std::vector<ggml_backend_dev_t> parse_device_list(const std::string &value) {
    std::vector<ggml_backend_dev_t> devices;
    auto dev_names = string_split<std::string>(value, ',');
    if (dev_names.empty()) {
        throw std::invalid_argument("no devices specified");
    }
    if (dev_names.size() == 1 && dev_names[0] == "none") {
        devices.push_back(nullptr);
    } else {
        for (const auto &device : dev_names) {
            auto *dev = ggml_backend_dev_by_name(device.c_str());
            if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                throw std::invalid_argument(string_format("invalid device: %s", device.c_str()));
            }
            devices.push_back(dev);
        }
        devices.push_back(nullptr);
    }
    return devices;
}

//
// Environment variable utils
//

template <typename T>
static typename std::enable_if<std::is_same<T, std::string>::value, void>::type get_env(std::string name, T &target) {
    char *value = std::getenv(name.c_str());
    target      = value ? std::string(value) : target;
}

template <typename T>
static typename std::enable_if<!std::is_same<T, bool>::value && std::is_integral<T>::value, void>::type get_env(std::string name, T &target) {
    char *value = std::getenv(name.c_str());
    target      = value ? std::stoi(value) : target;
}

template <typename T>
static typename std::enable_if<std::is_floating_point<T>::value, void>::type get_env(std::string name, T &target) {
    char *value = std::getenv(name.c_str());
    target      = value ? std::stof(value) : target;
}

template <typename T>
static typename std::enable_if<std::is_same<T, bool>::value, void>::type get_env(std::string name, T &target) {
    char *value = std::getenv(name.c_str());
    if (value) {
        std::string val(value);
        target = val == "1" || val == "true";
    }
}

template <typename T>
static typename std::enable_if<std::is_same<T, std::vector<ggml_backend_dev_t>>::value, void>::type get_env(std::string name, T &target) {
    char *value = std::getenv(name.c_str());
    if (value) {
        target = parse_device_list(value);
    }
}

static bool llama_box_params_parse(int argc, char **argv, llama_box_params &bparams) {
    // load dynamic backends
    ggml_backend_load_all();

    try {
        for (int i = 1; i < argc;) {
            const char *flag = argv[i++];

            if (*flag != '-') {
                continue;
            }

            // general //

            if (!strcmp(flag, "-h") || !strcmp(flag, "--help") || !strcmp(flag, "--usage")) {
                llama_box_params_print_usage(argc, argv, bparams);
                exit(0);
            }

            if (!strcmp(flag, "--version")) {
                fprintf(stderr, "version  : %s (%s)\n", LLAMA_BOX_BUILD_VERSION, LLAMA_BOX_COMMIT);
                fprintf(stderr, "compiler : %s\n", LLAMA_BOX_BUILD_COMPILER);
                fprintf(stderr, "target   : %s\n", LLAMA_BOX_BUILD_TARGET);
                fprintf(stderr, "vendor   : \n");
                fprintf(stderr, "  - llama.cpp %s (%d)\n", LLAMA_CPP_COMMIT, LLAMA_BOX_BUILD_NUMBER);
                fprintf(stderr, "  - stable-diffusion.cpp %s (%d)\n", STABLE_DIFFUSION_CPP_COMMIT, STABLE_DIFFUSION_CPP_BUILD_NUMBER);
                exit(0);
            }

            if (!strcmp(flag, "--system-info")) {
                fprintf(stderr, "%s\n", llama_print_system_info());
                exit(0);
            }

            if (llama_supports_gpu_offload()) {
                if (!strcmp(flag, "--list-devices")) {
                    printf("Available devices:\n");
                    for (size_t j = 0; j < ggml_backend_dev_count(); ++j) {
                        auto *dev = ggml_backend_dev_get(j);
                        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                            size_t free, total;
                            ggml_backend_dev_memory(dev, &free, &total);
                            printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), total / 1024 / 1024, free / 1024 / 1024);
                        }
                    }
                    exit(0);
                }
            }

            if (!strcmp(flag, "-v") || !strcmp(flag, "--verbose") || !strcmp(flag, "--log-verbose")) {
                bparams.gparams.verbosity = INT_MAX;
                common_log_set_verbosity_thold(INT_MAX);
                continue;
            }

            if (!strcmp(flag, "-lv") || !strcmp(flag, "--verbosity") || !strcmp(flag, "--log-verbosity")) {
                if (i == argc) {
                    missing("--log-verbosity");
                }
                char *arg                 = argv[i++];
                bparams.gparams.verbosity = std::stoi(std::string(arg));
                common_log_set_verbosity_thold(bparams.gparams.verbosity);
                continue;
            }

            if (!strcmp(flag, "--log-colors")) {
                common_log_set_colors(common_log_main(), true);
                continue;
            }

            // general //

            // server //

            if (!strcmp(flag, "--host")) {
                if (i == argc) {
                    missing("--host");
                }
                char *arg                = argv[i++];
                bparams.gparams.hostname = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "--port")) {
                if (i == argc) {
                    missing("--port");
                }
                char *arg            = argv[i++];
                bparams.gparams.port = std::stoi(std::string(arg));
                if (bparams.gparams.port < 0 || bparams.gparams.port > 65535) {
                    invalid("--port");
                }
                continue;
            }

            if (!strcmp(flag, "-to") || !strcmp(flag, "--timeout")) {
                if (i == argc) {
                    missing("--timeout");
                }
                char *arg                     = argv[i++];
                bparams.gparams.timeout_read  = std::stoi(std::string(arg));
                bparams.gparams.timeout_write = bparams.gparams.timeout_read;
                continue;
            }

            if (!strcmp(flag, "--threads-http")) {
                if (i == argc) {
                    missing("--threads-http");
                }
                char *arg                      = argv[i++];
                bparams.gparams.n_threads_http = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--conn-idle")) { // extend
                if (i == argc) {
                    missing("--conn-idle");
                }
                char *arg         = argv[i++];
                bparams.conn_idle = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--conn-keepalive")) { // extend
                if (i == argc) {
                    missing("--conn-keepalive");
                }
                char *arg              = argv[i++];
                bparams.conn_keepalive = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-m") || !strcmp(flag, "--model")) {
                if (i == argc) {
                    missing("--model");
                }
                char *arg             = argv[i++];
                bparams.gparams.model = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "-a") || !strcmp(flag, "--alias")) {
                if (i == argc) {
                    missing("--alias");
                }
                char *arg                   = argv[i++];
                bparams.gparams.model_alias = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "--lora")) {
                if (i == argc) {
                    missing("--lora");
                }
                char *arg = argv[i++];
                bparams.gparams.lora_adapters.push_back({
                    std::string(arg),
                    1.0f,
                });
                continue;
            }

            if (!strcmp(flag, "--lora-scaled")) {
                if (i == argc) {
                    missing("--lora-scaled");
                }
                char *n = argv[i++];
                if (i == argc) {
                    invalid("--lora-scaled");
                }
                char *s = argv[i++];
                bparams.gparams.lora_adapters.push_back({
                    std::string(n),
                    std::stof(std::string(s)),
                });
                continue;
            }

            if (!strcmp(flag, "--lora-init-without-apply")) {
                bparams.gparams.lora_init_without_apply = true;
                continue;
            }

            if (!strcmp(flag, "-s") || !strcmp(flag, "--seed")) {
                if (i == argc) {
                    missing("--seed");
                }
                char *arg                     = argv[i++];
                bparams.gparams.sampling.seed = std::stoul(std::string(arg));
                continue;
            }

            if (llama_supports_gpu_offload()) {
                if (!strcmp(flag, "-mg") || !strcmp(flag, "--main-gpu")) {
                    if (i == argc) {
                        missing("--main-gpu");
                    }
                    char *arg                = argv[i++];
                    bparams.gparams.main_gpu = std::stoi(std::string(arg));
                    if (bparams.gparams.main_gpu < 0 || bparams.gparams.main_gpu >= int32_t(llama_max_devices())) {
                        invalid("--main-gpu");
                    }
                    continue;
                }
            }

            if (!strcmp(flag, "-fa") || !strcmp(flag, "--flash-attn")) {
                bparams.gparams.flash_attn = true;
                continue;
            }

            if (!strcmp(flag, "--metrics")) {
                bparams.gparams.endpoint_metrics = true;
                continue;
            }

            if (!strcmp(flag, "--infill")) {
                bparams.endpoint_infill = true;
                continue;
            }

            if (!strcmp(flag, "--embedding") || !strcmp(flag, "--embeddings")) {
                bparams.gparams.embedding = true;
                continue;
            }

            if (!strcmp(flag, "--images")) {
                bparams.endpoint_images = true;
                continue;
            }

            if (!strcmp(flag, "--reranking") || !strcmp(flag, "--rerank")) {
                bparams.gparams.reranking = true;
                continue;
            }

            if (!strcmp(flag, "--slots")) {
                bparams.gparams.endpoint_slots = true;
                continue;
            }

            if (llama_supports_rpc()) {
                if (!strcmp(flag, "--rpc")) {
                    if (i == argc) {
                        missing("--rpc");
                    }
                    char *arg                   = argv[i++];
                    bparams.gparams.rpc_servers = arg;
                    continue;
                }
            }

            // server // completion//

            if (llama_supports_gpu_offload()) {
                if (!strcmp(flag, "-devd") || !strcmp(flag, "--device-draft")) {
                    if (i == argc) {
                        missing("--device-draft");
                    }
                    char *arg                           = argv[i++];
                    bparams.gparams.speculative.devices = parse_device_list(arg);
                    continue;
                }

                if (!strcmp(flag, "-ngl") || !strcmp(flag, "--gpu-layers") || !strcmp(flag, "--n-gpu-layers")) {
                    if (i == argc) {
                        missing("--gpu-layers");
                    }
                    char *arg                    = argv[i++];
                    bparams.gparams.n_gpu_layers = std::stoi(arg);
                    continue;
                }

                if (!strcmp(flag, "-sm") || !strcmp(flag, "--split-mode")) {
                    if (i == argc) {
                        missing("--split-mode");
                    }
                    char *arg = argv[i++];
                    if (!strcmp(arg, "none")) {
                        bparams.gparams.split_mode = LLAMA_SPLIT_MODE_NONE;
                    } else if (!strcmp(arg, "layer")) {
                        bparams.gparams.split_mode = LLAMA_SPLIT_MODE_LAYER;
                    } else if (!strcmp(arg, "row")) {
                        bparams.gparams.split_mode = LLAMA_SPLIT_MODE_ROW;
                    } else {
                        invalid("--split-mode");
                    }
                    continue;
                }

                if (!strcmp(flag, "-ts") || !strcmp(flag, "--tensor-split")) {
                    if (i == argc) {
                        missing("--tensor-split");
                    }
                    char *arg = argv[i++];
                    const std::regex regex{R"([,/]+)"};
                    std::string arg_s{arg};
                    std::sregex_token_iterator it{arg_s.begin(), arg_s.end(), regex, -1};
                    std::vector<std::string> split_arg{it, {}};
                    if (split_arg.size() >= llama_max_devices()) {
                        invalid("--tensor-split");
                    }
                    for (size_t j = 0; j < llama_max_devices(); ++j) {
                        if (j < split_arg.size()) {
                            bparams.gparams.tensor_split[j] = std::stof(split_arg[j]);
                        } else {
                            bparams.gparams.tensor_split[j] = 0.0f;
                        }
                    }
                    continue;
                }
            }

            if (!strcmp(flag, "--override-kv")) {
                if (i == argc) {
                    missing("--override-kv");
                }
                char *arg = argv[i++];
                if (!string_parse_kv_override(arg, bparams.gparams.kv_overrides)) {
                    invalid("--override-kv");
                }
                continue;
            }

            if (!strcmp(flag, "--slot-save-path")) {
                if (i == argc) {
                    missing("--slot-save-path");
                }
                char *arg = argv[i++];
                if (arg[0] == '\0') {
                    invalid("--slot-save-path");
                }
                std::string p(arg);
                if (p[p.size() - 1] != DIRECTORY_SEPARATOR) {
                    p += DIRECTORY_SEPARATOR;
                }
                bparams.gparams.slot_save_path = p;
                continue;
            }

            if (!strcmp(flag, "--chat-template")) {
                if (i == argc) {
                    missing("--chat-template");
                }
                char *arg = argv[i++];
                if (arg[0] == '\0') {
                    invalid("--chat-template");
                }
                std::string t(arg);
                if (t.size() > 20 && !common_chat_verify_template(t)) {
                    invalid("--chat-template");
                }
                bparams.gparams.enable_chat_template = true;
                bparams.gparams.chat_template        = t;
                continue;
            }

            if (!strcmp(flag, "--chat-template-file")) {
                if (i == argc) {
                    missing("--chat-template-file");
                }
                char *arg = argv[i++];
                std::ifstream file(arg);
                if (!file) {
                    invalid("--chat-template-file");
                }
                std::string t;
                std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), std::back_inserter(t));
                if (t.size() > 20 && !common_chat_verify_template(t)) {
                    invalid("--chat-template-file");
                }
                bparams.gparams.enable_chat_template = true;
                bparams.gparams.chat_template        = t;
                continue;
            }

            if (!strcmp(flag, "-sps") || !strcmp(flag, "--slot-prompt-similarity")) {
                if (i == argc) {
                    missing("--slot-prompt-similarity");
                }
                char *arg                              = argv[i++];
                bparams.gparams.slot_prompt_similarity = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-tps") || !strcmp(flag, "--tokens-per-second")) { // extend
                if (i == argc) {
                    missing("--tokens-per-second");
                }
                char *arg     = argv[i++];
                bparams.n_tps = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-t") || !strcmp(flag, "--threads")) {
                if (i == argc) {
                    missing("--threads");
                }
                char *arg                           = argv[i++];
                bparams.gparams.cpuparams.n_threads = std::stoi(std::string(arg));
                if (bparams.gparams.cpuparams.n_threads <= 0) {
                    bparams.gparams.cpuparams.n_threads = cpu_get_num_math();
                }
                continue;
            }

            if (!strcmp(flag, "-C") || !strcmp(flag, "--cpu-mask")) {
                if (i == argc) {
                    missing("--cpu-mask");
                }
                char *arg                            = argv[i++];
                bparams.gparams.cpuparams.mask_valid = true;
                if (!parse_cpu_mask(arg, bparams.gparams.cpuparams.cpumask)) {
                    invalid("--cpu-mask");
                }
                continue;
            }

            if (!strcmp(flag, "-Cr") || !strcmp(flag, "--cpu-range")) {
                if (i == argc) {
                    missing("--cpu-range");
                }
                char *arg                            = argv[i++];
                bparams.gparams.cpuparams.mask_valid = true;
                if (!parse_cpu_range(arg, bparams.gparams.cpuparams.cpumask)) {
                    invalid("--cpu-range");
                }
                continue;
            }

            if (!strcmp(flag, "--prio")) {
                if (i == argc) {
                    missing("--prio");
                }
                char *arg                          = argv[i++];
                bparams.gparams.cpuparams.priority = (enum ggml_sched_priority)std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--cpu-strict")) {
                if (i == argc) {
                    missing("--cpu-strict");
                }
                char *arg                            = argv[i++];
                bparams.gparams.cpuparams.strict_cpu = std::stoul(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--poll")) {
                if (i == argc) {
                    missing("--poll");
                }
                char *arg                      = argv[i++];
                bparams.gparams.cpuparams.poll = std::stoul(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-tb") || !strcmp(flag, "--threads-batch")) {
                if (i == argc) {
                    missing("--threads-batch");
                }
                char *arg                                 = argv[i++];
                bparams.gparams.cpuparams_batch.n_threads = std::stoi(std::string(arg));
                if (bparams.gparams.cpuparams_batch.n_threads <= 0) {
                    bparams.gparams.cpuparams_batch.n_threads = cpu_get_num_math();
                }
                continue;
            }

            if (!strcmp(flag, "-Cb") || !strcmp(flag, "--cpu-mask-batch")) {
                if (i == argc) {
                    missing("--cpu-mask-batch");
                }
                char *arg                                  = argv[i++];
                bparams.gparams.cpuparams_batch.mask_valid = true;
                if (!parse_cpu_mask(arg, bparams.gparams.cpuparams_batch.cpumask)) {
                    invalid("--cpu-mask-batch");
                }
                continue;
            }

            if (!strcmp(flag, "-Crb") || !strcmp(flag, "--cpu-range-batch")) {
                if (i == argc) {
                    missing("--cpu-range-batch");
                }
                char *arg                                  = argv[i++];
                bparams.gparams.cpuparams_batch.mask_valid = true;
                if (!parse_cpu_range(arg, bparams.gparams.cpuparams_batch.cpumask)) {
                    invalid("--cpu-range-batch");
                }
                continue;
            }

            if (!strcmp(flag, "--prio-batch")) {
                if (i == argc) {
                    missing("--prio-batch");
                }
                char *arg                                = argv[i++];
                bparams.gparams.cpuparams_batch.priority = (enum ggml_sched_priority)std::stoul(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--cpu-strict-batch")) {
                if (i == argc) {
                    missing("--cpu-strict-batch");
                }
                char *arg                                  = argv[i++];
                bparams.gparams.cpuparams_batch.strict_cpu = std::stoul(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--poll-batch")) {
                if (i == argc) {
                    missing("--poll-batch");
                }
                char *arg                            = argv[i++];
                bparams.gparams.cpuparams_batch.poll = std::stoul(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-c") || !strcmp(flag, "--ctx-size")) {
                if (i == argc) {
                    missing("--ctx-size");
                }
                char *arg             = argv[i++];
                bparams.gparams.n_ctx = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--no-context-shift")) {
                bparams.gparams.ctx_shift = false;
                continue;
            }

            if (!strcmp(flag, "-n") || !strcmp(flag, "--predict")) {
                if (i == argc) {
                    missing("--predict");
                }
                char *arg                 = argv[i++];
                bparams.gparams.n_predict = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-b") || !strcmp(flag, "--batch-size")) {
                if (i == argc) {
                    missing("--batch-size");
                }
                char *arg               = argv[i++];
                bparams.gparams.n_batch = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-ub") || !strcmp(flag, "--ubatch-size")) {
                if (i == argc) {
                    missing("--ubatch-size");
                }
                char *arg                = argv[i++];
                bparams.gparams.n_ubatch = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--keep")) {
                if (i == argc) {
                    missing("--keep");
                }
                char *arg              = argv[i++];
                bparams.gparams.n_keep = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-e") || !strcmp(flag, "--escape")) {
                bparams.gparams.escape = true;
                continue;
            }

            if (!strcmp(flag, "--no-escape")) {
                bparams.gparams.escape = false;
                continue;
            }

            if (!strcmp(flag, "--samplers")) {
                if (i == argc) {
                    missing("--samplers");
                }
                char *arg                         = argv[i++];
                const auto sampler_names          = string_split<std::string>(arg, ';');
                bparams.gparams.sampling.samplers = common_sampler_types_from_names(sampler_names, true);
                continue;
            }

            if (!strcmp(flag, "--sampling-seq")) {
                if (i == argc) {
                    missing("--sampling-seq");
                }
                char *arg                         = argv[i++];
                bparams.gparams.sampling.samplers = common_sampler_types_from_chars(arg);
                continue;
            }

            if (!strcmp(flag, "--penalize-nl")) {
                bparams.gparams.sampling.penalize_nl = true;
                continue;
            }

            if (!strcmp(flag, "--temp")) {
                if (i == argc) {
                    missing("--temp");
                }
                char *arg                     = argv[i++];
                bparams.gparams.sampling.temp = std::stof(std::string(arg));
                bparams.gparams.sampling.temp = std::max(bparams.gparams.sampling.temp, 0.0f);
                continue;
            }

            if (!strcmp(flag, "--top-k")) {
                if (i == argc) {
                    missing("--top-k");
                }
                char *arg                      = argv[i++];
                bparams.gparams.sampling.top_k = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--top-p")) {
                if (i == argc) {
                    missing("--top-p");
                }
                char *arg                      = argv[i++];
                bparams.gparams.sampling.top_p = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--min-p")) {
                if (i == argc) {
                    missing("--min-p");
                }
                char *arg                      = argv[i++];
                bparams.gparams.sampling.min_p = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--xtc-probability")) {
                if (i == argc) {
                    missing("--xtc-probability");
                }
                char *arg                                = argv[i++];
                bparams.gparams.sampling.xtc_probability = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--xtc-threshold")) {
                if (i == argc) {
                    missing("--xtc-threshold");
                }
                char *arg                              = argv[i++];
                bparams.gparams.sampling.xtc_threshold = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--typical")) {
                if (i == argc) {
                    missing("--typical");
                }
                char *arg                      = argv[i++];
                bparams.gparams.sampling.typ_p = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--repeat-last-n")) {
                if (i == argc) {
                    missing("--repeat-last-n");
                }
                char *arg                               = argv[i++];
                bparams.gparams.sampling.penalty_last_n = std::stoi(std::string(arg));
                bparams.gparams.sampling.n_prev         = std::max(bparams.gparams.sampling.n_prev, bparams.gparams.sampling.penalty_last_n);
                continue;
            }

            if (!strcmp(flag, "--repeat-penalty")) {
                if (i == argc) {
                    missing("--repeat-penalty");
                }
                char *arg                               = argv[i++];
                bparams.gparams.sampling.penalty_repeat = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--presence-penalty")) {
                if (i == argc) {
                    missing("--presence-penalty");
                }
                char *arg                                = argv[i++];
                bparams.gparams.sampling.penalty_present = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--frequency-penalty")) {
                if (i == argc) {
                    missing("--frequency-penalty");
                }
                char *arg                             = argv[i++];
                bparams.gparams.sampling.penalty_freq = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--dry-multiplier")) {
                if (i == argc) {
                    missing("--dry-multiplier");
                }
                char *arg                               = argv[i++];
                bparams.gparams.sampling.dry_multiplier = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--dry-base")) {
                if (i == argc) {
                    missing("--dry-base");
                }
                char *arg            = argv[i++];
                float potential_base = std::stof(std::string(arg));
                if (potential_base >= 1.0f) {
                    bparams.gparams.sampling.dry_multiplier = std::stof(std::string(arg));
                }
                continue;
            }

            if (!strcmp(flag, "--dry-allowed-length")) {
                if (i == argc) {
                    missing("--dry-allowed-length");
                }
                char *arg                                   = argv[i++];
                bparams.gparams.sampling.dry_allowed_length = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--dry-penalty-last-n")) {
                if (i == argc) {
                    missing("--dry-penalty-last-n");
                }
                char *arg                                   = argv[i++];
                bparams.gparams.sampling.dry_penalty_last_n = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--dry-sequence-breaker")) {
                if (i == argc) {
                    missing("--dry-sequence-breaker");
                }

                static bool defaults_cleared = false;
                if (!defaults_cleared) {
                    bparams.gparams.sampling.dry_sequence_breakers.clear();
                    defaults_cleared = true;
                }

                char *arg = argv[i++];
                if (!strcmp(arg, "none")) {
                    bparams.gparams.sampling.dry_sequence_breakers.clear();
                } else {
                    bparams.gparams.sampling.dry_sequence_breakers.emplace_back(arg);
                }
                continue;
            }

            if (!strcmp(flag, "--dynatemp-range")) {
                if (i == argc) {
                    missing("--dynatemp-range");
                }
                char *arg                               = argv[i++];
                bparams.gparams.sampling.dynatemp_range = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--dynatemp-exp")) {
                if (i == argc) {
                    missing("--dynatemp-exp");
                }
                char *arg                                  = argv[i++];
                bparams.gparams.sampling.dynatemp_exponent = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--mirostat")) {
                if (i == argc) {
                    missing("--mirostat");
                }
                char *arg                         = argv[i++];
                bparams.gparams.sampling.mirostat = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--mirostat-lr")) {
                if (i == argc) {
                    missing("--mirostat-lr");
                }
                char *arg                             = argv[i++];
                bparams.gparams.sampling.mirostat_eta = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--mirostat-ent")) {
                if (i == argc) {
                    missing("--mirostat-ent");
                }
                char *arg                             = argv[i++];
                bparams.gparams.sampling.mirostat_tau = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-l") || !strcmp(flag, "--logit-bias")) {
                if (i == argc) {
                    missing("--logit-bias");
                }
                char *arg = argv[i++];
                std::stringstream ss(arg);
                llama_token key;
                char sign;
                std::string value;
                if (ss >> key && ss >> sign && std::getline(ss, value) && (sign == '+' || sign == '-')) {
                    const float bias = std::stof(value) * ((sign == '-') ? -1.0f : 1.0f);
                    bparams.gparams.sampling.logit_bias.push_back({key, bias});
                } else {
                    invalid("--logit-bias");
                }
                continue;
            }

            if (!strcmp(flag, "--grammar")) {
                if (i == argc) {
                    missing("--grammar");
                }
                char *arg                        = argv[i++];
                bparams.gparams.sampling.grammar = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "--grammar-file")) {
                if (i == argc) {
                    missing("--grammar-file");
                }
                char *arg = argv[i++];
                std::ifstream file(arg);
                if (!file) {
                    invalid("--grammar-file");
                }
                std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(),
                          std::back_inserter(bparams.gparams.sampling.grammar));
                continue;
            }

            if (!strcmp(flag, "-j") || !strcmp(flag, "--json-schema")) {
                if (i == argc) {
                    missing("--json-schema");
                }
                char *arg                        = argv[i++];
                bparams.gparams.sampling.grammar = json_schema_to_grammar(json::parse(std::string(arg)));
                continue;
            }

            if (!strcmp(flag, "--rope-scaling")) {
                if (i == argc) {
                    missing("--rope-scaling");
                }
                char *arg = argv[i++];
                std::string value(arg);
                if (value == "none") {
                    bparams.gparams.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;
                } else if (value == "linear") {
                    bparams.gparams.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
                } else if (value == "yarn") {
                    bparams.gparams.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
                } else {
                    invalid("--rope-scaling");
                }
                continue;
            }

            if (!strcmp(flag, "--rope-scale")) {
                if (i == argc) {
                    missing("--rope-scale");
                }
                char *arg                       = argv[i++];
                bparams.gparams.rope_freq_scale = 1.0f / std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--rope-freq-base")) {
                if (i == argc) {
                    missing("--rope-freq-base");
                }
                char *arg                      = argv[i++];
                bparams.gparams.rope_freq_base = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--rope-freq-scale")) {
                if (i == argc) {
                    missing("--rope-freq-scale");
                }
                char *arg                       = argv[i++];
                bparams.gparams.rope_freq_scale = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--yarn-orig-ctx")) {
                if (i == argc) {
                    missing("--yarn-orig-ctx");
                }
                char *arg                     = argv[i++];
                bparams.gparams.yarn_orig_ctx = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--yarn-ext-factor")) {
                if (i == argc) {
                    missing("--yarn-ext-factor");
                }
                char *arg                       = argv[i++];
                bparams.gparams.yarn_ext_factor = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--yarn-attn-factor")) {
                if (i == argc) {
                    missing("--yarn-attn-factor");
                }
                char *arg                        = argv[i++];
                bparams.gparams.yarn_attn_factor = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--yarn-beta-fast")) {
                if (i == argc) {
                    missing("--yarn-beta-fast");
                }
                char *arg                      = argv[i++];
                bparams.gparams.yarn_beta_fast = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--yarn-beta-slow")) {
                if (i == argc) {
                    missing("--yarn-beta-slow");
                }
                char *arg                      = argv[i++];
                bparams.gparams.yarn_beta_slow = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-nkvo") || !strcmp(flag, "--no-kv-offload")) {
                bparams.gparams.no_kv_offload = true;
                continue;
            }

            if (!strcmp(flag, "--no-cache-prompt")) {
                bparams.cache_prompt = false;
                continue;
            }

            if (!strcmp(flag, "--cache-reuse")) {
                if (i == argc) {
                    missing("--cache-reuse");
                }
                char *arg                     = argv[i++];
                bparams.gparams.n_cache_reuse = std::stoi(std::string(arg));
                if (bparams.gparams.n_cache_reuse > 0) {
                    bparams.cache_prompt = true;
                }
                continue;
            }

            if (!strcmp(flag, "-ctk") || !strcmp(flag, "--cache-type-k")) {
                if (i == argc) {
                    missing("--cache-type-k");
                }
                char *arg                    = argv[i++];
                bparams.gparams.cache_type_k = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "-ctv") || !strcmp(flag, "--cache-type-v")) {
                if (i == argc) {
                    missing("--cache-type-v");
                }
                char *arg                    = argv[i++];
                bparams.gparams.cache_type_v = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "-dt") || !strcmp(flag, "--defrag-thold")) {
                if (i == argc) {
                    missing("--defrag-thold");
                }
                char *arg                    = argv[i++];
                bparams.gparams.defrag_thold = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-np") || !strcmp(flag, "--parallel")) {
                if (i == argc) {
                    missing("--parallel");
                }
                char *arg                  = argv[i++];
                bparams.gparams.n_parallel = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-nocb") || !strcmp(flag, "--no-cont-batching")) {
                bparams.gparams.cont_batching = false;
                continue;
            }

            if (!strcmp(flag, "--mmproj")) {
                if (i == argc) {
                    missing("--mmproj");
                }
                char *arg              = argv[i++];
                bparams.gparams.mmproj = std::string(arg);
                continue;
            }

            if (llama_supports_mlock()) {
                if (!strcmp(flag, "--mlock")) {
                    bparams.gparams.use_mlock = true;
                    continue;
                }
            }

            if (llama_supports_mmap()) {
                if (!strcmp(flag, "--no-mmap")) {
                    bparams.gparams.use_mmap = false;
                    continue;
                }
            }

            if (!strcmp(flag, "--numa")) {
                if (i == argc) {
                    missing("--numa");
                }
                char *arg = argv[i++];
                std::string value(arg);
                if (value == "distribute") {
                    bparams.gparams.numa = GGML_NUMA_STRATEGY_DISTRIBUTE;
                } else if (value == "isolate") {
                    bparams.gparams.numa = GGML_NUMA_STRATEGY_ISOLATE;
                } else if (value == "numactl") {
                    bparams.gparams.numa = GGML_NUMA_STRATEGY_NUMACTL;
                } else {
                    invalid("--numa");
                }
                continue;
            }

            if (!strcmp(flag, "--control-vector")) {
                if (i == argc) {
                    missing("--control-vector");
                }
                char *arg = argv[i++];
                bparams.gparams.control_vectors.push_back({1.0f, std::string(arg)});
                continue;
            }

            if (!strcmp(flag, "--control-vector-scaled")) {
                if (i == argc) {
                    missing("--control-vector-scaled");
                }
                char *n = argv[i++];
                if (i == argc) {
                    invalid("--control-vector-scaled");
                }
                char *s = argv[i++];
                bparams.gparams.control_vectors.push_back({std::stof(std::string(s)), std::string(n)});
                continue;
            }

            if (!strcmp(flag, "--control-vector-layer-range")) {
                if (i == argc) {
                    missing("--control-vector-layer-range");
                }
                char *s = argv[i++];
                if (i == argc) {
                    invalid("--control-vector-layer-range");
                }
                char *e                                    = argv[i++];
                bparams.gparams.control_vector_layer_start = std::stoi(std::string(s));
                bparams.gparams.control_vector_layer_end   = std::stoi(std::string(e));
                continue;
            }

            if (!strcmp(flag, "--no-warmup")) {
                bparams.gparams.warmup = false;
                continue;
            }

            if (!strcmp(flag, "--spm-infill")) {
                bparams.gparams.spm_infill = true;
                continue;
            }

            if (!strcmp(flag, "-sp") || !strcmp(flag, "--special")) {
                bparams.gparams.special = true;
                continue;
            }

            // server // completion // speculative //

            if (!strcmp(flag, "--draft") || !strcmp(flag, "--draft-max") || !strcmp(flag, "--draft-n")) {
                if (i == argc) {
                    missing("--draft-max");
                }
                char *arg                         = argv[i++];
                bparams.gparams.speculative.n_max = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--draft-min") || !strcmp(flag, "--draft-n-min")) {
                if (i == argc) {
                    missing("--draft-min");
                }
                char *arg                         = argv[i++];
                bparams.gparams.speculative.n_min = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--draft-p-min")) {
                if (i == argc) {
                    missing("--draft-p-min");
                }
                char *arg                         = argv[i++];
                bparams.gparams.speculative.p_min = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-md") || !strcmp(flag, "--model-draft")) {
                if (i == argc) {
                    missing("--model-draft");
                }
                char *arg                         = argv[i++];
                bparams.gparams.speculative.model = std::string(arg);
                continue;
            }

            if (llama_supports_gpu_offload()) {
                if (!strcmp(flag, "-dev") || !strcmp(flag, "--device")) {
                    if (i == argc) {
                        missing("--device");
                    }
                    char *arg               = argv[i++];
                    bparams.gparams.devices = parse_device_list(arg);
                    continue;
                }

                if (!strcmp(flag, "-ngld") || !strcmp(flag, "--gpu-layers-draft") || !strcmp(flag, "--n-gpu-layers-draft")) {
                    if (i == argc) {
                        missing("--gpu-layers-draft");
                    }
                    char *arg                                = argv[i++];
                    bparams.gparams.speculative.n_gpu_layers = std::stoi(arg);
                    continue;
                }
            }

            if (!strcmp(flag, "--lookup-ngram-min")) {
                if (i == argc) {
                    missing("--lookup-ngram-min");
                }
                char *arg                = argv[i++];
                bparams.lookup_ngram_min = std::stoi(std::string(arg));
                if (bparams.lookup_ngram_min < 1) {
                    invalid("--lookup-ngram-min");
                }
                if (bparams.lookup_ngram_min > LLAMA_NGRAM_MAX) {
                    invalid("--lookup-ngram-min");
                }
                continue;
            }

            if (!strcmp(flag, "-lcs") || !strcmp(flag, "--lookup-cache-static")) {
                if (i == argc) {
                    missing("--lookup-cache-static");
                }
                char *arg                           = argv[i++];
                bparams.gparams.lookup_cache_static = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "-lcd") || !strcmp(flag, "--lookup-cache-dynamic")) {
                if (i == argc) {
                    missing("--lookup-cache-dynamic");
                }
                char *arg                            = argv[i++];
                bparams.gparams.lookup_cache_dynamic = std::string(arg);
                continue;
            }

            // server // embedding //

            if (!strcmp(flag, "--pooling")) {
                if (i == argc) {
                    missing("--pooling");
                }
                char *arg = argv[i++];
                std::string value(arg);
                if (value == "none") {
                    bparams.gparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
                } else if (value == "mean") {
                    bparams.gparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;
                } else if (value == "cls") {
                    bparams.gparams.pooling_type = LLAMA_POOLING_TYPE_CLS;
                } else if (value == "last") {
                    bparams.gparams.pooling_type = LLAMA_POOLING_TYPE_LAST;
                } else if (value == "rank") {
                    bparams.gparams.pooling_type = LLAMA_POOLING_TYPE_RANK;
                } else {
                    invalid("--pooling");
                }
                continue;
            }

            // server // image //

            if (!strcmp(flag, "--image-max-batch")) {
                if (i == argc) {
                    missing("--image-max-batch");
                }
                char *arg                        = argv[i++];
                bparams.sdparams.max_batch_count = std::stoi(std::string(arg));
                if (bparams.sdparams.max_batch_count < 1) {
                    invalid("--image-max-batch");
                }
                continue;
            }

            if (!strcmp(flag, "--image-max-height")) {
                if (i == argc) {
                    missing("--image-max-height");
                }
                char *arg                   = argv[i++];
                bparams.sdparams.max_height = std::stoi(std::string(arg));
                if (bparams.sdparams.max_height < 256) {
                    invalid("--image-max-height");
                }
                if (bparams.sdparams.max_height % 64 != 0) {
                    invalid("--image-max-height");
                }
                continue;
            }

            if (!strcmp(flag, "--image-max-width")) {
                if (i == argc) {
                    missing("--image-max-width");
                }
                char *arg                  = argv[i++];
                bparams.sdparams.max_width = std::stoi(std::string(arg));
                if (bparams.sdparams.max_width < 256) {
                    invalid("--image-max-width");
                }
                if (bparams.sdparams.max_width % 64 != 0) {
                    invalid("--image-max-width");
                }
                continue;
            }

            if (!strcmp(flag, "--image-guidance")) {
                if (i == argc) {
                    missing("--image-guidance");
                }
                char *arg                 = argv[i++];
                bparams.sdparams.guidance = std::stof(std::string(arg));
                if (bparams.sdparams.guidance < 0.0f) {
                    invalid("--image-guidance");
                }
                continue;
            }

            if (!strcmp(flag, "--image-strength")) {
                if (i == argc) {
                    missing("--image-strength");
                }
                char *arg                 = argv[i++];
                bparams.sdparams.strength = std::stof(std::string(arg));
                if (bparams.sdparams.strength < 0.0f || bparams.sdparams.strength > 1.0f) {
                    invalid("--image-strength");
                }
                continue;
            }

            if (!strcmp(flag, "--image-sampler")) {
                if (i == argc) {
                    missing("--image-sampler");
                }
                char *arg                = argv[i++];
                bparams.sdparams.sampler = sd_argument_to_sample_method(arg);
                continue;
            }

            if (!strcmp(flag, "--image-sample-steps")) {
                if (i == argc) {
                    missing("--image-sample-steps");
                }
                char *arg                     = argv[i++];
                bparams.sdparams.sample_steps = std::stoi(std::string(arg));
                if (bparams.sdparams.sample_steps < 1) {
                    invalid("--image-sample-steps");
                }
                continue;
            }

            if (!strcmp(flag, "--image-cfg-scale")) {
                if (i == argc) {
                    missing("--image-cfg-scale");
                }
                char *arg                  = argv[i++];
                bparams.sdparams.cfg_scale = std::stof(std::string(arg));
                if (bparams.sdparams.cfg_scale < 1.0f) {
                    invalid("--image-cfg-scale");
                }
                continue;
            }

            if (!strcmp(flag, "--image-slg-scale")) {
                if (i == argc) {
                    missing("--image-slg-scale");
                }
                char *arg                  = argv[i++];
                bparams.sdparams.slg_scale = std::stof(std::string(arg));
                if (bparams.sdparams.slg_scale < 0.0f) {
                    invalid("--image-slg-scale");
                }
                continue;
            }

            if (!strcmp(flag, "--image-skip-layer")) {
                if (i == argc) {
                    missing("--image-skip-layer");
                }
                char *arg = argv[i++];
                auto lyr  = std::stoi(std::string(arg));
                if (lyr < 0) {
                    invalid("--image-skip-layer");
                }
                static bool defaults_cleared = false;
                if (!defaults_cleared) {
                    bparams.sdparams.slg_skip_layers.clear();
                    defaults_cleared = true;
                }

                bparams.sdparams.slg_skip_layers.push_back(lyr);
                continue;
            }

            if (!strcmp(flag, "--image-slg-start")) {
                if (i == argc) {
                    missing("--image-slg-start");
                }
                char *arg                  = argv[i++];
                bparams.sdparams.slg_start = std::stof(std::string(arg));
                if (bparams.sdparams.slg_start < 0.0f) {
                    invalid("--image-slg-start");
                }
                continue;
            }

            if (!strcmp(flag, "--image-slg-end")) {
                if (i == argc) {
                    missing("--image-slg-end");
                }
                char *arg                = argv[i++];
                bparams.sdparams.slg_end = std::stof(std::string(arg));
                if (bparams.sdparams.slg_end < 0.0f) {
                    invalid("--image-slg-end");
                }
                continue;
            }

            if (!strcmp(flag, "--image-schedule")) {
                if (i == argc) {
                    missing("--image-schedule");
                }
                char *arg                 = argv[i++];
                bparams.sdparams.schedule = sd_argument_to_schedule(arg);
                continue;
            }

            if (llama_supports_gpu_offload()) {
                if (!strcmp(flag, "--image-no-text-encoder-model-offload")) {
                    bparams.sdparams.text_encoder_model_offload = false;
                    continue;
                }
            }

            if (!strcmp(flag, "--image-clip-l-model")) {
                if (i == argc) {
                    missing("--image-clip-l-model");
                }
                char *arg                     = argv[i++];
                bparams.sdparams.clip_l_model = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "--image-clip-g-model")) {
                if (i == argc) {
                    missing("--image-clip-g-model");
                }
                char *arg                     = argv[i++];
                bparams.sdparams.clip_g_model = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "--image-t5xxl-model")) {
                if (i == argc) {
                    missing("--image-t5xxl-model");
                }
                char *arg                    = argv[i++];
                bparams.sdparams.t5xxl_model = std::string(arg);
                continue;
            }

            if (llama_supports_gpu_offload()) {
                if (!strcmp(flag, "--image-no-vae-model-offload")) {
                    bparams.sdparams.vae_model_offload = false;
                    continue;
                }
            }

            if (!strcmp(flag, "--image-vae-model")) {
                if (i == argc) {
                    missing("--image-vae-model");
                }
                char *arg                  = argv[i++];
                bparams.sdparams.vae_model = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "--image-vae-tiling")) {
                bparams.sdparams.vae_tiling = true;
                continue;
            }

            if (!strcmp(flag, "--image-taesd-model")) {
                if (i == argc) {
                    missing("--image-taesd-model");
                }
                char *arg                    = argv[i++];
                bparams.sdparams.taesd_model = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "--image-upscale-model")) {
                if (i == argc) {
                    missing("--image-upscale-model");
                }
                char *arg                      = argv[i++];
                bparams.sdparams.upscale_model = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "--image-upscale-repeats")) {
                if (i == argc) {
                    missing("--image-upscale-repeats");
                }
                char *arg                        = argv[i++];
                bparams.sdparams.upscale_repeats = std::stoi(std::string(arg));
                if (bparams.sdparams.upscale_repeats < 1) {
                    invalid("--image-upscale-repeats");
                }
                continue;
            }

            if (llama_supports_gpu_offload()) {
                if (!strcmp(flag, "--image-no-control-net-model-offload")) {
                    bparams.sdparams.control_model_offload = false;
                    continue;
                }
            }

            if (!strcmp(flag, "--image-control-net-model")) {
                if (i == argc) {
                    missing("--image-control-net-model");
                }
                char *arg                          = argv[i++];
                bparams.sdparams.control_net_model = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "--image-control-strength")) {
                if (i == argc) {
                    missing("--image-control-strength");
                }
                char *arg                         = argv[i++];
                bparams.sdparams.control_strength = std::stof(std::string(arg));
                if (bparams.sdparams.control_strength < 0.0f || bparams.sdparams.control_strength > 1.0f) {
                    invalid("--image-control-strength");
                }
                continue;
            }

            if (!strcmp(flag, "--image-control-canny")) {
                bparams.sdparams.control_canny = true;
                continue;
            }

            // server //

            // rpc-server //

            if (!strcmp(flag, "--rpc-server-host")) {
                if (i == argc) {
                    missing("--rpc-server-host");
                }
                char *arg                = argv[i++];
                bparams.rparams.hostname = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "--rpc-server-port")) {
                if (i == argc) {
                    missing("--rpc-server-port");
                }
                char *arg            = argv[i++];
                bparams.rparams.port = std::stoi(std::string(arg));
                if (bparams.rparams.port < 0 || bparams.rparams.port > 65535) {
                    invalid("--rpc-server-port");
                }
                continue;
            }

            if (llama_supports_gpu_offload()) {
                if (!strcmp(flag, "--rpc-server-main-gpu")) {
                    if (i == argc) {
                        missing("--rpc-server-main-gpu");
                    }
                    char *arg                = argv[i++];
                    bparams.rparams.main_gpu = std::stoi(std::string(arg));
                    if (bparams.rparams.main_gpu >= int32_t(llama_max_devices())) {
                        invalid("--rpc-server-main-gpu");
                    }
                    continue;
                }
            }

            if (!strcmp(flag, "--rpc-server-reserve-memory")) {
                if (i == argc) {
                    missing("--rpc-server-reserve-memory");
                }
                char *arg                      = argv[i++];
                bparams.rparams.reserve_memory = std::stoul(std::string(arg)) << 20;
                continue;
            }

            // rpc-server //

            unknown(flag);
        }
    } catch (const std::invalid_argument &ex) {
        fprintf(stderr, "%s\n", ex.what());
        return false;
    }

    // Retrieve params from environment variables
    get_env("LLAMA_ARG_MODEL", bparams.gparams.model);
    get_env("LLAMA_ARG_MODEL_ALIAS", bparams.gparams.model_alias);
    get_env("LLAMA_ARG_THREADS", bparams.gparams.cpuparams.n_threads);
    get_env("LLAMA_ARG_CTX_SIZE", bparams.gparams.n_ctx);
    get_env("LLAMA_ARG_N_PARALLEL", bparams.gparams.n_parallel);
    get_env("LLAMA_ARG_BATCH", bparams.gparams.n_batch);
    get_env("LLAMA_ARG_UBATCH", bparams.gparams.n_ubatch);
    get_env("LLAMA_ARG_DEVICE", bparams.gparams.devices);
    get_env("LLAMA_ARG_N_GPU_LAYERS", bparams.gparams.n_gpu_layers);
    get_env("LLAMA_ARG_THREADS_HTTP", bparams.gparams.n_threads_http);
    get_env("LLAMA_ARG_CACHE_PROMPT", bparams.cache_prompt);
    get_env("LLAMA_ARG_CACHE_REUSE", bparams.gparams.n_cache_reuse);
    get_env("LLAMA_ARG_CHAT_TEMPLATE", bparams.gparams.chat_template);
    get_env("LLAMA_ARG_N_PREDICT", bparams.gparams.n_predict);
    get_env("LLAMA_ARG_METRICS", bparams.gparams.endpoint_metrics);
    get_env("LLAMA_ARG_SLOTS", bparams.gparams.endpoint_slots);
    get_env("LLAMA_ARG_EMBEDDINGS", bparams.gparams.embedding);
    get_env("LLAMA_ARG_FLASH_ATTN", bparams.gparams.flash_attn);
    get_env("LLAMA_ARG_DEFRAG_THOLD", bparams.gparams.defrag_thold);
    get_env("LLAMA_ARG_CONT_BATCHING", bparams.gparams.cont_batching);
    get_env("LLAMA_ARG_HOST", bparams.gparams.hostname);
    get_env("LLAMA_ARG_PORT", bparams.gparams.port);
    get_env("LLAMA_ARG_DRAFT", bparams.gparams.speculative.n_max);
    get_env("LLAMA_ARG_MODEL_DRAFT", bparams.gparams.speculative.model);
    get_env("LLAMA_ARG_DEVICE_DRAFT", bparams.gparams.speculative.devices);
    get_env("LLAMA_ARG_N_GPU_LAYERS_DRAFT", bparams.gparams.speculative.n_gpu_layers);
    get_env("LLAMA_ARG_LOOKUP_NGRAM_MIN", bparams.lookup_ngram_min);
    get_env("LLAMA_ARG_LOOKUP_CACHE_STATIC", bparams.gparams.lookup_cache_static);
    get_env("LLAMA_ARG_LOOKUP_CACHE_DYNAMIC", bparams.gparams.lookup_cache_dynamic);
    get_env("LLAMA_ARG_RPC_SERVER_HOST", bparams.rparams.hostname);
    get_env("LLAMA_ARG_RPC_SERVER_PORT", bparams.rparams.port);
    get_env("LLAMA_LOG_VERBOSITY", bparams.gparams.verbosity);

    // Postprocess params
    postprocess_cpu_params(bparams.gparams.cpuparams, nullptr);
    postprocess_cpu_params(bparams.gparams.cpuparams_batch, &bparams.gparams.cpuparams);
    postprocess_cpu_params(bparams.gparams.speculative.cpuparams, &bparams.gparams.cpuparams);
    postprocess_cpu_params(bparams.gparams.speculative.cpuparams_batch, &bparams.gparams.cpuparams_batch);
    if (!bparams.gparams.devices.empty() && bparams.gparams.speculative.devices.empty()) {
        bparams.gparams.speculative.devices = bparams.gparams.devices;
    }

    if (!bparams.gparams.kv_overrides.empty()) {
        bparams.gparams.kv_overrides.emplace_back();
        bparams.gparams.kv_overrides.back().key[0] = 0;
    }

    if (bparams.endpoint_images) {
        bparams.gparams.enable_chat_template     = false;
        bparams.sdparams.model                   = bparams.gparams.model;
        bparams.sdparams.model_alias             = bparams.gparams.model_alias;
        bparams.sdparams.flash_attn              = bparams.gparams.flash_attn;
        bparams.sdparams.n_threads               = bparams.gparams.cpuparams.n_threads;
        bparams.sdparams.main_gpu                = bparams.gparams.main_gpu;
        bparams.sdparams.lora_init_without_apply = bparams.gparams.lora_init_without_apply;
        bparams.sdparams.lora_adapters           = bparams.gparams.lora_adapters;
    }

    return true;
}