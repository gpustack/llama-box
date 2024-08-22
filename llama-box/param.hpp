#pragma once

#include <cstdarg>
#include <utility>

#include "llama.cpp/common/common.h"
#include "llama.cpp/common/grammar-parser.h"
#include "llama.cpp/common/json-schema-to-grammar.h"
#define JSON_ASSERT GGML_ASSERT
#include "llama.cpp/common/json.hpp"
#include "llama.cpp/ggml/include/ggml.h"
#include "llama.cpp/include/llama.h"
#include "rpcserver.hpp"

// version
extern const char *LLAMA_BOX_GIT_TREE_STATE;
extern const char *LLAMA_BOX_GIT_VERSION;
extern const char *LLAMA_BOX_GIT_COMMIT;

using json = nlohmann::json;

struct llama_box_params {
    gpt_params gparams;
    rpc_server_params rparams;

    int32_t conn_idle = 60;       // connection idle in seconds
    int32_t conn_keepalive = 15;  // connection keep-alive in seconds
    int32_t n_tps = 0;            // maximum number of tokens per seconds
    int32_t lookup_ngram_min = 0; // minimum n-gram size for lookup cache
};

static int unknown(const char *flag) {
    throw std::invalid_argument("Unknown argument: " + std::string(flag));
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

    const auto &params = bparams.gparams;
    const auto &rparams = bparams.rparams;
    const auto &sparams = params.sparams;
    std::string sampler_type_chars;
    std::string sampler_type_names;
    for (const auto sampler_type : sparams.samplers_sequence) {
        sampler_type_chars += static_cast<char>(sampler_type);
        sampler_type_names += llama_sampling_type_to_str(sampler_type) + ";";
    }
    sampler_type_names.pop_back();

    // clang-format off
    std::vector<opt> opts;
    // general //
    opts.push_back({ "general" });
    opts.push_back({ "general",            "-h,    --help, --usage",        "print usage and exit" });
    opts.push_back({ "general",            "       --version",              "show version and build info" });
    opts.push_back({ "general",            "       --log-format {text,json}",
                                                                            "log output format: json or text (default: json)" });
    // general //
    // server //
    opts.push_back({ "server" });
    opts.push_back({ "server",             "       --host HOST",            "ip address to listen (default: %s)", params.hostname.c_str() });
    opts.push_back({ "server",             "       --port PORT",            "port to listen (default: %d)", params.port });
    opts.push_back({ "server",             "-to    --timeout N",            "server read/write timeout in seconds (default: %d)", params.timeout_read });
    opts.push_back({ "server",             "       --threads-http N",       "number of threads used to process HTTP requests (default: %d)", params.n_threads_http });
    opts.push_back({ "server",             "       --system-prompt-file FILE",
                                                                            "set a file to load a system prompt (initial prompt of all slots), this is useful for chat applications" });
    opts.push_back({ "server",             "       --metrics",              "enable prometheus compatible metrics endpoint (default: %s)", params.endpoint_metrics ? "enabled" : "disabled" });
    opts.push_back({ "server",             "       --infill",               "enable infill endpoint (default: %s)", params.infill? "enabled" : "disabled" });
    opts.push_back({ "server",             "       --embeddings",           "enable embedding endpoint (default: %s)", params.embedding ? "enabled" : "disabled" });
    opts.push_back({ "server",             "       --no-slots",             "disables slots monitoring endpoint (default: %s)", params.endpoint_slots ? "enabled" : "disabled" });
    opts.push_back({ "server",             "       --slot-save-path PATH",  "path to save slot kv cache (default: disabled)" });
    opts.push_back({ "server",             "       --chat-template JINJA_TEMPLATE",
                                                                            "set custom jinja chat template (default: template taken from model's metadata)\n"
                                                                            "only commonly used templates are accepted:\n"
                                                                            "https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template" });
    opts.push_back({ "server",             "       --chat-template-file FILE",
                                                                            "set a file to load a custom jinja chat template (default: template taken from model's metadata)" });
    opts.push_back({ "server",             "-sps,  --slot-prompt-similarity N",
                                                                            "how much the prompt of a request must match the prompt of a slot in order to use that slot (default: %.2f, 0.0 = disabled)\n", params.slot_prompt_similarity });
    opts.push_back({ "server",             "       --conn-idle N",          "server connection idle in seconds (default: %d)", bparams.conn_idle });
    opts.push_back({ "server",             "       --conn-keepalive N",     "server connection keep-alive in seconds (default: %d)", bparams.conn_keepalive });
    opts.push_back({ "server",             "-tps   --tokens-per-second N",  "maximum number of tokens per second (default: %d, 0 = disabled, -1 = try to detect)\n"
                                                                            "when enabled, limit the request within its X-Request-Tokens-Per-Second HTTP header", bparams.n_tps });
    opts.push_back({ "server",             "-m,    --model FILE",           "model path (default: %s)", DEFAULT_MODEL_PATH });
    opts.push_back({ "server",             "-a,    --alias NAME",           "model name alias (default: %s)", params.model_alias.c_str() });
    opts.push_back({ "server",             "-s,    --seed N",               "RNG seed (default: %d, use random seed for < 0)", params.seed });
    opts.push_back({ "server",             "-t,    --threads N",            "number of threads to use during generation (default: %d)", params.n_threads });
    opts.push_back({ "server",             "-tb,   --threads-batch N",      "number of threads to use during batch and prompt processing (default: same as --threads)" });
    opts.push_back({ "server",             "-c,    --ctx-size N",           "size of the prompt context (default: %d, 0 = loaded from model)", params.n_ctx });
    opts.push_back({ "server",             "-n,    --predict N",            "number of tokens to predict (default: %d, -1 = infinity, -2 = until context filled)", params.n_predict });
    opts.push_back({ "server",             "-b,    --batch-size N",         "logical maximum batch size (default: %d)", params.n_batch });
    opts.push_back({ "server",             "-ub,   --ubatch-size N",        "physical maximum batch size (default: %d)", params.n_ubatch });
    opts.push_back({ "server",             "       --keep N",               "number of tokens to keep from the initial prompt (default: %d, -1 = all)", params.n_keep });
    opts.push_back({ "server",             "       --chunks N",             "max number of chunks to process (default: %d, -1 = all)", params.n_chunks });
    opts.push_back({ "server",             "-fa,   --flash-attn",           "enable Flash Attention (default: %s)", params.flash_attn ? "enabled" : "disabled" });
    opts.push_back({ "server",             "-e,    --escape",               R"(process escapes sequences (\n, \r, \t, \', \", \\) (default: %s))", params.escape ? "true" : "false" });
    opts.push_back({ "server",             "       --no-escape",            "do not process escape sequences" });
    opts.push_back({ "server",             "       --samplers SAMPLERS",    "samplers that will be used for generation in the order, separated by \';\'\n"
                                                                            "(default: %s)", sampler_type_names.c_str() });
    opts.push_back({ "server",             "       --sampling-seq SEQUENCE",
                                                                            "simplified sequence for samplers that will be used (default: %s)", sampler_type_chars.c_str() });
    opts.push_back({ "server",             "       --penalize-nl",          "penalize newline tokens (default: %s)", sparams.penalize_nl ? "true" : "false" });
    opts.push_back({ "server",             "       --temp N",               "temperature (default: %.1f)", (double)sparams.temp });
    opts.push_back({ "server",             "       --top-k N",              "top-k sampling (default: %d, 0 = disabled)", sparams.top_k });
    opts.push_back({ "server",             "       --top-p N",              "top-p sampling (default: %.1f, 1.0 = disabled)", (double)sparams.top_p });
    opts.push_back({ "server",             "       --min-p N",              "min-p sampling (default: %.1f, 0.0 = disabled)", (double)sparams.min_p });
    opts.push_back({ "server",             "       --tfs N",                "tail free sampling, parameter z (default: %.1f, 1.0 = disabled)", (double)sparams.tfs_z });
    opts.push_back({ "server",             "       --typical N",            "locally typical sampling, parameter p (default: %.1f, 1.0 = disabled)", (double)sparams.typical_p });
    opts.push_back({ "server",             "       --repeat-last-n N",      "last n tokens to consider for penalize (default: %d, 0 = disabled, -1 = ctx_size)", sparams.penalty_last_n });
    opts.push_back({ "server",             "       --repeat-penalty N",     "penalize repeat sequence of tokens (default: %.1f, 1.0 = disabled)", (double)sparams.penalty_repeat });
    opts.push_back({ "server",             "       --presence-penalty N",   "repeat alpha presence penalty (default: %.1f, 0.0 = disabled)", (double)sparams.penalty_present });
    opts.push_back({ "server",             "       --frequency-penalty N",  "repeat alpha frequency penalty (default: %.1f, 0.0 = disabled)", (double)sparams.penalty_freq });
    opts.push_back({ "server",             "       --dynatemp-range N",     "dynamic temperature range (default: %.1f, 0.0 = disabled)", (double)sparams.dynatemp_range });
    opts.push_back({ "server",             "       --dynatemp-exp N",       "dynamic temperature exponent (default: %.1f)", (double)sparams.dynatemp_exponent });
    opts.push_back({ "server",             "       --mirostat N",           "use Mirostat sampling,\n"
                                                                            "Top K, Nucleus, Tail Free and Locally Typical samplers are ignored if used\n"
                                                                            "(default: %d, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)", sparams.mirostat });
    opts.push_back({ "server",             "       --mirostat-lr N",        "Mirostat learning rate, parameter eta (default: %.1f)", (double)sparams.mirostat_eta });
    opts.push_back({ "server",             "       --mirostat-ent N",       "Mirostat target entropy, parameter tau (default: %.1f)", (double)sparams.mirostat_tau });
    opts.push_back({ "server",             "-l     --logit-bias TOKEN_ID(+/-)BIAS",
                                                                            "modifies the likelihood of token appearing in the completion,\n"
                                                                            "i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',\n"
                                                                            "or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'" });
    opts.push_back({ "server",             "       --grammar GRAMMAR",      "BNF-like grammar to constrain generations (see samples in grammars/ dir) (default: '%s')", sparams.grammar.c_str() });
    opts.push_back({ "server",             "       --grammar-file FILE",    "file to read grammar from" });
    opts.push_back({ "server",             "-j,    --json-schema SCHEMA",
                                                                            "JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any JSON object\n"
                                                                            "For schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead" });
    opts.push_back({ "server",             "       --rope-scaling {none,linear,yarn}",
                                                                            "RoPE frequency scaling method, defaults to linear unless specified by the model" });
    opts.push_back({ "server",             "       --rope-scale N",         "RoPE context scaling factor, expands context by a factor of N" });
    opts.push_back({ "server",             "       --rope-freq-base N",     "RoPE base frequency, used by NTK-aware scaling (default: loaded from model)" });
    opts.push_back({ "server",             "       --rope-freq-scale N",    "RoPE frequency scaling factor, expands context by a factor of 1/N" });
    opts.push_back({ "server",             "       --yarn-orig-ctx N",      "YaRN: original context size of model (default: %d = model training context size)", params.yarn_orig_ctx });
    opts.push_back({ "server",             "       --yarn-ext-factor N",    "YaRN: extrapolation mix factor (default: %.1f, 0.0 = full interpolation)", (double)params.yarn_ext_factor });
    opts.push_back({ "server",             "       --yarn-attn-factor N",   "YaRN: scale sqrt(t) or attention magnitude (default: %.1f)", (double)params.yarn_attn_factor });
    opts.push_back({ "server",             "       --yarn-beta-fast N",     "YaRN: low correction dim or beta (default: %.1f)", (double)params.yarn_beta_fast });
    opts.push_back({ "server",             "       --yarn-beta-slow N",     "YaRN: high correction dim or alpha (default: %.1f)", (double)params.yarn_beta_slow });
    opts.push_back({ "server",             "-gan,  --grp-attn-n N",         "group-attention factor (default: %d)", params.grp_attn_n });
    opts.push_back({ "server",             "-gaw,  --grp-attn-w N",         "group-attention width (default: %.1f)", (double)params.grp_attn_w });
    opts.push_back({ "server",             "-nkvo, --no-kv-offload",        "disable KV offload" });
    opts.push_back({ "server",             "-ctk,  --cache-type-k TYPE",    "KV cache data type for K (default: %s)", params.cache_type_k.c_str() });
    opts.push_back({ "server",             "-ctv,  --cache-type-v TYPE",    "KV cache data type for V (default: %s)", params.cache_type_v.c_str() });
    opts.push_back({ "server",             "-dt,   --defrag-thold N",       "KV cache defragmentation threshold (default: %.1f, < 0 - disabled)", (double)params.defrag_thold });
    opts.push_back({ "server",             "-np,   --parallel N",           "number of parallel sequences to decode (default: %d)", params.n_parallel });
    opts.push_back({ "server",             "-cb,   --cont-batching",        "enable continuous batching (a.k.a dynamic batching) (default: %s)", params.cont_batching ? "enabled" : "disabled" });
    opts.push_back({ "server",             "-nocb, --no-cont-batching",     "disable continuous batching" });
    opts.push_back({ "server",             "       --mmproj FILE",          "path to a multimodal projector file for LLaVA" });
    if (llama_supports_mlock()) {
        opts.push_back({ "server",         "       --mlock",                "force system to keep model in RAM rather than swapping or compressing" });
    }
    if (llama_supports_mmap()) {
        opts.push_back({ "server",         "       --no-mmap",              "do not memory-map model (slower load but may reduce pageouts if not using mlock)" });
    }
    opts.push_back({ "server",             "       --numa TYPE",            "attempt optimizations that help on some NUMA systems\n"
                                                                            "  - distribute: spread execution evenly over all nodes\n"
                                                                            "  - isolate: only spawn threads on CPUs on the node that execution started on\n"
                                                                            "  - numactl: use the CPU map provided by numactl\n"
                                                                            "if run without this previously, it is recommended to drop the system page cache before using this\n"
                                                                            "see https://github.com/ggerganov/llama.cpp/issues/1437" });
    opts.push_back({ "server",             "       --override-kv KEY=TYPE:VALUE",
                                                                            "advanced option to override model metadata by key. may be specified multiple times.\n"
                                                                            "types: int, float, bool, str. example: --override-kv tokenizer.ggml.add_bos_token=bool:false" });
    opts.push_back({ "server",             "       --lora FILE",            "apply LoRA adapter (implies --no-mmap)" });
    opts.push_back({ "server",             "       --lora-scaled FILE SCALE",
                                                                            "apply LoRA adapter with user defined scaling S (implies --no-mmap)" });
    opts.push_back({ "server",             "       --lora-init-without-apply",
                                                                            "load LoRA adapters without applying them (apply later via POST /lora-adapters) (default: %s)", params.lora_init_without_apply ? "enabled" : "disabled" });
    opts.push_back({ "server",             "       --control-vector FILE",  "add a control vector" });
    opts.push_back({ "server",             "       --control-vector-scaled FILE SCALE",
                                                                            "add a control vector with user defined scaling SCALE" });
    opts.push_back({ "server",             "       --control-vector-layer-range START END",
                                                                            "layer range to apply the control vector(s) to, start and end inclusive" });
    opts.push_back({ "server",             "       --spm-infill",           "use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as some models prefer this (default: %s)", params.spm_infill ? "enabled" : "disabled" });
    opts.push_back({ "server",             "-sp,   --special",              "special tokens output enabled (default: %s)", params.special ? "true" : "false" });
    if (llama_supports_gpu_offload()) {
        opts.push_back({ "server",         "-ngl,  --gpu-layers N",         "number of layers to store in VRAM" });
        opts.push_back({ "server",         "-sm,   --split-mode SPLIT_MODE",
                                                                            "how to split the model across multiple GPUs, one of:\n"
                                                                            "  - none: use one GPU only\n"
                                                                            "  - layer (default): split layers and KV across GPUs\n"
                                                                            "  - row: split rows across GPUs" });
        opts.push_back({ "server",         "-ts,   --tensor-split SPLIT",
                                                                            "fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1" });
        opts.push_back({ "server",         "-mg,   --main-gpu N",           "the GPU to use for the model (with split-mode = none),\n"
                                                                            "or for intermediate results and KV (with split-mode = row) (default: %d)", params.main_gpu });
    }
    opts.push_back({ "server",             "       --rpc SERVERS",          "comma separated list of RPC servers" });
    // server // speculative //
    opts.push_back({ "server/speculative" });
    opts.push_back({ "server/speculative", "       --draft N",              "number of tokens to draft for speculative decoding (default: %d)", params.n_draft });
    opts.push_back({ "server/speculative", "-md,   --model-draft FNAME",    "draft model for speculative decoding (default: unused)" });
    opts.push_back({ "server/speculative", "-td,   --threads-draft N",      "number of threads to use during generation (default: same as --threads)" });
    opts.push_back({ "server/speculative", "-tbd,  --threads-batch-draft N",
                                                                            "number of threads to use during batch and prompt processing (default: same as --threads-draft)" });
    if (llama_supports_gpu_offload()) {
        opts.push_back({ "server/speculative",
                                           "-ngld, --gpu-layers-draft N",   "number of layers to store in VRAM for the draft model" });
    }
    opts.push_back({ "server/speculative", "       --lookup-ngram-min N",   "minimum n-gram size for lookup cache (default: %d, 0 = disabled)", bparams.lookup_ngram_min });
    opts.push_back({ "server/speculative", "-lcs,  --lookup-cache-static FILE",
                                                                            "path to static lookup cache to use for lookup decoding (not updated by generation)" });
    opts.push_back({ "server/speculative", "-lcd,  --lookup-cache-dynamic FILE",
                                                                            "path to dynamic lookup cache to use for lookup decoding (updated by generation)" });
    // server // speculative //
    // server //
    // rpc-server //
    opts.push_back({ "rpc-server" });
    opts.push_back({ "rpc-server",       "       --rpc-server-host HOST",   "ip address to rpc server listen (default: %s)", rparams.hostname.c_str() });
    opts.push_back({ "rpc-server",       "       --rpc-server-port PORT",   "port to rpc server listen (default: %d)", rparams.port });
    if (llama_supports_gpu_offload()) {
        opts.push_back({ "rpc-server",   "       --rpc-server-main-gpu N",  "the GPU to use for the rpc server (default: %d)", rparams.main_gpu });
    }
    opts.push_back({ "rpc-server",       "       --rpc-server-reserve-memory MEM",
                                                                            "reserve memory in MiB (default: %zu)", rparams.reserve_memory });
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
        size_t start = 0;
        size_t end = desc.find('\n');
        while (end != std::string::npos) {
            printf("%s\n%34s", desc.substr(start, end - start).c_str(), "");
            start = end + 1;
            end = desc.find('\n', start);
        }

        printf("%s\n", desc.substr(start).c_str());
    }
    printf("\n");
}

//
// Environment variable utils
//

template <typename T>
static typename std::enable_if<std::is_same<T, std::string>::value, void>::type
get_env(std::string name, T &target) {
    char *value = std::getenv(name.c_str());
    target = value ? std::string(value) : target;
}

template <typename T>
static
    typename std::enable_if<!std::is_same<T, bool>::value && std::is_integral<T>::value, void>::type
    get_env(std::string name, T &target) {
    char *value = std::getenv(name.c_str());
    target = value ? std::stoi(value) : target;
}

template <typename T>
static typename std::enable_if<std::is_floating_point<T>::value, void>::type
get_env(std::string name, T &target) {
    char *value = std::getenv(name.c_str());
    target = value ? std::stof(value) : target;
}

template <typename T>
static typename std::enable_if<std::is_same<T, bool>::value, void>::type get_env(std::string name,
                                                                                 T &target) {
    char *value = std::getenv(name.c_str());
    if (value) {
        std::string val(value);
        target = val == "1" || val == "true";
    }
}

static bool llama_box_params_parse(int argc, char **argv, llama_box_params &bparams) {
    try {
        for (int i = 1; i < argc;) {
            const char *flag = argv[i++];

            if (*flag != '-') {
                unknown(flag);
            }

            // general //

            if (!strcmp(flag, "-h") || !strcmp(flag, "--help") || !strcmp(flag, "--usage")) {
                llama_box_params_print_usage(argc, argv, bparams);
                exit(0);
            }

            if (!strcmp(flag, "--version")) {
                fprintf(stderr, "version: %s (%s)\n", LLAMA_BOX_GIT_VERSION, LLAMA_BOX_GIT_COMMIT);
                fprintf(stderr, "llama.cpp version: %d (%s)\n", LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
                fprintf(stderr, "built with %s for %s\n", LLAMA_COMPILER, LLAMA_BUILD_TARGET);
                exit(0);
            }

            if (!strcmp(flag, "--log-format")) {
                if (i == argc) {
                    missing("--log-format");
                }
                char *arg = argv[i++];
                if (!strcmp(arg, "json")) {
                    bparams.gparams.log_json = true;
                } else if (!strcmp(arg, "text")) {
                    bparams.gparams.log_json = false;
                } else {
                    unknown("--log-format");
                }
                continue;
            }

            // general //

            // server //

            if (!strcmp(flag, "--host")) {
                if (i == argc) {
                    missing("--host");
                }
                char *arg = argv[i++];
                bparams.gparams.hostname = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "--port")) {
                if (i == argc) {
                    missing("--port");
                }
                char *arg = argv[i++];
                bparams.gparams.port = std::stoi(std::string(arg));
                if (bparams.gparams.port <= 0 || bparams.gparams.port > 65535) {
                    invalid("--port");
                }
                continue;
            }

            if (!strcmp(flag, "-to") || !strcmp(flag, "--timeout")) {
                if (i == argc) {
                    missing("--timeout");
                }
                char *arg = argv[i++];
                bparams.gparams.timeout_read = std::stoi(std::string(arg));
                bparams.gparams.timeout_write = bparams.gparams.timeout_read;
                continue;
            }

            if (!strcmp(flag, "--threads-http")) {
                if (i == argc) {
                    missing("--threads-http");
                }
                char *arg = argv[i++];
                bparams.gparams.n_threads_http = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-spf") || !strcmp(flag, "--system-prompt-file")) {
                if (i == argc) {
                    missing("--system-prompt-file");
                }
                char *arg = argv[i++];
                std::ifstream file(arg);
                if (!file) {
                    invalid("--system-prompt-file");
                }
                std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(),
                          std::back_inserter(bparams.gparams.system_prompt));
                continue;
            }

            if (!strcmp(flag, "--metrics")) {
                bparams.gparams.endpoint_metrics = true;
                continue;
            }

            if (!strcmp(flag, "--infill")) {
                bparams.gparams.infill = true;
                continue;
            }

            if (!strcmp(flag, "--embedding") || !strcmp(flag, "--embeddings")) {
                bparams.gparams.embedding = true;
                continue;
            }

            if (!strcmp(flag, "--no-slots")) {
                bparams.gparams.endpoint_slots = false;
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
                if (t.size() > 20 && !llama_chat_verify_template(t)) {
                    invalid("--chat-template");
                }
                bparams.gparams.enable_chat_template = true;
                bparams.gparams.chat_template = t;
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
                std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(),
                          std::back_inserter(t));
                if (t.size() > 20 && !llama_chat_verify_template(t)) {
                    invalid("--chat-template-file");
                }
                bparams.gparams.enable_chat_template = true;
                bparams.gparams.chat_template = t;
                continue;
            }

            if (!strcmp(flag, "-sps") || !strcmp(flag, "--slot-prompt-similarity")) {
                if (i == argc) {
                    missing("--slot-prompt-similarity");
                }
                char *arg = argv[i++];
                bparams.gparams.slot_prompt_similarity = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--conn-idle")) { // extend
                if (i == argc) {
                    missing("--conn-idle");
                }
                char *arg = argv[i++];
                bparams.conn_idle = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--conn-keepalive")) { // extend
                if (i == argc) {
                    missing("--conn-keepalive");
                }
                char *arg = argv[i++];
                bparams.conn_keepalive = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-tps") || !strcmp(flag, "--tokens-per-second")) { // extend
                if (i == argc) {
                    missing("--tokens-per-second");
                }
                char *arg = argv[i++];
                bparams.n_tps = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-m") || !strcmp(flag, "--model")) {
                if (i == argc) {
                    missing("--model");
                }
                char *arg = argv[i++];
                bparams.gparams.model = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "-a") || !strcmp(flag, "--alias")) {
                if (i == argc) {
                    missing("--alias");
                }
                char *arg = argv[i++];
                bparams.gparams.model_alias = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "-s") || !strcmp(flag, "--seed")) {
                if (i == argc) {
                    missing("--seed");
                }
                char *arg = argv[i++];
                bparams.gparams.seed = std::stoul(std::string(arg));
                bparams.gparams.sparams.seed = bparams.gparams.seed;
                continue;
            }

            if (!strcmp(flag, "-t") || !strcmp(flag, "--threads")) {
                if (i == argc) {
                    missing("--threads");
                }
                char *arg = argv[i++];
                bparams.gparams.n_threads = std::stoi(std::string(arg));
                if (bparams.gparams.n_threads <= 0) {
                    bparams.gparams.n_threads = cpu_get_num_math();
                }
                continue;
            }

            if (!strcmp(flag, "-tb") || !strcmp(flag, "--threads-batch")) {
                if (i == argc) {
                    missing("--threads-batch");
                }
                char *arg = argv[i++];
                bparams.gparams.n_threads_batch = std::stoi(std::string(arg));
                if (bparams.gparams.n_threads_batch <= 0) {
                    bparams.gparams.n_threads_batch = cpu_get_num_math();
                }
                continue;
            }

            if (!strcmp(flag, "-c") || !strcmp(flag, "--ctx-size")) {
                if (i == argc) {
                    missing("--ctx-size");
                }
                char *arg = argv[i++];
                bparams.gparams.n_ctx = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-n") || !strcmp(flag, "--predict")) {
                if (i == argc) {
                    missing("--predict");
                }
                char *arg = argv[i++];
                bparams.gparams.n_predict = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-b") || !strcmp(flag, "--batch-size")) {
                if (i == argc) {
                    missing("--batch-size");
                }
                char *arg = argv[i++];
                bparams.gparams.n_batch = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-ub") || !strcmp(flag, "--ubatch-size")) {
                if (i == argc) {
                    missing("--ubatch-size");
                }
                char *arg = argv[i++];
                bparams.gparams.n_ubatch = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--keep")) {
                if (i == argc) {
                    missing("--keep");
                }
                char *arg = argv[i++];
                bparams.gparams.n_keep = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--chunks")) {
                if (i == argc) {
                    missing("--chunks");
                }
                char *arg = argv[i++];
                bparams.gparams.n_chunks = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-fa") || !strcmp(flag, "--flash-attn")) {
                bparams.gparams.flash_attn = true;
                continue;
            }

            if (!strcmp(flag, "-e") || !strcmp(flag, "--escape")) {
                bparams.gparams.escape = true;
                return true;
            }

            if (!strcmp(flag, "--no-escape")) {
                bparams.gparams.escape = false;
                continue;
            }

            if (!strcmp(flag, "--samplers")) {
                if (i == argc) {
                    missing("--samplers");
                }
                char *arg = argv[i++];
                const auto sampler_names = string_split(arg, ';');
                bparams.gparams.sparams.samplers_sequence =
                    llama_sampling_types_from_names(sampler_names, true);
                continue;
            }

            if (!strcmp(flag, "--sampling-seq")) {
                if (i == argc) {
                    missing("--sampling-seq");
                }
                char *arg = argv[i++];
                bparams.gparams.sparams.samplers_sequence = llama_sampling_types_from_chars(arg);
                continue;
            }

            if (!strcmp(flag, "--penalize-nl")) {
                bparams.gparams.sparams.penalize_nl = true;
                continue;
            }

            if (!strcmp(flag, "--temp")) {
                if (i == argc) {
                    missing("--temp");
                }
                char *arg = argv[i++];
                bparams.gparams.sparams.temp = std::stof(std::string(arg));
                bparams.gparams.sparams.temp = std::max(bparams.gparams.sparams.temp, 0.0f);
                continue;
            }

            if (!strcmp(flag, "--top-k")) {
                if (i == argc) {
                    missing("--top-k");
                }
                char *arg = argv[i++];
                bparams.gparams.sparams.top_k = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--top-p")) {
                if (i == argc) {
                    missing("--top-p");
                }
                char *arg = argv[i++];
                bparams.gparams.sparams.top_p = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--min-p")) {
                if (i == argc) {
                    missing("--min-p");
                }
                char *arg = argv[i++];
                bparams.gparams.sparams.min_p = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--tfs")) {
                if (i == argc) {
                    missing("--tfs");
                }
                char *arg = argv[i++];
                bparams.gparams.sparams.tfs_z = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--typical")) {
                if (i == argc) {
                    missing("--typical");
                }
                char *arg = argv[i++];
                bparams.gparams.sparams.typical_p = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--repeat-last-n")) {
                if (i == argc) {
                    missing("--repeat-last-n");
                }
                char *arg = argv[i++];
                bparams.gparams.sparams.penalty_last_n = std::stoi(std::string(arg));
                bparams.gparams.sparams.n_prev = std::max(bparams.gparams.sparams.n_prev,
                                                          bparams.gparams.sparams.penalty_last_n);
                continue;
            }

            if (!strcmp(flag, "--repeat-penalty")) {
                if (i == argc) {
                    missing("--repeat-penalty");
                }
                char *arg = argv[i++];
                bparams.gparams.sparams.penalty_repeat = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--presence-penalty")) {
                if (i == argc) {
                    missing("--presence-penalty");
                }
                char *arg = argv[i++];
                bparams.gparams.sparams.penalty_present = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--frequency-penalty")) {
                if (i == argc) {
                    missing("--frequency-penalty");
                }
                char *arg = argv[i++];
                bparams.gparams.sparams.penalty_freq = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--dynatemp-range")) {
                if (i == argc) {
                    missing("--dynatemp-range");
                }
                char *arg = argv[i++];
                bparams.gparams.sparams.dynatemp_range = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--dynatemp-exp")) {
                if (i == argc) {
                    missing("--dynatemp-exp");
                }
                char *arg = argv[i++];
                bparams.gparams.sparams.dynatemp_exponent = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--mirostat")) {
                if (i == argc) {
                    missing("--mirostat");
                }
                char *arg = argv[i++];
                bparams.gparams.sparams.mirostat = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--mirostat-lr")) {
                if (i == argc) {
                    missing("--mirostat-lr");
                }
                char *arg = argv[i++];
                bparams.gparams.sparams.mirostat_eta = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--mirostat-ent")) {
                if (i == argc) {
                    missing("--mirostat-ent");
                }
                char *arg = argv[i++];
                bparams.gparams.sparams.mirostat_tau = std::stof(std::string(arg));
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
                std::string value_str;
                if (ss >> key && ss >> sign && std::getline(ss, value_str) &&
                    (sign == '+' || sign == '-')) {
                    bparams.gparams.sparams.logit_bias[key] =
                        std::stof(value_str) * ((sign == '-') ? -1.0f : 1.0f);
                } else {
                    invalid("--logit-bias");
                }
                continue;
            }

            if (!strcmp(flag, "--grammar")) {
                if (i == argc) {
                    missing("--grammar");
                }
                char *arg = argv[i++];
                bparams.gparams.sparams.grammar = std::string(arg);
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
                          std::back_inserter(bparams.gparams.sparams.grammar));
                continue;
            }

            if (!strcmp(flag, "-j") || !strcmp(flag, "--json-schema")) {
                if (i == argc) {
                    missing("--json-schema");
                }
                char *arg = argv[i++];
                bparams.gparams.sparams.grammar =
                    json_schema_to_grammar(json::parse(std::string(arg)));
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
                char *arg = argv[i++];
                bparams.gparams.rope_freq_scale = 1.0f / std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--rope-freq-base")) {
                if (i == argc) {
                    missing("--rope-freq-base");
                }
                char *arg = argv[i++];
                bparams.gparams.rope_freq_base = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--rope-freq-scale")) {
                if (i == argc) {
                    missing("--rope-freq-scale");
                }
                char *arg = argv[i++];
                bparams.gparams.rope_freq_scale = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--yarn-orig-ctx")) {
                if (i == argc) {
                    missing("--yarn-orig-ctx");
                }
                char *arg = argv[i++];
                bparams.gparams.yarn_orig_ctx = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--yarn-ext-factor")) {
                if (i == argc) {
                    missing("--yarn-ext-factor");
                }
                char *arg = argv[i++];
                bparams.gparams.yarn_ext_factor = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--yarn-attn-factor")) {
                if (i == argc) {
                    missing("--yarn-attn-factor");
                }
                char *arg = argv[i++];
                bparams.gparams.yarn_attn_factor = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--yarn-beta-fast")) {
                if (i == argc) {
                    missing("--yarn-beta-fast");
                }
                char *arg = argv[i++];
                bparams.gparams.yarn_beta_fast = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "--yarn-beta-slow")) {
                if (i == argc) {
                    missing("--yarn-beta-slow");
                }
                char *arg = argv[i++];
                bparams.gparams.yarn_beta_slow = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-gan") || !strcmp(flag, "--grp-attn-n")) {
                if (i == argc) {
                    missing("--grp-attn-n");
                }
                char *arg = argv[i++];
                bparams.gparams.grp_attn_n = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-gaw") || !strcmp(flag, "--grp-attn-w")) {
                if (i == argc) {
                    missing("--grp-attn-w");
                }
                char *arg = argv[i++];
                bparams.gparams.grp_attn_w = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-nkvo") || !strcmp(flag, "--no-kv-offload")) {
                bparams.gparams.no_kv_offload = true;
                continue;
            }

            if (!strcmp(flag, "-ctk") || !strcmp(flag, "--cache-type-k")) {
                if (i == argc) {
                    missing("--cache-type-k");
                }
                char *arg = argv[i++];
                bparams.gparams.cache_type_k = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "-ctv") || !strcmp(flag, "--cache-type-v")) {
                if (i == argc) {
                    missing("--cache-type-v");
                }
                char *arg = argv[i++];
                bparams.gparams.cache_type_v = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "-dt") || !strcmp(flag, "--defrag-thold")) {
                if (i == argc) {
                    missing("--defrag-thold");
                }
                char *arg = argv[i++];
                bparams.gparams.defrag_thold = std::stof(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-np") || !strcmp(flag, "--parallel")) {
                if (i == argc) {
                    missing("--parallel");
                }
                char *arg = argv[i++];
                bparams.gparams.n_parallel = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-cb") || !strcmp(flag, "--cont-batching")) {
                bparams.gparams.cont_batching = true;
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
                char *arg = argv[i++];
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

            if (!strcmp(flag, "--lora")) {
                if (i == argc) {
                    missing("--lora");
                }
                char *arg = argv[i++];
                bparams.gparams.lora_adapters.push_back({
                    std::string(arg),
                    1.0f,
                });
                bparams.gparams.use_mmap = false;
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
                bparams.gparams.use_mmap = false;
                continue;
            }

            if (!strcmp(flag, "--lora-init-without-apply")) {
                bparams.gparams.lora_init_without_apply = true;
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
                bparams.gparams.control_vectors.push_back(
                    {std::stof(std::string(s)), std::string(n)});
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
                char *e = argv[i++];
                bparams.gparams.control_vector_layer_start = std::stoi(std::string(s));
                bparams.gparams.control_vector_layer_end = std::stoi(std::string(e));
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

            if (llama_supports_gpu_offload()) {
                if (!strcmp(flag, "-ngl") || !strcmp(flag, "--gpu-layers") ||
                    !strcmp(flag, "--n-gpu-layers")) {
                    if (i == argc) {
                        missing("--gpu-layers");
                    }
                    char *arg = argv[i++];
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

                if (!strcmp(flag, "-mg") || !strcmp(flag, "--main-gpu")) {
                    if (i == argc) {
                        missing("--main-gpu");
                    }
                    char *arg = argv[i++];
                    bparams.gparams.main_gpu = std::stoi(std::string(arg));
                    if (bparams.gparams.main_gpu < 0 ||
                        bparams.gparams.main_gpu >= int32_t(llama_max_devices())) {
                        invalid("--main-gpu");
                    }
                    continue;
                }
            }

            if (!strcmp(flag, "--rpc")) {
                if (i == argc) {
                    missing("--rpc");
                }
                char *arg = argv[i++];
                bparams.gparams.rpc_servers = arg;
                continue;
            }

            // server // speculative //

            if (!strcmp(flag, "--draft")) {
                if (i == argc) {
                    missing("--draft");
                }
                char *arg = argv[i++];
                bparams.gparams.n_draft = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-md") || !strcmp(flag, "--model-draft")) {
                if (i == argc) {
                    missing("--model-draft");
                }
                char *arg = argv[i++];
                bparams.gparams.model_draft = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "-td") || !strcmp(flag, "--threads-draft")) {
                if (i == argc) {
                    missing("--threads-draft");
                }
                char *arg = argv[i++];
                bparams.gparams.n_threads_draft = std::stoi(std::string(arg));
                if (bparams.gparams.n_threads_draft <= 0) {
                    bparams.gparams.n_threads_draft = cpu_get_num_math();
                }
                continue;
            }

            if (!strcmp(flag, "-tbd") || !strcmp(flag, "--threads-batch-draft")) {
                if (i == argc) {
                    missing("--threads-batch-draft");
                }
                char *arg = argv[i++];
                bparams.gparams.n_threads_batch_draft = std::stoi(std::string(arg));
                if (bparams.gparams.n_threads_batch_draft <= 0) {
                    bparams.gparams.n_threads_batch_draft = cpu_get_num_math();
                }
                continue;
            }

            if (llama_supports_gpu_offload()) {
                if (!strcmp(flag, "-ngld") || !strcmp(flag, "--gpu-layers-draft")) {
                    if (i == argc) {
                        missing("--gpu-layers-draft");
                    }
                    char *arg = argv[i++];
                    bparams.gparams.n_gpu_layers_draft = std::stoi(arg);
                    continue;
                }
            }

            if (!strcmp(flag, "--lookup-ngram-min")) {
                if (i == argc) {
                    missing("--lookup-ngram-min");
                }
                char *arg = argv[i++];
                bparams.lookup_ngram_min = std::stoi(std::string(arg));
                continue;
            }

            if (!strcmp(flag, "-lcs") || !strcmp(flag, "--lookup-cache-static")) {
                if (i == argc) {
                    missing("--lookup-cache-static");
                }
                char *arg = argv[i++];
                bparams.gparams.lookup_cache_static = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "-lcd") || !strcmp(flag, "--lookup-cache-dynamic")) {
                if (i == argc) {
                    missing("--lookup-cache-dynamic");
                }
                char *arg = argv[i++];
                bparams.gparams.lookup_cache_dynamic = std::string(arg);
                continue;
            }

            // server // speculative //

            // server //

            // rpc-server //

            if (!strcmp(flag, "--rpc-server-host")) {
                if (i == argc) {
                    missing("--rpc-server-host");
                }
                char *arg = argv[i++];
                bparams.rparams.hostname = std::string(arg);
                continue;
            }

            if (!strcmp(flag, "--rpc-server-port")) {
                if (i == argc) {
                    missing("--rpc-server-port");
                }
                char *arg = argv[i++];
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
                    char *arg = argv[i++];
                    bparams.rparams.main_gpu = std::stoi(std::string(arg));
                    if (bparams.rparams.main_gpu < 0 ||
                        bparams.rparams.main_gpu >= int32_t(llama_max_devices())) {
                        invalid("--rpc-server-main-gpu");
                    }
                    continue;
                }
            }

            if (!strcmp(flag, "--rpc-server-reserve-memory")) {
                if (i == argc) {
                    missing("--rpc-server-reserve-memory");
                }
                char *arg = argv[i++];
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

    if (!bparams.gparams.kv_overrides.empty()) {
        bparams.gparams.kv_overrides.emplace_back();
        bparams.gparams.kv_overrides.back().key[0] = 0;
    }

    if (bparams.gparams.n_threads_batch <= 0) {
        bparams.gparams.n_threads_batch = bparams.gparams.n_threads;
    }
    if (bparams.gparams.n_threads_draft <= 0) {
        bparams.gparams.n_threads_draft = bparams.gparams.n_threads;
    }
    if (bparams.gparams.n_threads_batch_draft <= 0) {
        bparams.gparams.n_threads_batch_draft = bparams.gparams.n_threads_draft;
    }

    // Retrieve params from environment variables
    get_env("LLAMA_ARG_MODEL", bparams.gparams.model);
    get_env("LLAMA_ARG_MODEL_ALIAS", bparams.gparams.model_alias);
    get_env("LLAMA_ARG_THREADS", bparams.gparams.n_threads);
    get_env("LLAMA_ARG_CTX_SIZE", bparams.gparams.n_ctx);
    get_env("LLAMA_ARG_N_PARALLEL", bparams.gparams.n_parallel);
    get_env("LLAMA_ARG_BATCH", bparams.gparams.n_batch);
    get_env("LLAMA_ARG_UBATCH", bparams.gparams.n_ubatch);
    get_env("LLAMA_ARG_N_GPU_LAYERS", bparams.gparams.n_gpu_layers);
    get_env("LLAMA_ARG_THREADS_HTTP", bparams.gparams.n_threads_http);
    get_env("LLAMA_ARG_CHAT_TEMPLATE", bparams.gparams.chat_template);
    get_env("LLAMA_ARG_N_PREDICT", bparams.gparams.n_predict);
    get_env("LLAMA_ARG_METRICS", bparams.gparams.endpoint_metrics);
    get_env("LLAMA_ARG_SLOTS", bparams.gparams.endpoint_slots);
    get_env("LLAMA_ARG_EMBEDDINGS", bparams.gparams.embedding);
    get_env("LLAMA_ARG_FLASH_ATTN", bparams.gparams.flash_attn);
    get_env("LLAMA_ARG_DEFRAG_THOLD", bparams.gparams.defrag_thold);
    get_env("LLAMA_ARG_DRAFT", bparams.gparams.n_draft);
    get_env("LLAMA_ARG_MODEL_DRAFT", bparams.gparams.model_draft);
    get_env("LLAMA_ARG_THREADS_DRAFT", bparams.gparams.n_threads_draft);
    get_env("LLAMA_ARG_N_GPU_LAYERS_DRAFT", bparams.gparams.n_gpu_layers_draft);
    get_env("LLAMA_ARG_LOOKUP_NGRAM_MIN", bparams.lookup_ngram_min);
    get_env("LLAMA_ARG_LOOKUP_CACHE_STATIC", bparams.gparams.lookup_cache_static);
    get_env("LLAMA_ARG_LOOKUP_CACHE_DYNAMIC", bparams.gparams.lookup_cache_dynamic);

    return true;
}