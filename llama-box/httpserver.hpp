// heads

#include <atomic>
#include <csignal>
#include <memory>
#include <unordered_map>
#include <utility>
#include <variant>

#include "concurrentqueue/blockingconcurrentqueue.h"
#include "readerwriterqueue/readerwriterqueue.h"

#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH 536870912
#define CPPHTTPLIB_TCP_NODELAY                         true
#include "llama.cpp/common/chat.h"
#include "llama.cpp/common/common.h"
#include "llama.cpp/common/ngram-cache.h"
#include "llama.cpp/common/sampling.h"
#include "llama.cpp/tools/mtmd/clip.h"
#include "llama.cpp/vendor/cpp-httplib/httplib.h"
#include "stable-diffusion.cpp/stable-diffusion.h"

#define SELF_PACKAGE 0
#include "z_multimodal.hpp"
#include "z_stablediffusion.hpp"
#include "z_utils.hpp"

// defines

#define HEADER_CACHE_CONTROL               "Cache-Control"
#define HEADER_CONNECTION                  "Connection"
#define HEADER_SERVER                      "SERVER"
#define HEADER_X_REQUEST_ID                "X-Request-ID"
#define HEADER_X_REQUEST_ACCEPTED_AT       "X-Request-Accepted-At"
#define HEADER_X_REQUEST_TOKENS_PER_SECOND "X-Request-Tokens-Per-Second"

using namespace moodycamel;
using json = nlohmann::json;

// types

struct httpserver_params {
    common_params          llm_params;
    stablediffusion_params sd_params;

    bool    cache_prompt        = true;
    bool    endpoint_images     = false;
    int32_t conn_idle           = 60;  // connection idle in seconds
    int32_t conn_keepalive      = 15;  // connection keep-alive in seconds
    int32_t n_tps               = 0;   // maximum number of tokens per seconds
    int32_t lookup_ngram_min    = 0;   // minimum n-gram size for lookup cache
    int32_t max_image_size      = 0;   // maximum image size for vision image processing
    int32_t max_projected_cache = 0;   // maximum number of projected embedding in cache
};

// implementations

// send_json, then close.
static inline int32_t send_json(const httplib::Request & request, httplib::Response & response,
                                httplib::StatusCode status, json & data) {
    if (request.is_connection_closed()) {
        response.status = httplib::RequestTimeout_408;
        return response.status;
    }
    json resp = data;
    if (status >= httplib::BadRequest_400) {
        resp = {
            { "error",  data                                                                     },
            { "detail", data.contains("message") ? data.at("message") : "Unknown error occurred" },
        };
    }
    response.status = status;
    response.set_content(resp.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
    return response.status;
}

// send_json, then close.
static inline int32_t send_json(const httplib::Request & request, httplib::Response & response,
                                httplib::StatusCode status, std::string message) {
    if (request.is_connection_closed()) {
        response.status = httplib::RequestTimeout_408;
        return response.status;
    }
    json data = {
        { "code",    status                          },
        { "message", message                         },
        { "type",    httplib::status_message(status) },
    };
    return send_json(request, response, status, data);
}

// send_string, then close.
static inline int32_t send_string(const httplib::Request & request, httplib::Response & response,
                                  httplib::StatusCode status, const std::string & message,
                                  const std::string & content_type = "") {
    if (request.is_connection_closed()) {
        response.status = httplib::RequestTimeout_408;
        return response.status;
    }
    response.status = status;
    response.set_content(message, content_type.empty() ? "text/plain" : content_type);
    return response.status;
}

// send_event_json, close if given status is not 100.
static inline int32_t send_event_json(httplib::DataSink & sink, httplib::StatusCode status, json & data) {
    if (!sink.is_writable()) {
        return httplib::RequestTimeout_408;
    }
    std::string event;
    std::string message;
    if (status >= httplib::BadRequest_400) {
        event           = "error";
        const json resp = {
            { "error",  data                                                                     },
            { "detail", data.contains("message") ? data.at("message") : "Unknown error occurred" },
        };
        message = resp.dump(-1, ' ', false, json::error_handler_t::replace);
    } else {
        event   = "data";
        message = data.dump(-1, ' ', false, json::error_handler_t::replace);
    }
    const std::string str = event + ": " + message + "\n\n";
    sink.write(str.c_str(), str.size());
    if (status != httplib::Continue_100) {
        sink.done();
    }
    return httplib::OK_200;
}

// send_event_string, close if given status is not 100.
static inline int32_t send_event_string(httplib::DataSink & sink, httplib::StatusCode status,
                                        const std::string & message) {
    if (!sink.is_writable()) {
        return httplib::RequestTimeout_408;
    }
    std::string event = "data";
    if (status >= httplib::BadRequest_400) {
        event = "error";
    }
    const std::string str = event + ": " + message + "\n\n";
    sink.write(str.c_str(), str.size());
    if (status != httplib::Continue_100) {
        sink.done();
    }
    return httplib::OK_200;
}

// normalize_seed, avoid black-box seed.
static inline uint32_t normalize_seed(uint32_t seed) {
    if (seed == LLAMA_DEFAULT_SEED) {
        return uint32_t(ggml_time_us());
    }
    return seed;
}

// prepare_sampling, returns llama.cpp sampling params.
static inline common_params_sampling prepare_sampling(const json & data, const common_params_sampling & defaults,
                                                      const llama_context * llm_ctx) {
    common_params_sampling params = defaults;  // copy
    if (!data.contains("samplers")) {
        return params;
    }

    {
        const json & samplers = data.at("samplers");
        if (samplers.is_array()) {
            params.samplers = common_sampler_types_from_names(samplers.get<std::vector<std::string>>(), false);
        } else if (samplers.is_string()) {
            params.samplers = common_sampler_types_from_chars(samplers.get<std::string>());
        }
    }
    params.top_k             = json_value(data, "top_k", defaults.top_k);
    params.top_p             = json_value(data, "top_p", defaults.top_p);
    params.min_p             = json_value(data, "min_p", defaults.min_p);
    params.top_n_sigma       = json_value(data, "top_n_sigma", defaults.top_n_sigma);
    params.xtc_probability   = json_value(data, "xtc_probability", defaults.xtc_probability);
    params.xtc_threshold     = json_value(data, "xtc_threshold", defaults.xtc_threshold);
    params.typ_p             = json_value(data, "typical_p", defaults.typ_p);
    params.temp              = json_value(data, "temperature", defaults.temp);
    params.dynatemp_range    = json_value(data, "dynatemp_range", defaults.dynatemp_range);
    params.dynatemp_exponent = json_value(data, "dynatemp_exponent", defaults.dynatemp_exponent);
    params.penalty_last_n    = json_value(data, "repeat_last_n", defaults.penalty_last_n);
    if (params.penalty_last_n <= -1) {
        params.penalty_last_n = int32_t(llama_n_ctx(llm_ctx));
    }
    params.penalty_repeat  = json_value(data, "repeat_penalty", defaults.penalty_repeat);
    params.penalty_freq    = json_value(data, "frequency_penalty", defaults.penalty_freq);
    params.penalty_present = json_value(data, "presence_penalty", defaults.penalty_present);
    params.dry_multiplier  = json_value(data, "dry_multiplier", defaults.dry_multiplier);
    params.dry_base        = json_value(data, "dry_base", defaults.dry_base);
    if (params.dry_base < 1.0f) {
        params.dry_base = defaults.dry_base;
    }
    params.dry_allowed_length = json_value(data, "dry_allowed_length", defaults.dry_allowed_length);
    params.dry_penalty_last_n = json_value(data, "dry_penalty_last_n", defaults.dry_penalty_last_n);
    if (params.dry_penalty_last_n <= -1) {
        params.dry_penalty_last_n = int32_t(llama_n_ctx(llm_ctx));
    }
    params.mirostat     = json_value(data, "mirostat", defaults.mirostat);
    params.mirostat_tau = json_value(data, "mirostat_tau", defaults.mirostat_tau);
    params.mirostat_eta = json_value(data, "mirostat_eta", defaults.mirostat_eta);
    params.seed         = normalize_seed(json_value(data, "seed", defaults.seed));
    params.n_probs      = json_value(data, "n_probs", defaults.n_probs);
    params.min_keep     = json_value(data, "min_keep", defaults.min_keep);
    if (data.contains("json_schema") && !data.contains("grammar")) {
        try {
            json schema    = json_value(data, "json_schema", json::object());
            params.grammar = json_schema_to_grammar(schema);
        } catch (const std::exception & e) {
            throw std::invalid_argument("Illegal param: \"json_schema\": " + std::string(e.what()));
        }
    } else if (data.contains("grammar")) {
        params.grammar = json_value(data, "grammar", defaults.grammar);
    }
    if (json_value(data, "ignore_eos", false)) {
        const llama_vocab * vocab     = llama_model_get_vocab(llama_get_model(llm_ctx));
        const llama_token   vocab_eos = llama_vocab_eos(vocab);
        if (vocab_eos != LLAMA_TOKEN_NULL) {
            params.logit_bias.push_back({ vocab_eos, -INFINITY });
        }
    }

    return params;
}

// prepare_sampling, returns stable-diffusion.cpp sampling params.
static inline stablediffusion_params_sampling prepare_sampling(const json &                            data,
                                                               const stablediffusion_params_sampling & defaults) {
    stablediffusion_params_sampling params = defaults;  // copy
    if (!data.contains("sampler") && !data.contains("sample_method")) {
        return params;
    }

    std::string sample_method_str = "euler_a";
    if (data.contains("sample_method")) {
        sample_method_str = data.at("sample_method");
    } else if (data.contains("sampler")) {
        sample_method_str = data.at("sampler");
    }
    params.sample_method   = sd_argument_to_sample_method(sample_method_str.c_str());
    int32_t sampling_steps = 10;
    if (data.contains("sampling_steps")) {
        sampling_steps = json_value(data, "sampling_steps", 10);
    } else if (data.contains("sample_steps")) {
        sampling_steps = json_value(data, "sample_steps", 10);
    }
    params.sampling_steps           = sampling_steps;
    std::string schedule_method_str = "default";
    if (data.contains("schedule_method")) {
        schedule_method_str = data.at("schedule_method");
    } else if (data.contains("scheduler")) {
        schedule_method_str = data.at("scheduler");
    } else if (data.contains("schedule")) {
        schedule_method_str = data.at("schedule");
    }
    params.schedule_method = sd_argument_to_schedule(schedule_method_str.c_str());
    params.seed            = normalize_seed(json_value(data, "seed", defaults.seed));
    params.guidance        = json_value(data, "guidance", defaults.guidance);
    params.cfg_scale       = json_value(data, "cfg_scale", defaults.cfg_scale);
    // TODO slg_skip_layers
    params.slg_scale       = json_value(data, "slg_scale", defaults.slg_scale);
    params.slg_start       = json_value(data, "slg_start", defaults.slg_start);
    params.slg_end         = json_value(data, "slg_end", defaults.slg_end);
    params.negative_prompt = json_value(data, "negative_prompt", defaults.negative_prompt);
    return params;
}

// prepare_sampling, returns stable-diffusion.cpp sampling params.
static inline stablediffusion_params_sampling prepare_sampling(const httplib::MultipartFormDataMap &   req,
                                                               const stablediffusion_params_sampling & defaults) {
    stablediffusion_params_sampling params = defaults;  // copy
    if (req.find("sampler") == req.end() && req.find("sample_method") == req.end()) {
        return params;
    }

    std::string sample_method_str = "euler_a";
    auto        item              = req.find("sample_method");
    if (item != req.end()) {
        sample_method_str = item->second.content;
    } else {
        item              = req.find("sampler");
        sample_method_str = item->second.content;
    }
    params.sample_method           = sd_argument_to_sample_method(sample_method_str.c_str());
    std::string sampling_steps_str = "10";
    item                           = req.find("sampling_steps");
    if (item != req.end()) {
        sampling_steps_str = item->second.content;
    } else {
        item               = req.find("sample_steps");
        sampling_steps_str = item->second.content;
    }
    try {
        params.sampling_steps = std::stoi(sampling_steps_str);
    } catch (...) {
        // NOP
    }
    std::string schedule_method_str = "default";
    item                            = req.find("schedule_method");
    if (item != req.end()) {
        schedule_method_str = item->second.content;
    } else {
        item = req.find("scheduler");
        if (item != req.end()) {
            schedule_method_str = item->second.content;
        } else {
            item = req.find("schedule");
            if (item != req.end()) {
                schedule_method_str = item->second.content;
            }
        }
    }
    params.schedule_method = sd_argument_to_schedule(schedule_method_str.c_str());
    item                   = req.find("seed");
    if (item != req.end()) {
        try {
            params.seed = normalize_seed(std::stoul(item->second.content));
        } catch (...) {
            // NOP
        }
    }
    item = req.find("guidance");
    if (item != req.end()) {
        try {
            params.guidance = std::stof(item->second.content);
        } catch (...) {
            // NOP
        }
    }
    item = req.find("cfg_scale");
    if (item != req.end()) {
        try {
            params.cfg_scale = std::stof(item->second.content);
        } catch (...) {
            // NOP
        }
    }
    item = req.find("slg_scale");
    if (item != req.end()) {
        try {
            params.slg_scale = std::stof(item->second.content);
        } catch (...) {
            // NOP
        }
    }
    // TODO slg_skip_layers
    item = req.find("slg_start");
    if (item != req.end()) {
        try {
            params.slg_start = std::stof(item->second.content);
        } catch (...) {
            // NOP
        }
    }
    item = req.find("slg_end");
    if (item != req.end()) {
        try {
            params.slg_end = std::stof(item->second.content);
        } catch (...) {
            // NOP
        }
    }
    item = req.find("negative_sampler");
    if (item != req.end()) {
        params.negative_prompt = item->second.content;
    }
    item = req.find("strength");
    if (item != req.end()) {
        try {
            params.strength = std::stof(item->second.content);
        } catch (...) {
            // NOP
        }
    }
    item = req.find("control_strength");
    if (item != req.end()) {
        try {
            params.control_strength = std::stof(item->second.content);
        } catch (...) {
            // NOP
        }
    }
    item = req.find("control_canny");
    if (item != req.end()) {
        try {
            params.control_canny = item->second.content == "true";
        } catch (...) {
            // NOP
        }
    }

    return params;
}

// sort_rerank_results, gives rerank JSON result.
static inline void sort_rerank_results(json & result, int32_t low, int32_t high) {
    if (low >= high) {
        return;
    }

    json    base = result[low];
    int32_t i = low, j = high;
    while (i != j) {
        while (i < j && json_value(result[j], "relevance_score", 0.0) <= json_value(base, "relevance_score", 0.0)) {
            j--;
        }
        while (i < j && json_value(result[i], "relevance_score", 0.0) >= json_value(base, "relevance_score", 0.0)) {
            i++;
        }
        if (i < j) {
            json temp = result[i];
            result[i] = result[j];
            result[j] = temp;
        }
    }
    result[low] = result[i];
    result[i]   = base;
    sort_rerank_results(result, low, i - 1);
    sort_rerank_results(result, i + 1, high);
}

// equal_lora returns true if both lora adapters are the same.
static inline bool equal_lora(const std::vector<common_adapter_lora_info> & l1,
                              const std::vector<common_adapter_lora_info> & l2) {
    if (l1.size() != l2.size()) {
        return false;
    }
    for (size_t i = 0; i < l1.size(); ++i) {
        // we don't check lora.path to reduce the time complexity
        if (l1[i].scale != l2[i].scale || l1[i].ptr != l2[i].ptr) {
            return false;
        }
    }
    return true;
}

// get_token_probabilities, returns token probabilities.
static inline std::vector<llama_token_data> get_token_probabilities(llama_context * ctx, int32_t idx) {
    std::vector<llama_token_data> cur;
    const auto *                  logits  = llama_get_logits_ith(ctx, idx);
    const int32_t                 n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));

    cur.resize(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur[token_id] = llama_token_data{ token_id, logits[token_id], 0.0f };
    }

    // sort tokens by logits
    std::sort(cur.begin(), cur.end(),
              [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; });

    // apply softmax
    float max_l   = cur[0].logit;
    float cum_sum = 0.0f;
    for (auto & i : cur) {
        float p = expf(i.logit - max_l);
        i.p     = p;
        cum_sum += p;
    }
    for (auto & i : cur) {
        i.p /= cum_sum;
    }

    return cur;
}

// gen_chat_completion_id, returns a random chat completion id.
static std::string gen_chat_completion_id() {
    return "chatcmpl-" + random_string();
}

// gen_completion_id, returns a random completion id.
static std::string gen_completion_id() {
    return "cmpl-" + random_string();
}

// gen_call_id, returns a random call id.
static std::string gen_call_id() {
    return "call-" + random_string();
}

// implementations // parser

enum req_type {
    REQ_TOKENIZE,
    REQ_DETOKENIZE,
    REQ_LEGACY_COMPLETE,
    REQ_CHAT_COMPLETE,
    REQ_EMBED,
    REQ_RERANK,
    REQ_IMAGE_GENERATE,
    REQ_IMAGE_EDIT,
    REQ_UNKNOWN,
};

struct breq {
  protected:
    std::string id;
    req_type    type = REQ_UNKNOWN;

  public:
    explicit breq(std::string id, req_type type) : id(std::move(id)), type(type) {}

    virtual ~breq() = default;

    /* LLAMA BOX */

    const char * get_id() { return id.c_str(); }

    [[nodiscard]] req_type get_type() const { return type; }

    /* OPEN AI */

    std::string model;
    int32_t     n = 1;
};

struct tokenize_req : breq {
    explicit tokenize_req(const std::string & id) : breq(id, REQ_TOKENIZE) {}

    /* LLAMA BOX */

    /* OPEN AI*/

    // std::string model;                                     // inherit
    json content;
    bool add_special = false;
    bool with_pieces = false;
};

static inline std::unique_ptr<tokenize_req> get_tokenize_req(const httplib::Request & request,
                                                             httplib::Response &      response,
                                                             const common_params &    params) {
    const std::string rid = response.get_header_value(HEADER_X_REQUEST_ID);
    const json        req = json::parse(request.body);
    if (!req.contains("content")) {
        throw std::invalid_argument(R"(Illegal param: "content" is required)");
    }
    if (!json_is_array_or_string(req.at("content"))) {
        throw std::invalid_argument(R"(Illegal param: "content" must be a string or a list)");
    }

    // print32_t the request for debugging
    if (common_log_verbosity_thold > 1) {
        json req_cp = req;
        if (common_log_verbosity_thold < 3) {
            req_cp["content"] = "...";
        }
        SRV_INF("rid %s | %s\n", rid.c_str(), req_cp.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    }

    std::unique_ptr<tokenize_req> ptr = std::make_unique<tokenize_req>(rid.c_str());

    ptr->model = json_value(req, "model", params.model_alias);

    ptr->content = req.at("content");

    ptr->add_special = json_value(req, "add_special", false);

    ptr->with_pieces = json_value(req, "with_pieces", false);

    return ptr;
}

struct detokenize_req : breq {
    explicit detokenize_req(const std::string & id) : breq(id, REQ_DETOKENIZE) {}

    /* LLAMA BOX */

    /* OPEN AI*/

    // std::string model;                                     // inherit
    json tokens;
};

static inline std::unique_ptr<detokenize_req> get_detokenize_req(const httplib::Request & request,
                                                                 httplib::Response &      response,
                                                                 const common_params &    params) {
    const std::string rid = response.get_header_value(HEADER_X_REQUEST_ID);
    const json        req = json::parse(request.body);
    if (!req.contains("tokens")) {
        throw std::invalid_argument("Illegal param: \"tokens\" is required");
    }
    if (!json_is_array_of_numbers(req.at("tokens"))) {
        throw std::invalid_argument("Illegal param: \"tokens\" must be a list of tokens");
    }

    // print32_t the request for debugging
    if (common_log_verbosity_thold > 1) {
        json req_cp = req;
        if (common_log_verbosity_thold < 3) {
            req_cp["tokens"] = "...";
        }
        SRV_INF("rid %s | %s\n", rid.c_str(), req_cp.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    }

    std::unique_ptr<detokenize_req> ptr = std::make_unique<detokenize_req>(rid.c_str());

    ptr->model = json_value(req, "model", params.model_alias);

    ptr->tokens = req.at("tokens");

    return ptr;
}

struct complete_req : breq {
    explicit complete_req(const std::string & id, req_type type) : breq(id, type) {}

    /* LLAMA BOX */

    // sample
    common_params_sampling                sampling;
    // lora
    std::vector<common_adapter_lora_info> lora_adapters;

    /* OPEN AI */

    // decode
    int32_t                  max_tokens = 0;
    int32_t                  logprobs   = -1;
    std::vector<std::string> stop;

    // stream
    bool stream         = false;
    json stream_options = {
        { "include_usage", true },
    };
};

struct legacy_complete_req : complete_req {
    explicit legacy_complete_req(const std::string & id) : complete_req(id, REQ_LEGACY_COMPLETE) {}

    /* LLAMA BOX */

    /* OPEN AI*/

    // std::string model;                                     // inherit
    json                          prompt;
    // int32_t best_of = 1;
    // bool echo = false;
    float                         frequency_penalty = 0.0f;
    std::vector<llama_logit_bias> logit_bias;
    // int32_t logprobs   = 0;                                // inherit
    // int32_t max_tokens = -1;                               // inherit
    // int32_t n = 1;                                         // inherit
    float                         presence_penalty = 0.0f;
    uint32_t                      seed             = LLAMA_DEFAULT_SEED;
    // std::vector<std::string> stop;                         // inherit
    // bool stream         = false;                           // inherit
    // json stream_options = {{"include_usage", true}};       // inherit
    // std::string suffix;
    float                         temperature      = 1.0;
    float                         top_p            = 1.0;
    // std::string user;
};

static inline std::unique_ptr<legacy_complete_req> get_legacy_complete_req(const httplib::Request &  request,
                                                                           httplib::Response &       response,
                                                                           const httpserver_params & hparams,
                                                                           const llama_context *     llm_ctx) {
    const common_params & params = hparams.llm_params;

    const std::string rid = response.get_header_value(HEADER_X_REQUEST_ID);
    const json        req = json::parse(request.body);
    if (!req.contains("prompt")) {
        throw std::invalid_argument("Illegal param: \"prompt\" is required");
    }

    // print32_t the request for debugging
    if (common_log_verbosity_thold > 1) {
        json req_cp = req;
        if (common_log_verbosity_thold < 3) {
            req_cp["prompt"] = "...";
        }
        SRV_INF("rid %s | %s\n", rid.c_str(), req_cp.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    }

    std::unique_ptr<legacy_complete_req> ptr = std::make_unique<legacy_complete_req>(rid.c_str());

    ptr->sampling = prepare_sampling(req, params.sampling, llm_ctx);

    if (req.contains("lora")) {
        const json & lora = req.at("lora");
        if (!lora.is_array()) {
            throw std::invalid_argument("Illegal param: \"lora\" must be a list");
        }
        ptr->lora_adapters = params.lora_adapters;
        // clear value
        for (common_adapter_lora_info & la : ptr->lora_adapters) {
            la.scale = 0.0f;
        }
        // set value
        int32_t max_id = int32_t(ptr->lora_adapters.size()) - 1;
        for (const json & l : lora) {
            if (!l.is_object()) {
                throw std::invalid_argument("Illegal param: \"lora\" must be a list of objects");
            }
            int32_t id    = json_value(l, "id", -1);
            float   scale = json_value(l, "scale", 0.0f);
            if (id < 0 || id > max_id) {
                throw std::invalid_argument("Illegal param: \"id\" must be in the range [0, " + std::to_string(max_id) +
                                            "]");
            }
            ptr->lora_adapters[id].scale = scale;
        }
    }

    ptr->model = json_value(req, "model", params.model_alias);

    ptr->prompt = req.at("prompt");

    ptr->frequency_penalty = json_value(req, "frequency_penalty", params.sampling.penalty_freq);

    if (req.contains("logit_bias")) {
        const json & logit_bias = req.at("logit_bias");
        if (!logit_bias.is_object()) {
            throw std::invalid_argument("Illegal param: \"logit_bias\" must be a map");
        }
        const llama_vocab * vocab      = llama_model_get_vocab(llama_get_model(llm_ctx));
        const int32_t       vocab_size = llama_vocab_n_tokens(vocab);
        for (const auto & el : logit_bias.items()) {
            llama_token tok = std::stoi(el.key());
            if (tok < 0) {
                throw std::invalid_argument("Illegal param: \"logit_bias\" keys must be integer string");
            } else if (tok >= vocab_size) {
                throw std::invalid_argument("Illegal param: \"logit_bias\" keys must be in the range [0, vocab_size)");
            }
            float bias = -INFINITY;
            if (el.value().is_number()) {
                bias = el.value().get<float>();
                if (bias < -100 || bias > 100) {
                    throw std::invalid_argument(
                        "Illegal param: \"logit_bias\" values must be in the range [-100, 100]");
                }
            } else if (el.value().is_boolean()) {
                if (el.value().get<bool>()) {
                    continue;
                }
            } else {
                throw std::invalid_argument("Illegal param: \"logit_bias\" values must be a number or boolean");
            }
            ptr->logit_bias.push_back({ tok, bias });
        }
    }

    if (req.contains("logprobs")) {
        ptr->logprobs = req.at("logprobs").get<int32_t>();
        if (ptr->logprobs < 0 || ptr->logprobs > 5) {
            throw std::invalid_argument("Illegal param: \"logprobs\" must be in the range [1, 5]");
        }
    }

    ptr->max_tokens = json_value(req, "max_tokens", ptr->max_tokens);
    if (ptr->max_tokens > int32_t(llama_n_ctx(llm_ctx))) {
        throw std::invalid_argument(
            "Illegal param: \"max_tokens\" must be less than or equal to the model's context length");
    }

    ptr->presence_penalty = json_value(req, "presence_penalty", params.sampling.penalty_present);

    ptr->seed = normalize_seed(json_value(req, "seed", params.sampling.seed));

    if (req.contains("stop")) {
        const json & stop = req.at("stop");
        if (stop.is_string()) {
            ptr->stop.push_back(stop.get<std::string>());
        } else if (stop.is_array()) {
            for (const json & s : stop) {
                if (!s.is_string()) {
                    throw std::invalid_argument("Illegal param: \"stop\" must be a list of strings");
                }
                ptr->stop.push_back(s.get<std::string>());
            }
        } else {
            throw std::invalid_argument("Illegal param: \"stop\" must be a string or a list of strings");
        }
    }

    ptr->stream = json_value(req, "stream", false);

    if (ptr->stream && req.contains("stream_options")) {
        const json & stream_options = req.at("stream_options");
        if (!stream_options.is_object()) {
            throw std::invalid_argument("Illegal param: \"stream_options\" must be an object");
        }
        for (const auto & el : stream_options.items()) {
            ptr->stream_options[el.key()] = el.value();
        }
    }

    ptr->temperature = json_value(req, "temperature", params.sampling.temp);

    ptr->top_p = json_value(req, "top_p", params.sampling.top_p);

    // merge sampling
    {
        ptr->sampling.penalty_freq    = ptr->frequency_penalty;
        ptr->sampling.n_probs         = ptr->logprobs;
        ptr->sampling.penalty_present = ptr->presence_penalty;
        ptr->sampling.logit_bias      = ptr->logit_bias;
        ptr->sampling.seed            = ptr->seed;
        ptr->sampling.temp            = ptr->temperature;
        ptr->sampling.top_p           = ptr->top_p;
    }

    return ptr;
}

struct clip_multimedia {
    clip_image_u8_ptr ptr;
    std::string       hash;
    bool              is_audio = false;

    clip_multimedia(clip_image_u8_ptr && ptr, std::string && hash, bool is_audio = false) :
        ptr(std::move(ptr)),
        hash(std::move(hash)),
        is_audio(is_audio) {}
};

static inline std::unique_ptr<clip_multimedia> get_clip_image(std::vector<uint8_t> && img_buff) {
    std::string hash;
    try {
        hash = hash_fnv(img_buff.data(), img_buff.size());
    } catch (std::exception & e) {
        LOG_WRN("failed to hash image: %s\n", e.what());
    }

    int32_t   w  = 0;
    int32_t   h  = 0;
    int32_t   c  = 0;
    uint8_t * dt = stbi_load_from_memory((const stbi_uc *) img_buff.data(), (int32_t) img_buff.size(), &w, &h, &c, 3);
    if (dt == nullptr) {
        throw std::invalid_argument("Illegal param: provided image is invalid: " + std::string(stbi_failure_reason()));
    }
    if (w <= 0 || h <= 0 || c < 3) {
        stbi_image_free(dt);
        throw std::invalid_argument("Illegal param: provided image is invalid");
    }

    clip_image_u8_ptr ptr(clip_image_u8_init());
    ptr->nx = w;
    ptr->ny = h;
    ptr->buf.resize(w * h * 3);
    std::memcpy(ptr->buf.data(), dt, ptr->buf.size());
    stbi_image_free(dt);

    return std::make_unique<clip_multimedia>(std::move(ptr), std::move(hash));
}

static inline std::unique_ptr<clip_multimedia> get_clip_audio(std::vector<uint8_t> && aud_buff) {
    std::string hash;
    try {
        hash = hash_fnv(aud_buff.data(), aud_buff.size());
    } catch (std::exception & e) {
        LOG_WRN("failed to hash audio: %s\n", e.what());
    }

    std::vector<float> dt;
    if (!decode_audio_from_buf(aud_buff.data(), aud_buff.size(), COMMON_SAMPLE_RATE, dt)) {
        throw std::invalid_argument("Illegal param: provided audio is invalid");
    }

    clip_image_u8_ptr ptr(clip_image_u8_init());
    ptr->nx = int(dt.size());
    ptr->ny = 1;
    ptr->buf.resize(dt.size() * sizeof(float));
    std::memcpy(ptr->buf.data(), dt.data(), ptr->buf.size());

    return std::make_unique<clip_multimedia>(std::move(ptr), std::move(hash), true);
}

struct chat_complete_req : complete_req {
    explicit chat_complete_req(const std::string & id) : complete_req(id, REQ_CHAT_COMPLETE) {}

    /* LLAMA BOX */

    std::vector<std::unique_ptr<clip_multimedia>> multimedias;  // images and audios
    // template
    common_chat_params                            chat_params;

    /* OPEN AI*/

    // std::string model;                                               // inherit
    std::vector<common_chat_msg>  messages;
    // bool store = false;
    // std::string reasoning_effort;
    // json metadata;
    float                         frequency_penalty = 0.0f;
    std::vector<llama_logit_bias> logit_bias;
    // bool logprobs = false;
    // int32_t top_logprobs          = 0;                               // inherit // migrate "logprobs"
    // int32_t max_tokens = -1;                                         // inherit // migrate "max_completion_tokens"
    // int32_t n = 1;                                                   // inherit
    // std::vector<std::string> modalities;
    // json prediction;
    // json audio;
    float                         presence_penalty = 0.0f;
    json                          response_format;
    uint32_t                      seed        = LLAMA_DEFAULT_SEED;
    // std::string service_tier;
    // std::vector<std::string> stop;                                   // inherit
    // bool stream         = false;                                     // inherit
    // json stream_options = {{"include_usage", true}};                 // inherit
    float                         temperature = 1.0;
    float                         top_p       = 1.0;
    std::vector<common_chat_tool> tools;                                               // migrate "functions"
    common_chat_tool_choice       tool_choice         = COMMON_CHAT_TOOL_CHOICE_NONE;  // migrate "function_call"
    bool                          parallel_tool_calls = true;
    // std::string user;
};

static inline std::unique_ptr<chat_complete_req> get_chat_complete_req(
    const httplib::Request & request, httplib::Response & response, const httpserver_params & hparams,
    const llama_context * llm_ctx, const bool support_tool_calls, const common_chat_templates * chat_templates) {
    const common_params & params = hparams.llm_params;

    const std::string rid = response.get_header_value(HEADER_X_REQUEST_ID);
    const json        req = json::parse(request.body);
    if (!req.contains("messages")) {
        throw std::invalid_argument("Illegal param: \"messages\" is required");
    } else if (!req.at("messages").is_array()) {
        throw std::invalid_argument("Illegal param: \"messages\" must be a list");
    }

    // print32_t the request for debugging
    if (common_log_verbosity_thold > 1) {
        json req_cp = req;
        if (common_log_verbosity_thold < 3) {
            req_cp["messages"] = "...";
        }
        SRV_INF("rid %s | %s\n", rid.c_str(), req_cp.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    }

    std::unique_ptr<chat_complete_req> ptr = std::make_unique<chat_complete_req>(rid.c_str());

    ptr->sampling = prepare_sampling(req, params.sampling, llm_ctx);

    if (req.contains("lora")) {
        const json & lora = req.at("lora");
        if (!lora.is_array()) {
            throw std::invalid_argument("Illegal param: \"lora\" must be a list");
        }
        ptr->lora_adapters = params.lora_adapters;
        // clear value
        for (common_adapter_lora_info & la : ptr->lora_adapters) {
            la.scale = 0.0f;
        }
        // set value
        int32_t max_id = int32_t(ptr->lora_adapters.size()) - 1;
        for (const json & l : lora) {
            if (!l.is_object()) {
                throw std::invalid_argument("Illegal param: \"lora\" must be a list of objects");
            }
            int32_t id    = json_value(l, "id", -1);
            float   scale = json_value(l, "scale", 0.0f);
            if (id < 0 || id > max_id) {
                throw std::invalid_argument("Illegal param: \"id\" must be in the range [0, " + std::to_string(max_id) +
                                            "]");
            }
            ptr->lora_adapters[id].scale = scale;
        }
    }

    ptr->model = json_value(req, "model", params.model_alias);

    {
        json messages = req.at("messages");
        for (const json & msg : messages) {
            std::string role = json_value(msg, "role", std::string());
            std::string content;
            // content
            if (msg.contains("content") && !msg.at("content").is_null()) {
                // string content
                if (msg.at("content").is_string()) {
                    content = msg.at("content").get<std::string>();
                }
                // array content
                else if (msg.at("content").is_array()) {
                    int32_t n_mtmd = 0;
                    for (const json & part : msg.at("content")) {
                        if (!part.is_object()) {
                            throw std::invalid_argument("Illegal param: \"content\" must be a list of objects");
                        }
                        std::string part_type = json_value(part, "type", std::string());
                        if (part_type.empty()) {
                            throw std::invalid_argument(R"(Illegal param: "content" item must have a "type" field)");
                        }
                        if (!part.contains(part_type)) {
                            throw std::invalid_argument(R"(Illegal param: "content" item with type ")" + part_type +
                                                        R"(" must have a field with the same name)");
                        }
                        // image_url
                        if (part_type == "image_url") {
                            // process image
                            const json & img = part.at("image_url");
                            if (img.empty() || !img.is_object()) {
                                throw std::invalid_argument(R"(Illegal param: "image_url" must be a non-empty object)");
                            }
                            std::string url = json_value(img, "url", std::string());
                            if (url.empty()) {
                                throw std::invalid_argument(
                                    R"(Illegal param: "image_url" must have a "url" field with non-empty value)");
                            }
                            if (url.find("data:image/") != std::string::npos) {
                                const std::string split = "base64,";
                                const size_t      idx   = url.find(split);
                                if (idx == std::string::npos) {
                                    throw std::invalid_argument(
                                        "Illegal param: \"url\" must be a valid base64-encoded image");
                                }
                                url = url.substr(idx + split.length());
                                if (img.empty()) {
                                    throw std::invalid_argument(
                                        "Illegal param: \"url\" is an empty image base64-encoded data");
                                }
                                try {
                                    std::vector<uint8_t> img_buff = decode_base64(url);
                                    ptr->multimedias.push_back(get_clip_image(std::move(img_buff)));
                                } catch (const std::exception & e) {
                                    throw std::invalid_argument(
                                        "Illegal param: \"url\" must be a valid base64-encoded image");
                                }
                            } else {
                                std::string host, path;
                                if (size_t pos = url.find("://"); pos == std::string::npos) {
                                    throw std::invalid_argument(
                                        "Illegal param: \"url\" must be a data URI or a valid URL");
                                } else {
                                    pos = url.find('/', pos + 3);
                                    if (pos == std::string::npos) {
                                        host = img;
                                        path = "/";
                                    } else {
                                        host = url.substr(0, pos);
                                        path = url.substr(pos);
                                    }
                                }
                                httplib::Client cli(host);
                                cli.set_connection_timeout(15, 0);                  // 15 seconds
                                cli.set_read_timeout(300, 0);                       // 5 minutes
                                cli.set_keep_alive(false);                          // close connection after request
                                cli.set_follow_location(true);                      // follow redirects
                                cli.set_default_headers({
                                    { "User-Agent", "llama-box" }
                                });               // set user-agent
                                cli.set_url_encode(true);                           // encode URL
                                cli.set_tcp_nodelay(true);                          // disable Nagle's algorithm
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
                                cli.enable_server_certificate_verification(false);  // disable SSL verification
#endif
                                httplib::Result resp = cli.Get(path);
                                if (!resp || resp->status != httplib::StatusCode::OK_200) {
                                    throw std::invalid_argument(
                                        R"(Illegal param: invalid "url", failed to fetch image from URL: )" + url +
                                        ", status: " + std::to_string(resp ? resp->status : -1) +
                                        ", reason: " + (resp ? resp->reason : "unknown"));
                                }
                                std::vector<uint8_t> img_buff(resp->body.begin(), resp->body.end());
                                ptr->multimedias.push_back(get_clip_image(std::move(img_buff)));
                            }
                            n_mtmd++;
                        }
                        // input_audio
                        else if (part_type == "input_audio") {
                            // process audio
                            const json & audio = part.at("input_audio");
                            if (audio.empty() || !audio.is_object()) {
                                throw std::invalid_argument(
                                    R"(Illegal param: "input_audio" must be a non-empty object)");
                            }
                            std::string format = json_value(audio, "format", std::string());
                            if (format != "wav" && format != "mp3") {
                                throw std::invalid_argument(
                                    R"(Illegal param: "input_audio" must have a "format" field with value "wav" or "mp3")");
                            }
                            std::string data = json_value(audio, "data", std::string());
                            if (data.empty()) {
                                throw std::invalid_argument(
                                    R"(Illegal param: "input_audio" must have a "data" field with non-empty value)");
                            }
                            try {
                                std::vector<uint8_t> aud_buff = decode_base64(data);
                                ptr->multimedias.push_back(get_clip_audio(std::move(aud_buff)));
                            } catch (const std::exception & e) {
                                throw std::invalid_argument(
                                    R"(Illegal param: "data" of "input_audio" must be a valid base64-encoded string)");
                            }
                            n_mtmd++;
                        }
                        // text
                        else if (part_type == "text") {
                            if (!content.empty()) {
                                content += "\n";
                            }
                            for (int32_t i = 0; i < n_mtmd; i++) {
                                content += "<MTMD/>\n";
                            }
                            content += json_value(part, "text", std::string());
                            n_mtmd = 0;
                        }
                        // other
                        else {
                            throw std::invalid_argument(R"(Illegal param: "content" item with type ")" + part_type +
                                                        R"(" is not supported)");
                        }
                    }
                    for (int32_t i = 0; i < n_mtmd; i++) {
                        content += "\n<MTMD/>";
                    }
                }
                // illegal
                else {
                    throw std::invalid_argument("Illegal param: invalid \"content\"");
                }
                ptr->messages.push_back({ role, content, {}, {}, "", "", "" });
            }
            // tool_calls
            else if (msg.contains("tool_calls") && !msg.at("tool_calls").is_null()) {
                // array tool_calls
                if (msg.at("tool_calls").is_array()) {
                    std::vector<common_chat_tool_call> chat_tcs;
                    for (const json & part : msg.at("tool_calls")) {
                        common_chat_tool_call chat_tc;
                        if (!part.contains("type") || part.at("type") != "function") {
                            continue;
                        }
                        if (!part.contains("function") || !part.at("function").is_object()) {
                            continue;
                        }
                        const json & func = part.at("function");
                        if (!func.contains("name") || !func.at("name").is_string()) {
                            continue;
                        }
                        if (!func.contains("arguments") || !func.at("arguments").is_string()) {
                            continue;
                        }
                        chat_tc.name      = func.at("name").get<std::string>();
                        chat_tc.arguments = func.at("arguments").get<std::string>();
                        if (func.contains("id")) {
                            chat_tc.id = func.at("id").get<std::string>();
                        }
                        chat_tcs.push_back(chat_tc);
                    }
                    ptr->messages.push_back({ role, "", {}, chat_tcs, "", "", "" });
                }
                // illegal
                else {
                    throw std::invalid_argument("Illegal param: invalid \"tool_calls\"");
                }
            }
            // illegal
            else {
                throw std::invalid_argument("Illegal param: missing 'content' or 'tool_calls' in \"messages\" item");
            }
        }
    }

    ptr->frequency_penalty = json_value(req, "frequency_penalty", params.sampling.penalty_freq);

    if (req.contains("logit_bias")) {
        const json & logit_bias = req.at("logit_bias");
        if (!logit_bias.is_object()) {
            throw std::invalid_argument("Illegal param: \"logit_bias\" must be a map");
        }
        const llama_vocab * vocab      = llama_model_get_vocab(llama_get_model(llm_ctx));
        const int32_t       vocab_size = llama_vocab_n_tokens(vocab);
        for (const auto & el : logit_bias.items()) {
            llama_token tok = std::stoi(el.key());
            if (tok < 0) {
                throw std::invalid_argument("Illegal param: \"logit_bias\" keys must be integer string");
            } else if (tok >= vocab_size) {
                throw std::invalid_argument("Illegal param: \"logit_bias\" keys must be in the range [0, vocab_size)");
            }
            float bias = -INFINITY;
            if (el.value().is_number()) {
                bias = el.value().get<float>();
                if (bias < -100 || bias > 100) {
                    throw std::invalid_argument(
                        "Illegal param: \"logit_bias\" values must be in the range [-100, 100]");
                }
            } else if (el.value().is_boolean()) {
                if (el.value().get<bool>()) {
                    continue;
                }
            } else {
                throw std::invalid_argument("Illegal param: \"logit_bias\" values must be a number or boolean");
            }
            ptr->logit_bias.push_back({ tok, bias });
        }
    }

    if (req.contains("logprobs")) {
        bool logprobs = json_value(req, "logprobs", false);
        if (logprobs) {
            ptr->logprobs = json_value(req, "top_logprobs", params.sampling.n_probs);
            if (ptr->logprobs < 0 || ptr->logprobs > 20) {
                throw std::invalid_argument("Illegal param: \"top_logprobs\" must be in the range [0, 20]");
            }
        }
    } else if (req.contains("top_logprobs")) {
        throw std::invalid_argument(R"(Illegal param: "top_logprobs" must use with "logprobs")");
    }

    if (req.contains("max_completion_tokens")) {
        ptr->max_tokens = json_value(req, "max_completion_tokens", ptr->max_tokens);
    } else {
        ptr->max_tokens = json_value(req, "max_tokens", ptr->max_tokens);
    }
    if (ptr->max_tokens > int32_t(llama_n_ctx(llm_ctx))) {
        throw std::invalid_argument(
            R"(Illegal param: "max_completion_tokens" or "max_tokens" must be less than or equal to the model's context length)");
    }

    ptr->presence_penalty = json_value(req, "presence_penalty", params.sampling.penalty_present);

    if (req.contains("response_format")) {
        ptr->response_format = req.at("response_format");
    }

    ptr->seed = normalize_seed(json_value(req, "seed", params.sampling.seed));

    if (req.contains("stop")) {
        const json & stop = req.at("stop");
        if (stop.is_string()) {
            ptr->stop.push_back(stop.get<std::string>());
        } else if (stop.is_array()) {
            for (const json & s : stop) {
                if (!s.is_string()) {
                    throw std::invalid_argument("Illegal param: \"stop\" must be a list of strings");
                }
                ptr->stop.push_back(s.get<std::string>());
            }
        } else if (!stop.is_null()) {
            throw std::invalid_argument("Illegal param: \"stop\" must be a string or a list of strings");
        }
    }

    ptr->stream = json_value(req, "stream", false);

    if (ptr->stream && req.contains("stream_options")) {
        const json & stream_options = req.at("stream_options");
        if (!stream_options.is_object()) {
            throw std::invalid_argument("Illegal param: \"stream_options\" must be an object");
        }
        for (const auto & el : stream_options.items()) {
            ptr->stream_options[el.key()] = el.value();
        }
    }

    ptr->temperature = json_value(req, "temperature", params.sampling.temp);

    ptr->top_p = json_value(req, "top_p", params.sampling.top_p);

    if (support_tool_calls) {
        // "tools" and "functions", migrate "functions" to "tools"
        if (req.contains("tools") && !req.contains("functions")) {
            const json & tools = req.at("tools");
            if (!tools.is_array()) {
                throw std::invalid_argument("Illegal param: \"tools\" must be an array");
            }
            for (const json & tool : tools) {
                if (!tool.contains("function")) {
                    continue;
                }
                const json & func = tool.at("function");
                if (!func.contains("name") || !func.at("name").is_string()) {
                    continue;
                }
                if (!func.contains("parameters") || !func.at("parameters").is_object()) {
                    continue;
                }
                std::string name        = func.at("name");
                std::string description = json_value(func, "description", std::string());
                std::string parameters  = func.at("parameters").dump(-1, ' ', false, json::error_handler_t::replace);
                ptr->tools.push_back({ name, description, parameters });
            }
        } else if (req.contains("functions")) {
            const json & functions = req.at("functions");
            if (!functions.is_array()) {
                throw std::invalid_argument("Illegal param: \"functions\" must be an array");
            }
            for (const json & func : functions) {
                if (!func.contains("name") || !func.at("name").is_string()) {
                    continue;
                }
                if (!func.contains("parameters") || !func.at("parameters").is_object()) {
                    continue;
                }
                std::string name        = json_value(func, "name", std::string());
                std::string description = json_value(func, "description", std::string());
                std::string parameters  = func.at("parameters").dump(-1, ' ', false, json::error_handler_t::replace);
                ptr->tools.push_back({ name, description, parameters });
            }
        }
        if (!ptr->tools.empty()) {
            ptr->tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
            // "tool_choice" and "function_call", migrate "function_call" to "tool_choice"
            if (req.contains("tool_choice") && !req.contains("function_call")) {
                const json & tc = req.at("tool_choice");
                if (tc.is_object() && tc.contains("function")) {
                    const json &      fc = tc.at("function");
                    const std::string fn = json_value(fc, "name", std::string());
                    ptr->tools.erase(std::remove_if(ptr->tools.begin(), ptr->tools.end(),
                                                    [fn](const common_chat_tool & t) { return t.name == fn; }),
                                     ptr->tools.end());
                    ptr->tool_choice =
                        ptr->tools.empty() ? COMMON_CHAT_TOOL_CHOICE_NONE : COMMON_CHAT_TOOL_CHOICE_REQUIRED;
                } else if (tc.is_string()) {
                    ptr->tool_choice = common_chat_tool_choice_parse_oaicompat(tc.get<std::string>());
                } else {
                    throw std::invalid_argument("Illegal param: \"tool_choice\" must be a string or an object");
                }
            } else if (req.contains("function_call")) {
                const json & fc = req.at("function_call");
                if (fc.is_object()) {
                    const std::string fn = json_value(fc, "name", std::string());
                    ptr->tools.erase(std::remove_if(ptr->tools.begin(), ptr->tools.end(),
                                                    [fn](const common_chat_tool & t) { return t.name == fn; }),
                                     ptr->tools.end());
                    ptr->tool_choice =
                        ptr->tools.empty() ? COMMON_CHAT_TOOL_CHOICE_NONE : COMMON_CHAT_TOOL_CHOICE_REQUIRED;
                } else if (fc.is_string()) {
                    ptr->tool_choice = common_chat_tool_choice_parse_oaicompat(fc.get<std::string>());
                } else {
                    throw std::invalid_argument("Illegal param: \"function_call\" must be a string or an object");
                }
            }
            // "parallel_tool_calls"
            ptr->parallel_tool_calls = json_value(req, "parallel_tool_calls", true) && support_tool_calls;
        }
    }

    {
        json json_schema;
        if (!ptr->response_format.empty()) {
            std::string response_type = json_value(ptr->response_format, "type", std::string());
            if (response_type == "json_object") {
                json_schema = json_value(ptr->response_format, "schema", json());
            } else if (response_type == "json_schema") {
                if (!ptr->response_format.contains("json_schema")) {
                    throw std::invalid_argument(
                        "Illegal param: using json schema response format must contain \"json_schema\"");
                }
                json_schema = json_value(ptr->response_format.at("json_schema"), "schema", json());
            } else if (!response_type.empty() && response_type != "text") {
                throw std::invalid_argument(
                    "Illegal param: \"response_format\" must be one of 'text', 'json_schema' or 'json_object', but "
                    "got: " +
                    response_type);
            }
        }

        common_chat_templates_inputs inputs;
        inputs.messages              = ptr->messages;
        inputs.tools                 = ptr->tools;
        inputs.tool_choice           = ptr->tool_choice;
        inputs.json_schema           = json_schema.is_null() ? "" : json_schema.dump();
        inputs.add_generation_prompt = json_value(req, "add_generation_prompt", true);
        inputs.use_jinja             = params.use_jinja;
        inputs.parallel_tool_calls   = ptr->parallel_tool_calls;
        inputs.enable_thinking       = params.reasoning_budget != 0;
        // NB(thxCode): common_chat_templates_apply2 is a patch.
        ptr->chat_params             = common_chat_templates_apply2(llama_get_model(llm_ctx), chat_templates, inputs);
        SRV_INFV(3, "rid %s | formatted prompt\n%s\n", rid.c_str(), ptr->chat_params.prompt.c_str());
    };

    // merge sampling
    {
        ptr->sampling.penalty_freq    = ptr->frequency_penalty;
        ptr->sampling.n_probs         = ptr->logprobs;
        ptr->sampling.penalty_present = ptr->presence_penalty;
        ptr->sampling.logit_bias      = ptr->logit_bias;
        ptr->sampling.seed            = ptr->seed;
        ptr->sampling.temp            = ptr->temperature;
        ptr->sampling.top_p           = ptr->top_p;
        if (!ptr->chat_params.grammar.empty()) {
            ptr->sampling.grammar      = ptr->chat_params.grammar;
            ptr->sampling.grammar_lazy = ptr->chat_params.grammar_lazy;
            const llama_vocab * vocab  = llama_model_get_vocab(llama_get_model(llm_ctx));
            for (const std::string & t : ptr->chat_params.preserved_tokens) {
                llama_tokens toks = common_tokenize(vocab, t, false, true);
                if (toks.size() == 1) {
                    ptr->sampling.preserved_tokens.insert(toks[0]);
                }
            }
            for (common_grammar_trigger & t : ptr->chat_params.grammar_triggers) {
                if (t.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                    const std::string & word = t.value;
                    llama_tokens        toks = common_tokenize(vocab, word, false, true);
                    if (toks.size() == 1) {
                        ptr->sampling.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN, word, toks[0] });
                        ptr->sampling.preserved_tokens.insert(toks[0]);
                        continue;
                    }
                }
                ptr->sampling.grammar_triggers.push_back(std::move(t));
            }
            for (const std::string & s : ptr->chat_params.additional_stops) {
                ptr->stop.push_back(s);
            }
        }
    }

    return ptr;
}

struct embed_req : breq {
    explicit embed_req(const std::string & id) : breq(id, REQ_EMBED) {}

    /* LLAMA BOX */

    /* OPEN AI*/

    // std::string model;                                     // inherit
    json        input;
    std::string encoding_format = "float";
};

static inline std::unique_ptr<embed_req> get_embed_req(const httplib::Request & request, httplib::Response & response,
                                                       const httpserver_params & hparams) {
    const common_params & params = hparams.llm_params;

    const std::string rid = response.get_header_value(HEADER_X_REQUEST_ID);
    const json        req = json::parse(request.body);
    if (!req.contains("input")) {
        throw std::invalid_argument("Illegal param: \"input\" is required");
    }

    // print32_t the request for debugging
    if (common_log_verbosity_thold > 1) {
        json req_cp = req;
        if (common_log_verbosity_thold < 3) {
            req_cp["input"] = "[...]";
        }
        SRV_INF("rid %s | %s\n", rid.c_str(), req_cp.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    }

    std::unique_ptr<embed_req> ptr = std::make_unique<embed_req>(rid.c_str());

    ptr->model = json_value(req, "model", params.model_alias);

    ptr->input = req.at("input");

    if (req.contains("encoding_format")) {
        ptr->encoding_format = req.at("encoding_format");
        if (ptr->encoding_format != "float" && ptr->encoding_format != "base64") {
            throw std::invalid_argument("Illegal param: \"encoding_format\" must be one of 'float' or 'base64'");
        }
    }

    return ptr;
}

struct rerank_req : breq {
    explicit rerank_req(const std::string & id) : breq(id, REQ_RERANK) {}

    /* LLAMA BOX */

    /* JINJA */

    // std::string model;                                     // inherit
    json              query;
    std::vector<json> documents;
    int32_t           top_n            = 1;
    bool              return_documents = false;
    bool              normalize        = false;
};

static inline std::unique_ptr<rerank_req> get_rerank_req(const httplib::Request & request, httplib::Response & response,
                                                         const httpserver_params & hparams) {
    const common_params & params = hparams.llm_params;

    const std::string rid = response.get_header_value(HEADER_X_REQUEST_ID);
    const json        req = json::parse(request.body);
    if (!req.contains("query")) {
        throw std::invalid_argument("Illegal param: \"query\" is required");
    }
    if (!req.contains("documents")) {
        throw std::invalid_argument("Illegal param: \"documents\" is required");
    }
    if (!req.at("documents").is_array() || req.at("documents").empty()) {
        throw std::invalid_argument("Illegal param: \"documents\" must be a list with at least one item");
    }

    // print32_t the request for debugging
    if (common_log_verbosity_thold > 1) {
        json req_cp = req;
        if (common_log_verbosity_thold < 3) {
            req_cp["query"]     = "...";
            req_cp["documents"] = "...";
        }
        SRV_INF("rid %s | %s\n", rid.c_str(), req_cp.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    }

    std::unique_ptr<rerank_req> ptr = std::make_unique<rerank_req>(rid.c_str());

    ptr->model = json_value(req, "model", params.model_alias);

    ptr->query = req.at("query");

    for (const json & doc : req.at("documents")) {
        if (doc.is_string()) {
            ptr->documents.push_back(doc);
        } else if (doc.is_object() && doc.contains("text")) {
            ptr->documents.push_back(doc.at("text"));
        } else {
            throw std::invalid_argument(
                "Illegal param: \"documents\" must be an array of strings or objects with 'text'");
        }
    }

    ptr->top_n = int32_t(std::min(json_value(req, "top_n", ptr->documents.size()), ptr->documents.size()));
    if (ptr->top_n <= 0) {
        throw std::invalid_argument("Illegal param: \"top_n\" must be greater than 0");
    }

    ptr->return_documents = json_value(req, "return_documents", true);

    ptr->normalize = json_value(req, "normalize", true);

    return ptr;
}

struct image_req : breq {
    explicit image_req(const std::string & id, req_type type) : breq(id, type) {}

    /* LLAMA BOX */

    [[nodiscard]] virtual const char * get_prompt() { return nullptr; }

    // sample
    stablediffusion_params_sampling       sampling;
    // lora
    std::vector<common_adapter_lora_info> lora_adapters;
    // stream
    bool                                  stream         = false;
    json                                  stream_options = {
        { "include_usage",  true  },
        { "chunk_result",   false },
        { "chunk_size",     4096  },
        { "preview",        false },
        { "preview_faster", false }, // deprecated
    };
};

struct image_generate_req : image_req {
    explicit image_generate_req(const std::string & id) : image_req(id, REQ_IMAGE_GENERATE) {}

    /* LLAMA BOX */

    [[nodiscard]] const char * get_prompt() override { return prompt.c_str(); }

    /* OPEN AI */

    std::string prompt;
    // std::string model;                                     // inherit
    // int32_t n = 1;                                         // inherit
    std::string quality         = "standard";
    std::string response_format = "b64_json";
    std::string size            = "512x512";
    std::string style           = "vivid";
    // std::string user;
};

static inline std::unique_ptr<image_generate_req> get_image_generate_req(const httplib::Request &  request,
                                                                         httplib::Response &       response,
                                                                         const httpserver_params & hparams) {
    const stablediffusion_params & params = hparams.sd_params;

    const std::string rid = response.get_header_value(HEADER_X_REQUEST_ID);
    const json        req = json::parse(request.body);
    if (!req.contains("prompt")) {
        throw std::invalid_argument("Illegal param: \"prompt\" is required");
    }

    // print32_t the request for debugging
    if (common_log_verbosity_thold > 1) {
        json req_cp = req;
        if (common_log_verbosity_thold < 3) {
            req_cp["prompt"] = "...";
        }
        SRV_INF("rid %s | %s\n", rid.c_str(), req_cp.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    }

    std::unique_ptr<image_generate_req> ptr = std::make_unique<image_generate_req>(rid.c_str());

    ptr->sampling = prepare_sampling(req, params.sampling);

    if (req.contains("lora")) {
        const json & lora = req.at("lora");
        if (!lora.is_array()) {
            throw std::invalid_argument("Illegal param: \"lora\" must be a list");
        }
        ptr->lora_adapters = params.lora_adapters;
        // clear value
        for (common_adapter_lora_info & la : ptr->lora_adapters) {
            la.scale = 0.0f;
        }
        // set value
        int32_t max_id = int32_t(ptr->lora_adapters.size()) - 1;
        for (const json & l : lora) {
            if (!l.is_object()) {
                throw std::invalid_argument("Illegal param: \"lora\" must be a list of objects");
            }
            int32_t id    = json_value(l, "id", -1);
            float   scale = json_value(l, "scale", 0.0f);
            if (id < 0 || id > max_id) {
                throw std::invalid_argument("Illegal param: \"id\" must be in the range [0, " + std::to_string(max_id) +
                                            "]");
            }
            ptr->lora_adapters[id].scale = scale;
        }
    }

    ptr->stream = json_value(req, "stream", false);

    if (ptr->stream && req.contains("stream_options")) {
        const json & stream_options = req.at("stream_options");
        if (!stream_options.is_object()) {
            throw std::invalid_argument("Illegal param: \"stream_options\" must be an object");
        }
        for (const auto & el : stream_options.items()) {
            ptr->stream_options[el.key()] = el.value();
        }
    }

    ptr->prompt = req.at("prompt");

    ptr->model = json_value(req, "model", params.model_alias);

    ptr->n = json_value(req, "n", 1);
    if (ptr->n <= 0) {
        throw std::invalid_argument("Illegal param: \"n\" must be greater than 0");
    } else if (ptr->n > params.max_batch_count) {
        throw std::invalid_argument("Illegal param: \"n\" must be less than or equal to " +
                                    std::to_string(params.max_batch_count));
    }

    ptr->quality = json_value(req, "quality", std::string("standard"));
    if (ptr->quality != "hd" && ptr->quality != "standard") {
        throw std::invalid_argument("Illegal param: \"quality\" must be one of 'hd' or 'standard'");
    }

    ptr->response_format = json_value(req, "response_format", std::string("b64_json"));
    if (ptr->response_format != "b64_json") {
        throw std::invalid_argument("Illegal param: \"response_format\" must be 'b64_json'");
    }

    ptr->size = json_value(req, "size", std::string("256x256"));
    {
        size_t pos = ptr->size.find('x');
        if (pos == std::string::npos) {
            throw std::invalid_argument("Illegal param: \"size\" must be in the format '{width}x{height}'");
        }
        int32_t width  = std::stoi(ptr->size.substr(0, pos));
        int32_t height = std::stoi(ptr->size.substr(pos + 1));
        if (width < 256 || height < 256) {
            throw std::invalid_argument("Illegal param: width and height of \"size\" must be at least 256");
        }
        if (width > params.sampling.width) {
            throw std::invalid_argument("Illegal param: width of \"size\" must be at most " +
                                        std::to_string(params.sampling.width));
        }
        if (height > params.sampling.height) {
            throw std::invalid_argument("Illegal param: height of \"size\" must be at most " +
                                        std::to_string(params.sampling.height));
        }
        if (width % 64 != 0 || height % 64 != 0) {
            throw std::invalid_argument("Illegal param: width and height of \"size\" must be multiples of 64");
        }
        ptr->sampling.width  = width;
        ptr->sampling.height = height;
    }

    ptr->style = json_value(req, "style", std::string("vivid"));
    if (ptr->style != "vivid" && ptr->style != "natural") {
        throw std::invalid_argument("Illegal param: \"style\" must be one of 'vivid' or 'natural'");
    }

    // merge sampling
    if (!req.contains("sampler") && !req.contains("sample_method")) {
        if (ptr->quality == "hd") {
            ptr->sampling.sampling_steps += 2;
            ptr->sampling.negative_prompt =
                ptr->sampling.negative_prompt.empty() ? "low quality" : ptr->sampling.negative_prompt + ", low quality";
        }
        if (ptr->style == "vivid") {
            ptr->sampling.negative_prompt =
                ptr->sampling.negative_prompt.empty() ? "not vivid" : ptr->sampling.negative_prompt + ", not vivid";
        } else {
            ptr->sampling.negative_prompt =
                ptr->sampling.negative_prompt.empty() ? "unnatural" : ptr->sampling.negative_prompt + ", unnatural";
        }
    }

    return ptr;
}

struct image_edit_req : image_req {
    explicit image_edit_req(const std::string & id) : image_req(id, REQ_IMAGE_EDIT) {}

    ~image_edit_req() override {
        if (sampling.control_img_buffer != nullptr) {
            stbi_image_free(sampling.control_img_buffer);
            sampling.control_img_buffer = nullptr;
        }
        if (sampling.init_img_buffer != nullptr) {
            stbi_image_free(sampling.init_img_buffer);
            sampling.init_img_buffer = nullptr;
        }
        if (sampling.mask_img_buffer != nullptr) {
            stbi_image_free(sampling.mask_img_buffer);
            sampling.mask_img_buffer = nullptr;
        }
    }

    /* LLAMA BOX */

    // control, temporary use, will not value in "process"
    std::vector<uint8_t> control;

    [[nodiscard]] const char * get_prompt() override { return prompt.c_str(); }

    /* OPEN AI */

    // image, temporary use, will not value in "process"
    std::vector<uint8_t> image;
    std::string          prompt;
    // mask, temporary use, will not value in "process"
    std::vector<uint8_t> mask;
    // std::string model;                                     // inherit
    // int32_t n = 1;                                         // inherit
    std::string          size            = "512x512";
    std::string          response_format = "b64_json";
    // std::string user;
};

static inline std::unique_ptr<image_edit_req> get_image_edit_req(const httplib::Request &  request,
                                                                 httplib::Response &       response,
                                                                 const httpserver_params & hparams) {
    const stablediffusion_params & params = hparams.sd_params;

    const std::string                     rid = response.get_header_value(HEADER_X_REQUEST_ID);
    const httplib::MultipartFormDataMap & req = request.files;
    if (req.find("prompt") == req.end()) {
        throw std::invalid_argument("Illegal param: \"prompt\" is required");
    } else if (req.find("image") == req.end()) {
        throw std::invalid_argument("Illegal param: \"image\" is required");
    }

    // print32_t the request for debugging
    if (common_log_verbosity_thold > 1) {
        json req_cp = json::object();
        for (const auto & el : req) {
            if (el.first == "image" || el.first == "mask" || el.first == "control") {
                req_cp[el.first] = "...";
            } else {
                req_cp[el.first] = el.second.content;
            }
        }
        if (common_log_verbosity_thold < 3) {
            req_cp["prompt"] = "...";
        }
        SRV_INF("rid %s | %s\n", rid.c_str(), req_cp.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    }

    std::unique_ptr<image_edit_req> ptr = std::make_unique<image_edit_req>(rid.c_str());

    ptr->sampling = prepare_sampling(req, params.sampling);

    auto item = req.find("lora");
    if (item != req.end()) {
        json lora;
        try {
            lora = json::parse(item->second.content);
            if (!lora.is_array()) {
                throw std::invalid_argument("Illegal param: \"lora\" must be a JSON list");
            }
        } catch (const std::exception &) {
            throw std::invalid_argument("Illegal param: \"lora\" must be a JSON list");
        }
        ptr->lora_adapters = params.lora_adapters;
        // clear value
        for (common_adapter_lora_info & la : ptr->lora_adapters) {
            la.scale = 0.0f;
        }
        // set value
        int32_t max_id = int32_t(ptr->lora_adapters.size()) - 1;
        for (const json & l : lora) {
            if (!l.is_object()) {
                throw std::invalid_argument("Illegal param: \"lora\" must be a list of objects");
            }
            int32_t id    = json_value(l, "id", -1);
            float   scale = json_value(l, "scale", 0.0f);
            if (id < 0 || id > max_id) {
                throw std::invalid_argument("Illegal param: \"id\" must be in the range [0, " + std::to_string(max_id) +
                                            "]");
            }
            ptr->lora_adapters[id].scale = scale;
        }
    }

    item = req.find("stream");
    if (item != req.end()) {
        ptr->stream = item->second.content == "true";
    }

    if (ptr->stream) {
        item = req.find("stream_options_include_usage");
        if (item != req.end()) {
            ptr->stream_options["include_usage"] = item->second.content == "true";
        }
        item = req.find("stream_options_chunk_result");
        if (item != req.end()) {
            try {
                ptr->stream_options["chunk_result"] = item->second.content == "true";
            } catch (...) {
                // NOP
            }
        }
        item = req.find("stream_options_chunk_size");
        if (item != req.end()) {
            try {
                ptr->stream_options["chunk_size"] = std::stol(item->second.content);
            } catch (...) {
                // NOP
            }
        }
        item = req.find("stream_options_preview");
        if (item != req.end()) {
            ptr->stream_options["preview"] = item->second.content == "true";
        }
        item = req.find("stream_options_preview_faster");  // deprecated
        if (item != req.end()) {
            ptr->stream_options["preview"] = item->second.content == "true";
        }
    }

    item = req.find("control");
    if (item != req.end()) {
        ptr->control.assign(item->second.content.begin(), item->second.content.end());
    }

    item = req.find("image");
    ptr->image.assign(item->second.content.begin(), item->second.content.end());

    item        = req.find("prompt");
    ptr->prompt = item->second.content;

    item = req.find("mask");
    if (item != req.end()) {
        ptr->mask.assign(item->second.content.begin(), item->second.content.end());
    }

    item = req.find("model");
    if (item != req.end()) {
        ptr->model = item->second.content;
    }

    item = req.find("n");
    if (item != req.end()) {
        try {
            ptr->n = std::stoi(item->second.content);
        } catch (...) {
            // NOP
        }
    }

    item = req.find("size");
    if (item != req.end()) {
        ptr->size  = item->second.content;
        size_t pos = ptr->size.find('x');
        if (pos == std::string::npos) {
            throw std::invalid_argument("Illegal param: \"size\" must be in the format '{width}x{height}'");
        }
        int32_t width  = std::stoi(ptr->size.substr(0, pos));
        int32_t height = std::stoi(ptr->size.substr(pos + 1));
        if (width < 256 || height < 256) {
            throw std::invalid_argument("Illegal param: width and height of \"size\" must be at least 256");
        }
        if (width > params.sampling.width) {
            throw std::invalid_argument("Illegal param: width of \"size\" must be at most " +
                                        std::to_string(params.sampling.width));
        }
        if (height > params.sampling.height) {
            throw std::invalid_argument("Illegal param: height of \"size\" must be at most " +
                                        std::to_string(params.sampling.height));
        }
        if (width % 64 != 0 || height % 64 != 0) {
            throw std::invalid_argument("Illegal param: width and height of \"size\" must be multiples of 64");
        }
        ptr->sampling.width  = width;
        ptr->sampling.height = height;
    }

    item = req.find("response_format");
    if (item != req.end()) {
        ptr->response_format = item->second.content;
        if (ptr->response_format != "b64_json") {
            throw std::invalid_argument("Illegal param: \"response_format\" must be 'b64_json'");
        }
    }

    // merge sampling
    {
#define FREE_IMG_BUFFER                                    \
    if (ptr->sampling.control_img_buffer != nullptr) {     \
        stbi_image_free(ptr->sampling.control_img_buffer); \
    }                                                      \
    if (ptr->sampling.init_img_buffer != nullptr) {        \
        stbi_image_free(ptr->sampling.init_img_buffer);    \
    }                                                      \
    if (ptr->sampling.mask_img_buffer != nullptr) {        \
        stbi_image_free(ptr->sampling.mask_img_buffer);    \
    }
        // control image process
        if (!ptr->control.empty()) {
            int32_t cc                       = 0;
            int32_t cw                       = 0;
            int32_t ch                       = 0;
            ptr->sampling.control_img_buffer = stbi_load_from_memory((const stbi_uc *) ptr->control.data(),
                                                                     (int) ptr->control.size(), &cw, &ch, &cc, 3);
            if (ptr->sampling.control_img_buffer == nullptr) {
                const char * reason = stbi_failure_reason();
                throw std::invalid_argument("Illegal param: \"control\" is not a valid image: " + std::string(reason));
            }
            if (cc < 3 || cw <= 0 || ch <= 0) {
                FREE_IMG_BUFFER;
                throw std::invalid_argument("Illegal param: \"control\" must be a valid RGB image");
            }
            ptr->sampling.height = ch;
            ptr->sampling.width  = cw;
        }
        // init image process
        int32_t iw = 0;
        int32_t ih = 0;
        int32_t ic = 0;
        ptr->sampling.init_img_buffer =
            stbi_load_from_memory((const stbi_uc *) ptr->image.data(), (int) ptr->image.size(), &iw, &ih, &ic, 3);
        if (ptr->sampling.init_img_buffer == nullptr) {
            FREE_IMG_BUFFER;
            const char * reason = stbi_failure_reason();
            throw std::invalid_argument("Illegal param: \"image\" is not a valid image: " + std::string(reason));
        }
        if (ic < 3 || iw <= 0 || ih <= 0) {
            FREE_IMG_BUFFER;
            throw std::invalid_argument("Illegal param: \"image\" must be a valid RGB image");
        }
        if (iw != ptr->sampling.width || ih != ptr->sampling.height) {
            // resize
            int32_t rw                   = ptr->sampling.width;
            int32_t rh                   = ptr->sampling.height;
            auto *  resized_image_buffer = (uint8_t *) malloc(rw * rh * 3);
            if (resized_image_buffer == nullptr) {
                FREE_IMG_BUFFER;
                throw std::invalid_argument("Illegal param: \"image\", failed to allocate memory for resizing");
            }
            if (!stbir_resize(ptr->sampling.init_img_buffer, iw, ih, 0, resized_image_buffer, rw, rh, 0,
                              STBIR_TYPE_UINT8,
                              3,                                                 // RGB
                              STBIR_ALPHA_CHANNEL_NONE,                          // no Alpha
                              0,                                                 // flags
                              STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,                // clamp edge mode
                              STBIR_FILTER_CATMULLROM, STBIR_FILTER_CATMULLROM,  // catmull-rom filter
                              STBIR_COLORSPACE_SRGB,                             // sRGB
                              nullptr)) {
                const char * reason = stbi_failure_reason();
                FREE_IMG_BUFFER;
                throw std::invalid_argument("Illegal param: \"image\", failed to resize: " + std::string(reason));
            }
            stbi_image_free(ptr->sampling.init_img_buffer);
            ptr->sampling.init_img_buffer = resized_image_buffer;
        }
        // mask image process
        if (!ptr->mask.empty()) {
            int32_t mw = 0;
            int32_t mh = 0;
            int32_t mc = 0;
            ptr->sampling.mask_img_buffer =
                stbi_load_from_memory((const stbi_uc *) ptr->mask.data(), (int) ptr->mask.size(), &mw, &mh, &mc, 1);
            if (ptr->sampling.mask_img_buffer == nullptr) {
                FREE_IMG_BUFFER;
                const char * reason = stbi_failure_reason();
                throw std::invalid_argument("Illegal param: \"mask\" is not a valid image: " + std::string(reason));
            }
            if (mc < 1 || mw <= 0 || mh <= 0) {
                FREE_IMG_BUFFER;
                throw std::invalid_argument("Illegal param: \"mask\" must be a valid gray scale image");
            }
            if (mw != ptr->sampling.width || mh != ptr->sampling.height) {
                int32_t rw                  = ptr->sampling.width;
                int32_t rh                  = ptr->sampling.height;
                auto *  resized_mask_buffer = (uint8_t *) malloc(rw * rh * 1);
                if (resized_mask_buffer == nullptr) {
                    FREE_IMG_BUFFER;
                    throw std::invalid_argument("Illegal param: \"mask\", failed to allocate memory for resizing");
                }
                if (!stbir_resize(ptr->sampling.mask_img_buffer, mw, mh, 0, resized_mask_buffer, rw, rh, 0,
                                  STBIR_TYPE_UINT8,
                                  1,                                             // GREY
                                  STBIR_ALPHA_CHANNEL_NONE,                      // no Alpha
                                  0,                                             // flags
                                  STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,            // clamp edge mode
                                  STBIR_FILTER_TRIANGLE, STBIR_FILTER_TRIANGLE,  // box filter
                                  STBIR_COLORSPACE_SRGB,                         // sRGB
                                  nullptr)) {
                    const char * reason = stbi_failure_reason();
                    FREE_IMG_BUFFER;
                    throw std::invalid_argument("Illegal param: \"mask\", failed to resize: " + std::string(reason));
                }
                stbi_image_free(ptr->sampling.mask_img_buffer);
                ptr->sampling.mask_img_buffer = resized_mask_buffer;
            }
        } else {
            ptr->sampling.mask_img_buffer = (uint8_t *) malloc(ptr->sampling.width * ptr->sampling.height * 1);
            if (ptr->sampling.mask_img_buffer == nullptr) {
                FREE_IMG_BUFFER;
                throw std::invalid_argument("Illegal param: \"image\", failed to allocate memory for processing ");
            }
            memset(ptr->sampling.mask_img_buffer, 255, ptr->sampling.width * ptr->sampling.height * 1);
        }

        // clear image buffer asap
        ptr->control.clear();
        ptr->image.clear();
        ptr->mask.clear();
#undef FREE_IMG_BUFFER
    }

    return ptr;
}

enum process_type {
    PROCESS_PREFILL,
    PROCESS_DECODE,
    PROCESS_UNKNOWN,
};

enum task_type {
    TASK_COMPLETIONS,
    TASK_EMBEDDINGS,
    TASK_IMAGES,
    TASK_UNKNOWN,
};

struct btask {
  protected:
    int32_t   id     = -1;
    task_type type   = TASK_UNKNOWN;
    int32_t   seq_id = -1;

  public:
    explicit btask(int32_t id, task_type type, const std::function<bool()> & is_connection_closed) :
        id(id),
        type(type),
        seq_id(id),
        is_connection_closed(is_connection_closed) {}

    virtual ~btask() = default;

    [[nodiscard]] int32_t get_id() const { return id; }

    [[nodiscard]] task_type get_type() const { return type; }

    void set_seq_id(int32_t new_seq_id) { seq_id = new_seq_id; }

    [[nodiscard]] int32_t get_seq_id() const { return seq_id; }

    [[nodiscard]] virtual std::string get_r_id() const {
        static std::string empty;
        return empty;
    }

    [[nodiscard]] virtual req_type get_r_type() const { return REQ_UNKNOWN; }

    [[nodiscard]] virtual std::vector<common_adapter_lora_info> & get_lora_adapters() {
        static std::vector<common_adapter_lora_info> empty;
        return empty;
    }

    [[nodiscard]] virtual bool is_stream() const { return false; }

    [[nodiscard]] virtual json & get_stream_options() const {
        static json empty;
        return empty;
    }

    std::function<bool()> is_connection_closed = []() {
        return true;
    };
};

struct completions_task : btask {
    explicit completions_task(int32_t id, const std::function<bool()> & is_connection_closed) :
        btask(id, TASK_COMPLETIONS, is_connection_closed) {}

    ~completions_task() override {
        if (sampler != nullptr) {
            common_sampler_free(sampler);
        }
        if (sampler_draft != nullptr) {
            common_sampler_free(sampler_draft);
        }
    }

    [[nodiscard]] std::string get_r_id() const override { return req->get_id(); }

    [[nodiscard]] req_type get_r_type() const override { return req->get_type(); }

    [[nodiscard]] std::vector<common_adapter_lora_info> & get_lora_adapters() override { return req->lora_adapters; }

    [[nodiscard]] bool is_stream() const override { return req->stream; }

    [[nodiscard]] json & get_stream_options() const override { return req->stream_options; }

    // input
    std::unique_ptr<RatelimitTokenBucket>                            token_bucket = nullptr;
    std::vector<std::variant<llama_tokens, llama_multimodal_tokens>> tokenized_prompts;
    common_chat_syntax                                               tokenized_prompts_syntax;
    bool                                                             tokenized_prompts_include_multimedias = false;
    bool                                                             tokenized_prompts_include_tools       = false;
    std::string                                                      cmpl_id;
    std::unique_ptr<complete_req>                                    req;

    // process
    llama_pos          pos             = 0;  // indicate the position at present
    llama_pos          pos_discard     = 0;  // count the discard position
    int32_t            i_batch_seq_end = 0;  // indicate the index of the batch seq end
    llama_tokens       processed_tokens;     // stores prompt tokens (if not prompt with images) and generated tokens
    int32_t            n_processed_detokenized = 0;  // indicate how many processed tokens are detokenized
    std::string        generated_finish_reason;      // indicate the reason of finish
    size_t             generated_text_keep_pos = std::string::npos;  // erase after call to_json
    std::string        generated_text;        // keep [generated_text_keep_pos,) after call to_json if streaming
    std::vector<json>  generated_tool_calls;  // erase after call to_json if streaming
    std::vector<float> generated_probs;       // erase after call get_probs_json if streaming
    std::vector<std::vector<std::pair<llama_token /* tok */, float /* prob */>>>
        generated_top_probs;                  // erase after call get_probs_json if streaming

    //// prefill
    int32_t n_prefilling_request = 0;  // indicate how many tokens need to be prefilled
    int32_t n_prefilled          = 0;  // indicate how many tokens have been prefilled
    int32_t n_prefilled_cached   = 0;  // indicate how many prefilled tokens are cached
    int64_t t_start_prefill      = 0;  // indicate the time when prefilling starts
    double  t_prefilled          = 0;  // indicate the time(ms) spent on prefilling
    double  p_prefilled_tps      = 0;

    //// decode
    int32_t                 n_decoding_budget     = 0;  // indicate how many tokens can be decoded
    int32_t                 n_decoded             = 0;  // indicate how many tokens have been decoded
    int64_t                 t_start_decode        = 0;  // indicate the time when decoding starts
    double                  t_decoded             = 0;  // indicate the time(ms) spent on decoding
    double                  p_decoded_tps         = 0;
    struct common_sampler * sampler               = nullptr;
    //// reasoning
    int32_t                 n_reasoning           = 0;  // indicate how many tokens are reasoning
    bool                    reasoning_start_found = false;
    bool                    reasoning_end_found   = false;
    bool                    reasoning_finished    = false;
    //// tool call
    bool                    tool_call_stop_fast   = false;  // collect from request
    ////// non-jinja too calls
    bool                    tool_call_start_found = false;

    //// speculative
    llama_tokens            drafted_tokens;          // store drafted tokens, clear before a new round drafting
    int32_t                 n_drafted          = 0;  // indicate how many tokens are drafted
    int32_t                 n_drafted_accepted = 0;  // indicate how many drafted tokens are accepted
    double                  p_drafted_apt      = 0;
    ////// draft-model speculative decoding
    struct common_sampler * sampler_draft      = nullptr;
    ////// model-free speculative decoding
    common_ngram_cache      ngram_cache;

    void push_generated_token(llama_context * llm_ctx, int32_t tok_idx, llama_token tok) {
        // store tokens
        processed_tokens.push_back(tok);

        // top logprobs
        if (req->logprobs > 0) {
            const std::vector<llama_token_data> cur = get_token_probabilities(llm_ctx, tok_idx);
            const size_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(llm_ctx)));
            const size_t n_probs = req->logprobs;
            // set probability for sampled token
            for (size_t i = 0; i < n_vocab; i++) {
                if (cur[i].id == tok) {
                    generated_probs.push_back(cur[i].p);
                    break;
                }
            }
            // set probability for top n_probs tokens
            generated_top_probs.emplace_back();
            for (size_t i = 0; i < std::min(n_vocab, n_probs); i++) {
                generated_top_probs[generated_top_probs.size() - 1].emplace_back(cur[i].id, cur[i].p);
            }
        }
    }

    json get_probs_json(const llama_context * llm_ctx) {
        if (generated_probs.empty()) {
            return {};
        }

        size_t tokens_size = processed_tokens.size();
        size_t probs_size  = generated_probs.size();

        json result;
        if (req->get_type() == REQ_CHAT_COMPLETE) {
            json content = json::array();

            for (size_t i = 0; i < probs_size; i++) {
                const llama_token id    = processed_tokens[tokens_size - probs_size + i];
                const std::string token = tokens_to_output_formatted_string(llm_ctx, id);
                float             token_logprob =
                    generated_probs[i] == 0.0f ? std::numeric_limits<float>::lowest() : std::log(generated_probs[i]);
                std::vector<unsigned char> token_bytes(token.begin(), token.end());
                json                       token_top_logprobs = json::array();
                for (const auto & tp : generated_top_probs[i]) {
                    const llama_token tp_id    = tp.first;
                    const std::string tp_token = tokens_to_output_formatted_string(llm_ctx, tp_id);
                    float             tp_token_logprob =
                        tp.second == 0.0f ? std::numeric_limits<float>::lowest() : std::log(tp.second);
                    std::vector<unsigned char> tp_token_bytes(tp_token.begin(), tp_token.end());
                    token_top_logprobs.push_back(json{
                        { "token",   tp_token         },
                        { "logprob", tp_token_logprob },
                        { "bytes",   tp_token_bytes   },
                    });
                }

                content.push_back(json{
                    { "token",        token              },
                    { "logprob",      token_logprob      },
                    { "bytes",        token_bytes        },
                    { "top_logprobs", token_top_logprobs },
                });
            }

            result = {
                { "content", content },
            };
        } else {
            json token_logprobs = json::array();
            json tokens         = json::array();
            json top_logprobs   = json::array();

            for (size_t i = 0; i < probs_size; i++) {
                const llama_token id    = processed_tokens[tokens_size - probs_size + i];
                const std::string token = tokens_to_output_formatted_string(llm_ctx, id);
                float             token_logprob =
                    generated_probs[i] == 0.0f ? std::numeric_limits<float>::lowest() : std::log(generated_probs[i]);
                json token_top_logprobs;
                for (const auto & tp : generated_top_probs[i]) {
                    const llama_token tp_id    = tp.first;
                    const std::string tp_token = tokens_to_output_formatted_string(llm_ctx, tp_id);
                    float             tp_token_logprob =
                        tp.second == 0.0f ? std::numeric_limits<float>::lowest() : std::log(tp.second);
                    token_top_logprobs[tp_token] = tp_token_logprob;
                }

                tokens.push_back(token);
                token_logprobs.push_back(token_logprob);
                top_logprobs.push_back(token_top_logprobs);
            }

            result = {
                { "tokens",         tokens         },
                { "token_logprobs", token_logprobs },
                { "top_logprobs",   top_logprobs   },
            };
        }

        // clean
        if (req->stream) {
            generated_probs.clear();
            generated_top_probs.clear();
        }

        return result;
    }

    json to_json(const llama_context * llm_ctx) {
        bool stop          = !generated_finish_reason.empty();
        bool include_usage = stop && json_value(req->stream_options, "include_usage", true);
        bool is_chat       = req->get_type() == REQ_CHAT_COMPLETE;

        json resp = {
            { "id",      cmpl_id            },
            { "created", std::time(nullptr) },
            { "model",   req->model         },
            { "usage",   json()             },
        };

        if (is_chat) {
            if (req->stream) {
                resp["object"] = "chat.completion.chunk";
            } else {
                resp["object"] = "chat.completion";
            }
        } else {
            resp["object"] = "text_completion";
        }

        if (include_usage) {
            resp["usage"] = {
                { "prompt_tokens", n_prefilled },
                { "completion_tokens", n_decoded },
                { "total_tokens", n_prefilled + n_decoded },
                {
                 "prompt_tokens_details", {
                        { "cached_tokens", n_prefilled_cached },
                    }, },
                {
                 "completion_tokens_details", {
                        { "reasoning_tokens", n_reasoning },
                        { "accepted_prediction_tokens", n_drafted_accepted },
                        { "rejected_prediction_tokens", n_drafted - n_drafted_accepted },
                    }, },
                // additional details
                { "time_to_first_token_ms", t_prefilled },
                { "time_per_output_token_ms", t_decoded / n_decoded },
                { "prompt_tokens_per_second", p_prefilled_tps },
                { "tokens_per_second", p_decoded_tps },
                { "draft_tokens", n_drafted },
                { "draft_tokens_acceptance", p_drafted_apt },
            };
        }

        std::string_view generated_text_send = std::string_view(generated_text).substr(0, generated_text_keep_pos);

        json choices = json::array();
        {
            json choice = {
                { "index",         0                                             },
                { "finish_reason", stop ? json(generated_finish_reason) : json() },
            };
            if (is_chat) {
                json delta_message = json{
                    { "content", generated_text_send },
                };
                if (!generated_tool_calls.empty()) {
                    if (generated_text_send.empty()) {
                        delta_message["content"] = json();
                    }
                    delta_message["tool_calls"] = generated_tool_calls;
                }
                if (req->stream) {
                    choice["delta"] = delta_message;
                } else {
                    delta_message["role"] = "assistant";
                    choice["message"]     = delta_message;
                }
            } else {
                choice["text"] = generated_text_send;
            };
            if (req->logprobs >= 0) {
                choice["logprobs"] = get_probs_json(llm_ctx);
            }
            choices.push_back(std::move(choice));
        }
        resp["choices"] = choices;

        // clean
        if (req->stream) {
            if (generated_text_keep_pos == std::string::npos) {
                generated_text.clear();
            } else if (generated_text_keep_pos != 0) {
                generated_text = generated_text.substr(generated_text_send.length());
            }
            generated_tool_calls.clear();
        }
        generated_text_keep_pos = std::string::npos;

        return resp;
    }
};

struct embeddings_task : btask {
    explicit embeddings_task(int32_t id, const std::function<bool()> & is_connection_closed) :
        btask(id, TASK_EMBEDDINGS, is_connection_closed) {}

    [[nodiscard]] std::string get_r_id() const override { return req->get_id(); }

    [[nodiscard]] req_type get_r_type() const override { return req->get_type(); }

    // input
    std::vector<llama_tokens> tokenized_inputs;
    std::unique_ptr<breq>     req;
    int32_t                   n_prefilling_request = 0;

    // process
    int32_t                         i_input_prefilled = 0;  // indicate the index which input has been prefilled
    int32_t                         i_batch_seq_end   = 0;  // indicate the index of the batch seq end
    std::vector<std::vector<float>> embeds;

    //// prefill
    int32_t n_prefilled     = 0;  // indicate how many tokens have been prefilled
    int64_t t_start_prefill = 0;  // indicate the time when prefilling starts
    double  t_prefilled     = 0;  // indicate the time(ms) spent on prefilling
    double  p_prefilled_tps = 0;
    int32_t n_min_prefilled = 0;
    int32_t n_max_prefilled = 0;

    json to_json() {
        auto n_seq = int32_t(embeds.size());

        json usage = {
            { "prompt_tokens",            n_prefilled     },
            { "total_tokens",             n_prefilled     },
            { "prompt_tokens_per_second", p_prefilled_tps },
            // additional details
            { "min_prompt_tokens",        n_min_prefilled },
            { "max_prompt_tokens",        n_max_prefilled },
        };

        if (req->get_type() == REQ_EMBED) {
            auto * dreq = dynamic_cast<embed_req *>(req.get());
            json   data = json::array();
            for (int32_t seq = 0; seq < n_seq; seq++) {
                json item = {
                    { "index",  seq         },
                    { "object", "embedding" },
                };
                if (dreq->encoding_format != "base64") {
                    item["embedding"] = embeds[seq];
                } else {
                    item["embedding"] =
                        encode_base64(reinterpret_cast<const unsigned char *>(embeds[seq].data()), embeds[seq].size());
                }
                data.push_back(item);
            }
            json resp = {
                { "created", std::time(nullptr) },
                { "model",   dreq->model        },
                { "object",  "list"             },
                { "data",    data               },
                { "usage",   usage              },
            };
            return resp;
        }

        auto * dreq    = dynamic_cast<rerank_req *>(req.get());
        json   results = json::array();
        for (int32_t seq = 0; seq < n_seq - (dreq->normalize ? 2 : 0); seq++) {
            json item = {
                { "index",           seq            },
                { "relevance_score", embeds[seq][0] },
            };
            if (dreq->return_documents) {
                item["document"] = dreq->documents[seq].is_string() ?
                                       json{
                                           { "text", dreq->documents[seq] }
                } :
                                       dreq->documents[seq];
            }
            results.push_back(item);
        }
        sort_rerank_results(results, 0, n_seq - 1 - (dreq->normalize ? 2 : 0));
        if (dreq->normalize) {
            float scr_max = std::max(embeds[n_seq - 2][0], results[0].at("relevance_score").get<float>());
            float scr_min = std::min(embeds[n_seq - 1][0], results[n_seq - 3].at("relevance_score").get<float>());
            float scr_dst = scr_max - scr_min;
            float a = 0.001, b = 0.998;
            if (scr_dst < 1e-6 || dreq->query.get<std::string>() ==
                                      dreq->documents[json_value(results[0], "index", 0)].get<std::string>()) {
                scr_dst = scr_max;
                scr_min = 0.0f;
                a = 0, b = 1;
            }
            for (int32_t seq = 0; seq < n_seq - 2 && seq < dreq->top_n; seq++) {
                auto scr                        = results[seq].at("relevance_score").get<float>();
                scr                             = a + (scr - scr_min) * b / scr_dst;
                results[seq]["relevance_score"] = scr;
            }
        }
        results.erase(results.begin() + dreq->top_n, results.end());
        json resp = {
            { "model",   dreq->model },
            { "results", results     },
            { "usage",   usage       },
        };
        return resp;
    }
};

struct images_task : btask {
    explicit images_task(int32_t id, const std::function<bool()> & is_connection_closed) :
        btask(id, TASK_IMAGES, is_connection_closed) {}

    [[nodiscard]] std::string get_r_id() const override { return req->get_id(); }

    [[nodiscard]] req_type get_r_type() const override { return req->get_type(); }

    [[nodiscard]] std::vector<common_adapter_lora_info> & get_lora_adapters() override { return req->lora_adapters; }

    [[nodiscard]] bool is_stream() const override { return req->stream; }

    [[nodiscard]] json & get_stream_options() const override { return req->stream_options; }

    // input
    std::unique_ptr<image_req> req;

    // process
    std::vector<std::unique_ptr<stablediffusion_sampling_stream>> streams;
    std::vector<std::string>                                      b64_jsons;
    std::vector<int32_t>                                          progressed_steps;
    std::vector<int32_t>                                          progress_steps;

    //// forward
    int32_t n_forward_steps = 0;  // indicate how many forwarded steps have been called
    int64_t t_start_forward = 0;  // indicate the time when forwarding starts
    double  t_forwarded     = 0;  // indicate the time(ms) spent on forwarding
    double  p_forwarded_sps = 0;

    //// reverse
    int32_t n_reverse_steps = 0;  // indicate how many reversed steps have been called
    int64_t t_start_reverse = 0;  // indicate the time when reversing starts
    double  t_reversed      = 0;  // indicate the time(ms) spent on reversing
    double  p_reversed_sps  = 0;

    json to_json(const int32_t seq) {
        bool all_seqs      = seq < 0;
        bool stop          = all_seqs || progress_steps[seq] == progressed_steps[seq];
        bool include_usage = stop && json_value(req->stream_options, "include_usage", true) &&
                             (!req->stream || seq == int32_t(progressed_steps.size() - 1));

        json resp = {
            { "created", std::time(nullptr) },
            { "model",   req->model         },
            { "object",  "list"             },
            { "usage",   json()             },
        };
        if (include_usage) {
            resp["usage"] = {
                { "time_to_process_ms",     t_forwarded                  },
                { "time_per_generation_ms", t_reversed / n_reverse_steps },
                { "generation_per_second",  p_reversed_sps               },
            };
        }
        json data = json::array();
        for (int32_t idx = 0; idx < int32_t(b64_jsons.size()); idx++) {
            if (!all_seqs && idx != seq) {
                continue;
            }
            json item = {
                { "index",            idx                                                                          },
                { "object",           "image"                                                                      },
                { "progressed_steps", progressed_steps[idx]                                                        },
                { "progress_steps",   progress_steps[idx]                                                          },
                { "progress",         stop ? 100 : float(progressed_steps[idx]) / float(progress_steps[idx]) * 100 },
                { "finish_reason",    stop ? "stop" : json()                                                       },
                { "b64_json",         std::move(b64_jsons[idx])                                                    },
            };
            data.push_back(std::move(item));
            if (!all_seqs) {
                break;
            }
        }
        resp["data"] = data;
        return resp;
    }
};

struct btask_result {
    explicit btask_result(httplib::StatusCode && status, json && result) : status(status), result(std::move(result)) {}

    httplib::StatusCode status = httplib::Continue_100;  // 100 continue (streaming), others finished.
    json                result;
};

// implementations // httpserver

struct httpserver_metrics {
    /* STABLE DIFFUSION */

    std::atomic<double>   t_image_forwarded_total      = 0;
    std::atomic<uint64_t> n_mtmd_steps_forwarded_total = 0;
    std::atomic<double>   t_image_reversed_total       = 0;
    std::atomic<uint64_t> n_mtmd_steps_reversed_total  = 0;

    /* LLAMA */

    std::atomic<double>   t_tokens_prefilled_total        = 0;
    std::atomic<uint64_t> n_tokens_prefilled_total        = 0;
    std::atomic<double>   t_tokens_decoded_total          = 0;
    std::atomic<uint64_t> n_tokens_decoded_total          = 0;
    std::atomic<uint64_t> n_tokens_drafted_total          = 0;
    std::atomic<uint64_t> n_tokens_drafted_accepted_total = 0;

    void on_mtmd_forwarded(double t, uint64_t n_steps) {
        t_image_forwarded_total      = t_image_forwarded_total + t;
        n_mtmd_steps_forwarded_total = n_mtmd_steps_forwarded_total + n_steps;
    }

    void on_mtmd_reversed(double t, uint64_t n_steps) {
        t_image_reversed_total      = t_image_reversed_total + t;
        n_mtmd_steps_reversed_total = n_mtmd_steps_reversed_total + n_steps;
    }

    void on_tokens_prefilled(double t, uint64_t n) {
        t_tokens_prefilled_total = t_tokens_prefilled_total + t;
        n_tokens_prefilled_total = n_tokens_prefilled_total + n;
    }

    void on_tokens_decoded(double t, uint64_t n, uint64_t n_drafted, uint64_t n_drafted_accepted) {
        t_tokens_decoded_total          = t_tokens_decoded_total + t;
        n_tokens_decoded_total          = n_tokens_decoded_total + n;
        n_tokens_drafted_total          = n_tokens_drafted_total + n_drafted;
        n_tokens_drafted_accepted_total = n_tokens_drafted_accepted_total + n_drafted_accepted;
    }
};

std::function<void(int)> httpserver_shutdown_handler;
std::atomic_flag         httpserver_is_terminating = ATOMIC_FLAG_INIT;

inline void httpserver_signal_handler(int32_t signal) {
    if (httpserver_is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C
        // twice this is for better developer experience, we can remove when the
        // server is stable enough
        SRV_WRN("%s", "received second interrupt, terminating immediately\n");
        exit(1);
    }
    httpserver_shutdown_handler(signal);
}

struct httpserver {
    explicit httpserver(httpserver_params & params) : params(params) {
        process_tasks =
            std::make_unique<BlockingConcurrentQueue<std::unique_ptr<btask>>>(params.llm_params.n_threads_http);
        process_task_results.resize(params.llm_params.n_threads_http);
        for (int32_t i = 0; i < params.llm_params.n_threads_http; i++) {
            process_task_results[i] = std::make_unique<BlockingReaderWriterQueue<std::unique_ptr<btask_result>>>();
        }

        llama_numa_init(params.llm_params.numa);
        llama_backend_init();
    }

    ~httpserver() {
        llama_batch_free(batch_text);
        llama_batch_free(batch_text_temp);
        if (llm_ctx != nullptr) {
            llama_detach_threadpool(llm_ctx);
        }
        if (llm_ctx_clip_v != nullptr) {
            clip_free(llm_ctx_clip_v);
        }
        if (llm_ctx_clip_a != nullptr) {
            clip_free(llm_ctx_clip_a);
        }

        if (llm_ctx_draft != nullptr) {
            llama_batch_free(batch_text_draft);
            llama_detach_threadpool(llm_ctx_draft);
        }

        if (threadpool != nullptr) {
            ggml_threadpool_free(threadpool);
            threadpool = nullptr;
        }
        if (threadpool_batch != nullptr) {
            ggml_threadpool_free(threadpool_batch);
            threadpool_batch = nullptr;
        }

        llama_backend_free();
    }

    bool load() {
        SRV_INF("loading model '%s'\n", params.llm_params.model.path.c_str());

        /* STABLE DIFFUSION */

        if (params.endpoint_images) {
            sd_init = common_sd_init_from_params(params.sd_params);
            sd_ctx  = sd_init.context.get();
            if (sd_ctx == nullptr) {
                SRV_ERR("failed to load stable diffusion model, '%s'\n", params.sd_params.model.c_str());
                return false;
            }

            if (params.sd_params.model_alias.empty()) {
                if (params.sd_params.model.find_last_of('/') != std::string::npos) {
                    params.sd_params.model_alias =
                        params.sd_params.model.substr(params.sd_params.model.find_last_of('/') + 1);
                } else if (params.sd_params.model.find_last_of('\\') != std::string::npos) {
                    params.sd_params.model_alias =
                        params.sd_params.model.substr(params.sd_params.model.find_last_of('\\') + 1);
                } else {
                    params.sd_params.model_alias = params.sd_params.model;
                }
            }
            if (params.sd_params.sampling.strength <= 0.0f) {
                params.sd_params.sampling.strength = sd_ctx->get_default_strength();
            }
            if (params.sd_params.sampling.sample_method >= N_SAMPLE_METHODS) {
                params.sd_params.sampling.sample_method = sd_ctx->get_default_sample_method();
            }
            if (params.sd_params.sampling.sampling_steps <= 0) {
                params.sd_params.sampling.sampling_steps = sd_ctx->get_default_sampling_steps();
            }
            if (params.sd_params.sampling.cfg_scale <= 0.0f) {
                params.sd_params.sampling.cfg_scale = sd_ctx->get_default_cfg_scale();
            }

            SRV_INF(
                "seed: %u, flash attn: %s, guidance: %.2f, strength: %.2f, sample method: %s, sampling steps: %d, cfg "
                "scale: %.2f, slg scale: %.2f, schedule method: %s\n",
                params.sd_params.seed, params.sd_params.flash_attn ? "true" : "false",
                params.sd_params.sampling.guidance, params.sd_params.sampling.strength,
                sd_sample_method_to_argument(params.sd_params.sampling.sample_method),
                params.sd_params.sampling.sampling_steps, params.sd_params.sampling.cfg_scale,
                params.sd_params.sampling.slg_scale,
                sd_schedule_to_argument(params.sd_params.sampling.schedule_method));

            return true;
        }

        /* LLAMA */

        if (params.llm_params.model_alias.empty()) {
            if (params.llm_params.model.path.find_last_of('/') != std::string::npos) {
                params.llm_params.model_alias =
                    params.llm_params.model.path.substr(params.llm_params.model.path.find_last_of('/') + 1);
            } else if (params.llm_params.model.path.find_last_of('\\') != std::string::npos) {
                params.llm_params.model_alias =
                    params.llm_params.model.path.substr(params.llm_params.model.path.find_last_of('\\') + 1);
            } else {
                params.llm_params.model_alias = params.llm_params.model.path;
            }
        }

        common_params llm_params = params.llm_params;
        llm_params.n_parallel    = params.llm_params.n_threads_http;
        llm_init                 = common_init_from_params(llm_params);
        llm_model                = llm_init.model.get();
        llm_ctx                  = llm_init.context.get();
        if (llm_model == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params.llm_params.model.path.c_str());
            return false;
        }
        llm_vocab            = llama_model_get_vocab(llm_model);
        llm_ctx_size         = int32_t(llama_n_ctx(llm_ctx));
        llm_ctx_embed_size   = llama_model_n_embd(llm_model);
        llm_kv_cache_limit   = llm_ctx_size - 1;
        llm_kv_cache_shift   = llama_memory_can_shift(llama_get_memory(llm_ctx));
        // NB(thxCode): llama_causal_attn is a patch.
        llm_model_casual     = llama_causal_attn(llm_ctx);
        llm_model_rope_mrope = llama_model_rope_type(llm_model) == LLAMA_ROPE_TYPE_MROPE;
        batch_view_max       = int32_t(llama_n_batch(llm_ctx));
        batch_text           = llama_batch_init(llm_ctx_size, 0, 1);
        batch_text_temp      = llama_batch_init(llm_ctx_size, 0, 1);

        // load multimodal projection model
        if (!params.llm_params.mmproj.path.empty()) {
            SRV_INF("loading multimodal projection model '%s'\n", params.llm_params.mmproj.path.c_str());

            if (params.llm_params.n_ctx < 2048) {
                SRV_WRN("%s", "n_ctx is too small for multimodal projection, setting to 2048\n");
                params.llm_params.n_ctx = 2048;
            }
            // NB(thxCode): clip_context_params is a patch.
            clip_context_params llm_params_clip{
                /* use_gpu */ params.llm_params.n_gpu_layers != 0,
                /* verbosity */ common_log_verbosity_thold > 3 ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_INFO,
                /* max_image_size */ params.max_image_size,
            };
            llm_init_clip = clip_init(params.llm_params.mmproj.path.c_str(), llm_params_clip);
            if (llm_init_clip.ctx_a == nullptr && llm_init_clip.ctx_v == nullptr) {
                SRV_ERR("failed to load multimodal project model, '%s'\n", params.llm_params.mmproj.path.c_str());
                return false;
            }
            llm_ctx_clip_v = llm_init_clip.ctx_v;
            llm_ctx_clip_a = llm_init_clip.ctx_a;

            // check multimodal projection model compatibility
            if (llm_ctx_clip_v != nullptr) {
                const int32_t n_embd_clip = clip_n_mmproj_embd(llm_ctx_clip_v);
                const int32_t n_embd      = llama_model_n_embd(llm_model);
                if (n_embd_clip != n_embd) {
                    SRV_ERR(
                        "multimodal projector embedding length is not equal to the model, n_embd_clip = %d, n_embd = "
                        "%d\n",
                        n_embd_clip, n_embd);
                    return false;
                }
            }
            if (llm_ctx_clip_a != nullptr) {
                const int32_t n_embd_clip = clip_n_mmproj_embd(llm_ctx_clip_a);
                const int32_t n_embd      = llama_model_n_embd(llm_model);
                if (n_embd_clip != n_embd) {
                    SRV_ERR(
                        "multimodal projector audio embedding length is not equal to the model, n_embd_clip = %d, "
                        "n_embd = %d\n",
                        n_embd_clip, n_embd);
                    return false;
                }
            }
        }

        // load the draft model if needed
        if (!params.llm_params.speculative.model.path.empty() && params.llm_params.speculative.n_max > 0) {
            if (!llm_kv_cache_shift) {
                SRV_ERR("%s", "draft model speculative decoding is not supported for non-shifting models\n");
                return false;
            }

            SRV_INF("loading draft model '%s'\n", params.llm_params.speculative.model.path.c_str());

            common_params llm_params_draft   = params.llm_params;
            llm_params_draft.n_parallel      = params.llm_params.n_threads_http;
            llm_params_draft.embedding       = false;
            llm_params_draft.model           = params.llm_params.speculative.model;
            llm_params_draft.n_gpu_layers    = params.llm_params.speculative.n_gpu_layers;
            llm_params_draft.cpuparams       = params.llm_params.speculative.cpuparams;
            llm_params_draft.cpuparams_batch = params.llm_params.speculative.cpuparams_batch;
            llm_params_draft.cache_type_k    = GGML_TYPE_F16;
            llm_params_draft.cache_type_v    = GGML_TYPE_F16;
            llm_params_draft.warmup          = false;
            llm_init_draft                   = common_init_from_params(llm_params_draft);
            llm_model_draft                  = llm_init_draft.model.get();
            llm_ctx_draft                    = llm_init_draft.context.get();
            if (llm_model_draft == nullptr) {
                SRV_ERR("failed to load draft model, '%s'\n", params.llm_params.speculative.model.path.c_str());
                return false;
            }
            llm_vocab_draft  = llama_model_get_vocab(llm_model_draft);
            batch_text_draft = llama_batch_init(int32_t(llama_n_ctx(llm_ctx_draft)), 0, 1);

            // check draft model compatibility if needed
            const bool vocab_type_draft = llama_vocab_type(llm_vocab_draft);
            const bool vocab_type       = llama_vocab_type(llm_vocab);
            if (vocab_type_draft != vocab_type) {
                SRV_ERR(
                    "draft model vocabulary type is not equal to the model, vocab_type_draft = %d, vocab_type = %d\n",
                    vocab_type_draft, vocab_type);
                return false;
            }
            if (llama_vocab_get_add_bos(llm_vocab_draft) != llama_vocab_get_add_bos(llm_vocab) ||
                llama_vocab_get_add_eos(llm_vocab_draft) != llama_vocab_get_add_eos(llm_vocab) ||
                llama_vocab_bos(llm_vocab_draft) != llama_vocab_bos(llm_vocab) ||
                llama_vocab_eos(llm_vocab_draft) != llama_vocab_eos(llm_vocab)) {
                SRV_ERR("%s", "draft model special tokens are not equal to the model\n");
                return false;
            }
        }

        // thread pool
        {
            struct ggml_threadpool_params tpp = ggml_threadpool_params_from_cpu_params(params.llm_params.cpuparams);
            threadpool                        = ggml_threadpool_new(&tpp);
            if (!threadpool) {
                SRV_ERR("threadpool create failed : n_threads %d\n", tpp.n_threads);
                return false;
            }

            struct ggml_threadpool_params tpp_batch =
                ggml_threadpool_params_from_cpu_params(params.llm_params.cpuparams_batch);
            threadpool_batch = ggml_threadpool_new(&tpp_batch);
            if (!threadpool_batch) {
                SRV_ERR("threadpool_batch create failed : n_threads %d\n", tpp_batch.n_threads);
                return false;
            }

            llama_attach_threadpool(llm_ctx, threadpool, threadpool_batch);
            if (llm_ctx_draft != nullptr) {
                llama_attach_threadpool(llm_ctx_draft, threadpool, threadpool_batch);
            }
        }

        // prompt cache
        cache_prompt = llm_model_casual && params.cache_prompt && llm_kv_cache_shift;
        if (cache_prompt) {
            cache_prompts.resize(params.llm_params.n_threads_http);
        }
        SRV_INF("prompt caching %s\n", cache_prompt ? "enabled" : (params.cache_prompt ? "unsupported" : "disabled"));

        // NB(thxCode): llama_model_arch_name is a patch.
        std::string arch_name = llama_model_arch_name(llm_model);

        need_end_eos = arch_name == "qwen3";

        if (!support_completion()) {
            return true;
        }

        // context shift
        shift_context = params.llm_params.ctx_shift && llm_kv_cache_shift;
        SRV_INF("context shifting %s\n", shift_context ? "enabled" : "disabled");

        // chat template
        {
            chat_templates = common_chat_templates_init(llm_model, params.llm_params.chat_template);

            // NB(thxCode): llama_chat_template_alias is a patch.
            std::string alias = llama_chat_template_alias(common_chat_templates_source(chat_templates.get()));

            if (params.llm_params.use_jinja) {
                // NB(thxCode): common_chat_templates_supports_tool_calls is a patch.
                support_tool_calls = common_chat_templates_supports_tool_calls(chat_templates.get());
            } else {
                bool get_token = false;
                // chatml / chatglm4
                if (alias == "chatml" || alias == "chatglm4") {
                    // <tool_call>
                    // {"name":"","arguments":{}}
                    // </tool_call>
                    get_token             = true;
                    support_tool_calls    = true;
                    tool_call_start_words = { "<tool_call>", "<tool>", "<tools>", "<function_call>" };
                    tool_call_start_trim  = true;
                    tool_call_end_words   = { "</tool_call>",   "</tool>",   "</tools>",   "</function_call>",
                                              "</tool_call>\n", "</tool>\n", "</tools>\n", "</function_call>\n" };
                    tool_call_end_trim    = true;
                }
                // mistral series
                else if (string_starts_with(alias, "mistral-")) {
                    // [TOOL_CALLS][{"name":"","arguments":{}}]
                    get_token             = true;
                    support_tool_calls    = true;
                    tool_call_start_words = { "[TOOL_CALLS]" };
                    tool_call_start_trim  = true;
                    tool_call_end_words   = { "}]", "}]\n", "}] " };
                    tool_call_end_trim    = true;
                }
                // llama3 / llama4
                else if (alias == "llama3" || alias == "llama4") {
                    // {"name":"","arguments":{}}
                    support_tool_calls    = true;
                    tool_call_start_words = { "{\"" };
                    tool_call_end_words   = { "}}", "}}\n", "}} " };
                }
                // granite
                else if (alias == "granite") {
                    // <tool_call>[{"name":"","arguments":{}}]
                    get_token             = true;
                    support_tool_calls    = true;
                    tool_call_start_words = { "<|tool_call|>", "<tool_call>" };
                    tool_call_start_trim  = true;
                    tool_call_end_words   = { "}]", "}]\n", "}] " };
                }
                // deepseek3
                else if (alias == "deepseek3") {
                    // <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>tool_name
                    //```json
                    //{"arg1": "some_value"}
                    //```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
                    get_token             = true;
                    support_tool_calls    = true;
                    tool_call_start_words = { "<｜tool▁calls▁begin｜>", "<｜tool▁call▁begin｜>",
                                              "<｜tool calls begin｜>", "<｜tool\\\\_calls\\\\_begin｜>",
                                              "<｜tool▁calls｜>" };
                    tool_call_start_trim  = true;
                    tool_call_end_words   = { "<｜tool▁call▁end｜>", "<｜tool▁calls▁end｜>", "<｜tool▁call▁end｜>\n",
                                              "<｜tool▁calls▁end｜>\n" };
                    tool_call_end_trim    = true;
                    tool_call_format      = "function";
                }
                if (get_token) {
                    for (const std::string & word : tool_call_start_words) {
                        llama_tokens ids = common_tokenize(llm_vocab, word, false, true);
                        if (ids.size() == 1) {
                            tool_call_start_tokens.push_back(ids[0]);
                        }
                    }
                    for (const std::string & word : tool_call_end_words) {
                        llama_tokens ids = common_tokenize(llm_vocab, word, false, true);
                        if (ids.size() == 1) {
                            tool_call_end_tokens.push_back(ids[0]);
                        }
                    }
                }
                if (!tool_call_start_words.empty()) {
                    for (const std::string & word : tool_call_start_words) {
                        tool_call_start_words_longest_length =
                            std::max(tool_call_start_words_longest_length, word.length());
                    }
                    tool_call_start_words_longest_length =
                        tool_call_start_words_longest_length +
                        int32_t(std::ceil(float(tool_call_start_words_longest_length) / 3.0));
                }
            }

            {
                if (alias == "deepseek3" || string_starts_with(arch_name, "qwen3")) {
                    llama_tokens ids = common_tokenize(llm_vocab, "<think>", false, true);
                    if (ids.size() == 1) {
                        reasoning_start_token = ids[0];
                    }
                    ids = common_tokenize(llm_vocab, "</think>", false, true);
                    if (ids.size() == 1) {
                        reasoning_end_token = ids[0];
                    }
                } else if (alias == "command-r") {
                    llama_tokens ids = common_tokenize(llm_vocab, "<|START_THINKING|>", false, true);
                    if (ids.size() == 1) {
                        reasoning_start_token = ids[0];
                    }
                    ids = common_tokenize(llm_vocab, "<|END_THINKING|>", false, true);
                    if (ids.size() == 1) {
                        reasoning_end_token = ids[0];
                    }
                }
                support_reasoning = params.llm_params.reasoning_budget != 0 &&
                                    reasoning_start_token != LLAMA_TOKEN_NULL &&
                                    reasoning_end_token != LLAMA_TOKEN_NULL;
                if (!support_reasoning) {
                    reasoning_start_token = LLAMA_TOKEN_NULL;
                    reasoning_end_token   = LLAMA_TOKEN_NULL;
                }
            }

            std::string prompt;
            {
                common_chat_templates_inputs inputs;
                inputs.messages = std::vector<common_chat_msg>({
                    { "system",    "You are a helpful assistant.",         {}, {}, "", "", "" },
                    { "user",      "Hello.",                               {}, {}, "", "", "" },
                    { "assistant", "Hi! How can I help you today?",        {}, {}, "", "", "" },
                    { "user",      "What is the weather like in Beijing?", {}, {}, "", "", "" },
                });
                if (support_tool_calls) {
                    // clang-format off
                    inputs.messages.push_back({ "assistant", "", {}, { { "get_weather", R"({"location":"Beijing"})", "123456789" } }, "", "", "" });
                    inputs.messages.push_back({ "tool", R"({"weather":"Sunny"})", {}, {}, "", "", "123456789" });
                    inputs.messages.push_back({ "assistant", "The weather is Sunny.", {}, {}, "", "", "123456789" });
                    inputs.messages.push_back({ "user", "What is the temperature in Beijing?", {}, {}, "", "", "" });
                    inputs.tools = std::vector<common_chat_tool>({
                        { "get_weather",     "",                                                   R"({"type":"object","properties":{"location":{"type":"string"}}})" },
                        { "get_temperature", "Return the temperature according to the location.",  R"({"type":"object","properties":{"location":{"type":"string"}}})" },
                    });
                    // clang-format on
                }
                inputs.tool_choice           = COMMON_CHAT_TOOL_CHOICE_NONE;
                inputs.add_generation_prompt = true;
                inputs.use_jinja             = params.llm_params.use_jinja;
                // NB(thxCode): common_chat_templates_apply2 is a patch.
                common_chat_params example   = common_chat_templates_apply2(llm_model, chat_templates.get(), inputs);
                prompt                       = example.prompt;
            }

            SRV_INF(
                "chat template, alias: %s, built-in: %s, jinja rendering: %s, tool call: %s, reasoning: %s, "
                "example:\n%s\n",
                alias.c_str(),
                // built-in
                params.llm_params.chat_template.empty() || !params.llm_params.use_jinja ? "true" : "false",
                // jinja rendering
                params.llm_params.use_jinja ? "enabled" : "disabled",
                // tool call
                support_tool_calls ? "supported" : "unsupported",
                // reasoning
                support_reasoning                       ? "supported" :
                params.llm_params.reasoning_budget != 0 ? "unsupported" :
                                                          "disabled",
                prompt.c_str());
            if (support_tool_calls) {
                SRV_INF(
                    "tool call trigger, "
                    "start: %s, end: %s\n",
                    // start
                    tool_call_start_tokens.empty() ? "words" : "tokens",
                    // end
                    tool_call_end_tokens.empty() ? "words" : "tokens");
            }
        }

        // sample tokens per second
        if (params.n_tps < 0) {
            SRV_INF("%s", "sampling tokens per second, this will take some time...\n");
            const int32_t    n_check             = std::min(llm_ctx_size, params.llm_params.n_ubatch);
            llama_tokens     check_prompt_tokens = { llama_vocab_bos(llm_vocab) };
            common_sampler * check_smpl          = common_sampler_init(llm_model, params.llm_params.sampling);
            int64_t          t_start_decode      = ggml_time_us();
            int32_t          n_check_decoded     = 0;
            while (true) {
                auto i = int32_t(check_prompt_tokens.size());
                if (i >= n_check) {
                    break;
                }
                if (llama_decode(llm_ctx, llama_batch_get_one(&check_prompt_tokens[i - 1], 1))) {
                    break;
                }
                n_check_decoded++;
                const int32_t id = common_sampler_sample(check_smpl, llm_ctx, 0);
                if (llama_vocab_is_eog(llm_vocab, id)) {
                    break;
                }
                common_sampler_accept(check_smpl, id, false);
                check_prompt_tokens.push_back(id);
            }
            params.n_tps = ceil(1.e3 / (double(ggml_time_us() - t_start_decode) / 1.e3) * n_check_decoded);
            common_sampler_free(check_smpl);
            llama_memory_clear(llama_get_memory(llm_ctx), true);
            llama_synchronize(llm_ctx);
            llama_perf_context_reset(llm_ctx);
            SRV_INF("sampled tokens per second, tps = %d\n", params.n_tps);
        }

        return true;
    }

    int32_t start() {
        SRV_INF("%s", "starting\n");

        std::shared_ptr<httplib::Server>     server = std::make_shared<httplib::Server>();
        std::shared_ptr<httplib::ThreadPool> thread_pool =
            std::make_shared<httplib::ThreadPool>(params.llm_params.n_threads_http + 1);

        // register routes
#define HANDLER(handler)                                                  \
    [&](const httplib::Request & request, httplib::Response & response) { \
        handler(request, response);                                       \
    }
        server->Get("/health", HANDLER(handle_health));
        if (params.llm_params.endpoint_metrics) {
            server->Get("/metrics", HANDLER(handle_metrics));
        }
        if (!params.llm_params.lora_adapters.empty()) {
            server->Get("/lora-adapters", HANDLER(handle_lora_adapters));
        }
        server->Get("/v1/models", HANDLER(handle_models));
        /* STABLE DIFFUSION */
        if (support_image()) {
            server->Post("/v1/images/:category", HANDLER(handle_images));
        }
        /* LLAMA */
        else {
            server->Post("/tokenize", HANDLER(handle_tokenize));
            server->Post("/detokenize", HANDLER(handle_detokenize));
            if (support_completion()) {
                server->Post("/v1/completions", HANDLER(handle_legacy_completions));
                server->Post("/v1/chat/completions", HANDLER(handle_chat_completions));
            }
            if (support_embedding()) {
                server->Post("/v1/embeddings", HANDLER(handle_embeddings));
            }
            if (support_reranking()) {
                server->Post("/v1/rerank", HANDLER(handle_rerank));
            }
        }
#undef _HANDLER

        // register reconcile loop
        thread_pool->enqueue([&]() {
            server->wait_until_ready();
            if (!server->is_running()) {
                SRV_FUNC_ERR("start", "%s", "server is not ready\n");
                server->stop();
                return;
            }
            SRV_FUNC_INF("start", "%s", "server is ready\n");
            reconcile_loop(server);
        });

        // register shutdown handler
        httpserver_shutdown_handler = [&](int) {
            SRV_FUNC_INF("start", "%s", "server is stopping\n");
            server->stop();
        };
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
        struct sigaction sigint_action{};

        sigint_action.sa_handler = httpserver_signal_handler;
        sigemptyset(&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, nullptr);
        sigaction(SIGTERM, &sigint_action, nullptr);
#elif defined(_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (httpserver_signal_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

        // configure server
        server->set_read_timeout(params.llm_params.timeout_read);
        server->set_write_timeout(params.llm_params.timeout_write);
        server->set_payload_max_length(CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH);
        server->set_idle_interval(params.conn_idle);
        server->set_keep_alive_timeout(params.conn_keepalive);
        server->set_exception_handler(
            [&](const httplib::Request & request, httplib::Response & response, const std::exception_ptr & ep) {
                httplib::StatusCode code = httplib::InternalServerError_500;
                std::string         message;
                try {
                    std::rethrow_exception(ep);
                } catch (const std::invalid_argument & re) {
                    code    = httplib::BadRequest_400;
                    message = re.what();
                } catch (const std::exception & e) {
                    message = e.what();
                } catch (...) {
                    message = "Unknown error occurred";
                }
                send_json(request, response, code, message);
                // logging
                std::string rid = response.get_header_value(HEADER_X_REQUEST_ID);
                uint64_t    rct = ggml_time_us() - response.get_header_value_u64(HEADER_X_REQUEST_ACCEPTED_AT);
                SRV_FUNC_INF("exception", "rid %s | %4s %s %s:%d | status %d | cost %.2f%s | %s\n", rid.c_str(),
                             request.method.c_str(), request.path.c_str(), request.remote_addr.c_str(),
                             request.remote_port, response.status, double(rct) / (rct > 1.e6 ? 1.e6 : 1.e3),
                             rct > 1.e6 ? "s" : "ms", message.c_str());
            });
        server->set_logger([&](const httplib::Request & request, const httplib::Response & response) {
            // logging
            if (request.path != "/health") {
                std::string rid    = response.get_header_value(HEADER_X_REQUEST_ID);
                uint64_t    rct    = ggml_time_us() - response.get_header_value_u64(HEADER_X_REQUEST_ACCEPTED_AT);
                int32_t     status = response.status;
                if (request.is_connection_closed()) {
                    status = httplib::RequestTimeout_408;
                }
                if (status >= httplib::BadRequest_400) {
                    SRV_FUNC_ERR(" response", "rid %s | %4s %s %s:%d | status %d | cost %.2f%s | %s\n", rid.c_str(),
                                 request.method.c_str(), request.path.c_str(), request.remote_addr.c_str(),
                                 request.remote_port, status, double(rct) / (rct > 1.e6 ? 1.e6 : 1.e3),
                                 rct > 1.e6 ? "s" : "ms", request.is_connection_closed() ? "closed" : "opened");
                } else {
                    SRV_FUNC_INF(" response", "rid %s | %4s %s %s:%d | status %d | cost %.2f%s | %s\n", rid.c_str(),
                                 request.method.c_str(), request.path.c_str(), request.remote_addr.c_str(),
                                 request.remote_port, status, double(rct) / (rct > 1.e6 ? 1.e6 : 1.e3),
                                 rct > 1.e6 ? "s" : "ms", request.is_connection_closed() ? "closed" : "opened");
                }
            }
        });
        server->set_pre_routing_handler([&](const httplib::Request & request, httplib::Response & response) {
            response.set_header(HEADER_SERVER, "llama-box/" + std::string(LLAMA_BOX_BUILD_VERSION));
            response.set_header("Access-Control-Allow-Origin", request.get_header_value("Origin", "*"));
            if (request.method == "OPTIONS") {
                response.set_header("Access-Control-Allow-Credentials", "true");
                response.set_header("Access-Control-Allow-Methods", "OPTIONS, HEAD, GET, POST, PUT, DELETE, PATCH");
                response.set_header("Access-Control-Allow-Headers", request.get_header_value("Content-Type", "*"));
                return httplib::Server::HandlerResponse::Handled;
            }
            // logging
            if (request.path != "/health") {
                std::string now = std::to_string(ggml_time_us());
                std::string rid = request.get_header_value(HEADER_X_REQUEST_ID, now.c_str());
                response.set_header(HEADER_X_REQUEST_ID, rid);
                response.set_header(HEADER_X_REQUEST_ACCEPTED_AT, now);
                if (request.path != "/health") {
                    SRV_FUNC_INF("  request", "rid %s | %4s %s %s:%d\n", rid.c_str(), request.method.c_str(),
                                 request.path.c_str(), request.remote_addr.c_str(), request.remote_port);
                }
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
        server->new_task_queue = [&thread_pool] {
            return thread_pool.get();
        };

        // listening on socket
        if (string_ends_with(std::string(params.llm_params.hostname), ".sock")) {
            SRV_INF("listening sock = %s\n", params.llm_params.hostname.c_str());
            server->set_address_family(AF_UNIX);
            server->bind_to_port(params.llm_params.hostname, 1);
            return server->listen_after_bind();
        }

        // listening on port
        SRV_INF("listening host = %s, port = %d\n", params.llm_params.hostname.c_str(), params.llm_params.port);
        return server->listen(params.llm_params.hostname, params.llm_params.port);
    }

  private:
    //
    // Attributes
    //

    httpserver_params                                                                      params;
    httpserver_metrics                                                                     metrics;
    std::unique_ptr<BlockingConcurrentQueue<std::unique_ptr<btask>>>                       process_tasks;
    std::vector<std::unique_ptr<BlockingReaderWriterQueue<std::unique_ptr<btask_result>>>> process_task_results;

    // lora
    std::vector<common_adapter_lora_info> lora_adapters;

    /* STABLE DIFFUSION */

    // model
    common_sd_init_result     sd_init;
    stablediffusion_context * sd_ctx = nullptr;

    /* LLAMA */

    // model
    common_init_result  llm_init;
    llama_model *       llm_model             = nullptr;
    llama_context *     llm_ctx               = nullptr;
    const llama_vocab * llm_vocab             = nullptr;
    int32_t             llm_ctx_size          = 0;
    int32_t             llm_ctx_embed_size    = 0;
    int32_t             llm_kv_cache_used     = 0;  // include llm_kv_cache_inactive
    int32_t             llm_kv_cache_inactive = 0;
    int32_t             llm_kv_cache_limit    = 0;
    bool                llm_kv_cache_shift    = false;
    bool                llm_model_casual      = true;
    bool                llm_model_rope_mrope  = false;
    int32_t             batch_view_max        = 0;
    llama_batch         batch_text            = {};
    llama_batch         batch_text_temp       = {};

    // model addition

    struct cache_prompt_entry {
        llama_tokens tokens;
        bool         used        = false;
        llama_pos    pos         = 0;  // position
        llama_pos    pos_discard = 0;  // count the discard position
    };

    bool                            cache_prompt  = false;
    bool                            shift_context = false;
    common_chat_templates_ptr       chat_templates;
    std::vector<cache_prompt_entry> cache_prompts;

    // embedding
    bool need_end_eos = false;

    // clip model
    std::mutex       llm_ctx_clip_mtx;
    clip_init_result llm_init_clip  = {};
    clip_ctx *       llm_ctx_clip_v = nullptr;
    clip_ctx *       llm_ctx_clip_a = nullptr;

    struct cache_multimodal_entry {
        std::vector<llama_multimodal_tokens> tokens;
        int64_t                              last_used = 0;
    };

    std::unordered_map<std::string, cache_multimodal_entry> cache_multimodals;

    // speculative decoding
    common_init_result  llm_init_draft;
    llama_model *       llm_model_draft  = nullptr;
    llama_context *     llm_ctx_draft    = nullptr;
    const llama_vocab * llm_vocab_draft  = nullptr;
    llama_batch         batch_text_draft = {};

    // thread pool
    ggml_threadpool * threadpool       = nullptr;
    ggml_threadpool * threadpool_batch = nullptr;

    // tool calls
    bool                     support_tool_calls                   = false;
    // non-jinja tool calls
    llama_tokens             tool_call_start_tokens               = {};
    std::vector<std::string> tool_call_start_words                = {};
    size_t                   tool_call_start_words_longest_length = 0;
    bool                     tool_call_start_trim                 = false;
    llama_tokens             tool_call_end_tokens                 = {};
    std::vector<std::string> tool_call_end_words                  = {};
    bool                     tool_call_end_trim                   = false;
    std::string              tool_call_format                     = "json";

    // reasoning
    bool        support_reasoning     = false;
    llama_token reasoning_start_token = LLAMA_TOKEN_NULL;
    llama_token reasoning_end_token   = LLAMA_TOKEN_NULL;

    static inline int32_t get_task_id() {
        thread_local static int32_t id = -1;
        if (id == -1) {
            static std::atomic<int32_t> next{ 0 };
            id = next++;
        }
        return id;
    }

    inline bool support_tokenize() const { return llm_vocab != nullptr; }

    inline bool support_completion() const {
        return llm_ctx != nullptr && llm_model_casual && !params.llm_params.reranking;
    }

    inline bool support_embedding() const { return llm_ctx != nullptr && params.llm_params.embedding; }

    inline bool support_reranking() const {
        return llm_ctx != nullptr && !llm_model_casual && params.llm_params.reranking;
    }

    inline bool support_image() const { return sd_ctx != nullptr; }

    inline void shift_completion_task_cache(completions_task * task) {
        if (!llm_kv_cache_shift) {
            return;
        }

        // try to reduce the kv cache of the storing cache

        if (cache_prompt) {
            int32_t cache_id  = -1;
            int32_t cache_pos = 0;
            for (int32_t i = 0; i < params.llm_params.n_threads_http; i++) {
                cache_prompt_entry & cache = cache_prompts.at(i);
                if (!cache.used && cache.pos > cache_pos) {
                    cache_id  = i;
                    cache_pos = cache.pos;
                }
            }
            if (cache_id != -1) {
                const int32_t n_keep    = params.llm_params.n_keep + 1;
                const int32_t n_left    = cache_pos - n_keep;
                int32_t       n_discard = std::min(n_left >> 2, params.llm_params.n_ubatch);
                if (n_discard <= 4) {
                    return;
                }
                llama_memory_seq_rm(llama_get_memory(llm_ctx), cache_id, n_keep, n_keep + n_discard);
                llama_memory_seq_add(llama_get_memory(llm_ctx), cache_id, n_keep + n_discard, cache_pos, -n_discard);
                if (llm_ctx_draft != nullptr) {
                    llama_memory_seq_rm(llama_get_memory(llm_ctx_draft), cache_id, n_keep, n_keep + n_discard);
                    llama_memory_seq_add(llama_get_memory(llm_ctx_draft), cache_id, n_keep + n_discard, cache_pos,
                                         -n_discard);
                }
                SRV_WRN(
                    "squash kv cache, "
                    "seq = %d, [%d, %d) -> [%d, %d)\n",
                    cache_id, n_discard, cache_pos, n_keep, cache_pos - n_discard);

                // stats
                cache_prompt_entry & cache = cache_prompts.at(cache_id);
                cache.pos -= n_discard;
                cache.pos_discard += n_discard;
                llm_kv_cache_used -= n_discard;
                llm_kv_cache_inactive -= n_discard;
                return;
            }
        }

        if (task == nullptr) {
            return;
        }

        // otherwise, reduce the kv cache of the given task

        const int32_t n_keep    = params.llm_params.n_keep + 1;
        const int32_t n_left    = task->pos - n_keep;
        const int32_t n_discard = std::min(n_left >> 2, params.llm_params.n_ubatch);
        if (n_discard <= 4) {
            return;
        }
        const std::string rid    = task->get_r_id();
        const int32_t     seq_id = task->get_seq_id();

        llama_memory_seq_rm(llama_get_memory(llm_ctx), seq_id, n_keep, n_keep + n_discard);
        llama_memory_seq_add(llama_get_memory(llm_ctx), seq_id, n_keep + n_discard, task->pos, -n_discard);
        if (llm_ctx_draft != nullptr) {
            llama_memory_seq_rm(llama_get_memory(llm_ctx_draft), seq_id, n_keep, n_keep + n_discard);
            llama_memory_seq_add(llama_get_memory(llm_ctx_draft), seq_id, n_keep + n_discard, task->pos, -n_discard);
        }
        SRV_WRN(
            "rid %s | shift kv cache, "
            "seq = %d, [%d, %d) -> [%d, %d)\n",
            rid.c_str(), seq_id, n_keep + n_discard, task->pos, n_keep, task->pos - n_discard);

        // stats
        task->pos -= n_discard;
        task->pos_discard += n_discard;
        llm_kv_cache_used -= n_discard;
    }

    inline int32_t decode_completion_task_batch(llama_context * input_ctx, llama_batch & input_batch,
                                                const std::vector<std::unique_ptr<btask>> & batch_task_ptrs) {
        // decoded results:
        // -3 compute failed,
        // -2 allocate failed,
        // -1 no tokens,
        //  0 ok,
        //  1 no kv cache,
        //  2 compute aborted
        int32_t                decoded       = 0;
        bool                   is_mrope_view = llm_model_rope_mrope && input_batch.n_tokens > batch_view_max;
        std::vector<llama_pos> mrope_pos_view;

        for (int32_t i_batch_view = 0; i_batch_view < input_batch.n_tokens; i_batch_view += batch_view_max) {
            const int32_t n_batch_view = std::min(batch_view_max, input_batch.n_tokens - i_batch_view);
            llama_batch   input_batch_view;
            if (input_batch.embd == nullptr) {
                input_batch_view = {
                    n_batch_view,
                    input_batch.token + i_batch_view,
                    nullptr,
                    input_batch.pos + i_batch_view,
                    input_batch.n_seq_id + i_batch_view,
                    input_batch.seq_id + i_batch_view,
                    input_batch.logits + i_batch_view,
                };
            } else if (is_mrope_view) {
                mrope_pos_view.clear();
                mrope_pos_view.reserve(n_batch_view * 4);
                for (int32_t i_pos_view = 0; i_pos_view < 4; i_pos_view++) {
                    size_t pos_view_offset = i_pos_view * input_batch.n_tokens + i_batch_view;
                    mrope_pos_view.insert(mrope_pos_view.end(), input_batch.pos + pos_view_offset,
                                          input_batch.pos + pos_view_offset + n_batch_view);
                }
                input_batch_view = {
                    n_batch_view,
                    nullptr,
                    input_batch.embd + i_batch_view * llm_ctx_embed_size,
                    mrope_pos_view.data(),
                    input_batch.n_seq_id + i_batch_view,
                    input_batch.seq_id + i_batch_view,
                    input_batch.logits + i_batch_view,
                };
            } else {
                input_batch_view = {
                    n_batch_view,
                    nullptr,
                    input_batch.embd + i_batch_view * llm_ctx_embed_size,
                    input_batch.pos + i_batch_view,
                    input_batch.n_seq_id + i_batch_view,
                    input_batch.seq_id + i_batch_view,
                    input_batch.logits + i_batch_view,
                };
            }
            decoded = llama_decode(input_ctx, input_batch_view);
            for (int32_t retries = 3; decoded == 1 && retries > 0; retries--) {
                SRV_WRN("failed to decode in batch, try shifting kv cache within %d times\n", retries);
                // find a task in the same batch who has largest pos and finished prefilling
                completions_task * task       = nullptr;
                int32_t            target_pos = -1;
                for (const std::unique_ptr<btask> & task_ptr : batch_task_ptrs) {
                    auto * candidate = dynamic_cast<completions_task *>(task_ptr.get());
                    if (candidate->n_decoded > 0 && candidate->pos > target_pos) {
                        task       = candidate;
                        target_pos = candidate->pos;
                    }
                }
                // shift the target task if found
                if (task != nullptr) {
                    int32_t pos_previous = task->pos;
                    shift_completion_task_cache(task);
                    task->pos -= (pos_previous - task->pos);
                    // adjust batch pos
                    input_batch.pos[task->i_batch_seq_end] = task->pos - 1;
                }
                // otherwise, shift the cache
                else {
                    shift_completion_task_cache(nullptr);
                }
                // decode again
                decoded = llama_decode(input_ctx, input_batch_view);
            }
            if (decoded != 0) {
                break;
            }
        }

        return decoded;
    }

    //
    // Logics
    //
#if defined(linux) || defined(__linux) || defined(__linux__)
#    define PIN_THREAD                                                        \
        cpu_set_t cpu_mask;                                                   \
        CPU_ZERO(&cpu_mask);                                                  \
        CPU_SET(sched_getcpu(), &cpu_mask);                                   \
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_mask); \
        sched_param cpu_sched_param{ 80 };                                    \
        pthread_setschedparam(pthread_self(), SCHED_FIFO, &cpu_sched_param);
#else
#    define PIN_THREAD
#endif

    void reconcile_loop(const std::shared_ptr<httplib::Server> & server) {
        PIN_THREAD;

        while (server->is_running()) {
            reconcile();
        }
    }

    void reconcile() {
        // dequeue tasks
        std::vector<std::unique_ptr<btask>> task_ptrs;
        task_ptrs.resize(params.llm_params.n_threads_http);
        size_t n_dequeue_tasks =
            process_tasks->wait_dequeue_bulk_timed(task_ptrs.data(), params.llm_params.n_threads_http, 3000000);
        if (n_dequeue_tasks == 0) {
            return;
        }

        // batch tasks
        task_type                           batch_task_type    = TASK_UNKNOWN;
        process_type                        batch_process_type = PROCESS_UNKNOWN;
        std::vector<std::unique_ptr<btask>> batch_task_ptrs;
        batch_task_ptrs.reserve(n_dequeue_tasks);
        for (auto & task_ptr : task_ptrs) {
            if (task_ptr == nullptr) {
                break;
            }

            const int32_t     tid   = task_ptr->get_id();
            const task_type   ttype = task_ptr->get_type();
            const std::string rid   = task_ptr->get_r_id();
            const req_type    rtype = task_ptr->get_r_type();

            // filter
            if (batch_task_type == TASK_UNKNOWN) {
                batch_task_type = ttype;

                //// completions or embeddings
                if (batch_task_type != TASK_IMAGES) {
                    // set batch task type
                    llama_set_embeddings(llm_ctx, batch_task_type == TASK_EMBEDDINGS);
                    // apply lora adapters, only need to do it once per batch
                    if (!equal_lora(task_ptr->get_lora_adapters(), lora_adapters)) {
                        lora_adapters = task_ptr->get_lora_adapters();  // copy
                        try {
                            common_set_adapter_lora(llm_ctx, lora_adapters);
                        } catch (const std::exception & e) {
                            SRV_ERR("rid %s | batching, failed to apply lora %s\n", rid.c_str(), e.what());
                        }
                    }
                    // clean batch for later adding
                    common_batch_clear(batch_text);
                    if (llm_ctx_draft != nullptr) {
                        common_batch_clear(batch_text_draft);
                    }
                }

                //// images
                else {
                    // apply lora adapters, only need to do it once per batch
                    if (!equal_lora(task_ptr->get_lora_adapters(), lora_adapters)) {
                        lora_adapters = task_ptr->get_lora_adapters();  // copy
                        try {
                            sd_ctx->apply_lora_adapters(lora_adapters);
                        } catch (const std::exception & e) {
                            SRV_ERR("rid %s | batching, failed to apply lora %s\n", rid.c_str(), e.what());
                        }
                    }
                }
            } else if (batch_task_type != ttype) {
                SRV_INF(
                    "rid %s | "
                    "batching, waiting previous batch finished: not the same kind batch\n",
                    rid.c_str());
                process_tasks->enqueue(std::move(task_ptr));
                continue;
            } else if (!equal_lora(task_ptr->get_lora_adapters(), lora_adapters)) {
                SRV_INF(
                    "rid %s | "
                    "batching, waiting previous batch finished: lora adapters not matched\n",
                    rid.c_str());
                process_tasks->enqueue(std::move(task_ptr));
                continue;
            }

            // batch
            int32_t seq_id = task_ptr->get_seq_id();

            if (batch_task_type != TASK_IMAGES) {
                /**
                 * completions
                 */

                if (batch_task_type == TASK_COMPLETIONS) {
                    auto * task = dynamic_cast<completions_task *>(task_ptr.get());
                    if (task->processed_tokens.capacity() == 0) {
                        task->processed_tokens.reserve(task->n_decoding_budget >= INT32_MAX ?
                                                           task->n_prefilling_request :
                                                           task->n_decoding_budget);
                    }

                    // prefill first (n_prefilled < n_prefilling_request)
                    if (batch_process_type == PROCESS_UNKNOWN && task->n_prefilled < task->n_prefilling_request) {
                        // filter
                        if (llm_kv_cache_used - llm_kv_cache_inactive + task->n_prefilling_request >
                            llm_kv_cache_limit) {
                            SRV_DBG(
                                "rid %s | "
                                "batching, waiting previous batch finished: not enough space to place all tokens\n",
                                rid.c_str());
                            process_tasks->enqueue(std::move(task_ptr));
                            continue;
                        }

                        batch_process_type = PROCESS_PREFILL;

                        // prepare cache - prefix cache
                        if (task->n_prefilled == 0 && cache_prompt) {
                            llama_tokens tokens;
                            // get tokens of plain text
                            if (!task->tokenized_prompts_include_multimedias) {
                                tokens = std::get<llama_tokens>(task->tokenized_prompts[0]);
                            }
                            // get tokens of multimedia
                            else {
                                for (const auto & tokenized_prompt : task->tokenized_prompts) {
                                    if (std::holds_alternative<llama_tokens>(tokenized_prompt)) {
                                        llama_tokens tokenized_text = std::get<llama_tokens>(tokenized_prompt);
                                        tokens.insert(tokens.end(), tokenized_text.begin(), tokenized_text.end());
                                    } else {
                                        llama_multimodal_tokens tokenized_mtmd =
                                            std::get<llama_multimodal_tokens>(tokenized_prompt);
                                        llama_tokens dummy_tokens(tokenized_mtmd.n_pos, tokenized_mtmd.dummy_token);
                                        tokens.insert(tokens.end(), dummy_tokens.begin(), dummy_tokens.end());
                                    }
                                }
                            }
                            int32_t   seq_lcp_id          = seq_id;
                            size_t    seq_lcp_l           = 0;
                            llama_pos seq_lcp_pos         = 0;
                            llama_pos seq_lcp_pos_discard = 0;
                            for (int32_t i = 0; i < params.llm_params.n_threads_http; i++) {
                                cache_prompt_entry & cache = cache_prompts.at(i);
                                if (!cache.used) {
                                    size_t lcp_l = common_lcp(cache.tokens, tokens);
                                    if (lcp_l > seq_lcp_l) {
                                        seq_lcp_id          = i;
                                        seq_lcp_l           = lcp_l;
                                        seq_lcp_pos         = cache.pos;
                                        seq_lcp_pos_discard = cache.pos_discard;
                                    }
                                }
                            }
                            seq_id = seq_lcp_id;
                            // miss cache
                            if (seq_lcp_l == 0) {
                                SRV_INFV(2,
                                         "rid %s | miss prompt cache, "
                                         "seq = %d, next_pos = 0\n",
                                         rid.c_str(), seq_id);
                            }
                            // hit cache but need to redirect
                            else if (seq_lcp_pos_discard != 0 && seq_lcp_l == tokens.size()) {
                                SRV_INFV(2,
                                         "rid %s | hit prompt cache, "
                                         "seq = %d, next_pos = 0\n",
                                         rid.c_str(), seq_id);
                            }
                            // hit cache
                            else {
                                int32_t   cached              = int32_t(seq_lcp_l) - 1;
                                llama_pos pos                 = std::min(seq_lcp_pos - 1, cached);
                                task->pos                     = pos;
                                task->n_processed_detokenized = cached;
                                task->n_prefilled             = cached;
                                task->n_prefilled_cached      = cached;
                                SRV_INFV(2,
                                         "rid %s | hit prompt cache, "
                                         "seq = %d, cached = %d, next_pos = %d\n",
                                         rid.c_str(), seq_id, cached, pos);
                            }
                            // mark cache
                            cache_prompt_entry & cache = cache_prompts.at(seq_id);
                            llm_kv_cache_used -= cache.pos;
                            llm_kv_cache_inactive -= cache.pos;
                            cache.used        = true;
                            cache.pos         = 0;
                            cache.pos_discard = task->pos > 0 ? cache.pos_discard : 0;
                            task->set_seq_id(seq_id);

                            // clean kv cache
                            llama_memory_seq_rm(llama_get_memory(llm_ctx), seq_id, task->pos, -1);
                            if (llm_ctx_draft != nullptr) {
                                llama_memory_seq_rm(llama_get_memory(llm_ctx_draft), seq_id, task->pos, -1);
                            }
                            SRV_DBG(
                                "rid %s | prefix cache, "
                                "clean kv cache, seq %d = [%d, end)\n",
                                rid.c_str(), seq_id, task->pos);
                            llm_kv_cache_used += task->pos;
                        }

                        // batching
                        const auto n_prompt    = int32_t(task->tokenized_prompts.size());
                        int32_t    c_prefilled = 0;
                        if (n_prompt > 1) {
                            // process n-1 prompts
                            for (int32_t i_prompt = 0; i_prompt < (n_prompt - 1); i_prompt++) {
                                // text
                                if (std::holds_alternative<llama_tokens>(task->tokenized_prompts[i_prompt])) {
                                    llama_tokens tokenized_text =
                                        std::get<llama_tokens>(task->tokenized_prompts[i_prompt]);
                                    const auto    n_text   = int32_t(tokenized_text.size());
                                    const int32_t n_text_s = task->n_prefilled - c_prefilled;
                                    if (n_text_s < n_text) {
                                        const int32_t n_text_d = n_text - n_text_s;
                                        // in batch
                                        for (int32_t i_text = n_text_s; i_text < n_text; i_text++) {
                                            const llama_token tok = tokenized_text[i_text];
                                            common_batch_add(batch_text_temp, tok, task->pos, { seq_id }, false);
                                            task->pos++;
                                        }
                                        task->n_prefilled += n_text_d;
                                        llm_kv_cache_used += n_text_d;
                                        // decode immediately
                                        const int32_t decoded_text =
                                            decode_completion_task_batch(llm_ctx, batch_text_temp, batch_task_ptrs);
                                        common_batch_clear(batch_text_temp);
                                        if (decoded_text != 0) {
                                            SRV_ERR(
                                                "rid %s | decode vision text, failed to decode, try again, "
                                                "increasing context size or reducing requests: result = %d\n",
                                                rid.c_str(), decoded_text);
                                            break;
                                        }
                                    }
                                    // accumulate prefilled
                                    c_prefilled += n_text;
                                    // append processed tokens
                                    task->processed_tokens.insert(task->processed_tokens.end(), tokenized_text.begin(),
                                                                  tokenized_text.end());
                                }
                                // multimedia
                                else {
                                    auto tokenized_mtmd =
                                        std::get<llama_multimodal_tokens>(std::move(task->tokenized_prompts[i_prompt]));
                                    const int32_t n_mtmd   = tokenized_mtmd.n_tokens;
                                    const int32_t n_mtmd_s = task->n_prefilled - c_prefilled;
                                    if (n_mtmd_s < tokenized_mtmd.n_pos) {
                                        const int32_t                n_mtmd_d = tokenized_mtmd.n_pos;
                                        // in batch
                                        llama_multimodal_embed_batch batch_mtmd;
                                        //// mrope
                                        if (llm_model_rope_mrope) {
                                            std::vector<llama_pos> pos(n_mtmd * 4);
                                            // vision (2d)
                                            if (!tokenized_mtmd.is_audio) {
                                                clip_image_size & is = tokenized_mtmd.size;
                                                const int32_t     ps = clip_get_patch_size(llm_ctx_clip_v) * 2;
                                                const int32_t     ph = is.height / ps + (is.height % ps > 0);
                                                const int32_t     pw = is.width / ps + (is.width % ps > 0);
                                                for (int32_t y = 0; y < ph; y++) {
                                                    for (int32_t x = 0; x < pw; x++) {
                                                        const int i         = y * pw + x;
                                                        pos[i]              = task->pos;
                                                        pos[i + n_mtmd * 1] = task->pos + y;
                                                        pos[i + n_mtmd * 2] = task->pos + x;
                                                        pos[i + n_mtmd * 3] = 0;
                                                    }
                                                }
                                            }
                                            // audio (1d)
                                            else {
                                                for (int32_t i = 0; i < n_mtmd; i++) {
                                                    pos[i]              = task->pos + i;
                                                    pos[i + n_mtmd * 1] = task->pos + i;
                                                    pos[i + n_mtmd * 2] = task->pos + i;
                                                    pos[i + n_mtmd * 3] = 0;
                                                }
                                            }
                                            batch_mtmd = llama_multimodal_embed_batch(tokenized_mtmd.embed.data(),
                                                                                      n_mtmd, std::move(pos), seq_id);
                                        }
                                        //// non-mrope
                                        else {
                                            batch_mtmd = llama_multimodal_embed_batch(tokenized_mtmd.embed.data(),
                                                                                      n_mtmd, task->pos, seq_id);
                                        }
                                        task->pos += n_mtmd_d;
                                        task->n_prefilled += n_mtmd_d;
                                        llm_kv_cache_used += n_mtmd_d;
                                        // decode immediately
                                        if (llm_ctx_clip_v != nullptr && clip_is_gemma3(llm_ctx_clip_v)) {
                                            llama_set_causal_attn(llm_ctx, false);
                                        }
                                        const int32_t decoded_image =
                                            decode_completion_task_batch(llm_ctx, batch_mtmd.temp, batch_task_ptrs);
                                        if (llm_ctx_clip_v != nullptr && clip_is_gemma3(llm_ctx_clip_v)) {
                                            llama_set_causal_attn(llm_ctx, false);
                                        }
                                        if (decoded_image != 0) {
                                            SRV_ERR(
                                                "rid %s | decode vision image, failed to decode, try again, "
                                                "increasing context size or reducing requests: result = %d\n",
                                                rid.c_str(), decoded_image);
                                            break;
                                        }
                                    }
                                    // accumulate prefilled
                                    c_prefilled += tokenized_mtmd.n_pos;
                                    // append processed tokens
                                    llama_tokens dummy_tokens(tokenized_mtmd.n_pos, tokenized_mtmd.dummy_token);
                                    task->processed_tokens.insert(task->processed_tokens.end(), dummy_tokens.begin(),
                                                                  dummy_tokens.end());
                                }
                            }
                        }
                        llama_tokens  tokenized_text = std::get<llama_tokens>(task->tokenized_prompts[n_prompt - 1]);
                        const auto    n_text         = int32_t(tokenized_text.size());
                        const int32_t n_text_s       = task->n_prefilled - c_prefilled;
                        const int32_t n_text_d       = n_text - n_text_s;
                        // in batch
                        for (int32_t i_text = n_text_s; i_text < n_text; i_text++) {
                            const llama_token tok = tokenized_text[i_text];
                            const bool        emb = i_text + 1 == n_text;
                            common_batch_add(batch_text, tok, task->pos, { seq_id }, emb);
                            if (llm_ctx_draft != nullptr) {
                                common_batch_add(batch_text_draft, tok, task->pos, { seq_id }, emb);
                            }
                            task->pos++;
                        }
                        task->n_prefilled += n_text_d;
                        llm_kv_cache_used += n_text_d;
                        // abort if incomplete prefilling
                        if (task->n_prefilled < task->n_prefilling_request) {
                            // clean kv cache
                            llama_memory_seq_rm(llama_get_memory(llm_ctx), seq_id, 0, -1);
                            if (llm_ctx_draft != nullptr) {
                                llama_memory_seq_rm(llama_get_memory(llm_ctx_draft), seq_id, 0, -1);
                            }
                            SRV_DBG(
                                "rid %s | prefill, incomplete, "
                                "clean kv cache, seq %d = [0, end)",
                                rid.c_str(), seq_id);
                            llm_kv_cache_used -= task->pos;
                            // output
                            json data = {
                                { "message",
                                 "failed to prefill, try again, "
                                  "increasing context size or reducing requests" }
                            };
                            process_task_results[tid]->enqueue(
                                std::make_unique<btask_result>(httplib::InternalServerError_500, std::move(data)));
                            continue;
                        }
                        // append processed tokens
                        task->processed_tokens.insert(task->processed_tokens.end(), tokenized_text.begin(),
                                                      tokenized_text.end());
                        // save for cache prompts,
                        // so we need to mark the base in n_processed_detokenized
                        task->n_processed_detokenized = task->n_prefilling_request;
                        SRV_DBG("rid %s | batching, decode, seq = %d\n", rid.c_str(), seq_id);

                        task->i_batch_seq_end = (batch_text.n_tokens + (batch_view_max - 1)) % batch_view_max;
                        batch_task_ptrs.push_back(std::move(task_ptr));
                    }

                    // decode next (n_decoded > 0)
                    else if (batch_process_type != PROCESS_PREFILL && task->n_decoded > 0 &&
                             batch_text.n_tokens + params.llm_params.speculative.n_max < batch_view_max) {
                        // token throttling
                        if (task->token_bucket != nullptr) {
                            if (!task->token_bucket->try_acquire()) {
                                process_tasks->enqueue(std::move(task_ptr));
                                continue;
                            }
                        }

                        batch_process_type = PROCESS_DECODE;

                        // prepare cache - truncate cache
                        if (llm_kv_cache_used >= llm_kv_cache_limit) {
                            shift_completion_task_cache(task);
                        }

                        // batching
                        common_batch_add(batch_text, task->processed_tokens.back(), task->pos, { seq_id }, true);
                        task->pos++;
                        llm_kv_cache_used++;
                        if (!task->drafted_tokens.empty()) {
                            for (const llama_token & tok : task->drafted_tokens) {
                                common_batch_add(batch_text, tok, task->pos, { seq_id }, true);
                                task->pos++;
                                llm_kv_cache_used++;
                            }
                        }

                        if (task->n_decoded == 1) {
                            SRV_DBG("rid %s | batching, decode next, seq = %d\n", rid.c_str(), seq_id);
                        }

                        task->i_batch_seq_end = batch_text.n_tokens - 1;
                        batch_task_ptrs.push_back(std::move(task_ptr));
                    }

                    // otherwise, wait for next
                    else {
                        SRV_DBG(
                            "rid %s | "
                            "batching, waiting previous batch finished: different processing type\n",
                            rid.c_str());
                        process_tasks->enqueue(std::move(task_ptr));
                    }

                    continue;
                }

                /**
                 * embeddings
                 */

                auto * task = dynamic_cast<embeddings_task *>(task_ptr.get());

                // prefill first
                const bool emb     = rtype == REQ_EMBED && llama_pooling_type(llm_ctx) == LLAMA_POOLING_TYPE_NONE;
                const auto n_input = int32_t(task->tokenized_inputs.size());
                if (task->i_input_prefilled < n_input) {
                    llama_tokens tokenized_input = task->tokenized_inputs[task->i_input_prefilled];
                    const auto   n_pos           = int32_t(tokenized_input.size());
                    // allow batch's tokens size be equal to llm_ctx_size
                    if (batch_text.n_tokens + n_pos > llm_ctx_size) {
                        SRV_INF("rid %s | batching, not enough space to fill, waiting\n", rid.c_str());
                        continue;
                    }
                    // avoid smaller batch size
                    else if (batch_text.n_tokens + n_pos <= seq_id) {
                        seq_id = batch_text.n_tokens + n_pos - 1;
                        task->set_seq_id(seq_id);
                    }

                    // prepare cache - clean cache
                    if (cache_prompt) {
                        llama_memory_seq_rm(llama_get_memory(llm_ctx), seq_id, 0, -1);
                        if (llm_ctx_draft != nullptr) {
                            llama_memory_seq_rm(llama_get_memory(llm_ctx_draft), seq_id, 0, -1);
                        }
                        SRV_INFV(2,
                                 "rid %s | batching, clean cache, "
                                 "clean kv cache, seq %d = [0, end)",
                                 rid.c_str(), seq_id);

                        cache_prompt_entry & cache = cache_prompts.at(seq_id);
                        llm_kv_cache_used -= cache.pos;
                        llm_kv_cache_inactive -= cache.pos;
                        cache.used        = false;
                        cache.pos         = 0;
                        cache.pos_discard = 0;
                    }

                    for (llama_pos pos = 0; pos < n_pos; pos++) {
                        common_batch_add(batch_text, tokenized_input[pos], pos, { seq_id }, emb);
                    }
                    task->n_prefilled += n_pos;
                    task->n_min_prefilled = task->n_min_prefilled == 0 ? n_pos : std::min(task->n_min_prefilled, n_pos);
                    task->n_max_prefilled = std::max(task->n_max_prefilled, n_pos);

                    task->i_input_prefilled++;

                    task->i_batch_seq_end = batch_text.n_tokens - 1;
                    batch_task_ptrs.push_back(std::move(task_ptr));
                }

                SRV_DBG("rid %s | batching, decode, seq = %d\n", rid.c_str(), seq_id);

                continue;
            }

            /**
             * images
             */

            auto * task = dynamic_cast<images_task *>(task_ptr.get());

            // forward
            const int32_t n_repeat = task->req->n;

            if (task->n_forward_steps == 0) {
                task->streams.resize(n_repeat);
                task->b64_jsons.resize(n_repeat);
                task->progressed_steps.resize(n_repeat);
                task->progress_steps.resize(n_repeat);
                // init stream
                for (int32_t n = 0; n < n_repeat; n++) {
                    stablediffusion_params_sampling sampling = task->req->sampling;  // copy
                    sampling.seed += n;
                    std::unique_ptr<stablediffusion_sampling_stream> stream =
                        sd_ctx->generate_stream(task->req->get_prompt(), sampling);
                    task->streams[n] = std::move(stream);
                    task->n_forward_steps++;
                }
            }

            batch_task_ptrs.push_back(std::move(task_ptr));

            SRV_DBG("rid %s | batching, reverse, seq = %d\n", rid.c_str(), seq_id);
        }

        // process tasks

        if (batch_task_type != TASK_IMAGES) {
            /**
             * completions
             */

            if (batch_task_type == TASK_COMPLETIONS) {
                // decode
                if (batch_text.n_tokens > 0) {
                    const int32_t decoded = decode_completion_task_batch(llm_ctx, batch_text, batch_task_ptrs);
                    if (decoded != 0) {
                        SRV_ERR(
                            "decode in batch, failed to decode, try again, "
                            "increasing context size or reducing parallel: result = %d\n",
                            decoded);
                        for (const std::unique_ptr<btask> & task_ptr : batch_task_ptrs) {
                            auto *            task   = dynamic_cast<completions_task *>(task_ptr.get());
                            const std::string rid    = task->get_r_id();
                            const int32_t     seq_id = task->get_seq_id();
                            // clean kv cache
                            llama_memory_seq_rm(llama_get_memory(llm_ctx), seq_id, 0, -1);
                            if (llm_ctx_draft != nullptr) {
                                llama_memory_seq_rm(llama_get_memory(llm_ctx_draft), seq_id, 0, -1);
                            }
                            SRV_INFV(2,
                                     "rid %s | decode, "
                                     "clean kv cache, seq %d = [0, end)",
                                     rid.c_str(), seq_id);
                            llm_kv_cache_used -= task->pos;
                            // output
                            json data = {
                                { "message",
                                 "failed to decode, try again, "
                                  "increasing context size or reducing parallel" }
                            };
                            process_task_results[task->get_id()]->enqueue(
                                std::make_unique<btask_result>(httplib::InternalServerError_500, std::move(data)));
                        }
                        return;
                    }
                }
                // speculative - draft
                // NB(thxCode): we don't need to decode in a loop like above,
                // as the previous llm_ctx decode also shift the draft kv cache during failure decoding.
                if (batch_text_draft.n_tokens > 0) {
                    const int32_t decoded_draft =
                        decode_completion_task_batch(llm_ctx_draft, batch_text_draft, batch_task_ptrs);
                    if (decoded_draft != 0) {
                        // NB(thxCode): we should not reach here.
                        SRV_ERR(
                            "decode draft in batch, failed to decode, try increasing context size "
                            "or reducing parallel: result = %d\n",
                            decoded_draft);
                        for (auto & task_ptr : batch_task_ptrs) {
                            auto *            task   = dynamic_cast<completions_task *>(task_ptr.get());
                            const std::string rid    = task->get_r_id();
                            const int32_t     seq_id = task->get_seq_id();
                            // clean kv cache
                            llama_memory_seq_rm(llama_get_memory(llm_ctx), seq_id, 0, -1);
                            llama_memory_seq_rm(llama_get_memory(llm_ctx_draft), seq_id, 0, -1);
                            SRV_INFV(2,
                                     "rid %s | decode, "
                                     "clean kv cache, seq %d = [0, end)",
                                     rid.c_str(), seq_id);
                            llm_kv_cache_used -= task->pos;
                            // output
                            json data = {
                                { "message",
                                 "failed to decode draft, try again, "
                                  "increasing context size or reducing parallel" }
                            };
                            process_task_results[task_ptr->get_id()]->enqueue(
                                std::make_unique<btask_result>(httplib::InternalServerError_500, std::move(data)));
                        }
                        return;
                    }
                }
                // sample
                for (auto & task_ptr : batch_task_ptrs) {
                    auto *            task   = dynamic_cast<completions_task *>(task_ptr.get());
                    const int32_t     tid    = task->get_id();
                    const std::string rid    = task->get_r_id();
                    const int32_t     seq_id = task->get_seq_id();
                    // sample token
                    //// default
                    if (task->drafted_tokens.empty()) {
                        const int32_t     tok_idx = task->i_batch_seq_end;
                        const llama_token tok     = common_sampler_sample(task->sampler, llm_ctx, tok_idx);
                        common_sampler_accept(task->sampler, tok, true);
                        task->push_generated_token(llm_ctx, tok_idx, tok);
                        task->n_decoded++;
                        task->n_decoding_budget--;
                    }
                    //// include drafted tokens
                    else {
                        // +1 for main model decoded token
                        for (int32_t j = 0, s = int32_t(task->drafted_tokens.size()); j < s + 1; ++j) {
                            // greedy verification only
                            const int32_t     tok_idx = task->i_batch_seq_end - s + j;
                            const llama_token tok     = common_sampler_sample(task->sampler, llm_ctx, tok_idx);
                            common_sampler_accept(task->sampler, tok, true);
                            task->push_generated_token(llm_ctx, tok_idx, tok);
                            task->n_decoded++;
                            task->n_decoding_budget--;
                            if (j < s) {
                                if (tok != task->drafted_tokens[j]) {
                                    int32_t d = s - j;
                                    // back pos to the correct position
                                    task->pos -= d;
                                    // stats drafted tokens size
                                    task->n_decoded += d;
                                    task->n_decoding_budget -= d;
                                    // clean kv cache
                                    llama_memory_seq_rm(llama_get_memory(llm_ctx), seq_id, task->pos, -1);
                                    if (llm_ctx_draft != nullptr) {
                                        llama_memory_seq_rm(llama_get_memory(llm_ctx_draft), seq_id, task->pos, -1);
                                    }
                                    SRV_INFV(2,
                                             "rid %s | decode, "
                                             "clean kv cache, seq %d = [%d, end)",
                                             rid.c_str(), seq_id, task->pos);
                                    llm_kv_cache_used -= d;
                                    break;
                                }
                                task->n_drafted_accepted++;
                            }
                        }
                    }
                    // speculative - lookup
                    if (params.lookup_ngram_min > 0) {
                        common_ngram_cache_update(task->ngram_cache, params.lookup_ngram_min, LLAMA_NGRAM_MAX,
                                                  task->processed_tokens, 1, false);
                    }
                    // stats
                    if (task->n_decoded == 1) {
                        task->t_start_decode = ggml_time_us();
                        task->t_prefilled    = double(task->t_start_decode - task->t_start_prefill) / 1.e3;
                        metrics.on_tokens_prefilled(task->t_prefilled, task->n_prefilled);
                        task->p_prefilled_tps = 1.e3 / task->t_prefilled * task->n_prefilled;
                    }
                    // postprocess
                    bool send_text = false;
                    {
                        const int32_t n_generated_tokens_s = task->n_processed_detokenized;
                        const int32_t n_generated_tokens_e = int32_t(task->processed_tokens.size());
                        std::string   sampled_str;
                        for (; task->n_processed_detokenized < n_generated_tokens_e; task->n_processed_detokenized++) {
                            llama_token tok = task->processed_tokens[task->n_processed_detokenized];
                            // accept special token
                            bool        special =
                                params.llm_params.special || task->req->sampling.preserved_tokens.find(tok) !=
                                                                 task->req->sampling.preserved_tokens.end();
                            sampled_str += common_token_to_piece(llm_ctx, tok, special);
                            // check if the token is a reasoning token
                            if (support_reasoning && !task->reasoning_finished) {
                                if (!task->reasoning_start_found) {
                                    task->reasoning_start_found = tok == reasoning_start_token;
                                } else if (!task->reasoning_end_found) {
                                    if (tok == reasoning_end_token) {
                                        task->reasoning_end_found = true;
                                    } else {
                                        task->n_reasoning++;
                                    }
                                } else if (!task->reasoning_finished) {
                                    task->reasoning_finished = true;
                                    task->generated_text.clear();  // avoid to remember the thinking content
                                }
                            }
                        }
                        task->generated_text += sampled_str;
                        send_text = get_position_of_utf8(task->generated_text) == task->generated_text.size();
                        if (send_text && common_log_verbosity_thold > 5) {
                            SRV_DBG("rid %s | sampled str: %s\n", rid.c_str(), escape_string(sampled_str).c_str());
                        }
                        // check stop
                        //// check stop word or tool call
                        if (send_text) {
                            // has stop words
                            for (const std::string & word : task->req->stop) {
                                size_t pos =
                                    task->generated_text.find(word, task->generated_text.size() - sampled_str.size());
                                if (pos != std::string::npos) {
                                    SRV_DBG("rid %s | stopped by word\n", rid.c_str());
                                    task->generated_finish_reason = "stop";
                                    task->generated_text_keep_pos = pos;
                                    break;
                                }
                            }
                            // find tool call
                            if (task->tokenized_prompts_include_tools && task->reasoning_finished) {
                                //// jinja
                                if (params.llm_params.use_jinja) {
                                    if (common_sampler_grammer_lazy_triggered(task->sampler)) {
                                        send_text                 = false;
                                        std::string functions_str = task->generated_text;
                                        if (!functions_str.empty()) {
                                            try {
                                                common_chat_msg msg = common_chat_parse(functions_str, false,
                                                                                        task->tokenized_prompts_syntax);
                                                if (!msg.tool_calls.empty()) {
                                                    for (const common_chat_tool_call & tc : msg.tool_calls) {
                                                        task->generated_tool_calls.push_back({
                                                            { "type",     "function"                                },
                                                            { "function",
                                                             { { "name", tc.name }, { "arguments", tc.arguments } } },
                                                            { "id",       tc.id.empty() ? gen_call_id() : tc.id     },
                                                        });
                                                    }
                                                    if (task->tool_call_stop_fast) {
                                                        SRV_DBG("rid %s | stopped by tool call\n", rid.c_str());
                                                        task->generated_finish_reason =
                                                            "tool_calls";  // send_text = true;
                                                    }
                                                    // eat the rest of the text
                                                    task->generated_text_keep_pos = std::string::npos;
                                                    task->generated_text.clear();
                                                }
                                            } catch (const std::exception & e) {
                                                task->generated_text_keep_pos = 0;
                                            }
                                        }
                                    }
                                }
                                //// non-jinja
                                else {
                                    ////// found tool call start
                                    if (!task->tool_call_start_found) {
                                        if (!tool_call_start_tokens.empty()) {
                                            // stop sending text if the start token found
                                            for (int32_t i = n_generated_tokens_s; i < n_generated_tokens_e; i++) {
                                                for (const llama_token & token : tool_call_start_tokens) {
                                                    if (task->processed_tokens[i] == token) {
                                                        task->tool_call_start_found = true;
                                                        if (!sampled_str.empty() && tool_call_start_trim) {
                                                            // trim the start word if needed
                                                            for (const std::string & sw : tool_call_start_words) {
                                                                if (size_t sp = task->generated_text.find(sw);
                                                                    sp != std::string::npos) {
                                                                    task->generated_text_keep_pos = sp;
                                                                    // trim the start word
                                                                    task->generated_text = task->generated_text.erase(
                                                                        sp, sp + sw.length());
                                                                }
                                                            }
                                                        }
                                                        break;
                                                    }
                                                }
                                                if (task->tool_call_start_found) {
                                                    break;
                                                }
                                            }
                                        } else if (!tool_call_start_words.empty()) {
                                            // stop sending text if the start word found
                                            if (task->generated_text.size() <= tool_call_start_words_longest_length) {
                                                send_text = false;
                                            } else {
                                                for (const std::string & sw : tool_call_start_words) {
                                                    if (size_t sp = task->generated_text.find(sw);
                                                        sp != std::string::npos) {
                                                        task->tool_call_start_found   = true;
                                                        task->generated_text_keep_pos = sp;
                                                        if (tool_call_start_trim) {
                                                            // trim the start word
                                                            task->generated_text =
                                                                task->generated_text.erase(sp, sp + sw.length());
                                                        }
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    ////// found tool call end
                                    else {
                                        send_text = false;
                                        std::string functions_str;
                                        if (!tool_call_end_tokens.empty()) {
                                            for (int32_t i = n_generated_tokens_e - 1; i >= n_generated_tokens_s; --i) {
                                                for (const llama_token & token : tool_call_end_tokens) {
                                                    if (task->processed_tokens[i] == token) {
                                                        size_t sp     = task->generated_text_keep_pos;
                                                        sp            = sp == std::string::npos ? 0 : sp;
                                                        functions_str = task->generated_text.substr(sp);
                                                        break;
                                                    }
                                                }
                                                if (!functions_str.empty()) {
                                                    if (tool_call_end_trim) {
                                                        for (const std::string & ew : tool_call_end_words) {
                                                            if (size_t ep = functions_str.rfind(ew);
                                                                ep != std::string::npos) {
                                                                functions_str = functions_str.substr(0, ep);
                                                            }
                                                        }
                                                    }
                                                    break;
                                                }
                                            }
                                        } else if (!tool_call_end_words.empty()) {
                                            for (const std::string & ew : tool_call_end_words) {
                                                if (size_t ep = task->generated_text.rfind(ew);
                                                    ep != std::string::npos) {
                                                    if (!tool_call_end_trim) {
                                                        ep += ew.length();
                                                    }
                                                    size_t sp     = task->generated_text_keep_pos;
                                                    sp            = sp == std::string::npos ? 0 : sp;
                                                    functions_str = task->generated_text.substr(sp, ep);
                                                    break;
                                                }
                                            }
                                        }
                                        if (!functions_str.empty()) {
                                            try {
                                                auto append_tool_calls = [&](json & function) {
                                                    if (!function.is_object()) {
                                                        throw std::runtime_error("function is an object");
                                                    }
                                                    if (!function.contains("name")) {
                                                        throw std::runtime_error("function does not contain \"name\"");
                                                    }
                                                    if (!function.contains("arguments")) {
                                                        throw std::runtime_error(
                                                            "function does not contain \"arguments\"");
                                                    }
                                                    if (!function.at("arguments").is_string()) {
                                                        function["arguments"] =
                                                            function.at("arguments")
                                                                .dump(-1, ' ', false, json::error_handler_t::replace);
                                                    }
                                                    json tool_call = {
                                                        { "type",     "function"    },
                                                        { "function", function      },
                                                        { "id",       gen_call_id() },
                                                    };
                                                    task->generated_tool_calls.push_back(tool_call);
                                                };
                                                // json
                                                if (tool_call_format == "json") {
                                                    json functions = json::parse(functions_str);
                                                    if (functions.is_array()) {
                                                        for (auto & function : functions) {
                                                            append_tool_calls(function);
                                                        }
                                                    } else {
                                                        append_tool_calls(functions);
                                                    }
                                                }
                                                // function
                                                else {
                                                    const std::string name_s = "function";
                                                    const std::string func_s = "```json\n";
                                                    const std::string func_e = "```";
                                                    size_t            sp     = functions_str.find(name_s);
                                                    for (; sp != std::string::npos;) {
                                                        sp += name_s.length();
                                                        size_t ep = functions_str.find(func_s, sp);
                                                        if (ep == std::string::npos) {
                                                            break;  // incomplete
                                                        }
                                                        json fn{};
                                                        fn["name"] = functions_str.substr(sp, ep - sp - 1);
                                                        sp         = ep + func_s.length();
                                                        ep         = functions_str.find(func_e, sp);
                                                        if (ep == std::string::npos) {
                                                            break;  // incomplete
                                                        }
                                                        fn["arguments"] = functions_str.substr(sp, ep - sp - 1);
                                                        append_tool_calls(fn);
                                                        sp = ep + func_e.length();
                                                        sp = functions_str.find(name_s, sp);
                                                    }
                                                }
                                                if (!task->generated_tool_calls.empty()) {
                                                    if (task->tool_call_stop_fast) {
                                                        SRV_DBG("rid %s | stopped by tool call\n", rid.c_str());
                                                        task->generated_finish_reason =
                                                            "tool_calls";  // send_text = true;
                                                    }
                                                    // eat the rest of the text
                                                    task->generated_text_keep_pos = std::string::npos;
                                                    task->generated_text.clear();
                                                }
                                            } catch (const std::exception & e) {
                                                task->generated_text_keep_pos = 0;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        //// check eog or budget
                        if (task->generated_finish_reason.empty()) {
                            // end of generation
                            if (llama_vocab_is_eog(llm_vocab, task->processed_tokens.back())) {
                                if (task->generated_tool_calls.empty()) {
                                    SRV_DBG("rid %s | stopped by EOG\n", rid.c_str());
                                    task->generated_finish_reason = "stop";
                                } else {
                                    SRV_DBG("rid %s | stopped by tool call\n", rid.c_str());
                                    task->generated_finish_reason = "tool_calls";
                                }
                                task->generated_text_keep_pos = task->generated_text.size();
                            }
                            // no enough budget
                            else if (task->n_decoding_budget <= 0) {
                                SRV_DBG("rid %s | stopped by length\n", rid.c_str());
                                task->generated_finish_reason = "length";
                                task->generated_text_keep_pos = task->generated_text.size();
                            }
                        }
                    }
                    // continue if not finished
                    bool opened = true;
                    if (task->generated_finish_reason.empty()) {
                        // stream outputting
                        if (send_text && task_ptr->is_stream()) {
                            json data = task->to_json(llm_ctx);
                            process_task_results[tid]->enqueue(
                                std::make_unique<btask_result>(httplib::Continue_100, std::move(data)));
                        }
                        // speculative
                        if (!task->tokenized_prompts_include_multimedias) {
                            task->drafted_tokens.clear();
                            //// draft
                            if (llm_ctx_draft != nullptr) {
                                // clean batch for later adding
                                common_batch_clear(batch_text_draft);
                                common_batch_add(batch_text_draft, task->processed_tokens.back(), task->pos, { seq_id },
                                                 true);
                                int32_t decoded_draft = llama_decode(llm_ctx_draft, batch_text_draft);
                                if (decoded_draft != 0) {
                                    SRV_ERR(
                                        "rid %s | decode draft, failed to decode, try again, "
                                        "increasing context size or reducing requests: result = %d\n",
                                        rid.c_str(), decoded_draft);
                                    // output
                                    json data = {
                                        { "message",
                                         "failed to decode draft, try again, "
                                          "increasing context size or reducing parallel" }
                                    };
                                    process_task_results[tid]->enqueue(std::make_unique<btask_result>(
                                        httplib::InternalServerError_500, std::move(data)));
                                    continue;
                                }
                                // speculative in n_max times
                                for (int32_t j = 0; j < params.llm_params.speculative.n_max; ++j) {
                                    const llama_token tok =
                                        common_sampler_sample(task->sampler_draft, llm_ctx_draft, 0, true);
                                    const llama_token_data_array * cur_p =
                                        common_sampler_get_candidates(task->sampler_draft);
                                    if (cur_p->data[0].p < params.llm_params.speculative.p_min) {
                                        break;
                                    }
                                    common_sampler_accept(task->sampler_draft, tok, true);
                                    task->n_drafted++;
                                    if (llama_vocab_is_eog(llm_vocab_draft, tok)) {
                                        break;
                                    }
                                    task->drafted_tokens.push_back(tok);
                                    // clean batch for later adding
                                    common_batch_clear(batch_text_draft);
                                    common_batch_add(batch_text_draft, tok, task->pos + 1 + j, { seq_id }, true);
                                    decoded_draft = llama_decode(llm_ctx_draft, batch_text_draft);
                                    if (decoded_draft != 0) {
                                        SRV_INFV(2,
                                                 "rid %s | decode draft, "
                                                 "clean kv cache, seq %d = [0, end)",
                                                 rid.c_str(), seq_id);
                                        break;
                                    }
                                }
                                // ignore if less than n_min
                                if (int32_t(task->drafted_tokens.size()) < params.llm_params.speculative.n_min) {
                                    task->drafted_tokens.clear();
                                }
                            }
                            //// lookup ngram
                            if (params.lookup_ngram_min > 0) {
                                size_t n_drafted = task->drafted_tokens.size();
                                if (n_drafted == 0) {
                                    task->drafted_tokens.push_back(task->processed_tokens.back());
                                }
                                common_ngram_cache ngram_cache_empty;
                                common_ngram_cache_draft(task->processed_tokens, task->drafted_tokens,
                                                         params.llm_params.speculative.n_max, params.lookup_ngram_min,
                                                         LLAMA_NGRAM_MAX, task->ngram_cache, ngram_cache_empty,
                                                         ngram_cache_empty);
                                if (n_drafted == 0) {
                                    task->drafted_tokens.erase(task->drafted_tokens.begin());
                                }
                                task->n_drafted += int32_t(task->drafted_tokens.size() - n_drafted);
                            }
                        }
                        // enqueue
                        if (!task_ptr->is_connection_closed()) {
                            process_tasks->enqueue(std::move(task_ptr));
                            continue;
                        }
                        opened = false;
                    }
                    // stats
                    task->t_decoded = double(ggml_time_us() - task->t_start_decode) / 1.e3;
                    metrics.on_tokens_decoded(task->t_decoded, task->n_decoded, task->n_drafted,
                                              task->n_drafted_accepted);
                    task->p_decoded_tps = 1.e3 / task->t_decoded * task->n_decoded;
                    task->p_drafted_apt =
                        task->n_drafted == 0 ? 0.0 : double(task->n_drafted_accepted) / double(task->n_drafted);
                    // output
                    if (opened) {
                        json data = task->to_json(llm_ctx);
                        process_task_results[tid]->enqueue(
                            std::make_unique<btask_result>(httplib::OK_200, std::move(data)));
                    }
                    SRV_INF(
                        "rid %s | "
                        "prefill_t = %d, prefill_cached_t = %d, prefill_tps = %.2f tps, ttft = %.2fms, "
                        "decode_t = %d, decode_tps = %.2f tps, tpot = %.2fms, "
                        "draft_t = %d, draft_apt = %.2f%%, "
                        "total_t = %d, "
                        "stop_r = %s\n",
                        rid.c_str(), task->n_prefilled, task->n_prefilled_cached, task->p_prefilled_tps,
                        task->t_prefilled, task->n_decoded, task->p_decoded_tps,
                        task->t_decoded / double(task->n_decoded), task->n_drafted, task->p_drafted_apt * 100,
                        task->n_prefilled + task->n_decoded, opened ? task->generated_finish_reason.c_str() : "closed");
                    // clean kv cache
                    if (!cache_prompt) {
                        llama_memory_seq_rm(llama_get_memory(llm_ctx), seq_id, 0, -1);
                        if (llm_ctx_draft != nullptr) {
                            llama_memory_seq_rm(llama_get_memory(llm_ctx_draft), seq_id, 0, -1);
                        }
                        SRV_INFV(2,
                                 "rid %s | decode in batch, "
                                 "clean kv cache, seq %d = [0, end)\n",
                                 rid.c_str(), seq_id);
                        llm_kv_cache_used -= task->pos;
                    }
                    // cache prompt
                    else {
                        cache_prompt_entry & cache = cache_prompts.at(seq_id);
                        cache.tokens.swap(task->processed_tokens);
                        cache.used = false;
                        cache.pos  = task->pos;
                        cache.pos_discard += task->pos_discard;
                        llm_kv_cache_inactive += task->pos;
                    }
                }
                return;
            }

            /**
             * embeddings
             */

            const int32_t decoded = llama_decode(llm_ctx, batch_text);
            if (decoded != 0) {
                SRV_ERR(
                    "decode in batch, failed to decode, try again, "
                    "increasing context size or reducing parallel: result = %d\n",
                    decoded);
                // clean kv cache
                llama_memory_clear(llama_get_memory(llm_ctx), true);
                // output
                for (auto & task_ptr : batch_task_ptrs) {
                    json data = {
                        { "message",
                         "failed to decode, try again, "
                          "increasing context size or reducing parallel" }
                    };
                    process_task_results[task_ptr->get_id()]->enqueue(
                        std::make_unique<btask_result>(httplib::InternalServerError_500, std::move(data)));
                }
                return;
            }
            for (auto & task_ptr : batch_task_ptrs) {
                auto *            task    = dynamic_cast<embeddings_task *>(task_ptr.get());
                const int32_t     tid     = task->get_id();
                const std::string rid     = task->get_r_id();
                const req_type    rtype   = task->get_r_type();
                const int32_t     seq_id  = task->get_seq_id();
                // get embeddings
                const size_t      n_input = task->tokenized_inputs.size();
                task->embeds.reserve(n_input);
                if (rtype == REQ_EMBED) {
                    task->embeds.emplace_back(llm_ctx_embed_size, 0.0f);
                } else {
                    task->embeds.emplace_back(1, -1.e6);
                }
                const float * embed = llama_get_embeddings_seq(llm_ctx, seq_id);
                if (embed == nullptr) {
                    embed = llama_get_embeddings_ith(llm_ctx, task->i_batch_seq_end);
                }
                if (embed == nullptr) {
                    SRV_WRN("rid %s | decode in batch, failed to get embeddings\n", rid.c_str());
                    continue;
                }
                if (rtype == REQ_EMBED) {
                    // NB(thxCode): normalize embeddings result.
                    common_embd_normalize(embed, task->embeds[task->embeds.size() - 1].data(), llm_ctx_embed_size, 2);
                } else {
                    task->embeds[task->embeds.size() - 1][0] = embed[0];
                }
                // clean kv cache
                if (!cache_prompt) {
                    llama_memory_seq_rm(llama_get_memory(llm_ctx), seq_id, 0, -1);
                    if (llm_ctx_draft != nullptr) {
                        llama_memory_seq_rm(llama_get_memory(llm_ctx_draft), seq_id, 0, -1);
                    }
                    SRV_INFV(2,
                             "rid %s | decode in batch, "
                             "clean kv cache, seq %d = [0, end)\n",
                             rid.c_str(), seq_id);
                }
                // continue if not finished
                bool opened = true;
                if (task->embeds.size() < n_input) {
                    if (!task_ptr->is_connection_closed()) {
                        process_tasks->enqueue(std::move(task_ptr));
                        continue;
                    }
                    opened = false;
                }
                // stats
                task->t_prefilled = double(ggml_time_us() - task->t_start_prefill) / 1.e3;
                metrics.on_tokens_prefilled(task->t_prefilled, task->n_prefilled);
                task->p_prefilled_tps = 1.e3 / task->t_prefilled * task->n_prefilled;
                // output
                if (opened) {
                    json data = task->to_json();
                    process_task_results[tid]->enqueue(
                        std::make_unique<btask_result>(httplib::OK_200, std::move(data)));
                }
                SRV_INF(
                    "rid %s | "
                    "prefill_t = %d, prefill_tps = %.2f tps, ttft = %.2fms, "
                    "total_t = %d, min_prompt_t = %d, max_prompt_t = %d, "
                    "stop_r = %s\n",
                    rid.c_str(), task->n_prefilled, task->p_prefilled_tps, task->t_prefilled, task->n_prefilled,
                    task->n_min_prefilled, task->n_max_prefilled, opened ? "stop" : "closed");
            }
            return;
        }

        /**
         * images
         */

        for (auto & task_ptr : batch_task_ptrs) {
            auto *     task    = dynamic_cast<images_task *>(task_ptr.get());
            const bool preview = json_value(task->req->stream_options, "preview", false) ||
                                 json_value(task->req->stream_options, "preview_faster", false);
            const int32_t     tid      = task->get_id();
            const std::string rid      = task->get_r_id();
            const int32_t     n_repeat = task->req->n;
            // stats
            if (task->progressed_steps[0] == 0) {
                task->t_start_reverse = ggml_time_us();
                task->t_forwarded     = double(task->t_start_reverse - task->t_start_forward) / 1.e3;
                metrics.on_mtmd_forwarded(task->t_forwarded, task->n_forward_steps);
                task->p_forwarded_sps = 1.e3 / task->t_forwarded * task->n_forward_steps;
            }
            bool incomplete = false;
            // reverse
            for (int32_t n = 0; n < n_repeat; n++) {
                // skip if finished
                if (task->progressed_steps[n] > 0 && task->progressed_steps[n] == task->progress_steps[n]) {
                    continue;
                }
                // sample
                stablediffusion_sampling_stream * stream   = task->streams[n].get();
                uint64_t                          start_at = ggml_time_us();
                incomplete                                 = sd_ctx->sample_stream(stream);
                uint64_t rct                               = ggml_time_us() - start_at;
                task->n_reverse_steps++;
                const auto & [progressed_steps, progress_steps] = sd_ctx->progress_stream(stream);
                SRV_INFV(3, "rid %s | reversed, seq = %d, n = %d, progress = %03i/%03i, cost = %.2f%s\n", rid.c_str(),
                         tid, n, progressed_steps, progress_steps, double(rct) / (rct > 1.e6 ? 1.e6 : 1.e3),
                         rct > 1.e6 ? "s" : "ms");
                task->progressed_steps[n] = progressed_steps;
                task->progress_steps[n]   = progress_steps;
                // output
                if (incomplete) {
                    // stream outputting
                    if (task->is_stream()) {
                        // get preview image
                        if (preview) {
                            auto        preview_img = sd_ctx->preview_image_stream(stream, true);
                            std::string b64_json    = encode_base64(preview_img->data, preview_img->size);
                            task->b64_jsons[n]      = std::move(b64_json);
                        }
                        json data = task->to_json(n);
                        process_task_results[tid]->enqueue(
                            std::make_unique<btask_result>(httplib::Continue_100, std::move(data)));
                    }
                } else {
                    // get generated image
                    auto        generated_img = sd_ctx->result_image_stream(stream);
                    std::string b64_json      = encode_base64(generated_img->data, generated_img->size);
                    task->b64_jsons[n]        = std::move(b64_json);
                    // stream outputting, but not the last one
                    if (task->is_stream() && n + 1 < n_repeat) {
                        json data = task->to_json(n);
                        process_task_results[tid]->enqueue(
                            std::make_unique<btask_result>(httplib::Continue_100, std::move(data)));
                    }
                }
            }
            // continue if not finished
            bool opened = true;
            if (incomplete) {
                if (!task_ptr->is_connection_closed()) {
                    process_tasks->enqueue(std::move(task_ptr));
                    continue;
                }
                opened = false;
            }
            // stats
            task->t_reversed = double(ggml_time_us() - task->t_start_reverse) / 1.e3;
            metrics.on_mtmd_reversed(task->t_reversed, task->n_reverse_steps);
            task->p_reversed_sps = 1.e3 / task->t_reversed * task->n_reverse_steps;
            // output
            if (opened) {
                json data;
                if (task->is_stream()) {
                    data = task->to_json(n_repeat - 1);
                } else {
                    data = task->to_json(-1);
                }
                process_task_results[tid]->enqueue(std::make_unique<btask_result>(httplib::OK_200, std::move(data)));
            }
            SRV_INF(
                "rid %s | "
                "forward_s = %d, forward_sps = %.2f sps, "
                "reverse_s = %d, reverse_sps = %.2f sps, "
                "stop_r = %s\n",
                rid.c_str(), task->n_forward_steps, task->p_forwarded_sps, task->n_reverse_steps, task->p_reversed_sps,
                opened ? "stop" : "closed");
        }
    }

    int32_t process(const httplib::Request & request, httplib::Response & response,
                    std::unique_ptr<btask> && task_ptr) {
        PIN_THREAD;

        const int32_t     tid        = task_ptr->get_id();
        const task_type   ttype      = task_ptr->get_type();
        const std::string rid        = task_ptr->get_r_id();
        const req_type    rtype      = task_ptr->get_r_type();
        const bool        stream     = task_ptr->is_stream();
        const bool        chunk      = stream && json_value(task_ptr->get_stream_options(), "chunk", false);
        const int32_t     chunk_size = json_value(task_ptr->get_stream_options(), "chunk_size", 4096);

        // enqueue task
        process_tasks->enqueue(std::move(task_ptr));

        // non-streaming
        if (!stream) {
            // dequeue result
            std::unique_ptr<btask_result> result_ptr;
            process_task_results[tid]->wait_dequeue(result_ptr);

            // output result
            int32_t status = send_json(request, response, result_ptr->status, result_ptr->result);
            if (status != httplib::OK_200) {
                SRV_ERR("rid %s | failed to send response, status = %d\n", rid.c_str(), status);
            }
            return status;
        }

        // streaming
        const auto on_chunk = [=](size_t, httplib::DataSink & sink) {
            // dequeue result
            std::unique_ptr<btask_result> result_ptr;
            process_task_results[tid]->wait_dequeue(result_ptr);

            // output result
            //// completions or embeddings
            if (ttype != TASK_IMAGES) {
                int32_t status = send_event_json(sink, result_ptr->status, result_ptr->result);
                if (status != httplib::OK_200) {
                    SRV_FUNC_ERR("process", "rid %s | failed to send event response, status = %d\n", rid.c_str(),
                                 status);
                    return false;
                }
                if (result_ptr->status != httplib::Continue_100) {
                    return false;
                }
            }
            //// images
            else {
                std::string b64_json = std::move(result_ptr->result["data"][0]["b64_json"]);
                if (!chunk || b64_json.empty()) {
                    int32_t status = send_event_json(sink, result_ptr->status, result_ptr->result);
                    if (status != httplib::OK_200) {
                        SRV_FUNC_ERR("process", "rid %s | failed to send event response, status = %d\n", rid.c_str(),
                                     status);
                        return false;
                    }
                } else {
                    const int32_t progressed_steps          = result_ptr->result["data"][0]["progressed_steps"];
                    const int32_t progress_steps            = result_ptr->result["data"][0]["progress_steps"];
                    const size_t  chunk_send                = b64_json.size() / chunk_size + 1;
                    const float   chunk_send_progress_base  = float(progressed_steps - 1) / float(progress_steps);
                    const float   chunk_send_progress_scale = 1 / float(progress_steps);
                    size_t        chunk_sent                = 0;
                    while (!b64_json.empty()) {
                        chunk_sent++;
                        float chunk_send_progress = chunk_send_progress_base +
                                                    float(chunk_sent) / float(chunk_send) * chunk_send_progress_scale;
                        result_ptr->result["data"][0]["progress"] = chunk_send_progress * 100;
                        result_ptr->result["data"][0]["b64_json"] = b64_json.substr(0, chunk_size);
                        b64_json                                  = b64_json.substr(chunk_size);
                        // send
                        int32_t status                            = send_event_json(
                            sink, !b64_json.empty() ? httplib::Continue_100 : result_ptr->status, result_ptr->result);
                        if (status != httplib::OK_200) {
                            SRV_FUNC_ERR("process", "rid %s | failed to send event response, status = %d\n",
                                         rid.c_str(), status);
                            return false;
                        }
                    }
                }
                if (result_ptr->status != httplib::Continue_100) {
                    return false;
                }
            }

            return true;
        };
        response.set_header(HEADER_CACHE_CONTROL, "no-cache, no-store, no-transform");
        response.set_header(HEADER_CONNECTION, "close");
        response.set_chunked_content_provider("text/event-stream", on_chunk);
        return httplib::OK_200;
    }

    std::vector<llama_multimodal_tokens> cache_tokenize_multimedia(const char *                        rid,
                                                                   std::unique_ptr<clip_multimedia> && mtmd) {
        if (llm_ctx_clip_v == nullptr && llm_ctx_clip_a == nullptr) {
            return {};
        }

        std::string type = mtmd->is_audio ? "audio" : "image";

        std::vector<llama_multimodal_tokens> result;

        std::unique_lock<std::mutex> lock(llm_ctx_clip_mtx);

        // check if resource hash is empty or cache is disabled.
        if (mtmd->hash.empty() || params.max_projected_cache <= 0) {
            SRV_INFV(2,
                     "rid %s | tokenizing, "
                     "type = %s, hash = %s\n",
                     rid, type.c_str(), mtmd->hash.c_str());
            if (mtmd->is_audio) {
                if (llm_ctx_clip_a == nullptr) {
                    SRV_ERR("rid %s | tokenizing, audio clip is not initialized\n", rid);
                    return result;
                }
                result = tokenize_audio(llm_ctx_clip_a, params.llm_params.cpuparams.n_threads, mtmd->ptr.get());
            } else {
                if (llm_ctx_clip_v == nullptr) {
                    SRV_ERR("rid %s | tokenizing, vision clip is not initialized\n", rid);
                    return result;
                }
                result = tokenize_image(llm_ctx_clip_v, params.llm_params.cpuparams.n_threads, mtmd->ptr.get());
            }
            if (common_log_verbosity_thold >= 2) {
                int32_t n_tokens     = 0;
                int32_t n_pos        = 0;
                size_t  n_embed_size = 0;
                for (const auto & token : result) {
                    n_tokens += token.n_tokens;
                    n_pos += token.n_pos;
                    n_embed_size += token.embed.size() * sizeof(float);
                }
                SRV_INF(
                    "rid %s | tokenized,  "
                    "type = %s, hash = %s, n_tokens = %d, n_pos = %d, n_embed_size = %zu kib\n",
                    rid, type.c_str(), mtmd->hash.c_str(), n_tokens, n_pos, n_embed_size >> 10);
            }
        }
        // check if resource is already cached.
        else if (auto hit = cache_multimodals.find(mtmd->hash); hit != cache_multimodals.end()) {
            hit->second.last_used = ggml_time_us();
            result                = hit->second.tokens;
            if (common_log_verbosity_thold >= 2) {
                int32_t n_tokens     = 0;
                int32_t n_pos        = 0;
                size_t  n_embed_size = 0;
                for (const auto & token : result) {
                    n_tokens += token.n_tokens;
                    n_pos += token.n_pos;
                    n_embed_size += token.embed.size() * sizeof(float);
                }
                SRV_INF(
                    "rid %s | cached,     "
                    "type = %s, hash = %s, n_tokens = %d, n_pos = %d, n_embed_size = %zu kib\n",
                    rid, type.c_str(), mtmd->hash.c_str(), n_tokens, n_pos, n_embed_size >> 10);
            }
        }
        // cache resource.
        else {
            // evict the oldest image if the cache is full.
            if (int32_t(cache_multimodals.size()) >= params.max_projected_cache) {
                // find the oldest image,
                // remove it and add the new image.
                auto oldest_it = cache_multimodals.begin();
                for (auto it = cache_multimodals.begin(); it != cache_multimodals.end(); ++it) {
                    if (it->second.last_used < oldest_it->second.last_used) {
                        oldest_it = it;
                    }
                }
                auto oldest_mtmd = oldest_it->second.tokens;
                if (common_log_verbosity_thold >= 2) {
                    int32_t n_tokens     = 0;
                    int32_t n_pos        = 0;
                    size_t  n_embed_size = 0;
                    for (const auto & token : oldest_mtmd) {
                        n_tokens += token.n_tokens;
                        n_pos += token.n_pos;
                        n_embed_size += token.embed.size() * sizeof(float);
                    }
                    SRV_INF(
                        "rid %s | decached,   "
                        "type = %s, hash = %s, n_tokens = %d, n_pos = %d, n_embed_size = %zu kib\n",
                        rid, type.c_str(), oldest_it->first.c_str(), n_tokens, n_pos, n_embed_size >> 10);
                }
                cache_multimodals.erase(oldest_it);
            }
            SRV_INFV(2,
                     "rid %s | tokenizing, "
                     "type = %s, hash = %s\n",
                     rid, type.c_str(), mtmd->hash.c_str());
            if (mtmd->is_audio) {
                if (llm_ctx_clip_a == nullptr) {
                    SRV_ERR("rid %s | tokenizing, audio clip is not initialized\n", rid);
                    return result;
                }
                result = tokenize_audio(llm_ctx_clip_a, params.llm_params.cpuparams.n_threads, mtmd->ptr.get());
            } else {
                if (llm_ctx_clip_v == nullptr) {
                    SRV_ERR("rid %s | tokenizing, vision clip is not initialized\n", rid);
                    return result;
                }
                result = tokenize_image(llm_ctx_clip_v, params.llm_params.cpuparams.n_threads, mtmd->ptr.get());
            }
            if (common_log_verbosity_thold >= 2) {
                int32_t n_tokens     = 0;
                int32_t n_pos        = 0;
                size_t  n_embed_size = 0;
                for (const auto & token : result) {
                    n_tokens += token.n_tokens;
                    n_pos += token.n_pos;
                    n_embed_size += token.embed.size() * sizeof(float);
                }
                SRV_INF(
                    "rid %s | tokenized,  "
                    "type = %s, hash = %s, n_tokens = %d, n_pos = %d, n_embed_size = %zu kib\n",
                    rid, type.c_str(), mtmd->hash.c_str(), n_tokens, n_pos, n_embed_size >> 10);
            }
            cache_multimodals[mtmd->hash] = { result, ggml_time_us() };
        }

        lock.unlock();

        return result;
    }

    //
    // Routes
    //

    static int32_t handle_health(const httplib::Request & request, httplib::Response & response) {
        json resp = {
            { "status", "ok" },
        };
        return send_json(request, response, httplib::OK_200, resp);
    }

    int32_t handle_metrics(const httplib::Request & request, httplib::Response & response) {
        double   t_image_forwarded_total         = metrics.t_image_forwarded_total.load();
        uint64_t n_mtmd_steps_forwarded_total    = metrics.n_mtmd_steps_forwarded_total.load();
        double   t_image_reversed_total          = metrics.t_image_reversed_total.load();
        uint64_t n_mtmd_steps_reversed_total     = metrics.n_mtmd_steps_reversed_total.load();
        double   t_tokens_prefilled_total        = metrics.t_tokens_prefilled_total.load();
        uint64_t n_tokens_prefilled_total        = metrics.n_tokens_prefilled_total.load();
        double   t_tokens_decoded_total          = metrics.t_tokens_decoded_total.load();
        uint64_t n_tokens_decoded_total          = metrics.n_tokens_decoded_total.load();
        uint64_t n_tokens_drafted_total          = metrics.n_tokens_drafted_total.load();
        uint64_t n_tokens_drafted_accepted_total = metrics.n_tokens_drafted_accepted_total.load();

        const json all_metrics_def = {
            {
             "counter", {
                    /* STABLE DIFFUSION */
                    {
                        { "name", "image_forward_total" },
                        { "help", "Number of image forwarded (steps) in diffusion processing." },
                        { "value", n_mtmd_steps_forwarded_total },
                    },
                    {
                        { "name", "image_forward_seconds_total" },
                        { "help", "Image forward process time." },
                        { "value", t_image_forwarded_total / 1.e3 },
                    },
                    {
                        { "name", "image_reverse_total" },
                        { "help", "Number of image reversed (steps) in diffusion processing." },
                        { "value", n_mtmd_steps_reversed_total },
                    },
                    {
                        { "name", "image_reverse_seconds_total" },
                        { "help", "Image reverse process time." },
                        { "value", t_image_reversed_total / 1.e3 },
                    },

                    /* LLAMA */

                    {
                        { "name", "tokens_prefill_total" },
                        { "help", "Number of prompt tokens processed." },
                        { "value", n_tokens_prefilled_total },
                    },
                    {
                        { "name", "tokens_prefill_seconds_total" },
                        { "help", "Prompt process time." },
                        { "value", t_tokens_prefilled_total / 1.e3 },
                    },
                    {
                        { "name", "tokens_decode_total" },
                        { "help", "Number of generation tokens processed." },
                        { "value", n_tokens_decoded_total },
                    },
                    {
                        { "name", "tokens_decode_seconds_total" },
                        { "help", "Predict process time." },
                        { "value", t_tokens_decoded_total / 1.e3 },
                    },
                    {
                        { "name", "tokens_drafted_total" },
                        { "help", "Number of speculative decoding tokens processed." },
                        { "value", n_tokens_drafted_total },
                    },
                    {
                        { "name", "tokens_drafted_accepted_total" },
                        { "help", "Number of speculative decoding tokens to be accepted." },
                        { "value", n_tokens_drafted_accepted_total },
                    },
                }, },
            {
             "gauge",        {
                    /* STABLE DIFFUSION */

                    {
                        { "name", "image_forward_steps_per_second" },
                        { "help", "Average image forwarded diffusion throughput in steps/s." },
                        { "value", n_mtmd_steps_forwarded_total ?
                                       1.e3 / double(t_image_forwarded_total) * double(n_mtmd_steps_forwarded_total) :
                                       0. },
                    },
                    {
                        { "name", "image_reverse_steps_per_second" },
                        { "help", "Average image reversed diffusion throughput in steps/s." },
                        { "value", n_mtmd_steps_reversed_total ?
                                       1.e3 / double(t_image_reversed_total) * double(n_mtmd_steps_reversed_total) :
                                       0. },
                    },

                    /* LLAMA */

                    {
                        { "name", "tokens_prefill_per_second" },
                        { "help", "Average prompt throughput in tokens/s." },
                        { "value", n_tokens_prefilled_total ?
                                       1.e3 / double(t_tokens_prefilled_total) * double(n_tokens_prefilled_total) :
                                       0. },
                    },
                    {
                        { "name", "tokens_decode_per_second" },
                        { "help", "Average generation throughput in tokens/s." },
                        { "value", n_tokens_decoded_total ?
                                       1.e3 / double(t_tokens_decoded_total) * double(n_tokens_decoded_total) :
                                       0. },
                    },
                    {
                        { "name", "kv_cache_usage_ratio" },
                        { "help", "KV-cache usage. 1 means 100 percent usage." },
                        { "value",
                          support_completion() ? double(llama_kv_self_used_cells(llm_ctx)) / llm_ctx_size : 0 },
                    },
                    {
                        { "name", "kv_cache_tokens" },
                        { "help", "KV-cache tokens." },
                        { "value", support_completion() ? llama_kv_self_n_tokens(llm_ctx) : 0 },
                    },
                }, },
        };

        std::stringstream metrics_stream;
        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();
            for (const auto & metric_def : metrics_def) {
                const std::string & name  = metric_def.at("name");
                const std::string & help  = metric_def.at("help");
                const json &        value = metric_def.at("value");
                metrics_stream << "# HELP llamabox:" << name << " " << help << "\n"
                               << "# TYPE llamabox:" << name << " " << type << "\n"
                               << "llamabox:" << name << " " << value << "\n";
            }
        }

        const std::string metrics_str = metrics_stream.str();
        return send_string(request, response, httplib::OK_200, metrics_str, "text/plain; version=0.0.4");
    }

    int32_t handle_tokenize(const httplib::Request & request, httplib::Response & response) {
        if (!support_tokenize()) {
            return send_string(request, response, httplib::Forbidden_403,
                               "You are not allowed to do tokenize from this model");
        }

        std::unique_ptr<tokenize_req> req = get_tokenize_req(request, response, params.llm_params);

        json tokens_json = json::array();
        {
            llama_tokens tokens = tokenize_prompt(llm_vocab, req->content, req->add_special, true);
            if (req->with_pieces) {
                for (const llama_token & id : tokens) {
                    std::string piece = common_token_to_piece(llm_ctx, id);
                    // if valid UTF-8, store as string
                    if (string_is_utf8(piece)) {
                        tokens_json.push_back({
                            { "id",    id    },
                            { "piece", piece }
                        });
                        continue;
                    }
                    // otherwise, store as array of byte values
                    json piece_json = json::array();
                    for (unsigned char c : piece) {
                        piece_json.push_back(static_cast<int>(c));
                    }
                    tokens_json.push_back({
                        { "id",    id         },
                        { "piece", piece_json }
                    });
                }
            } else {
                tokens_json = tokens;
            }
        }

        json resp = {
            { "model",  req->model  },
            { "tokens", tokens_json },
        };
        return send_json(request, response, httplib::OK_200, resp);
    }

    int32_t handle_detokenize(const httplib::Request & request, httplib::Response & response) {
        if (!support_tokenize()) {
            return send_string(request, response, httplib::Forbidden_403,
                               "You are not allowed to do detokenize from this model");
        }

        std::unique_ptr<detokenize_req> req = get_detokenize_req(request, response, params.llm_params);

        const json content_json = common_detokenize(llm_ctx, req->tokens, false);

        json resp = {
            { "model",   req->model   },
            { "content", content_json },
        };
        return send_json(request, response, httplib::OK_200, resp);
    }

    int32_t handle_lora_adapters(const httplib::Request & request, httplib::Response & response) {
        json resp = json::array();
        for (size_t i = 0; i < params.llm_params.lora_adapters.size(); ++i) {
            common_adapter_lora_info & l = params.llm_params.lora_adapters[i];
            resp.push_back({
                { "id",    i       },
                { "path",  l.path  },
                { "scale", l.scale },
            });
        }
        return send_json(request, response, httplib::OK_200, resp);
    }

    int32_t handle_models(const httplib::Request & request, httplib::Response & response) {
        json metadata_json;
        /* STABLE DIFFUSION */
        if (support_image()) {
            std::pair<int, int> img_size = sd_ctx->get_default_image_size();
            metadata_json                = {
                { "n_slot", params.sd_params.n_parallel },
                { "seed", int32_t(params.sd_params.sampling.seed) },
                { "max_batch_count", params.sd_params.max_batch_count },
                { "max_height", params.sd_params.sampling.height },
                { "max_width", params.sd_params.sampling.width },
                { "default_height", std::min(img_size.first, params.sd_params.sampling.height) },
                { "default_width", std::min(img_size.second, params.sd_params.sampling.width) },
                { "guidance", params.sd_params.sampling.guidance },
                { "strength", params.sd_params.sampling.strength },
                { "sample_method", sd_sample_method_to_argument(params.sd_params.sampling.sample_method) },
                { "sampling_steps", params.sd_params.sampling.sampling_steps },
                { "cfg_scale", params.sd_params.sampling.cfg_scale },
                { "slg_scale", params.sd_params.sampling.slg_scale },
                { "slg_skip_layers", params.sd_params.sampling.slg_skip_layers },
                { "slg_start", params.sd_params.sampling.slg_start },
                { "slg_end", params.sd_params.sampling.slg_end },
                { "schedule_method", sd_schedule_to_argument(params.sd_params.sampling.schedule_method) },
                { "negative_prompt", params.sd_params.sampling.negative_prompt },
            };
        }
        /* LLAMA */
        else {
            metadata_json = {
                { "vocab_type",            llama_vocab_type(llm_vocab)                      },
                { "n_vocab",               llama_vocab_n_tokens(llm_vocab)                  },
                { "n_ctx_train",           llama_model_n_ctx_train(llm_model)               },
                { "n_embd",                llama_model_n_embd(llm_model)                    },
                { "n_params",              llama_model_n_params(llm_model)                  },
                { "size",                  llama_model_size(llm_model)                      },
                { "n_ctx",                 llm_ctx_size                                     },
                { "n_slot",                1                                                },
                { "n_slot_ctx",            llm_ctx_size                                     },
                { "ctx_shift",             shift_context                                    },
                { "prompt_cache",          cache_prompt                                     },
                { "seed",                  int32_t(params.llm_params.sampling.seed)         },
                { "temperature",           params.llm_params.sampling.temp                  },
                { "dynatemp_range",        params.llm_params.sampling.dynatemp_range        },
                { "dynatemp_exponent",     params.llm_params.sampling.dynatemp_exponent     },
                { "top_k",                 params.llm_params.sampling.top_k                 },
                { "top_p",                 params.llm_params.sampling.top_p                 },
                { "min_p",                 params.llm_params.sampling.min_p                 },
                { "top_n_sigma",           params.llm_params.sampling.top_n_sigma           },
                { "xtc_probability",       params.llm_params.sampling.xtc_probability       },
                { "xtc_threshold",         params.llm_params.sampling.xtc_threshold         },
                { "typical_p",             params.llm_params.sampling.typ_p                 },
                { "repeat_last_n",         params.llm_params.sampling.penalty_last_n        },
                { "repeat_penalty",        params.llm_params.sampling.penalty_repeat        },
                { "presence_penalty",      params.llm_params.sampling.penalty_present       },
                { "frequency_penalty",     params.llm_params.sampling.penalty_freq          },
                { "dry_multiplier",        params.llm_params.sampling.dry_multiplier        },
                { "dry_base",              params.llm_params.sampling.dry_base              },
                { "dry_allowed_length",    params.llm_params.sampling.dry_allowed_length    },
                { "dry_penalty_last_n",    params.llm_params.sampling.dry_penalty_last_n    },
                { "dry_sequence_breakers", params.llm_params.sampling.dry_sequence_breakers },
                { "mirostat",              params.llm_params.sampling.mirostat              },
                { "mirostat_tau",          params.llm_params.sampling.mirostat_tau          },
                { "mirostat_eta",          params.llm_params.sampling.mirostat_eta          },
                { "support_vision",        llm_ctx_clip_v != nullptr                        },
                { "support_audio",         llm_ctx_clip_a != nullptr                        },
                { "support_speculative",   llm_ctx_draft != nullptr                         },
                { "support_tool_calls",    support_tool_calls                               },
                { "support_reasoning",     support_reasoning                                },
            };
        }

        json resp = {
            { "object", "list" },
            {
             "data", {
                    {
                        { "id", support_image() ? params.sd_params.model_alias : params.llm_params.model_alias },
                        { "object", "model" },
                        { "created", std::time(nullptr) },
                        { "owned_by", "llama-box" },
                        { "meta", metadata_json },
                    },
                }, },
        };
        return send_json(request, response, httplib::OK_200, resp);
    }

    int32_t handle_legacy_completions(const httplib::Request & request, httplib::Response & response) {
        if (!support_completion()) {
            return send_string(request, response, httplib::Forbidden_403,
                               "You are not allowed to do completions from this model");
        }

        std::unique_ptr<RatelimitTokenBucket> token_bucket = nullptr;
        if (params.n_tps > 0) {
            std::string tps_str = request.get_header_value(HEADER_X_REQUEST_TOKENS_PER_SECOND, "");
            if (!tps_str.empty()) {
                int32_t tps = params.n_tps;
                try {
                    tps = std::stoi(tps_str);
                } catch (std::exception &) {
                }
                // if the request less than or equals to 0, use the default tokens per second.
                if (tps <= 0) {
                    tps = params.n_tps;
                }
                // if the request exceeds the maximum tokens per second, return 410 Gone
                else if (tps > params.n_tps) {
                    return send_string(request, response, httplib::Gone_410,
                                       "The request exceeds the maximum tokens per second");
                }
                token_bucket = std::make_unique<RatelimitTokenBucket>(tps, tps);
            }
        }

        std::unique_ptr<legacy_complete_req> req = get_legacy_complete_req(request, response, params, llm_ctx);

        int32_t n_prefilling_request = 0;

        std::vector<std::variant<llama_tokens, llama_multimodal_tokens>> tokenized_prompts;
        /* PLAIN TEXT */
        {
            llama_tokens tokenized_prompt = tokenize_prompt(llm_vocab, req->prompt, true, true);
            n_prefilling_request          = int32_t(tokenized_prompt.size());
            if (n_prefilling_request >= llm_ctx_size) {
                if (!shift_context) {
                    SRV_ERR(
                        "rid %s | prompt tokens size exceeds the context size, please enable context shift "
                        "or reduce prompt size, prefill_t = %d, n_ctx = %d\n",
                        req->get_id(), n_prefilling_request, llm_ctx_size);
                    return send_json(request, response, httplib::BadRequest_400,
                                     "Illegal param: prompt tokens size exceeds the context size");
                }
                const int32_t n_left       = llm_ctx_size - params.llm_params.n_keep;
                const int32_t n_block_size = n_left >> 1;
                const int32_t n_block_erased =
                    (n_prefilling_request - params.llm_params.n_keep - n_block_size) / n_block_size;
                if (n_block_erased > 0) {
                    SRV_WRN(
                        "rid %s | prompt tokens size exceeds the context size, "
                        "shifting context [%d, %d) -> [%d, %d)\n",
                        req->get_id(), params.llm_params.n_keep + n_block_size * n_block_erased, n_prefilling_request,
                        params.llm_params.n_keep, n_prefilling_request - n_block_size * n_block_erased);
                    tokenized_prompt.erase(
                        tokenized_prompt.begin() + params.llm_params.n_keep,
                        tokenized_prompt.begin() + params.llm_params.n_keep + n_block_size * n_block_erased);
                }
                n_prefilling_request = int32_t(tokenized_prompt.size());
            }
            tokenized_prompts.emplace_back(std::move(tokenized_prompt));
        }

        if (n_prefilling_request == 0) {
            return send_json(request, response, httplib::BadRequest_400, "Illegal param: empty completions tokens");
        }

        int32_t n_decoding_budget = llm_ctx_size;
        if (req->max_tokens > 0) {
            n_decoding_budget = req->max_tokens;
        } else if (req->max_tokens < 0) {
            n_decoding_budget = INT32_MAX;
        }

        common_sampler * sampler       = nullptr;
        common_sampler * sampler_draft = nullptr;
        {
            sampler = common_sampler_init(llm_model, req->sampling);
            if (sampler == nullptr) {
                return send_json(request, response, httplib::BadRequest_400, "Illegal param: \"sampling\" is invalid");
            }
            if (llm_ctx_draft != nullptr) {
                common_params_sampling sampling_draft = {};
                sampling_draft.seed                   = req->sampling.seed;
                sampling_draft.no_perf                = false;
                sampling_draft.top_k                  = 10;
                sampling_draft.samplers               = {
                    COMMON_SAMPLER_TYPE_TOP_K,
                };
                sampler_draft = common_sampler_init(llm_model, sampling_draft);
                if (sampler_draft == nullptr) {
                    common_sampler_free(sampler);
                    return send_json(request, response, httplib::BadRequest_400,
                                     "Illegal param: \"sampling\" is invalid");
                }
            }
        }

        std::unique_ptr<completions_task> task =
            std::make_unique<completions_task>(get_task_id(), request.is_connection_closed);
        task->token_bucket         = std::move(token_bucket);
        task->tokenized_prompts    = std::move(tokenized_prompts);
        task->n_prefilling_request = n_prefilling_request;
        task->n_decoding_budget    = n_decoding_budget;
        task->sampler              = sampler;
        task->sampler_draft        = sampler_draft;
        task->cmpl_id              = gen_completion_id();
        task->reasoning_finished   = !support_reasoning;
        task->req                  = std::move(req);
        task->t_start_prefill      = ggml_time_us();

        SRV_INFV(2, "rid %s | prefill_t = %d, decode_budget = %d\n", task->get_r_id().c_str(),
                 task->n_prefilling_request, task->n_decoding_budget);

        return process(request, response, std::move(task));
    }

    int32_t handle_chat_completions(const httplib::Request & request, httplib::Response & response) {
        if (!support_completion()) {
            return send_string(request, response, httplib::Forbidden_403,
                               "You are not allowed to do chat operation from this model");
        }

        std::unique_ptr<RatelimitTokenBucket> token_bucket = nullptr;
        if (params.n_tps > 0) {
            std::string tps_str = request.get_header_value(HEADER_X_REQUEST_TOKENS_PER_SECOND, "");
            if (!tps_str.empty()) {
                int32_t tps = params.n_tps;
                try {
                    tps = std::stoi(tps_str);
                } catch (std::exception &) {
                }
                // if the request less than or equals to 0, use the default tokens per second.
                if (tps <= 0) {
                    tps = params.n_tps;
                }
                // if the request exceeds the maximum tokens per second, return 410 Gone
                else if (tps > params.n_tps) {
                    return send_string(request, response, httplib::Gone_410,
                                       "The request exceeds the maximum tokens per second");
                }
                token_bucket = std::make_unique<RatelimitTokenBucket>(tps, tps);
            }
        }

        std::unique_ptr<chat_complete_req> req =
            get_chat_complete_req(request, response, params, llm_ctx, support_tool_calls, chat_templates.get());

        int32_t n_prefilling_request = 0;

        std::vector<std::variant<llama_tokens, llama_multimodal_tokens>> tokenized_prompts;
        /* PLAIN TEXT */
        if (req->multimedias.empty()) {
            llama_tokens tokenized_prompt = tokenize_prompt(llm_vocab, req->chat_params.prompt, true, true);
            n_prefilling_request          = int32_t(tokenized_prompt.size());
            if (n_prefilling_request >= llm_ctx_size) {
                if (!shift_context) {
                    SRV_ERR(
                        "rid %s | prompt tokens size exceeds the context size, please enable context shift "
                        "or reduce prompt size, prefill_t = %d, n_ctx = %d\n",
                        req->get_id(), n_prefilling_request, llm_ctx_size);
                    return send_json(request, response, httplib::BadRequest_400,
                                     "Illegal param: prompt tokens size exceeds the context size");
                }
                const int32_t n_left       = llm_ctx_size - params.llm_params.n_keep;
                const int32_t n_block_size = n_left >> 1;
                const int32_t n_block_erased =
                    (n_prefilling_request - params.llm_params.n_keep - n_block_size) / n_block_size;
                if (n_block_erased > 0) {
                    SRV_WRN(
                        "rid %s | prompt tokens size exceeds the context size, "
                        "shifting context [%d, %d) -> [%d, %d)\n",
                        req->get_id(), params.llm_params.n_keep + n_block_size * n_block_erased, n_prefilling_request,
                        params.llm_params.n_keep, n_prefilling_request - n_block_size * n_block_erased);
                    tokenized_prompt.erase(
                        tokenized_prompt.begin() + params.llm_params.n_keep,
                        tokenized_prompt.begin() + params.llm_params.n_keep + n_block_size * n_block_erased);
                }
                n_prefilling_request = int32_t(tokenized_prompt.size());
            }
            tokenized_prompts.emplace_back(std::move(tokenized_prompt));
        }
        /* VISION */
        else {
            const std::string mtmd_sign = "<MTMD/>";

            std::string prompt = req->chat_params.prompt;

            auto    n_mtmd   = int32_t(req->multimedias.size());
            int32_t i_mtmd   = -1;
            size_t  mtmd_pos = prompt.find(mtmd_sign);
            while (mtmd_pos != std::string::npos && ++i_mtmd < n_mtmd) {
                // process text
                if (const std::string text = prompt.substr(0, mtmd_pos); !text.empty()) {
                    llama_tokens tokenized_text = common_tokenize(llm_vocab, text, tokenized_prompts.empty(), true);
                    n_prefilling_request += int32_t(tokenized_text.size());
                    tokenized_prompts.emplace_back(std::move(tokenized_text));
                }

                // process multimedia
                std::vector<llama_multimodal_tokens> tokenized_mtmds =
                    cache_tokenize_multimedia(req->get_id(), std::move(req->multimedias[i_mtmd]));
                if (tokenized_mtmds.empty()) {
                    return send_string(request, response, httplib::InternalServerError_500,
                                       "Failed to embed the multimedia");
                }
                bool add_bos = tokenized_prompts.empty();
                //// vision
                if (!tokenized_mtmds.front().is_audio) {
                    // minicpmv
                    if (clip_is_minicpmv(llm_ctx_clip_v) != 0) {
                        // format:
                        // 2.5: <image> (overview image) </image>
                        //      <slice>
                        //          <image> (tile image) </image> ... <image> (tile image) </image>\n
                        //          <image> (tile image) </image> ... <image> (tile image) </image>\n
                        //          ...
                        //      </slice>
                        // 2.6: <image> (overview image) </image>
                        //      <slice> (tile image) </slice> <slice> (tile image) </slice>\n
                        //      <slice> (tile image) </slice> <slice> (tile image) </slice>\n
                        //      ...

                        // <image>
                        llama_tokens tokenized_text = common_tokenize(llm_vocab, "<image>", add_bos, true);
                        n_prefilling_request += int32_t(tokenized_text.size());
                        tokenized_prompts.emplace_back(std::move(tokenized_text));
                        // overview <MTMD/>
                        llama_multimodal_tokens overview_tokenized_image = tokenized_mtmds.front();
                        clip_image_size         grid_size                = overview_tokenized_image.grid_size;
                        n_prefilling_request += int32_t(overview_tokenized_image.n_pos);
                        tokenized_prompts.emplace_back(std::move(overview_tokenized_image));
                        // </image>
                        tokenized_text = common_tokenize(llm_vocab, "</image>", false, true);
                        n_prefilling_request += int32_t(tokenized_text.size());
                        tokenized_prompts.emplace_back(std::move(tokenized_text));
                        // process tile images
                        tokenized_mtmds.erase(tokenized_mtmds.begin());
                        if (!tokenized_mtmds.empty()) {
                            const int32_t n_x     = grid_size.width;
                            const int32_t n_y     = grid_size.height;
                            const int32_t version = clip_is_minicpmv(llm_ctx_clip_v);
                            if (version < 3) {
                                // <slice>
                                tokenized_text = common_tokenize(llm_vocab, "<slice>", false, true);
                                n_prefilling_request += int32_t(tokenized_text.size());
                                tokenized_prompts.emplace_back(std::move(tokenized_text));
                            }
                            std::string ifmt = "<slice>";
                            std::string ofmt = "</slice>";
                            if (version < 3) {
                                ifmt = "<image>";
                                ofmt = "</image>";
                            }
                            for (int y = 0; y < n_y; y++) {
                                for (int x = 0; x < n_x; x++) {
                                    // <slice> | <image>
                                    tokenized_text = common_tokenize(llm_vocab, ifmt, false, true);
                                    n_prefilling_request += int32_t(tokenized_text.size());
                                    tokenized_prompts.emplace_back(std::move(tokenized_text));
                                    // tile <MTMD/>
                                    llama_multimodal_tokens tokenized_mtmd = tokenized_mtmds[y * n_x + x];
                                    n_prefilling_request += int32_t(tokenized_mtmd.n_pos);
                                    tokenized_prompts.emplace_back(std::move(tokenized_mtmd));
                                    // </slice> | </image>
                                    tokenized_text = common_tokenize(llm_vocab, ofmt, false, true);
                                    n_prefilling_request += int32_t(tokenized_text.size());
                                    tokenized_prompts.emplace_back(std::move(tokenized_text));
                                }
                                if (y != n_y - 1) {
                                    // \n
                                    tokenized_text = common_tokenize(llm_vocab, "\n", false, true);
                                    n_prefilling_request += int32_t(tokenized_text.size());
                                    tokenized_prompts.emplace_back(std::move(tokenized_text));
                                }
                            }
                            if (version < 3) {
                                // </slice>
                                tokenized_text = common_tokenize(llm_vocab, "</slice>", false, true);
                                n_prefilling_request += int32_t(tokenized_text.size());
                                tokenized_prompts.emplace_back(std::move(tokenized_text));
                            }
                        }
                    }
                    // llama4
                    // NB(thxCode): clip_is_llama4 is a patch.
                    else if (clip_is_llama4(llm_ctx_clip_v)) {
                        // format:
                        // <|image_start|>
                        // (tile image) <|tile_x_separator|> (tile image) <|tile_x_separator|> ... <|tile_y_separator|>
                        // (tile image) <|tile_x_separator|> (tile image) <|tile_x_separator|> ... <|tile_y_separator|>
                        // ...
                        // <|tile_y_separator|>
                        // <|image|> (overview image)
                        // <|image_end|>

                        // <|image_start|>
                        llama_tokens tokenized_text = common_tokenize(llm_vocab, "<|image_start|>", add_bos, true);
                        n_prefilling_request += int32_t(tokenized_text.size());
                        tokenized_prompts.emplace_back(std::move(tokenized_text));
                        llama_multimodal_tokens overview_tokenized_image = tokenized_mtmds.front();
                        clip_image_size         grid_size                = overview_tokenized_image.grid_size;
                        // process tile images
                        tokenized_mtmds.erase(tokenized_mtmds.begin());
                        if (!tokenized_mtmds.empty()) {
                            const int32_t n_x = grid_size.width;
                            const int32_t n_y = grid_size.height;
                            for (int y = 0; y < n_y; y++) {
                                for (int x = 0; x < n_x; x++) {
                                    // slice <MTMD/>
                                    llama_multimodal_tokens tokenized_mtmd = tokenized_mtmds[y * n_x + x];
                                    n_prefilling_request += int32_t(tokenized_mtmd.n_pos);
                                    tokenized_prompts.emplace_back(std::move(tokenized_mtmd));
                                    // <|tile_x_separator|>
                                    if (x != n_x - 1) {
                                        tokenized_text =
                                            common_tokenize(llm_vocab, "<|tile_x_separator|>", false, true);
                                        n_prefilling_request += int32_t(tokenized_text.size());
                                        tokenized_prompts.emplace_back(std::move(tokenized_text));
                                    }
                                }
                                // <|tile_y_separator|>
                                tokenized_text = common_tokenize(llm_vocab, "<|tile_y_separator|>", false, true);
                                n_prefilling_request += int32_t(tokenized_text.size());
                                tokenized_prompts.emplace_back(std::move(tokenized_text));
                            }
                        }
                        // <|image|>
                        tokenized_text = common_tokenize(llm_vocab, "<|image|>", add_bos, true);
                        n_prefilling_request += int32_t(tokenized_text.size());
                        tokenized_prompts.emplace_back(std::move(tokenized_text));
                        // overview <MTMD/>
                        n_prefilling_request += int32_t(overview_tokenized_image.n_pos);
                        tokenized_prompts.emplace_back(std::move(overview_tokenized_image));
                        // <|image_end|>
                        tokenized_text = common_tokenize(llm_vocab, "<|image_end|>", false, true);
                        n_prefilling_request += int32_t(tokenized_text.size());
                        tokenized_prompts.emplace_back(std::move(tokenized_text));
                    }
                    // qwen2vl
                    else if (clip_is_qwen2vl(llm_ctx_clip_v)) {
                        // format:
                        // <|vision_start|> (image) <|vision_end|>

                        // <|vision_start|>
                        llama_tokens tokenized_text = common_tokenize(llm_vocab, "<|vision_start|>", add_bos, true);
                        n_prefilling_request += int32_t(tokenized_text.size());
                        tokenized_prompts.emplace_back(std::move(tokenized_text));
                        // <MTMD/>
                        llama_multimodal_tokens tokenized_mtmd = tokenized_mtmds.front();
                        n_prefilling_request += int32_t(tokenized_mtmd.n_pos);
                        tokenized_prompts.emplace_back(std::move(tokenized_mtmd));
                        // <|vision_end|>
                        tokenized_text = common_tokenize(llm_vocab, "<|vision_end|>", false, true);
                        n_prefilling_request += int32_t(tokenized_text.size());
                        tokenized_prompts.emplace_back(std::move(tokenized_text));
                    }
                    // gemma3
                    else if (clip_is_gemma3(llm_ctx_clip_v)) {
                        // format:
                        // <|start_of_image|> (image) <|end_of_image|>

                        // <|start_of_image|>
                        llama_tokens tokenized_text = common_tokenize(llm_vocab, "<|start_of_image|>", add_bos, true);
                        n_prefilling_request += int32_t(tokenized_text.size());
                        tokenized_prompts.emplace_back(std::move(tokenized_text));
                        // <MTMD/>
                        llama_multimodal_tokens tokenized_mtmd = tokenized_mtmds.front();
                        n_prefilling_request += int32_t(tokenized_mtmd.n_pos);
                        tokenized_prompts.emplace_back(std::move(tokenized_mtmd));
                        // <|end_of_image|>
                        tokenized_text = common_tokenize(llm_vocab, "<|end_of_image|>", false, true);
                        n_prefilling_request += int32_t(tokenized_text.size());
                        tokenized_prompts.emplace_back(std::move(tokenized_text));
                    }
                    // smolvlm
                    // NB(thxCode): clip_is_smolvlm is a patch.
                    else if (clip_is_smolvlm(llm_ctx_clip_v)) {
                        // format:
                        // <fake_token_around_image><global-img> (image) <fake_token_around_image>

                        // <fake_token_around_image><global-img>
                        llama_tokens tokenized_text =
                            common_tokenize(llm_vocab, "<fake_token_around_image><global-img>", add_bos, true);
                        n_prefilling_request += int32_t(tokenized_text.size());
                        tokenized_prompts.emplace_back(std::move(tokenized_text));
                        // <MTMD/>
                        llama_multimodal_tokens tokenized_mtmd = tokenized_mtmds.front();
                        n_prefilling_request += int32_t(tokenized_mtmd.n_pos);
                        tokenized_prompts.emplace_back(std::move(tokenized_mtmd));
                        // <fake_token_around_image>
                        tokenized_text = common_tokenize(llm_vocab, "<fake_token_around_image>", false, true);
                        n_prefilling_request += int32_t(tokenized_text.size());
                        tokenized_prompts.emplace_back(std::move(tokenized_text));
                    }
                    // pixtral
                    // NB(thxCode): clip_is_pixtral is a patch.
                    else if (clip_is_pixtral(llm_ctx_clip_v)) {
                        // format:
                        // (image) [IMG_END]

                        // <MTMD/>
                        llama_multimodal_tokens tokenized_mtmd = tokenized_mtmds.front();
                        n_prefilling_request += int32_t(tokenized_mtmd.n_pos);
                        tokenized_prompts.emplace_back(std::move(tokenized_mtmd));
                        // [IMG_END]
                        llama_tokens tokenized_text = common_tokenize(llm_vocab, "[IMG_END]", false, true);
                        n_prefilling_request += int32_t(tokenized_text.size());
                        tokenized_prompts.emplace_back(std::move(tokenized_text));
                    }
                    // internvl
                    // NB(thxCode): clip_is_internvl is a patch.
                    else if (clip_is_internvl(llm_ctx_clip_v)) {
                        // format:
                        // <img> (image) </img>

                        // <img>
                        llama_tokens tokenized_text = common_tokenize(llm_vocab, "<img>", add_bos, true);
                        n_prefilling_request += int32_t(tokenized_text.size());
                        tokenized_prompts.emplace_back(std::move(tokenized_text));
                        // <MTMD/>
                        llama_multimodal_tokens tokenized_mtmd = tokenized_mtmds.front();
                        n_prefilling_request += int32_t(tokenized_mtmd.n_pos);
                        tokenized_prompts.emplace_back(std::move(tokenized_mtmd));
                        // </img>
                        tokenized_text = common_tokenize(llm_vocab, "</img>", false, true);
                        n_prefilling_request += int32_t(tokenized_text.size());
                        tokenized_prompts.emplace_back(std::move(tokenized_text));
                    }
                    // others
                    else {
                        // <MTMD/>
                        for (llama_multimodal_tokens & tokenized_mtmd : tokenized_mtmds) {
                            n_prefilling_request += int32_t(tokenized_mtmd.n_pos);
                            tokenized_prompts.emplace_back(std::move(tokenized_mtmd));
                        }
                    }
                }
                //// audio
                else {
                    // qwen2 audio
                    // NB(thxCode): clip_is_qwen2a is a patch.
                    if (clip_is_qwen2a(llm_ctx_clip_a)) {
                        // format:
                        // <|audio_bos|> (audio) <|audio_eos|>

                        // <|audio_bos|>
                        llama_tokens tokenized_text = common_tokenize(llm_vocab, "<|audio_bos|>", add_bos, true);
                        n_prefilling_request += int32_t(tokenized_text.size());
                        tokenized_prompts.emplace_back(std::move(tokenized_text));
                        // <MTMD/>
                        for (llama_multimodal_tokens & tokenized_mtmd : tokenized_mtmds) {
                            n_prefilling_request += int32_t(tokenized_mtmd.n_pos);
                            tokenized_prompts.emplace_back(std::move(tokenized_mtmd));
                        }
                        // <|audio_eos|>
                        tokenized_text = common_tokenize(llm_vocab, "<|audio_eos|>", false, true);
                        n_prefilling_request += int32_t(tokenized_text.size());
                        tokenized_prompts.emplace_back(std::move(tokenized_text));
                    }
                    // others
                    else {
                        // <MTMD/>
                        for (llama_multimodal_tokens & tokenized_mtmd : tokenized_mtmds) {
                            n_prefilling_request += int32_t(tokenized_mtmd.n_pos);
                            tokenized_prompts.emplace_back(std::move(tokenized_mtmd));
                        }
                    }
                }

                prompt   = prompt.substr(mtmd_pos + mtmd_sign.size());
                mtmd_pos = prompt.find(mtmd_sign);
            }

            // process remain text
            if (!prompt.empty()) {
                llama_tokens tokenized_text = common_tokenize(llm_vocab, prompt, false, true);
                n_prefilling_request += int32_t(tokenized_text.size());
                tokenized_prompts.emplace_back(std::move(tokenized_text));
            }

            if (n_prefilling_request >= llm_ctx_size) {
                SRV_ERR(
                    "rid %s | prompt tokens size exceeds the context size, please increase the context size "
                    "or reduce prompt size, prefill_t = %d, n_ctx = %d\n",
                    req->get_id(), n_prefilling_request, llm_ctx_size);
                return send_json(request, response, httplib::BadRequest_400,
                                 "Illegal param: prompt tokens size exceeds the context size");
            }
        }

        if (n_prefilling_request == 0) {
            return send_json(request, response, httplib::BadRequest_400, "Illegal param: empty completions tokens");
        }

        bool tokenized_prompts_include_multimedias = !req->multimedias.empty();
        req->multimedias.clear();  // release multimedias asap
        req->multimedias.shrink_to_fit();

        bool tokenized_prompts_include_tools = !req->tools.empty();

        int32_t n_decoding_budget = llm_ctx_size;
        if (req->max_tokens > 0) {
            n_decoding_budget = req->max_tokens;
        } else if (req->max_tokens < 0) {
            n_decoding_budget = INT32_MAX;
        }

        common_sampler * sampler       = nullptr;
        common_sampler * sampler_draft = nullptr;
        {
            sampler = common_sampler_init(llm_model, req->sampling);
            if (sampler == nullptr) {
                return send_json(request, response, httplib::BadRequest_400, "Illegal param: \"sampling\" is invalid");
            }
            if (llm_ctx_draft != nullptr) {
                common_params_sampling sampling_draft;
                sampling_draft.seed     = req->sampling.seed;
                sampling_draft.no_perf  = false;
                sampling_draft.top_k    = 10;
                sampling_draft.samplers = {
                    COMMON_SAMPLER_TYPE_TOP_K,
                };
                sampler_draft = common_sampler_init(llm_model, sampling_draft);
                if (sampler_draft == nullptr) {
                    common_sampler_free(sampler);
                    return send_json(request, response, httplib::BadRequest_400,
                                     "Illegal param: \"sampling\" is invalid");
                }
            }
        }

        bool tool_call_stop_fast = !req->parallel_tool_calls || req->tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

        // NB(thxCode): disable reasoning process if we need to generate tool calls in jinja.
        bool reasoning_finished = !support_reasoning || (params.llm_params.use_jinja &&
                                                         req->tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED &&
                                                         tokenized_prompts_include_tools);

        std::unique_ptr<completions_task> task =
            std::make_unique<completions_task>(get_task_id(), request.is_connection_closed);
        task->token_bucket                          = std::move(token_bucket);
        task->tokenized_prompts                     = std::move(tokenized_prompts);
        task->tokenized_prompts_syntax.format       = req->chat_params.format;
        task->tokenized_prompts_include_multimedias = tokenized_prompts_include_multimedias;
        task->tokenized_prompts_include_tools       = tokenized_prompts_include_tools;
        task->n_decoding_budget                     = n_decoding_budget;
        task->n_prefilling_request                  = n_prefilling_request;
        task->sampler                               = sampler;
        task->sampler_draft                         = sampler_draft;
        task->tool_call_stop_fast                   = tool_call_stop_fast;
        task->cmpl_id                               = gen_chat_completion_id();
        task->reasoning_finished                    = reasoning_finished;
        task->req                                   = std::move(req);
        task->t_start_prefill                       = ggml_time_us();

        SRV_INFV(2, "rid %s | prefill_t = %d, decode_budget = %d\n", task->get_r_id().c_str(),
                 task->n_prefilling_request, task->n_decoding_budget);

        return process(request, response, std::move(task));
    }

    int32_t handle_embeddings(const httplib::Request & request, httplib::Response & response) {
        if (!support_embedding()) {
            return send_json(request, response, httplib::Forbidden_403,
                             "You are not allowed to do embedding from this model");
        }

        std::unique_ptr<embed_req> req = get_embed_req(request, response, params);

        const llama_token tok_eos = llama_vocab_eos(llm_vocab);

        int32_t n_prefilling_request = 0;

        std::vector<llama_tokens> tokenized_inputs = tokenize_prompts(llm_vocab, req->input, true, true);
        for (size_t i = 0; i < tokenized_inputs.size(); i++) {
            if (need_end_eos || tokenized_inputs[i].empty()) {
                tokenized_inputs[i].push_back(tok_eos);
            }
            auto n_pos = int32_t(tokenized_inputs[i].size());
            if (n_pos > llm_ctx_size) {
                if (!shift_context) {
                    return send_json(request, response, httplib::BadRequest_400,
                                     "Illegal param: \"input\" tokens size exceeds the context size");
                }
                SRV_WRN(
                    "rid %s | input item %zu tokens size exceeds the context size, "
                    "shifting context [%d, %d) -> [0, %d)\n",
                    req->get_id(), i, n_pos - llm_ctx_size, n_pos, llm_ctx_size);
                tokenized_inputs[i].erase(tokenized_inputs[i].begin(), tokenized_inputs[i].end() - llm_ctx_size);
                n_pos = llm_ctx_size;
            }
            n_prefilling_request += n_pos;
        }

        if (n_prefilling_request == 0) {
            return send_json(request, response, httplib::BadRequest_400, "Illegal param: empty embedding tokens");
        }

        std::unique_ptr<embeddings_task> task =
            std::make_unique<embeddings_task>(get_task_id(), request.is_connection_closed);
        task->tokenized_inputs     = std::move(tokenized_inputs);
        task->n_prefilling_request = n_prefilling_request;
        task->req                  = std::move(req);
        task->t_start_prefill      = ggml_time_us();

        return process(request, response, std::move(task));
    }

    int32_t handle_rerank(const httplib::Request & request, httplib::Response & response) {
        if (!support_reranking()) {
            return send_json(request, response, httplib::Forbidden_403,
                             "You are not allowed to do reranking from this model");
        }

        std::unique_ptr<rerank_req> req = get_rerank_req(request, response, params);

        const llama_token tok_bos        = llama_vocab_bos(llm_vocab);
        const llama_token tok_eos        = llama_vocab_eos(llm_vocab);
        const llama_token tok_sep        = llama_vocab_sep(llm_vocab);
        const size_t      n_tok_addition = 4;

        int32_t n_prefilling_request = 0;

        llama_tokens tokenized_query = tokenize_prompt(llm_vocab, req->query, false, true);
        if (req->normalize && tokenized_query.size() * 2 + n_tok_addition > size_t(llm_ctx_size)) {
            return send_json(
                request, response, httplib::BadRequest_400,
                R"(Illegal param: "query" length exceeds the context size, disable "normalize" to bypass this check)");
        }
        auto decorate = [&](const llama_tokens & tokenized_document) {
            auto n_pos = int32_t(tokenized_query.size() + tokenized_document.size() + n_tok_addition);
            if (n_pos > llm_ctx_size) {
                throw std::invalid_argument(
                    R"(Illegal param: the sum of the lengths of "query" and "document" exceeds the context size)");
            }
            n_prefilling_request += n_pos;
            // format input: [BOS]query[EOS][SEP]document[EOS]
            llama_tokens tokenized_input;
            tokenized_input.reserve(n_pos);
            tokenized_input.push_back(tok_bos);
            tokenized_input.insert(tokenized_input.end(), tokenized_query.begin(), tokenized_query.end());
            tokenized_input.push_back(tok_sep);
            tokenized_input.insert(tokenized_input.end(), tokenized_document.begin(), tokenized_document.end());
            tokenized_input.push_back(tok_sep);
            tokenized_input.push_back(tok_eos);
            return tokenized_input;
        };
        std::vector<llama_tokens> tokenized_inputs;
        // tokenized_inputs[0] is the query with document 0,
        // tokenized_inputs[1] is the query with document 1,
        // ...,
        // tokenized_inputs[length-2] is the query with itself when normalizing,
        // tokenized_inputs[length-1] is the query with unknown token when normalizing.
        for (const json & document : req->documents) {
            llama_tokens tokenized_document = tokenize_prompt(llm_vocab, document, false, true);
            tokenized_inputs.emplace_back(decorate(tokenized_document));
        }
        if (req->normalize) {
            tokenized_inputs.emplace_back(decorate(tokenized_query));
            // NB(thxCode): llama_vocab_unk is a patch.
            tokenized_inputs.emplace_back(decorate({ llama_vocab_unk(llm_vocab) }));
        }

        if (n_prefilling_request == 0) {
            return send_json(request, response, httplib::BadRequest_400, "Illegal param: empty reranking tokens");
        }

        std::unique_ptr<embeddings_task> task =
            std::make_unique<embeddings_task>(get_task_id(), request.is_connection_closed);
        task->tokenized_inputs     = std::move(tokenized_inputs);
        task->n_prefilling_request = n_prefilling_request;
        task->req                  = std::move(req);
        task->t_start_prefill      = ggml_time_us();

        return process(request, response, std::move(task));
    }

    int32_t handle_images(const httplib::Request & request, httplib::Response & response) {
        if (!support_image()) {
            return send_json(request, response, httplib::Forbidden_403,
                             "You are not allowed to do image operation from this model");
        }

        const std::string category = request.path_params.at("category");
        if (category != "generations" && category != "edits") {
            return send_json(request, response, httplib::Forbidden_403,
                             "You are not allowed to do image operation from this model");
        }

        std::unique_ptr<images_task> task = std::make_unique<images_task>(get_task_id(), request.is_connection_closed);
        if (category == "generations") {
            std::unique_ptr<image_generate_req> req = get_image_generate_req(request, response, params);
            task->req                               = std::move(req);
        } else {
            if (!request.is_multipart_form_data()) {
                return send_json(request, response, httplib::BadRequest_400,
                                 "Illegal request: multipart/form-data content type is required");
            }
            std::unique_ptr<image_edit_req> req = get_image_edit_req(request, response, params);
            task->req                           = std::move(req);
        }
        task->t_start_forward = ggml_time_us();

        return process(request, response, std::move(task));
    }
};

static int32_t start_httpserver(httpserver_params & params) {
    httpserver srv(params);

    if (!srv.load()) {
        SRV_ERR("%s", "failed to load\n");
        return -1;
    }

    return srv.start();
}
