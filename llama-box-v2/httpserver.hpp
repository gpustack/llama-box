// heads

#include <atomic>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <fstream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

#include "hv/HttpClient.h"
#include "hv/HttpServer.h"
#include "hv/hasync.h"
#include "hv/hlog.h"
#include "hv/hthread.h"
#include "hv/hthreadpool.h"
#include "llama.cpp/common/chat.h"
#include "llama.cpp/common/common.h"
#include "llama.cpp/common/ngram-cache.h"
#include "llama.cpp/common/sampling.h"
#include "llama.cpp/examples/llava/clip.h"
#include "llama.cpp/examples/llava/llava.h"
#include "llama.cpp/ggml/src/ggml-backend-impl.h"
#include "stable-diffusion.cpp/model.h"
#include "stable-diffusion.cpp/stable-diffusion.h"

#include "stablediffusion.hpp"
#include "utils.hpp"

// defines

#define HEADER_CACHE_CONTROL "Cache-Control"
#define HEADER_CONNECTION "Connection"
#define HEADER_CONTENT_TYPE "Content-Type"
#define HEADER_SERVER "SERVER"
#define HEADER_X_REQUEST_ID "X-Request-ID"
#define HEADER_X_REQUEST_ACCEPTED_AT "X-Request-Accepted-At"
#define HEADER_X_REQUEST_TOKENS_PER_SECOND "X-Request-Tokens-Per-Second"

using namespace hv;

// types

struct v2_httpserver_params {
    common_params llm_params;
    v2_stablediffusion_params sd_params;

    int32_t n_parallel       = 1;
    bool force_context_shift = false; // use context shift even if not allowed
    bool cache_prompt        = true;
    bool endpoint_images     = false;
    int32_t conn_keepalive   = 15; // connection keep-alive in seconds
    int32_t n_tps            = 0;  // maximum number of tokens per seconds
    int32_t lookup_ngram_min = 0;  // minimum n-gram size for lookup cache
    int32_t max_image_size   = 0;  // maximum image size for vision image processing
};

// implementations

// send_error_json, then close.
static inline int send_error_json(const HttpResponseWriterPtr &writer, http_status code, Json &data) {
    if (!writer->isConnected()) {
        return HTTP_STATUS_REQUEST_TIMEOUT;
    }
    const Json resp = {
        {"error", data},
        {"detail", data.contains("message") ? data.at("message") : "Unknown error occurred"},
    };
    if (writer->state < hv::HttpResponseWriter::SEND_HEADER) {
        writer->Begin();
        writer->WriteStatus(code);
        writer->WriteHeader(HEADER_CONTENT_TYPE, "application/json");
    }
    if (writer->response->GetHeader(HEADER_CONTENT_TYPE) != "text/event-stream") {
        int32_t ret = writer->End(resp.dump(-1, ' ', false, Json::error_handler_t::replace));
        if (ret < 0) {
            return HTTP_STATUS_REQUEST_TIMEOUT;
        }
        return code;
    }
    // SSE
    int32_t ret = writer->write("error: " + resp.dump(-1, ' ', false, Json::error_handler_t::replace) + "\n\n");
    if (ret < 0) {
        writer->close(true);
        return HTTP_STATUS_REQUEST_TIMEOUT;
    }
    writer->close(true);
    return code;
}

// send_error_json, then close.
static inline int send_error_json(const HttpResponseWriterPtr &writer, http_status code, std::string message) {
    if (!writer->isConnected()) {
        return HTTP_STATUS_REQUEST_TIMEOUT;
    }
    Json data = {
        {"code", code},
        {"message", message},
        {"type", http_status_str(code)},
    };
    return send_error_json(writer, code, data);
}

// send_json, then close.
static inline int send_json(const HttpResponseWriterPtr &writer, const Json &data) {
    if (!writer->isConnected()) {
        return HTTP_STATUS_REQUEST_TIMEOUT;
    }
    if (writer->state < hv::HttpResponseWriter::SEND_HEADER) {
        writer->Begin();
        writer->WriteStatus(HTTP_STATUS_OK);
        writer->WriteHeader(HEADER_CONTENT_TYPE, "application/json");
    }
    if (writer->response->GetHeader(HEADER_CONTENT_TYPE) != "text/event-stream") {
        int32_t ret = writer->End(data.dump(-1, ' ', false, Json::error_handler_t::replace));
        if (ret < 0) {
            return HTTP_STATUS_REQUEST_TIMEOUT;
        }
        return HTTP_STATUS_OK;
    }
    // SSE
    int32_t ret = writer->write("data: " + data.dump(-1, ' ', false, Json::error_handler_t::replace) + "\n\n");
    if (ret < 0) {
        writer->close(true);
        return HTTP_STATUS_REQUEST_TIMEOUT;
    }
    writer->close(true);
    return HTTP_STATUS_OK;
}

// send_error_string, then close.
static inline int send_error_string(const HttpResponseWriterPtr &writer, http_status code, const std::string &message, const std::string &content_type = "") {
    if (!writer->isConnected()) {
        return HTTP_STATUS_REQUEST_TIMEOUT;
    }
    if (writer->state < hv::HttpResponseWriter::SEND_HEADER) {
        writer->Begin();
        writer->WriteStatus(code);
        writer->WriteHeader(HEADER_CONTENT_TYPE, content_type.empty() ? "text/plain" : content_type);
    }
    if (writer->response->GetHeader(HEADER_CONTENT_TYPE) != "text/event-stream") {
        int32_t ret = writer->End(message);
        if (ret < 0) {
            return HTTP_STATUS_REQUEST_TIMEOUT;
        }
        return code;
    }
    // SSE
    int32_t ret = writer->write("error: " + message + "\n\n");
    if (ret < 0) {
        writer->close(true);
        return HTTP_STATUS_REQUEST_TIMEOUT;
    }
    writer->close(true);
    return code;
}

// send_string, then close.
static inline int send_string(const HttpResponseWriterPtr &writer, const std::string &message, const std::string &content_type = "") {
    if (!writer->isConnected()) {
        return HTTP_STATUS_REQUEST_TIMEOUT;
    }
    if (writer->state < hv::HttpResponseWriter::SEND_HEADER) {
        writer->Begin();
        writer->WriteStatus(HTTP_STATUS_OK);
        writer->WriteHeader(HEADER_CONTENT_TYPE, content_type.empty() ? "text/plain" : content_type);
    }
    if (writer->response->GetHeader(HEADER_CONTENT_TYPE) != "text/event-stream") {
        int32_t ret = writer->End(message);
        if (ret < 0) {
            return HTTP_STATUS_REQUEST_TIMEOUT;
        }
        return HTTP_STATUS_OK;
    }
    // SSE
    int32_t ret = writer->write("data: " + message + "\n\n");
    if (ret < 0) {
        writer->close(true);
        return HTTP_STATUS_REQUEST_TIMEOUT;
    }
    writer->close(true);
    return HTTP_STATUS_OK;
}

// send_event_json, but not close.
static inline int send_event_json(const HttpResponseWriterPtr &writer, const Json &data) {
    if (!writer->isConnected()) {
        return HTTP_STATUS_REQUEST_TIMEOUT;
    }
    if (writer->state < hv::HttpResponseWriter::SEND_HEADER) {
        writer->Begin();
        writer->WriteStatus(HTTP_STATUS_OK);
        writer->WriteHeader(HEADER_CACHE_CONTROL, "no-cache, no-store, no-transform");
        writer->WriteHeader(HEADER_CONNECTION, "close");
        writer->EndHeaders(HEADER_CONTENT_TYPE, "text/event-stream");
    }
    int32_t ret = writer->write("data: " + data.dump(-1, ' ', false, Json::error_handler_t::replace) + "\n\n");
    if (ret < 0) {
        writer->close(true);
        return HTTP_STATUS_REQUEST_TIMEOUT;
    }
    return HTTP_STATUS_OK;
}

// normalize_seed, avoid black-box seed.
static inline uint32_t normalize_seed(uint32_t seed) {
    if (seed == LLAMA_DEFAULT_SEED) {
        return uint32_t(ggml_time_us());
    }
    return seed;
}

// prepare_sampling, returns llama.cpp sampling params.
static inline common_params_sampling prepare_sampling(const Json &data, const common_params_sampling &defaults, const llama_context *ctx) {
    common_params_sampling params = defaults; // copy
    if (!data.contains("samplers")) {
        return params;
    }

    {
        Json samplers = data.at("samplers");
        if (samplers.is_array()) {
            params.samplers = common_sampler_types_from_names(samplers, false);
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
        params.penalty_last_n = int32_t(llama_n_ctx(ctx));
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
        params.dry_penalty_last_n = int32_t(llama_n_ctx(ctx));
    }
    params.mirostat     = json_value(data, "mirostat", defaults.mirostat);
    params.mirostat_tau = json_value(data, "mirostat_tau", defaults.mirostat_tau);
    params.mirostat_eta = json_value(data, "mirostat_eta", defaults.mirostat_eta);
    params.seed         = normalize_seed(json_value(data, "seed", defaults.seed));
    params.n_probs      = json_value(data, "n_probs", defaults.n_probs);
    params.min_keep     = json_value(data, "min_keep", defaults.min_keep);
    if (data.contains("json_schema") && !data.contains("grammar")) {
        try {
            Json schema    = json_value(data, "json_schema", Json::object());
            params.grammar = json_schema_to_grammar(schema);
        } catch (const std::exception &e) {
            throw std::invalid_argument("Illegal param: \"json_schema\": " + std::string(e.what()));
        }
    } else if (data.contains("grammar")) {
        params.grammar = json_value(data, "grammar", defaults.grammar);
    }
    if (json_value(data, "ignore_eos", false)) {
        const llama_vocab *vocab    = llama_model_get_vocab(llama_get_model(ctx));
        const llama_token vocab_eos = llama_vocab_eos(vocab);
        if (vocab_eos != LLAMA_TOKEN_NULL) {
            params.logit_bias.push_back({vocab_eos, -INFINITY});
        }
    }

    return params;
}

// prepare_sampling, returns stable-diffusion.cpp sampling params.
static inline v2_stablediffusion_params_sampling prepare_sampling(const Json &data, const v2_stablediffusion_params_sampling &defaults) {
    v2_stablediffusion_params_sampling params = defaults; // copy
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
static inline v2_stablediffusion_params_sampling prepare_sampling(const MultiPart &req, const v2_stablediffusion_params_sampling &defaults) {
    v2_stablediffusion_params_sampling params = defaults; // copy
    if (req.find("sampler") == req.end() && req.find("sample_method") == req.end()) {
        return params;
    }

    std::string sample_method_str = "euler_a";
    auto item                     = req.find("sample_method");
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
static inline void sort_rerank_results(Json &result, int32_t low, int32_t high) {
    if (low >= high) {
        return;
    }

    Json base = result[low];
    int i = low, j = high;
    while (i != j) {
        while (i < j && json_value(result[j], "score", 0.0) <= json_value(base, "score", 0.0))
            j--;
        while (i < j && json_value(result[i], "score", 0.0) >= json_value(base, "score", 0.0))
            i++;
        if (i < j) {
            Json temp = result[i];
            result[i] = result[j];
            result[j] = temp;
        }
    }
    result[low] = result[i];
    result[i]   = base;
    sort_rerank_results(result, low, i - 1);
    sort_rerank_results(result, i + 1, high);
}

// common_batch_add_with_mrope, mocks common_batch_add but works in mrope.
static inline void common_batch_add_with_mrope(struct llama_batch &batch, llama_token id, llama_pos st_pos_id, int32_t n_eval, const std::vector<llama_seq_id> &seq_ids, bool logits) {
    GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

    batch.token[batch.n_tokens] = id;
    for (int i = 0; i < 4; i++) {
        if (i == 3) {
            st_pos_id = 0;
        }
        batch.pos[batch.n_tokens + n_eval * i] = st_pos_id;
    }
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t j = 0; j < seq_ids.size(); ++j) {
        batch.seq_id[batch.n_tokens][j] = seq_ids[j];
    }
    batch.logits[batch.n_tokens] = logits;
    batch.n_tokens++;
}

// equal_lora returns true if both lora adapters are the same.
static inline bool equal_lora(const std::vector<common_adapter_lora_info> &l1, const std::vector<common_adapter_lora_info> &l2) {
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
static inline std::vector<llama_token_data> get_token_probabilities(llama_context *ctx, int idx) {
    std::vector<llama_token_data> cur;
    const auto *logits = llama_get_logits_ith(ctx, idx);
    const int n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));

    cur.resize(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
    }

    // sort tokens by logits
    std::sort(cur.begin(), cur.end(), [](const llama_token_data &a, const llama_token_data &b) {
        return a.logit > b.logit;
    });

    // apply softmax
    float max_l   = cur[0].logit;
    float cum_sum = 0.0f;
    for (size_t i = 0; i < cur.size(); ++i) {
        float p  = expf(cur[i].logit - max_l);
        cur[i].p = p;
        cum_sum += p;
    }
    for (size_t i = 0; i < cur.size(); ++i) {
        cur[i].p /= cum_sum;
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
    explicit breq(std::string rid, req_type type)
        : rid(std::move(rid)), type(type) {
    }

    virtual ~breq() = default;

    const char *get_rid() {
        return rid.c_str();
    }

    [[nodiscard]] req_type get_type() const {
        return type;
    }

    [[nodiscard]] virtual std::string get_model() const {
        return "";
    }

    [[nodiscard]] virtual int32_t get_n() const {
        return 1;
    }

  protected:
    std::string rid;
    req_type type = REQ_UNKNOWN;
};

struct tokenize_req : breq {
    explicit tokenize_req(const std::string &rid)
        : breq(rid, REQ_TOKENIZE) {
    }

    /* LLAMA BOX */

    [[nodiscard]] std::string get_model() const override {
        return model;
    }

    /* OPEN AI*/

    std::string model;
    Json content;
    bool add_special = false;
    bool with_pieces = false;
};

static inline std::unique_ptr<tokenize_req> get_tokenize_req(const HttpContextPtr &ctx, const common_params &params) {
    const std::string rid = ctx->response->GetHeader(HEADER_X_REQUEST_ID);
    const Json req        = ctx->request->GetJson();
    if (!req.contains("content")) {
        throw std::invalid_argument("Illegal param: \"content\" is required");
    }
    if (!json_is_array_or_string(req.at("content"))) {
        throw std::invalid_argument("Illegal param: \"content\" must be a string or a list");
    }

    // print the request for debugging
    if (common_log_verbosity_thold > 1) {
        Json req_cp = req;
        if (common_log_verbosity_thold < 2) {
            req_cp["content"] = "...";
        }
        SRV_INF("rid %s | %s\n", rid.c_str(), req_cp.dump(-1, ' ', false, Json::error_handler_t::replace).c_str());
    }

    std::unique_ptr<tokenize_req> ptr = std::make_unique<tokenize_req>(rid);

    ptr->model = json_value(req, "model", params.model_alias);

    ptr->content = req.at("content");

    ptr->add_special = json_value(req, "add_special", false);

    ptr->with_pieces = json_value(req, "with_pieces", false);

    return ptr;
}

struct detokenize_req : breq {
    explicit detokenize_req(const std::string &rid)
        : breq(rid, REQ_DETOKENIZE) {
    }

    /* LLAMA BOX */

    [[nodiscard]] std::string get_model() const override {
        return model;
    }

    /* OPEN AI*/

    std::string model;
    Json tokens;
};

static inline std::unique_ptr<detokenize_req> get_detokenize_req(const HttpContextPtr &ctx, const common_params &params) {
    const std::string rid = ctx->response->GetHeader(HEADER_X_REQUEST_ID);
    const Json req        = ctx->request->GetJson();
    if (!req.contains("tokens")) {
        throw std::invalid_argument("Illegal param: \"tokens\" is required");
    }
    if (!json_is_array_of_numbers(req.at("tokens"))) {
        throw std::invalid_argument("Illegal param: \"tokens\" must be a list of tokens");
    }

    // print the request for debugging
    if (common_log_verbosity_thold > 1) {
        Json req_cp = req;
        if (common_log_verbosity_thold < 2) {
            req_cp["tokens"] = "...";
        }
        SRV_INF("rid %s | %s\n", rid.c_str(), req_cp.dump(-1, ' ', false, Json::error_handler_t::replace).c_str());
    }

    std::unique_ptr<detokenize_req> ptr = std::make_unique<detokenize_req>(rid);

    ptr->model = json_value(req, "model", params.model_alias);

    ptr->tokens = req.at("tokens");

    return ptr;
}

struct complete_req : breq {
    explicit complete_req(const std::string &rid, req_type type)
        : breq(rid, type) {
    }

    /* LLAMA BOX */

    // sample
    common_params_sampling sampling;
    // lora
    std::vector<common_adapter_lora_info> lora_adapters;

    /* OPEN AI */

    // decode
    int32_t max_tokens = -1;
    int32_t logprobs   = -1;
    std::vector<std::string> stop;

    // stream
    bool stream         = false;
    Json stream_options = {
        {"include_usage", true},
    };
};

struct legacy_complete_req : complete_req {
    explicit legacy_complete_req(const std::string &rid)
        : complete_req(rid, REQ_LEGACY_COMPLETE) {
    }

    /* LLAMA BOX */

    [[nodiscard]] std::string get_model() const override {
        return model;
    }

    /* OPEN AI*/

    std::string model;
    Json prompt;
    // int32_t best_of = 1;
    // bool echo = false;
    float frequency_penalty = 0.0f;
    std::vector<llama_logit_bias> logit_bias;
    // int32_t logprobs   = 0;                                // inherit
    // int32_t max_tokens = -1;                               // inherit
    // int32_t n = 1;
    float presence_penalty = 0.0f;
    uint32_t seed          = LLAMA_DEFAULT_SEED;
    // std::vector<std::string> stop;                         // inherit
    // bool stream         = false;                           // inherit
    // Json stream_options = {{"include_usage", true}};       // inherit
    // std::string suffix;
    float temperature = 1.0;
    float top_p       = 1.0;
    // std::string user;
};

static inline std::unique_ptr<legacy_complete_req> get_legacy_complete_req(const HttpContextPtr &ctx, const v2_httpserver_params &hparams, const llama_context *llm_ctx) {
    const common_params &params = hparams.llm_params;

    const std::string rid = ctx->response->GetHeader(HEADER_X_REQUEST_ID);
    const Json req        = ctx->request->GetJson();
    if (!req.contains("prompt")) {
        throw std::invalid_argument("Illegal param: \"prompt\" is required");
    }

    // print the request for debugging
    if (common_log_verbosity_thold > 1) {
        Json req_cp = req;
        if (common_log_verbosity_thold < 2) {
            req_cp["prompt"] = "...";
        }
        SRV_INF("rid %s | %s\n", rid.c_str(), req_cp.dump(-1, ' ', false, Json::error_handler_t::replace).c_str());
    }

    std::unique_ptr<legacy_complete_req> ptr = std::make_unique<legacy_complete_req>(rid);

    ptr->sampling = prepare_sampling(req, params.sampling, llm_ctx);

    if (req.contains("lora")) {
        const Json &lora = req.at("lora");
        if (!lora.is_array()) {
            throw std::invalid_argument("Illegal param: \"lora\" must be a list");
        }
        ptr->lora_adapters = params.lora_adapters;
        // clear value
        for (common_adapter_lora_info &la : ptr->lora_adapters) {
            la.scale = 0.0f;
        }
        // set value
        int32_t max_id = int32_t(ptr->lora_adapters.size()) - 1;
        for (const Json &l : lora) {
            if (!l.is_object()) {
                throw std::invalid_argument("Illegal param: \"lora\" must be a list of objects");
            }
            int32_t id  = json_value(l, "id", -1);
            float scale = json_value(l, "scale", 0.0f);
            if (id < 0 || id > max_id) {
                throw std::invalid_argument("Illegal param: \"id\" must be in the range [0, " + std::to_string(max_id) + "]");
            }
            ptr->lora_adapters[id].scale = scale;
        }
    }

    ptr->model = json_value(req, "model", params.model_alias);

    ptr->prompt = req.at("prompt");

    ptr->frequency_penalty = json_value(req, "frequency_penalty", params.sampling.penalty_freq);

    if (req.contains("logit_bias")) {
        const Json &logit_bias = req.at("logit_bias");
        if (!logit_bias.is_object()) {
            throw std::invalid_argument("Illegal param: \"logit_bias\" must be a map");
        }
        const llama_vocab *vocab = llama_model_get_vocab(llama_get_model(llm_ctx));
        const int32_t vocab_size = llama_vocab_n_tokens(vocab);
        for (const auto &el : logit_bias.items()) {
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
                    throw std::invalid_argument("Illegal param: \"logit_bias\" values must be in the range [-100, 100]");
                }
            } else if (el.value().is_boolean()) {
                if (el.value().get<bool>()) {
                    continue;
                }
            } else {
                throw std::invalid_argument("Illegal param: \"logit_bias\" values must be a number or boolean");
            }
            ptr->logit_bias.push_back({tok, bias});
        }
    }

    if (req.contains("logprobs")) {
        ptr->logprobs = req.at("logprobs").get<int32_t>();
        if (ptr->logprobs < 0 || ptr->logprobs > 5) {
            throw std::invalid_argument("Illegal param: \"logprobs\" must be in the range [1, 5]");
        }
    }

    ptr->max_tokens = json_value(req, "max_tokens", params.n_predict);
    if (ptr->max_tokens <= 0) {
        ptr->max_tokens = int32_t(llama_n_ctx(llm_ctx));
    } else if (ptr->max_tokens > int32_t(llama_n_ctx(llm_ctx))) {
        throw std::invalid_argument("Illegal param: \"max_tokens\" must be less than or equal to the model's context length");
    }

    ptr->presence_penalty = json_value(req, "presence_penalty", params.sampling.penalty_present);

    ptr->seed = normalize_seed(json_value(req, "seed", params.sampling.seed));

    if (req.contains("stop")) {
        const Json &stop = req.at("stop");
        if (stop.is_string()) {
            ptr->stop.push_back(stop.get<std::string>());
        } else if (stop.is_array()) {
            for (const Json &s : stop) {
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
        const Json &stream_options = req.at("stream_options");
        if (!stream_options.is_object()) {
            throw std::invalid_argument("Illegal param: \"stream_options\" must be an object");
        }
        for (const auto &el : stream_options.items()) {
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

struct chat_complete_req : complete_req {
    explicit chat_complete_req(const std::string &rid)
        : complete_req(rid, REQ_CHAT_COMPLETE) {
    }

    /* LLAMA BOX */

    [[nodiscard]] std::string get_model() const override {
        return model;
    }

    // images, temporary use, will not value in "process"
    std::vector<std::unique_ptr<clip_image_u8>> images;
    common_chat_params chat_params;
    // tool calls
    llama_token tool_call_start_token = LLAMA_TOKEN_NULL;
    std::vector<std::string> tool_call_start_words;
    size_t tool_call_start_words_longest_length = 0;

    /* OPEN AI*/

    std::string model;
    std::vector<common_chat_msg> messages;
    // bool store = false;
    // std::string reasoning_effort;
    // Json metadata;
    float frequency_penalty = 0.0f;
    std::vector<llama_logit_bias> logit_bias;
    // bool logprobs = false;
    // int32_t top_logprobs          = 0;                               // inherit // migrate "logprobs"
    // int32_t max_tokens = -1;                                         // inherit // migrate "max_completion_tokens"
    // int32_t n = 1;
    // std::vector<std::string> modalities;
    // Json prediction;
    // Json audio;
    float presence_penalty = 0.0f;
    Json response_format;
    uint32_t seed = LLAMA_DEFAULT_SEED;
    // std::string service_tier;
    // std::vector<std::string> stop;                                   // inherit
    // bool stream         = false;                                     // inherit
    // Json stream_options = {{"include_usage", true}};                 // inherit
    float temperature = 1.0;
    float top_p       = 1.0;
    std::vector<common_chat_tool> tools;                                // migrate "functions"
    common_chat_tool_choice tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE; // migrate "function_call"
    bool parallel_tool_calls            = false;
    // std::string user;
};

static inline std::unique_ptr<clip_image_u8> get_clip_image(std::vector<uint8_t> &&img_buff, const int32_t max_image_size) {
    int w       = 0;
    int h       = 0;
    int c       = 0;
    uint8_t *dt = stbi_load_from_memory((const stbi_uc *)img_buff.data(), (int32_t)img_buff.size(), &w, &h, &c, 3);
    if (dt == nullptr) {
        throw std::invalid_argument("Illegal param: provided image is invalid: " + std::string(stbi_failure_reason()));
    }

    if (c < 3) {
        stbi_image_free(dt);
        throw std::invalid_argument("Illegal param: provided image must be a valid RGB image");
    }

    std::unique_ptr<clip_image_u8> img = std::make_unique<clip_image_u8>();

    int m = MAX(w, h);
    if (max_image_size < 0 || m <= max_image_size) {
        img->nx  = w;
        img->ny  = h;
        img->buf = std::vector<uint8_t>(dt, dt + w * h * 3);
        return img;
    }

    float nr = float(max_image_size) / float(m);
    int nw   = MAX(int(std::ceil(float(w) * nr)), 1);
    int nh   = MAX(int(std::ceil(float(h) * nr)), 1);

    auto *ndt    = (uint8_t *)malloc(nw * nh * 3);
    bool resized = stbir_resize(
        dt, w, h, 0,
        ndt, nw, nh, 0,
        STBIR_TYPE_UINT8,
        3,                                                // RGB
        STBIR_ALPHA_CHANNEL_NONE,                         // no Alpha
        0,                                                // flags
        STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,               // clamp edge mode
        STBIR_FILTER_CATMULLROM, STBIR_FILTER_CATMULLROM, // catmull-rom filter
        STBIR_COLORSPACE_SRGB,                            // sRGB
        nullptr);
    stbi_image_free(dt);
    if (!resized) {
        throw std::runtime_error("Illegal param: provide image exceeds the max image size, but failed to resize: " + std::string(stbi_failure_reason()));
    }

    img->nx  = nw;
    img->ny  = nh;
    img->buf = std::vector<uint8_t>(ndt, ndt + nw * nh * 3);
    return img;
}

static inline std::unique_ptr<chat_complete_req> get_chat_complete_req(const HttpContextPtr &ctx, const v2_httpserver_params &hparams, const llama_context *llm_ctx, const bool support_tool_calls, const common_chat_templates *chat_templates) {
    const common_params &params = hparams.llm_params;

    const std::string rid = ctx->response->GetHeader(HEADER_X_REQUEST_ID);
    const Json req        = ctx->request->GetJson();
    if (!req.contains("messages")) {
        throw std::invalid_argument("Illegal param: \"messages\" is required");
    } else if (!req.at("messages").is_array()) {
        throw std::invalid_argument("Illegal param: \"messages\" must be a list");
    }

    // print the request for debugging
    if (common_log_verbosity_thold > 1) {
        Json req_cp = req;
        if (common_log_verbosity_thold < 2) {
            req_cp["messages"] = "...";
        }
        SRV_INF("rid %s | %s\n", rid.c_str(), req_cp.dump(-1, ' ', false, Json::error_handler_t::replace).c_str());
    }

    std::unique_ptr<chat_complete_req> ptr = std::make_unique<chat_complete_req>(rid);

    ptr->sampling = prepare_sampling(req, params.sampling, llm_ctx);

    if (req.contains("lora")) {
        const Json &lora = req.at("lora");
        if (!lora.is_array()) {
            throw std::invalid_argument("Illegal param: \"lora\" must be a list");
        }
        ptr->lora_adapters = params.lora_adapters;
        // clear value
        for (common_adapter_lora_info &la : ptr->lora_adapters) {
            la.scale = 0.0f;
        }
        // set value
        int32_t max_id = int32_t(ptr->lora_adapters.size()) - 1;
        for (const Json &l : lora) {
            if (!l.is_object()) {
                throw std::invalid_argument("Illegal param: \"lora\" must be a list of objects");
            }
            int32_t id  = json_value(l, "id", -1);
            float scale = json_value(l, "scale", 0.0f);
            if (id < 0 || id > max_id) {
                throw std::invalid_argument("Illegal param: \"id\" must be in the range [0, " + std::to_string(max_id) + "]");
            }
            ptr->lora_adapters[id].scale = scale;
        }
    }

    ptr->model = json_value(req, "model", params.model_alias);

    {
        Json messages = req.at("messages");
        for (const Json &msg : messages) {
            std::string role = json_value(msg, "role", std::string());
            std::string content;
            if (msg.contains("content") && !msg.at("content").is_null()) {
                if (msg.at("content").is_string()) {
                    content = msg.at("content").get<std::string>();
                } else if (msg.at("content").is_array()) {
                    int32_t n_img = 0;
                    for (const Json &part : msg.at("content")) {
                        if (part.contains("type") && part.at("type") == "image_url") {
                            // process image
                            std::string img = json_value(part.at("image_url"), "url", std::string());
                            if (img.find("data:image/") != std::string::npos) {
                                const std::string split = "base64,";
                                const size_t idx        = img.find(split);
                                if (idx == std::string::npos) {
                                    throw std::invalid_argument("Illegal param: \"image_url\" must be a valid base64-encoded image");
                                }
                                img = img.substr(idx + split.length());
                                if (img.empty()) {
                                    throw std::invalid_argument("Illegal param: \"image_url\" is an empty image base64-encoded data");
                                }
                                try {
                                    std::vector<uint8_t> img_buff           = decode_base64(img);
                                    std::unique_ptr<clip_image_u8> clip_img = get_clip_image(std::move(img_buff), hparams.max_image_size);
                                    ptr->images.push_back(std::move(clip_img));
                                } catch (const std::exception &e) {
                                    throw std::invalid_argument("Illegal param: \"image_url\" must be a valid base64-encoded image");
                                }
                            } else {
                                std::string host, path;
                                if (size_t pos = img.find("://"); pos == std::string::npos) {
                                    throw std::invalid_argument("Illegal param: \"image_url\" must be a data URI or a valid URL");
                                } else {
                                    pos = img.find('/', pos + 3);
                                    if (pos == std::string::npos) {
                                        host = img;
                                        path = "/";
                                    } else {
                                        host = img.substr(0, pos);
                                        path = img.substr(pos);
                                    }
                                }
                                auto req                   = std::make_unique<HttpRequest>();
                                req->url                   = img;
                                req->connect_timeout       = 15;
                                req->timeout               = 300;
                                req->headers["User-Agent"] = "llama-box";
                                req->method                = HTTP_GET;
                                auto cli                   = http_client_new();
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
                                hssl_ctx_opt_t ssl_param;
                                ssl_param.verify_peer = 0;
                                http_client_new_ssl_ctx(cli, &ssl_param);
#endif
                                auto resp = std::make_unique<HttpResponse>();
                                if (int ret = http_client_send(cli, req.get(), resp.get()); ret != 0) {
                                    throw std::invalid_argument("Illegal param: invalid \"image_url\", failed to fetch image from URL: " + img + ", result: " + std::string(http_client_strerror(ret)));
                                }
                                if (http_status sc = resp->status_code; sc != HTTP_STATUS_OK) {
                                    throw std::invalid_argument("Illegal param: invalid \"image_url\", failed to fetch image from URL: " + img + ", status: " + std::string(http_status_str(sc)));
                                }
                                std::vector<uint8_t> img_buff(resp->body.begin(), resp->body.end());
                                std::unique_ptr<clip_image_u8> clip_img = get_clip_image(std::move(img_buff), hparams.max_image_size);
                                ptr->images.push_back(std::move(clip_img));
                            }
                            n_img++;
                            continue;
                        }
                        if (part.contains("text")) {
                            if (!content.empty()) {
                                content += "\n";
                            }
                            for (int32_t i = 0; i < n_img; i++) {
                                content += "<--IMAGE-->\n";
                            }
                            content += part.at("text").get<std::string>();
                            n_img = 0;
                        }
                    }
                    for (int32_t i = 0; i < n_img; i++) {
                        content += "\n<--IMAGE-->";
                    }
                } else {
                    throw std::invalid_argument("Illegal param: invalid \"content\"");
                }
                ptr->messages.push_back({role, content, {}, {}, "", "", ""});
            } else if (msg.contains("tool_calls") && !msg.at("tool_calls").is_null()) {
                if (msg.at("tool_calls").is_array()) {
                    std::vector<common_chat_tool_call> chat_tcs;
                    for (const Json &part : msg.at("tool_calls")) {
                        common_chat_tool_call chat_tc;
                        if (!part.contains("type") || part.at("type") != "function") {
                            continue;
                        }
                        if (!part.contains("function") || !part.at("function").is_object()) {
                            continue;
                        }
                        const Json &func = part.at("function");
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
                    ptr->messages.push_back({role, "", {}, chat_tcs, "", "", ""});
                } else {
                    throw std::invalid_argument("Illegal param: invalid \"tool_calls\"");
                }
            } else {
                throw std::invalid_argument("Illegal param: missing 'content' or 'tool_calls' in \"messages\" item");
            }
        }
    }

    ptr->frequency_penalty = json_value(req, "frequency_penalty", params.sampling.penalty_freq);

    if (req.contains("logit_bias")) {
        const Json &logit_bias = req.at("logit_bias");
        if (!logit_bias.is_object()) {
            throw std::invalid_argument("Illegal param: \"logit_bias\" must be a map");
        }
        const llama_vocab *vocab = llama_model_get_vocab(llama_get_model(llm_ctx));
        const int32_t vocab_size = llama_vocab_n_tokens(vocab);
        for (const auto &el : logit_bias.items()) {
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
                    throw std::invalid_argument("Illegal param: \"logit_bias\" values must be in the range [-100, 100]");
                }
            } else if (el.value().is_boolean()) {
                if (el.value().get<bool>()) {
                    continue;
                }
            } else {
                throw std::invalid_argument("Illegal param: \"logit_bias\" values must be a number or boolean");
            }
            ptr->logit_bias.push_back({tok, bias});
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
        ptr->max_tokens = json_value(req, "max_completion_tokens", params.n_predict);
    } else {
        ptr->max_tokens = json_value(req, "max_tokens", params.n_predict);
    }
    if (ptr->max_tokens <= 0) {
        ptr->max_tokens = int32_t(llama_n_ctx(llm_ctx));
    } else if (ptr->max_tokens > int32_t(llama_n_ctx(llm_ctx))) {
        throw std::invalid_argument(R"(Illegal param: "max_completion_tokens" or "max_tokens" must be less than or equal to the model's context length)");
    }

    ptr->presence_penalty = json_value(req, "presence_penalty", params.sampling.penalty_present);

    if (req.contains("response_format")) {
        ptr->response_format = req.at("response_format");
    }

    ptr->seed = normalize_seed(json_value(req, "seed", params.sampling.seed));

    if (req.contains("stop")) {
        const Json &stop = req.at("stop");
        if (stop.is_string()) {
            ptr->stop.push_back(stop.get<std::string>());
        } else if (stop.is_array()) {
            for (const Json &s : stop) {
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
        const Json &stream_options = req.at("stream_options");
        if (!stream_options.is_object()) {
            throw std::invalid_argument("Illegal param: \"stream_options\" must be an object");
        }
        for (const auto &el : stream_options.items()) {
            ptr->stream_options[el.key()] = el.value();
        }
    }

    ptr->temperature = json_value(req, "temperature", params.sampling.temp);

    ptr->top_p = json_value(req, "top_p", params.sampling.top_p);

    if (support_tool_calls) {
        // "tools" and "functions", migrate "functions" to "tools"
        if (req.contains("tools") && !req.contains("functions")) {
            const Json &tools = req.at("tools");
            if (!tools.is_array()) {
                throw std::invalid_argument("Illegal param: \"tools\" must be an array");
            }
            for (const Json &tool : tools) {
                if (!tool.contains("function")) {
                    continue;
                }
                const Json &func = tool.at("function");
                if (!func.contains("name") || !func.at("name").is_string()) {
                    continue;
                }
                if (!func.contains("parameters") || !func.at("parameters").is_object()) {
                    continue;
                }
                std::string name        = func.at("name");
                std::string description = json_value(func, "description", std::string());
                std::string parameters  = func.at("parameters").dump(-1, ' ', false, Json::error_handler_t::replace);
                ptr->tools.push_back({name, description, parameters});
            }
        } else if (req.contains("functions")) {
            const Json &functions = req.at("functions");
            if (!functions.is_array()) {
                throw std::invalid_argument("Illegal param: \"functions\" must be an array");
            }
            for (const Json &func : functions) {
                if (!func.contains("name") || !func.at("name").is_string()) {
                    continue;
                }
                if (!func.contains("parameters") || !func.at("parameters").is_object()) {
                    continue;
                }
                std::string name        = json_value(func, "name", std::string());
                std::string description = json_value(func, "description", std::string());
                std::string parameters  = func.at("parameters").dump(-1, ' ', false, Json::error_handler_t::replace);
                ptr->tools.push_back({name, description, parameters});
            }
        }
        if (!ptr->tools.empty()) {
            ptr->tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
            // "tool_choice" and "function_call", migrate "function_call" to "tool_choice"
            if (req.contains("tool_choice") && !req.contains("function_call")) {
                const Json &tc = req.at("tool_choice");
                if (tc.is_object() && tc.contains("function")) {
                    const Json &fc       = tc.at("function");
                    const std::string fn = json_value(fc, "name", std::string());
                    ptr->tools.erase(
                        std::remove_if(ptr->tools.begin(), ptr->tools.end(), [fn](const common_chat_tool &t) { return t.name == fn; }),
                        ptr->tools.end());
                    ptr->tool_choice = ptr->tools.empty() ? COMMON_CHAT_TOOL_CHOICE_NONE : COMMON_CHAT_TOOL_CHOICE_REQUIRED;
                } else if (tc.is_string()) {
                    ptr->tool_choice = common_chat_tool_choice_parse_oaicompat(tc.get<std::string>());
                } else {
                    throw std::invalid_argument("Illegal param: \"tool_choice\" must be a string or an object");
                }
            } else if (req.contains("function_call")) {
                const Json &fc = req.at("function_call");
                if (fc.is_object()) {
                    const std::string fn = json_value(fc, "name", std::string());
                    ptr->tools.erase(
                        std::remove_if(ptr->tools.begin(), ptr->tools.end(), [fn](const common_chat_tool &t) { return t.name == fn; }),
                        ptr->tools.end());
                    ptr->tool_choice = ptr->tools.empty() ? COMMON_CHAT_TOOL_CHOICE_NONE : COMMON_CHAT_TOOL_CHOICE_REQUIRED;
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
        Json json_schema;
        if (!ptr->response_format.empty()) {
            std::string response_type = json_value(ptr->response_format, "type", std::string());
            if (response_type == "json_object") {
                json_schema = json_value(ptr->response_format, "schema", Json());
            } else if (response_type == "json_schema") {
                if (!ptr->response_format.contains("json_schema")) {
                    throw std::invalid_argument("Illegal param: using json schema response format must contain \"json_schema\"");
                }
                json_schema = json_value(ptr->response_format.at("json_schema"), "schema", Json());
            } else if (!response_type.empty() && response_type != "text") {
                throw std::invalid_argument("Illegal param: \"response_format\" must be one of 'text', 'json_schema' or 'json_object', but got: " + response_type);
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
        // NB(thxCode): common_chat_templates_apply2 is a patch.
        ptr->chat_params = common_chat_templates_apply2(llama_get_model(llm_ctx), chat_templates, inputs);
        if (common_log_verbosity_thold > 2) {
            SRV_INF("rid %s | formatted prompt\n%s\n", rid.c_str(), ptr->chat_params.prompt.c_str());
        }
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
            const llama_vocab *vocab   = llama_model_get_vocab(llama_get_model(llm_ctx));
            for (const std::string &t : ptr->chat_params.preserved_tokens) {
                llama_tokens toks = common_tokenize(vocab, t, /* add_special= */ false, /* parse_special= */ true);
                if (toks.size() == 1) {
                    ptr->sampling.preserved_tokens.insert(toks[0]);
                }
            }
            for (const common_grammar_trigger &t : ptr->chat_params.grammar_triggers) {
                if (t.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                    const std::string &word = t.value;
                    llama_tokens toks       = common_tokenize(vocab, word, /* add_special= */ false, /* parse_special= */ true);
                    if (toks.size() == 1) {
                        ptr->sampling.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN, word, toks[0]});
                        ptr->sampling.preserved_tokens.insert(toks[0]);
                        ptr->tool_call_start_token = toks[0];
                        continue;
                    }
                    ptr->sampling.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word, LLAMA_TOKEN_NULL});
                    ptr->tool_call_start_words.push_back(word);
                    ptr->tool_call_start_words_longest_length = MAX(ptr->tool_call_start_words_longest_length, word.length());
                    continue;
                }
                ptr->sampling.grammar_triggers.push_back(t);
            }
            if (!ptr->tool_call_start_words.empty()) {
                ptr->tool_call_start_words_longest_length = ptr->tool_call_start_words_longest_length + int32_t(std::ceil(float(ptr->tool_call_start_words_longest_length) / 3.0));
            }
            if (ptr->sampling.grammar_lazy && ptr->sampling.grammar_triggers.empty()) {
                throw std::invalid_argument("no triggers set for lazy grammar");
            }
            for (const std::string &s : ptr->chat_params.additional_stops) {
                ptr->stop.push_back(s);
            }
        }
    }

    return ptr;
}

struct embed_req : breq {
    explicit embed_req(const std::string &rid)
        : breq(rid, REQ_EMBED) {
    }

    /* LLAMA BOX */

    [[nodiscard]] std::string get_model() const override {
        return model;
    }

    /* OPEN AI*/

    std::string model;
    Json input;
    std::string encoding_format = "float";
};

static inline std::unique_ptr<embed_req> get_embed_req(const HttpContextPtr &ctx, const v2_httpserver_params &hparams) {
    const common_params &params = hparams.llm_params;

    const std::string rid = ctx->response->GetHeader(HEADER_X_REQUEST_ID);
    const Json req        = ctx->request->GetJson();
    if (!req.contains("input")) {
        throw std::invalid_argument("Illegal param: \"input\" is required");
    }

    // print the request for debugging
    if (common_log_verbosity_thold > 1) {
        Json req_cp = req;
        if (common_log_verbosity_thold < 2) {
            req_cp["input"] = "[...]";
        }
        SRV_INF("rid %s | %s\n", rid.c_str(), req_cp.dump(-1, ' ', false, Json::error_handler_t::replace).c_str());
    }

    std::unique_ptr<embed_req> ptr = std::make_unique<embed_req>(rid);

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
    explicit rerank_req(const std::string &rid)
        : breq(rid, REQ_RERANK) {
    }

    /* LLAMA BOX */

    [[nodiscard]] std::string get_model() const override {
        return model;
    }

    /* JINJA */

    std::string model;
    Json query;
    std::vector<Json> documents;
    int32_t top_n         = 1;
    bool return_documents = false;
    bool normalize        = false;
};

static inline std::unique_ptr<rerank_req> get_rerank_req(const HttpContextPtr &ctx, const v2_httpserver_params &hparams) {
    const common_params &params = hparams.llm_params;

    const std::string rid = ctx->response->GetHeader(HEADER_X_REQUEST_ID);
    const Json req        = ctx->request->GetJson();
    if (!req.contains("query")) {
        throw std::invalid_argument("Illegal param: \"query\" is required");
    }
    if (!req.contains("documents")) {
        throw std::invalid_argument("Illegal param: \"documents\" is required");
    }
    if (!req.at("documents").is_array() || req.at("documents").empty()) {
        throw std::invalid_argument("Illegal param: \"documents\" must be a list with at least one item");
    }

    // print the request for debugging
    if (common_log_verbosity_thold > 1) {
        Json req_cp = req;
        if (common_log_verbosity_thold < 2) {
            req_cp["query"]     = "...";
            req_cp["documents"] = "...";
        }
        SRV_INF("rid %s | %s\n", rid.c_str(), req_cp.dump(-1, ' ', false, Json::error_handler_t::replace).c_str());
    }

    std::unique_ptr<rerank_req> ptr = std::make_unique<rerank_req>(rid);

    ptr->model = json_value(req, "model", params.model_alias);

    ptr->query = req.at("query");

    for (const Json &doc : req.at("documents")) {
        if (doc.is_string()) {
            ptr->documents.push_back(doc);
        } else if (doc.is_object() && doc.contains("text")) {
            ptr->documents.push_back(doc.at("text"));
        } else {
            throw std::invalid_argument("Illegal param: \"documents\" must be an array of strings or objects with 'text'");
        }
    }

    ptr->top_n = int32_t(MIN(json_value(req, "top_n", ptr->documents.size()), ptr->documents.size()));
    if (ptr->top_n <= 0) {
        throw std::invalid_argument("Illegal param: \"top_n\" must be greater than 0");
    }

    ptr->return_documents = json_value(req, "return_documents", true);

    ptr->normalize = json_value(req, "normalize", true);

    return ptr;
}

struct image_req : breq {
    explicit image_req(const std::string &rid, req_type type)
        : breq(rid, type) {
    }

    /* LLAMA BOX */

    [[nodiscard]] virtual const char *get_prompt() {
        return nullptr;
    }

    // sample
    v2_stablediffusion_params_sampling sampling;
    // lora
    std::vector<common_adapter_lora_info> lora_adapters;
    // stream
    bool stream         = false;
    Json stream_options = {
        {"include_usage", true},
        {"chunk_result", false},
        {"chunk_size", 4096},
        {"preview", false},
        {"preview_faster", false}, // deprecated
    };
};

struct image_generate_req : image_req {
    explicit image_generate_req(const std::string &rid)
        : image_req(rid, REQ_IMAGE_GENERATE) {
    }

    /* LLAMA BOX */

    /* OPEN AI */

    [[nodiscard]] std::string get_model() const override {
        return model;
    }

    [[nodiscard]] int32_t get_n() const override {
        return n;
    }

    [[nodiscard]] const char *get_prompt() override {
        return prompt.c_str();
    }

    std::string prompt;
    std::string model;
    int32_t n                   = 1;
    std::string quality         = "standard";
    std::string response_format = "b64_json";
    std::string size            = "512x512";
    std::string style           = "vivid";
    // std::string user;
};

static inline std::unique_ptr<image_generate_req> get_image_generate_req(const HttpContextPtr &ctx, const v2_httpserver_params &hparams) {
    const v2_stablediffusion_params &params = hparams.sd_params;

    const std::string rid = ctx->response->GetHeader(HEADER_X_REQUEST_ID);
    const Json req        = ctx->request->GetJson();
    if (!req.contains("prompt")) {
        throw std::invalid_argument("Illegal param: \"prompt\" is required");
    }

    // print the request for debugging
    if (common_log_verbosity_thold > 1) {
        Json req_cp = req;
        if (common_log_verbosity_thold < 2) {
            req_cp["prompt"] = "...";
        }
        SRV_INF("rid %s | %s\n", rid.c_str(), req_cp.dump(-1, ' ', false, Json::error_handler_t::replace).c_str());
    }

    std::unique_ptr<image_generate_req> ptr = std::make_unique<image_generate_req>(rid);

    ptr->sampling = prepare_sampling(req, params.sampling);

    if (req.contains("lora")) {
        const Json &lora = req.at("lora");
        if (!lora.is_array()) {
            throw std::invalid_argument("Illegal param: \"lora\" must be a list");
        }
        ptr->lora_adapters = params.lora_adapters;
        // clear value
        for (common_adapter_lora_info &la : ptr->lora_adapters) {
            la.scale = 0.0f;
        }
        // set value
        int32_t max_id = int32_t(ptr->lora_adapters.size()) - 1;
        for (const Json &l : lora) {
            if (!l.is_object()) {
                throw std::invalid_argument("Illegal param: \"lora\" must be a list of objects");
            }
            int32_t id  = json_value(l, "id", -1);
            float scale = json_value(l, "scale", 0.0f);
            if (id < 0 || id > max_id) {
                throw std::invalid_argument("Illegal param: \"id\" must be in the range [0, " + std::to_string(max_id) + "]");
            }
            ptr->lora_adapters[id].scale = scale;
        }
    }

    ptr->stream = json_value(req, "stream", false);

    if (ptr->stream && req.contains("stream_options")) {
        const Json &stream_options = req.at("stream_options");
        if (!stream_options.is_object()) {
            throw std::invalid_argument("Illegal param: \"stream_options\" must be an object");
        }
        for (const auto &el : stream_options.items()) {
            ptr->stream_options[el.key()] = el.value();
        }
    }

    ptr->prompt = req.at("prompt");

    ptr->model = json_value(req, "model", params.model_alias);

    ptr->n = json_value(req, "n", 1);
    if (ptr->n <= 0) {
        throw std::invalid_argument("Illegal param: \"n\" must be greater than 0");
    } else if (ptr->n > params.max_batch_count) {
        throw std::invalid_argument("Illegal param: \"n\" must be less than or equal to " + std::to_string(params.max_batch_count));
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
            throw std::invalid_argument("Illegal param: width of \"size\" must be at most " + std::to_string(params.sampling.width));
        }
        if (height > params.sampling.height) {
            throw std::invalid_argument("Illegal param: height of \"size\" must be at most " + std::to_string(params.sampling.height));
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
            ptr->sampling.negative_prompt = ptr->sampling.negative_prompt.empty() ? "low quality" : ptr->sampling.negative_prompt + ", low quality";
        }
        if (ptr->style == "vivid") {
            ptr->sampling.negative_prompt = ptr->sampling.negative_prompt.empty() ? "not vivid" : ptr->sampling.negative_prompt + ", not vivid";
        } else {
            ptr->sampling.negative_prompt = ptr->sampling.negative_prompt.empty() ? "unnatural" : ptr->sampling.negative_prompt + ", unnatural";
        }
    }

    return ptr;
}

struct image_edit_req : image_req {
    explicit image_edit_req(const std::string &rid)
        : image_req(rid, REQ_IMAGE_EDIT) {
    }

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

    /* OPEN AI */

    [[nodiscard]] std::string get_model() const override {
        return model;
    }

    [[nodiscard]] int32_t get_n() const override {
        return n;
    }

    [[nodiscard]] const char *get_prompt() override {
        return prompt.c_str();
    }

    /* OPEN AI */

    // image, temporary use, will not value in "process"
    std::vector<uint8_t> image;
    std::string prompt;
    // mask, temporary use, will not value in "process"
    std::vector<uint8_t> mask;
    std::string model;
    int32_t n                   = 1;
    std::string size            = "512x512";
    std::string response_format = "b64_json";
    // std::string user;
};

static inline std::unique_ptr<image_edit_req> get_image_edit_req(const HttpContextPtr &ctx, const v2_httpserver_params &hparams) {
    const v2_stablediffusion_params &params = hparams.sd_params;

    const std::string rid = ctx->response->GetHeader(HEADER_X_REQUEST_ID);
    const MultiPart &req  = ctx->request->GetForm();
    if (req.find("prompt") == req.end()) {
        throw std::invalid_argument("Illegal param: \"prompt\" is required");
    } else if (req.find("image") == req.end()) {
        throw std::invalid_argument("Illegal param: \"image\" is required");
    }

    // print the request for debugging
    if (common_log_verbosity_thold > 1) {
        Json req_cp = Json::object();
        for (const auto &el : req) {
            if (el.first == "image" || el.first == "mask" || el.first == "control") {
                req_cp[el.first] = "...";
            } else {
                req_cp[el.first] = el.second.content;
            }
        }
        if (common_log_verbosity_thold < 2) {
            req_cp["prompt"] = "...";
        }
        SRV_INF("rid %s | %s\n", rid.c_str(), req_cp.dump(-1, ' ', false, Json::error_handler_t::replace).c_str());
    }

    std::unique_ptr<image_edit_req> ptr = std::make_unique<image_edit_req>(rid);

    ptr->sampling = prepare_sampling(req, params.sampling);

    auto item = req.find("lora");
    if (item != req.end()) {
        Json lora;
        try {
            lora = Json::parse(item->second.content);
            if (!lora.is_array()) {
                throw std::invalid_argument("Illegal param: \"lora\" must be a JSON list");
            }
        } catch (const std::exception &) {
            throw std::invalid_argument("Illegal param: \"lora\" must be a JSON list");
        }
        ptr->lora_adapters = params.lora_adapters;
        // clear value
        for (common_adapter_lora_info &la : ptr->lora_adapters) {
            la.scale = 0.0f;
        }
        // set value
        int32_t max_id = int32_t(ptr->lora_adapters.size()) - 1;
        for (const Json &l : lora) {
            if (!l.is_object()) {
                throw std::invalid_argument("Illegal param: \"lora\" must be a list of objects");
            }
            int32_t id  = json_value(l, "id", -1);
            float scale = json_value(l, "scale", 0.0f);
            if (id < 0 || id > max_id) {
                throw std::invalid_argument("Illegal param: \"id\" must be in the range [0, " + std::to_string(max_id) + "]");
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
        item = req.find("stream_options_preview_faster"); // deprecated
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
            throw std::invalid_argument("Illegal param: width of \"size\" must be at most " + std::to_string(params.sampling.width));
        }
        if (height > params.sampling.height) {
            throw std::invalid_argument("Illegal param: height of \"size\" must be at most " + std::to_string(params.sampling.height));
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
            int cc                           = 0;
            int cw                           = 0;
            int ch                           = 0;
            ptr->sampling.control_img_buffer = stbi_load_from_memory((const stbi_uc *)ptr->control.data(), (int)ptr->control.size(), &cw, &ch, &cc, 3);
            if (ptr->sampling.control_img_buffer == nullptr) {
                const char *reason = stbi_failure_reason();
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
        int iw                        = 0;
        int ih                        = 0;
        int ic                        = 0;
        ptr->sampling.init_img_buffer = stbi_load_from_memory((const stbi_uc *)ptr->image.data(), (int)ptr->image.size(), &iw, &ih, &ic, 3);
        if (ptr->sampling.init_img_buffer == nullptr) {
            FREE_IMG_BUFFER;
            const char *reason = stbi_failure_reason();
            throw std::invalid_argument("Illegal param: \"image\" is not a valid image: " + std::string(reason));
        }
        if (ic < 3 || iw <= 0 || ih <= 0) {
            FREE_IMG_BUFFER;
            throw std::invalid_argument("Illegal param: \"image\" must be a valid RGB image");
        }
        if (iw != ptr->sampling.width || ih != ptr->sampling.height) {
            // resize
            int rw                     = ptr->sampling.width;
            int rh                     = ptr->sampling.height;
            auto *resized_image_buffer = (uint8_t *)malloc(rw * rh * 3);
            if (resized_image_buffer == nullptr) {
                FREE_IMG_BUFFER;
                throw std::invalid_argument("Illegal param: \"image\", failed to allocate memory for resizing");
            }
            if (!stbir_resize(ptr->sampling.init_img_buffer, iw, ih, 0,
                              resized_image_buffer, rw, rh, 0,
                              STBIR_TYPE_UINT8,
                              3,                                                // RGB
                              STBIR_ALPHA_CHANNEL_NONE,                         // no Alpha
                              0,                                                // flags
                              STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,               // clamp edge mode
                              STBIR_FILTER_CATMULLROM, STBIR_FILTER_CATMULLROM, // catmull-rom filter
                              STBIR_COLORSPACE_SRGB,                            // sRGB
                              nullptr)) {
                const char *reason = stbi_failure_reason();
                FREE_IMG_BUFFER;
                throw std::invalid_argument("Illegal param: \"image\", failed to resize: " + std::string(reason));
            }
            stbi_image_free(ptr->sampling.init_img_buffer);
            ptr->sampling.init_img_buffer = resized_image_buffer;
        }
        // mask image process
        if (!ptr->mask.empty()) {
            int mw                        = 0;
            int mh                        = 0;
            int mc                        = 0;
            ptr->sampling.mask_img_buffer = stbi_load_from_memory((const stbi_uc *)ptr->mask.data(), (int)ptr->mask.size(), &mw, &mh, &mc, 1);
            if (ptr->sampling.mask_img_buffer == nullptr) {
                FREE_IMG_BUFFER;
                const char *reason = stbi_failure_reason();
                throw std::invalid_argument("Illegal param: \"mask\" is not a valid image: " + std::string(reason));
            }
            if (mc < 1 || mw <= 0 || mh <= 0) {
                FREE_IMG_BUFFER;
                throw std::invalid_argument("Illegal param: \"mask\" must be a valid gray scale image");
            }
            if (mw != ptr->sampling.width || mh != ptr->sampling.height) {
                int rw                    = ptr->sampling.width;
                int rh                    = ptr->sampling.height;
                auto *resized_mask_buffer = (uint8_t *)malloc(rw * rh * 1);
                if (resized_mask_buffer == nullptr) {
                    FREE_IMG_BUFFER;
                    throw std::invalid_argument("Illegal param: \"mask\", failed to allocate memory for resizing");
                }
                if (!stbir_resize(ptr->sampling.mask_img_buffer, mw, mh, 0,
                                  resized_mask_buffer, rw, rh, 0,
                                  STBIR_TYPE_UINT8,
                                  1,                                            // GREY
                                  STBIR_ALPHA_CHANNEL_NONE,                     // no Alpha
                                  0,                                            // flags
                                  STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,           // clamp edge mode
                                  STBIR_FILTER_TRIANGLE, STBIR_FILTER_TRIANGLE, // box filter
                                  STBIR_COLORSPACE_SRGB,                        // sRGB
                                  nullptr)) {
                    const char *reason = stbi_failure_reason();
                    FREE_IMG_BUFFER;
                    throw std::invalid_argument("Illegal param: \"mask\", failed to resize: " + std::string(reason));
                }
                stbi_image_free(ptr->sampling.mask_img_buffer);
                ptr->sampling.mask_img_buffer = resized_mask_buffer;
            }
        } else {
            ptr->sampling.mask_img_buffer = (uint8_t *)malloc(ptr->sampling.width * ptr->sampling.height * 1);
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

enum task_type {
    TASK_COMPLETIONS,
    TASK_EMBEDDINGS,
    TASK_IMAGES,
    TASK_UNKNOWN,
};

struct btask {
    explicit btask(int32_t tid, task_type type)
        : tid(tid), type(type) {
    }

    virtual ~btask() = default;

    [[nodiscard]] int32_t get_tid() const {
        return tid;
    }

    [[nodiscard]] task_type get_type() const {
        return type;
    }

  protected:
    int32_t tid    = -1;
    task_type type = TASK_UNKNOWN;
};

struct completions_task : btask {
    explicit completions_task(int32_t tid)
        : btask(tid, TASK_COMPLETIONS) {
    }

    ~completions_task() override {
        if (sampler != nullptr) {
            common_sampler_free(sampler);
        }
        if (sampler_draft != nullptr) {
            common_sampler_free(sampler_draft);
        }
    }

    // input
    std::unique_ptr<RatelimitTokenBucket> token_bucket;
    std::vector<std::variant<llama_tokens, std::unique_ptr<llava_image_embed>>> tokenized_prompts;
    common_chat_format tokenized_prompts_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    bool tokenized_prompts_include_images       = false;
    bool tokenized_prompts_include_tools        = false;
    llama_token tool_call_start_token           = LLAMA_TOKEN_NULL; // move from chat_complete_req
    std::vector<std::string> tool_call_start_words;                 // move from chat_complete_req
    size_t tool_call_start_words_longest_length = 0;                // move from chat_complete_req
    std::string cmpl_id;
    std::unique_ptr<complete_req> req;

    // prefill
    int32_t n_prefilling       = 0; // indicate how many tokens need to be prefilled
    int32_t n_prefilled        = 0; // indicate how many tokens have been prefilled
    int32_t n_prefilled_cached = 0; // indicate how many prefilled tokens are cached
    int64_t t_start_prefill    = 0; // indicate the time when prefilling starts
    double t_prefilled         = 0; // indicate the time(ms) spent on prefilling
    double p_prefilled_tps     = 0;

    // decode
    int32_t n_decoding_budget      = 0; // indicate how many tokens can be decoded
    int32_t n_decoded              = 0; // indicate how many tokens have been decoded
    int32_t n_decoded_processed    = 0; // indicate how many decoded tokens are processed
    int64_t t_start_decode         = 0; // indicate the time when decoding starts
    double t_decoded               = 0; // indicate the time(ms) spent on decoding
    double p_decoded_tps           = 0;
    struct common_sampler *sampler = nullptr;
    //// reasoning
    int32_t n_reasoning        = 0; // indicate how many tokens are reasoning
    bool reasoning_start_found = false;
    bool reasoning_end_found   = false;
    bool reasoning_finished    = false;
    //// tool call
    bool tool_call_start_found = false;
    ////// non-jinja too calls
    std::string tool_call_start_found_word;

    // speculative
    llama_tokens sampled_draft;
    int32_t n_drafted          = 0; // indicate how many tokens are drafted
    int32_t n_drafted_accepted = 0; // indicate how many drafted tokens are accepted
    double p_drafted_apt       = 0;
    //// draft-model speculative decoding
    struct common_sampler *sampler_draft = nullptr;
    //// model-free speculative decoding
    int32_t lookup_ngram_min = 0;
    common_ngram_cache ctx_ngram_cache;

    // output
    std::string generated_finish_reason;
    llama_tokens generated_tokens;
    std::string generated_text;                                                                       // erase after call to_json if streaming
    std::vector<Json> generated_tool_calls;                                                           // erase after call to_json if streaming
    std::vector<float> generated_probs;                                                               // erase after call get_probs_json if streaming
    std::vector<std::vector<std::pair<llama_token /* tok */, float /* prob */>>> generated_top_probs; // erase after call get_probs_json if streaming

    Json get_probs_json(const llama_context *ctx) {
        if (generated_probs.empty()) {
            return {};
        }

        size_t tokens_size = generated_tokens.size();
        size_t probs_size  = generated_probs.size();

        Json result;
        if (req->get_type() == REQ_CHAT_COMPLETE) {
            Json content = Json::array();

            for (size_t i = 0; i < probs_size; i++) {
                const llama_token id    = generated_tokens[tokens_size - probs_size + i];
                const std::string token = tokens_to_output_formatted_string(ctx, id);
                float token_logprob     = generated_probs[i] == 0.0f ? std::numeric_limits<float>::lowest() : std::log(generated_probs[i]);
                std::vector<unsigned char> token_bytes(token.begin(), token.end());
                Json token_top_logprobs = Json::array();
                for (const auto &tp : generated_top_probs[i]) {
                    const llama_token tp_id    = tp.first;
                    const std::string tp_token = tokens_to_output_formatted_string(ctx, tp_id);
                    float tp_token_logprob     = tp.second == 0.0f ? std::numeric_limits<float>::lowest() : std::log(tp.second);
                    std::vector<unsigned char> tp_token_bytes(tp_token.begin(), tp_token.end());
                    token_top_logprobs.push_back(Json{
                        {"token", tp_token},
                        {"logprob", tp_token_logprob},
                        {"bytes", tp_token_bytes},
                    });
                }

                content.push_back(Json{
                    {"token", token},
                    {"logprob", token_logprob},
                    {"bytes", token_bytes},
                    {"top_logprobs", token_top_logprobs},
                });
            }

            result = {
                {"content", content},
            };
        } else {
            Json token_logprobs = Json::array();
            Json tokens         = Json::array();
            Json top_logprobs   = Json::array();

            for (size_t i = 0; i < probs_size; i++) {
                const llama_token id    = generated_tokens[tokens_size - probs_size + i];
                const std::string token = tokens_to_output_formatted_string(ctx, id);
                float token_logprob     = generated_probs[i] == 0.0f ? std::numeric_limits<float>::lowest() : std::log(generated_probs[i]);
                Json token_top_logprobs;
                for (const auto &tp : generated_top_probs[i]) {
                    const llama_token tp_id      = tp.first;
                    const std::string tp_token   = tokens_to_output_formatted_string(ctx, tp_id);
                    float tp_token_logprob       = tp.second == 0.0f ? std::numeric_limits<float>::lowest() : std::log(tp.second);
                    token_top_logprobs[tp_token] = tp_token_logprob;
                }

                tokens.push_back(token);
                token_logprobs.push_back(token_logprob);
                top_logprobs.push_back(token_top_logprobs);
            }

            result = {
                {"tokens", tokens},
                {"token_logprobs", token_logprobs},
                {"top_logprobs", top_logprobs},
            };
        }

        // clean
        if (req->stream) {
            generated_probs.clear();
            generated_top_probs.clear();
        }

        return result;
    }

    Json to_json(const llama_context *ctx) {
        bool stop          = !generated_finish_reason.empty();
        bool include_usage = stop && json_value(req->stream_options, "include_usage", true);
        bool is_chat       = req->get_type() == REQ_CHAT_COMPLETE;

        Json resp = {
            {"id", cmpl_id},
            {"created", std::time(nullptr)},
            {"model", req->get_model()},
            {"usage", Json()},
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
                {"prompt_tokens", n_prefilled},
                {"completion_tokens", n_decoded},
                {"total_tokens", n_prefilled + n_decoded},
                {
                    "prompt_tokens_details",
                    {
                        {"cached_tokens", n_prefilled_cached},
                    },
                },
                {
                    "completion_tokens_details",
                    {
                        {"reasoning_tokens", n_reasoning},
                        {"accepted_prediction_tokens", n_drafted},
                        {"rejected_prediction_tokens", n_drafted - n_drafted_accepted},
                    },
                },
                // additional details
                {"time_to_first_token_ms", t_prefilled},
                {"time_per_output_token_ms", t_decoded / n_decoded},
                {"prompt_tokens_per_second", p_prefilled_tps},
                {"tokens_per_second", p_decoded_tps},
                {"draft_tokens", n_drafted},
                {"draft_tokens_acceptance", p_drafted_apt},
            };
        }

        Json choices = Json::array();
        {
            Json choice = {
                {"index", 0},
                {"finish_reason", stop ? Json(generated_finish_reason) : Json()},
            };
            if (is_chat) {
                Json delta_message = Json{
                    {"content", generated_text},
                };
                if (!generated_tool_calls.empty()) {
                    if (generated_text.empty()) {
                        delta_message["content"] = Json();
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
                choice["text"] = generated_text;
            };
            if (req->logprobs >= 0) {
                choice["logprobs"] = get_probs_json(ctx);
            }
            choices.push_back(std::move(choice));
        }
        resp["choices"] = choices;

        // clean
        if (req->stream) {
            generated_text.clear();
            generated_tool_calls.clear();
        }

        return resp;
    }
};

struct embeddings_task : btask {
    explicit embeddings_task(int32_t tid)
        : btask(tid, TASK_EMBEDDINGS) {
    }

    // input
    std::vector<llama_tokens> tokenized_inputs;
    std::unique_ptr<breq> req;
    int32_t n_prefilling = 0;

    // prefill
    int32_t n_prefilled     = 0; // indicate how many tokens have been prefilled
    int64_t t_start_prefill = 0; // indicate the time when prefilling starts
    double t_prefilled      = 0; // indicate the time(ms) spent on prefilling
    double p_prefilled_tps  = 0;
    int32_t n_min_prefilled = 0;
    int32_t n_max_prefilled = 0;

    // output
    std::vector<std::vector<float>> embeds;

    Json to_json() {
        auto n_seq = int32_t(embeds.size());

        Json usage = {
            {"prompt_tokens", n_prefilled},
            {"total_tokens", n_prefilled},
            {"prompt_tokens_per_second", p_prefilled_tps},
            // additional details
            {"min_prompt_tokens", n_min_prefilled},
            {"max_prompt_tokens", n_max_prefilled},
        };

        if (req->get_type() == REQ_EMBED) {
            auto *dreq = dynamic_cast<embed_req *>(req.get());
            Json data  = Json::array();
            for (int32_t seq = 0; seq < n_seq; seq++) {
                Json item = {
                    {"index", seq},
                    {"object", "embedding"},
                };
                if (dreq->encoding_format != "base64") {
                    item["embedding"] = embeds[seq];
                } else {
                    item["embedding"] = encode_base64(reinterpret_cast<const unsigned char *>(embeds[seq].data()), embeds[seq].size());
                }
                data.push_back(item);
            }
            Json resp = {
                {"created", std::time(nullptr)},
                {"model", dreq->model},
                {"object", "list"},
                {"data", data},
                {"usage", usage},
            };
            return resp;
        }

        auto *dreq   = dynamic_cast<rerank_req *>(req.get());
        Json results = Json::array();
        for (int32_t seq = 0; seq < n_seq - (dreq->normalize ? 2 : 0); seq++) {
            Json item = {
                {"index", seq},
                {"score", embeds[seq][0]},
            };
            if (dreq->return_documents) {
                item["document"] = dreq->documents[seq].is_string() ? Json{{"text", dreq->documents[seq]}} : dreq->documents[seq];
            }
            results.push_back(item);
        }
        sort_rerank_results(results, 0, n_seq - 1 - (dreq->normalize ? 2 : 0));
        if (dreq->normalize) {
            float scr_max = MAX(embeds[n_seq - 2][0], results[0].at("score").get<float>());
            float scr_min = MIN(embeds[n_seq - 1][0], results[n_seq - 3].at("score").get<float>());
            float scr_dst = scr_max - scr_min;
            float a = 0.001, b = 0.998;
            if (scr_dst < 1e-6 || dreq->query.get<std::string>() == dreq->documents[json_value(results[0], "index", 0)].get<std::string>()) {
                scr_dst = scr_max;
                scr_min = 0.0f;
                a = 0, b = 1;
            }
            for (int32_t seq = 0; seq < n_seq - 2 && seq < dreq->top_n; seq++) {
                auto scr              = results[seq].at("score").get<float>();
                scr                   = a + (scr - scr_min) * b / scr_dst;
                results[seq]["score"] = scr;
            }
        }
        results.erase(results.begin() + dreq->top_n, results.end());
        Json resp = {
            {"model", dreq->model},
            {"results", results},
            {"usage", usage},
        };
        return resp;
    }
};

struct images_task : btask {
    explicit images_task(int32_t tid)
        : btask(tid, TASK_IMAGES) {
    }

    // input
    std::unique_ptr<image_req> req;

    // forward
    int32_t n_forward_steps = 0; // indicate how many forwarded steps have been called`
    int64_t t_start_forward = 0; // indicate the time when forwarding starts
    double t_forwarded      = 0; // indicate the time(ms) spent on forwaring
    double p_forwarded_sps  = 0;

    // reverse
    int32_t n_reverse_steps = 0; // indicate how many reversed steps have been called
    int64_t t_start_reverse = 0; // indicate the time when reversing starts
    double t_reversed       = 0; // indicate the time(ms) spent on reversing
    double p_reversed_sps   = 0;

    // output
    std::vector<std::string> b64_jsons;
    std::vector<int32_t> progressed_steps;
    std::vector<int32_t> progress_steps;

    Json to_json(const int32_t seq) {
        bool all_seqs      = seq < 0;
        bool stop          = all_seqs || progress_steps[seq] == progressed_steps[seq];
        bool include_usage = stop && json_value(req->stream_options, "include_usage", true) && (!req->stream || seq == int32_t(progressed_steps.size() - 1));

        Json resp = {
            {"created", std::time(nullptr)},
            {"model", req->get_model()},
            {"object", "list"},
            {"usage", Json()},
        };
        if (include_usage) {
            resp["usage"] = {
                {"time_to_process_ms", t_forwarded},
                {"time_per_generation_ms", t_reversed / n_reverse_steps},
                {"generation_per_second", p_reversed_sps},
            };
        }
        Json data = Json::array();
        for (int32_t idx = 0; idx < int32_t(b64_jsons.size()); idx++) {
            if (!all_seqs && idx != seq) {
                continue;
            }
            Json item = {
                {"index", idx},
                {"object", "image"},
                {"progressed_steps", progressed_steps[idx]},
                {"progress_steps", progress_steps[idx]},
                {"progress", stop ? 100 : float(progressed_steps[idx]) / float(progress_steps[idx]) * 100},
                {"finish_reason", stop ? "stop" : Json()},
                {"b64_json", std::move(b64_jsons[idx])},
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

// implementations // httpserver

struct httpserver_metrics {
    /* STABLE DIFFUSION */

    std::atomic<double> t_image_forwarded_total         = 0;
    std::atomic<uint64_t> n_image_steps_forwarded_total = 0;
    std::atomic<double> t_image_reversed_total          = 0;
    std::atomic<uint64_t> n_image_steps_reversed_total  = 0;

    /* LLAMA */

    std::atomic<double> t_tokens_prefilled_total          = 0;
    std::atomic<uint64_t> n_tokens_prefilled_total        = 0;
    std::atomic<double> t_tokens_decoded_total            = 0;
    std::atomic<uint64_t> n_tokens_decoded_total          = 0;
    std::atomic<uint64_t> n_tokens_drafted_total          = 0;
    std::atomic<uint64_t> n_tokens_drafted_accepted_total = 0;

    void on_image_forwarded(double t, uint64_t n_steps) {
        t_image_forwarded_total       = t_image_forwarded_total + t;
        n_image_steps_forwarded_total = n_image_steps_forwarded_total + n_steps;
    }

    void on_image_reversed(double t, uint64_t n_steps) {
        t_image_reversed_total       = t_image_reversed_total + t;
        n_image_steps_reversed_total = n_image_steps_reversed_total + n_steps;
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

struct httpserver {
    explicit httpserver(v2_httpserver_params &params)
        : params(params) {
        processing_loop = std::make_unique<HThreadPool>(1, 1);

        llama_numa_init(params.llm_params.numa);
        llama_backend_init();
    }

    ~httpserver() {
        llama_batch_free(batch);
        if (llm_ctx != nullptr) {
            llama_detach_threadpool(llm_ctx);
        }
        if (llm_ctx_clip != nullptr) {
            clip_free(llm_ctx_clip);
        }

        if (llm_ctx_draft != nullptr) {
            llama_batch_free(batch_draft);
            llama_detach_threadpool(llm_ctx_draft);
        }
        if (params.lookup_ngram_min > 0 && !params.llm_params.lookup_cache_dynamic.empty()) {
            common_ngram_cache_save(ngram_cache_dynamic, params.llm_params.lookup_cache_dynamic);
        }
        ngram_cache_static.clear();
        ngram_cache_dynamic.clear();

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
        SRV_INF("loading model '%s'\n", params.llm_params.model.c_str());

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
                    params.sd_params.model_alias = params.sd_params.model.substr(params.sd_params.model.find_last_of('/') + 1);
                } else if (params.sd_params.model.find_last_of('\\') != std::string::npos) {
                    params.sd_params.model_alias = params.sd_params.model.substr(params.sd_params.model.find_last_of('\\') + 1);
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

            SRV_INF("seed: %u, flash attn: %s, guidance: %.2f, strength: %.2f, sample method: %s, sampling steps: %d, cfg scale: %.2f, slg scale: %.2f, schedule method: %s\n",
                    params.sd_params.seed,
                    params.sd_params.flash_attn ? "true" : "false",
                    params.sd_params.sampling.guidance,
                    params.sd_params.sampling.strength,
                    sd_sample_method_to_argument(params.sd_params.sampling.sample_method),
                    params.sd_params.sampling.sampling_steps,
                    params.sd_params.sampling.cfg_scale,
                    params.sd_params.sampling.slg_scale,
                    sd_schedule_to_argument(params.sd_params.sampling.schedule_method));

            return true;
        }

        /* LLAMA */

        if (params.llm_params.model_alias.empty()) {
            if (params.llm_params.model.find_last_of('/') != std::string::npos) {
                params.llm_params.model_alias = params.llm_params.model.substr(params.llm_params.model.find_last_of('/') + 1);
            } else if (params.llm_params.model.find_last_of('\\') != std::string::npos) {
                params.llm_params.model_alias = params.llm_params.model.substr(params.llm_params.model.find_last_of('\\') + 1);
            } else {
                params.llm_params.model_alias = params.llm_params.model;
            }
        }

        // load multimodal projection model
        if (!params.llm_params.mmproj.empty()) {
            if (params.llm_params.n_ctx < 2048) {
                SRV_WRN("%s", "n_ctx is too small for multimodal projection, setting to 2048\n");
                params.llm_params.n_ctx = 2048;
            }
            // NB(thxCode): clip_context_params is a patch.
            clip_context_params llm_params_clip{
                /* max_image_size */ params.max_image_size,
                /* use_gpu */ params.llm_params.n_gpu_layers != 0,
                /* verbosity */ common_log_verbosity_thold,
            };
            llm_ctx_clip = clip_init(params.llm_params.mmproj.c_str(), llm_params_clip);
            if (llm_ctx_clip == nullptr) {
                SRV_ERR("failed to load multimodal project model, '%s'\n", params.llm_params.mmproj.c_str());
                return false;
            }
        }

        // load the draft model if needed
        if (!params.llm_params.speculative.model.empty() && params.llm_params.speculative.n_max > 0) {
            SRV_INF("loading draft model '%s'\n", params.llm_params.speculative.model.c_str());

            common_params llm_params_draft   = params.llm_params;
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
                SRV_ERR("failed to load draft model, '%s'\n", params.llm_params.speculative.model.c_str());
                return false;
            }
            llm_vocab_draft = llama_model_get_vocab(llm_model_draft);
            batch_draft     = llama_batch_init(int32_t(llama_n_ctx(llm_ctx_draft)), 0, 1);
        }

        // load the ngram cache if needed
        if (params.lookup_ngram_min > 0) {
            if (!params.llm_params.lookup_cache_static.empty()) {
                try {
                    ngram_cache_static = common_ngram_cache_load(params.llm_params.lookup_cache_static);
                } catch (std::ifstream::failure const &) {
                    SRV_ERR("failed to load static ngram cache, '%s'\n", params.llm_params.lookup_cache_static.c_str());
                    return false;
                }
            }
            if (!params.llm_params.lookup_cache_dynamic.empty()) {
                try {
                    ngram_cache_dynamic = common_ngram_cache_load(params.llm_params.lookup_cache_dynamic);
                } catch (std::ifstream::failure const &) {
                    SRV_WRN("failed to load dynamic ngram cache, '%s'\n", params.llm_params.lookup_cache_dynamic.c_str());
                }
            }
        }
        llm_init  = common_init_from_params(params.llm_params);
        llm_model = llm_init.model.get();
        llm_ctx   = llm_init.context.get();
        if (llm_model == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params.llm_params.model.c_str());
            return false;
        }
        llm_vocab = llama_model_get_vocab(llm_model);
        batch     = llama_batch_init(int32_t(llama_n_ctx(llm_ctx)), 0, 1);

        // check multimodal projection model compatibility
        if (llm_ctx_clip != nullptr) {
            const int32_t n_embd_clip = clip_n_mmproj_embd(llm_ctx_clip);
            const int32_t n_embd      = llama_model_n_embd(llm_model);
            if (n_embd_clip != n_embd) {
                SRV_ERR("multimodal projector embedding length is not equal to the model, n_embd_clip = %d, n_embd = %d\n", n_embd_clip, n_embd);
                return false;
            }
        }

        // check draft model compatibility if needed
        if (llm_ctx_draft != nullptr) {
            const bool vocab_type_draft = llama_vocab_type(llm_vocab_draft);
            const bool vocab_type       = llama_vocab_type(llm_vocab);
            if (vocab_type_draft != vocab_type) {
                SRV_ERR("draft model vocabulary type is not equal to the model, vocab_type_draft = %d, vocab_type = %d\n", vocab_type_draft, vocab_type);
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

            struct ggml_threadpool_params tpp_batch = ggml_threadpool_params_from_cpu_params(params.llm_params.cpuparams_batch);
            threadpool_batch                        = ggml_threadpool_new(&tpp_batch);
            if (!threadpool_batch) {
                SRV_ERR("threadpool_batch create failed : n_threads %d\n", tpp_batch.n_threads);
                return false;
            }

            llama_attach_threadpool(llm_ctx, threadpool, threadpool_batch);
            if (llm_ctx_draft != nullptr) {
                llama_attach_threadpool(llm_ctx_draft, threadpool, threadpool_batch);
            }
        }

        if (!support_completion()) {
            return true;
        }

        cache_prompt = params.cache_prompt;
        // disable prompt caching if using clip model.
        cache_prompt = cache_prompt && llm_ctx_clip == nullptr;
        // disable prompt caching if disallowing context shifting.
        cache_prompt = cache_prompt && params.llm_params.ctx_shift;
        // disable prompt caching if using remote backend.
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                if (ggml_backend_reg_name(reg) == std::string("RPC")) {
                    cache_prompt = false;
                    break;
                }
            }
        }
        SRV_INF("prompt caching %s\n", cache_prompt ? "enabled" : (params.cache_prompt ? "unsupported" : "disabled"));

        // chat template
        {
            chat_templates = common_chat_templates_init(llm_model, params.llm_params.chat_template);

            // NB(thxCode): llama_chat_template_alias is a patch.
            std::string alias = llama_chat_template_alias(common_chat_templates_source(chat_templates.get()));

            if (params.llm_params.use_jinja) {
                // NB(thxCode): common_chat_templates_supports_tool_calls is a patch.
                support_tool_calls = common_chat_templates_supports_tool_calls(chat_templates.get());
            } else {
                if (alias == "chatml" || alias == "chatglm4") {
                    // <tool_call>
                    // {"name":"","arguments":{}}
                    // </tool_call>
                    support_tool_calls = true;
                    llama_tokens ids   = common_tokenize(llm_vocab, "<tool_call>", false, true);
                    if (ids.size() == 1) {
                        tool_call_start_token = ids[0];
                    } else {
                        tool_call_start_words = {"<tool_call>", "<tool_call>\n"};
                        tool_call_start_trim  = true;
                    }
                    ids = common_tokenize(llm_vocab, "</tool_call>", false, true);
                    if (ids.size() == 1) {
                        tool_call_end_token = ids[0];
                    } else {
                        tool_call_end_words = {"</tool_call>", "</tool_call>\n", "</tool_call> "};
                        tool_call_end_trim  = true;
                    }
                } else if (string_starts_with(alias, "mistral-")) {
                    // [TOOL_CALLS][{"name":"","arguments":{}}]
                    support_tool_calls = true;
                    llama_tokens ids   = common_tokenize(llm_vocab, "[TOOL_CALLS]", false, true);
                    if (ids.size() == 1) {
                        tool_call_start_token = ids[0];
                    } else {
                        tool_call_start_words = {"[TOOL_CALLS]"};
                        tool_call_start_trim  = true;
                    }
                    tool_call_end_words = {"}]", "}]\n", "}] "};
                } else if (alias == "llama3") {
                    // {"name":"","arguments":{}}
                    support_tool_calls    = true;
                    tool_call_start_words = {"{\""};
                    tool_call_end_words   = {"}}", "}}\n", "}} "};
                } else if (alias == "granite") {
                    // <tool_call>[{"name":"","arguments":{}}]
                    support_tool_calls    = true;
                    tool_call_start_words = {"<tool_call>"};
                    tool_call_start_trim  = true;
                    tool_call_end_words   = {"}]", "}]\n", "}] "};
                }
                if (!tool_call_start_words.empty()) {
                    for (const std::string &word : tool_call_start_words) {
                        tool_call_start_words_longest_length = MAX(tool_call_start_words_longest_length, word.length());
                    }
                    tool_call_start_words_longest_length = tool_call_start_words_longest_length + int32_t(std::ceil(float(tool_call_start_words_longest_length) / 3.0));
                }
            }

            {
                if (alias == "deepseek3") {
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
                support_reasoning = reasoning_start_token != LLAMA_TOKEN_NULL && reasoning_end_token != LLAMA_TOKEN_NULL;
                if (!support_reasoning) {
                    reasoning_start_token = LLAMA_TOKEN_NULL;
                    reasoning_end_token   = LLAMA_TOKEN_NULL;
                }
            }

            std::string prompt;
            {
                common_chat_templates_inputs inputs;
                inputs.messages = std::vector<common_chat_msg>({
                    {"system", "You are a helpful assistant.", {}, {}, "", "", ""},
                    {"user", "Hello.", {}, {}, "", "", ""},
                    {"assistant", "Hi! How can I help you today?", {}, {}, "", "", ""},
                    {"user", "What is the weather like in Beijing?", {}, {}, "", "", ""},
                });
                if (support_tool_calls) {
                    inputs.messages.push_back({"assistant", "", {}, {{"get_weather", R"({"location":"Beijing"})", "123456789"}}, "", "", ""});
                    inputs.messages.push_back({"tool", R"({"weather":"Sunny"})", {}, {}, "", "", "123456789"});
                    inputs.messages.push_back({"assistant", "The weather is Sunny.", {}, {}, "", "", "123456789"});
                    inputs.tools = std::vector<common_chat_tool>({
                        {"get_weather", "", R"({"type":"object","properties":{"location":{"type":"string"}}})"},
                        {"get_temperature", "Return the temperature according to the location.", R"({"type":"object","properties":{"location":{"type":"string"}}})"},
                    });
                }
                inputs.tool_choice           = COMMON_CHAT_TOOL_CHOICE_NONE;
                inputs.add_generation_prompt = true;
                inputs.use_jinja             = params.llm_params.use_jinja;
                // NB(thxCode): common_chat_templates_apply2 is a patch.
                common_chat_params example = common_chat_templates_apply2(llm_model, chat_templates.get(), inputs);
                prompt                     = example.prompt;
            }

            SRV_INF("chat template, built-in: %s, jinja rendering: %s, tool call: %s, reasoning: %s, example:\n%s\n",
                    params.llm_params.chat_template.empty() || !params.llm_params.use_jinja ? "true" : "false",
                    params.llm_params.use_jinja ? "enabled" : "disabled",
                    support_tool_calls ? "supported" : "unsupported",
                    support_reasoning ? "supported" : "unsupported",
                    prompt.c_str());
        }

        // sample tokens per second
        if (params.n_tps < 0) {
            SRV_INF("%s", "sampling tokens per second, this will take some time...\n");
            const int32_t n_check            = MIN(int32_t(llama_n_ctx(llm_ctx)), params.llm_params.n_ubatch);
            llama_tokens check_prompt_tokens = {llama_vocab_bos(llm_vocab)};
            common_sampler *check_smpl       = common_sampler_init(llm_model, params.llm_params.sampling);
            int64_t t_start_decoding         = ggml_time_us();
            int32_t n_check_decoded          = 0;
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
            params.n_tps = ceil(1.e3 / (double(ggml_time_us() - t_start_decoding) / 1.e3) * n_check_decoded);
            common_sampler_free(check_smpl);
            llama_kv_self_clear(llm_ctx);
            llama_synchronize(llm_ctx);
            llama_perf_context_reset(llm_ctx);
            SRV_INF("sampled tokens per second, tps = %d\n", params.n_tps);
        }

        return true;
    }

    int start() {
        SRV_INF("%s", "starting\n");
        // register routes
        HttpService service;
        service.enable_access_log = false;
        service.keepalive_timeout = params.conn_keepalive * 1000;
        service.preprocessor      = preprocessor;
        service.postprocessor     = postprocessor;
        service.AllowCORS();
#define SAFE_HANDLER(HANDLER)                                                                                 \
    [this](const HttpContextPtr &ctx) {                                                                       \
        try {                                                                                                 \
            return this->HANDLER(ctx);                                                                        \
        } catch (const std::invalid_argument &re) {                                                           \
            return send_error_json(ctx->writer, HTTP_STATUS_BAD_REQUEST, re.what());                          \
        } catch (const std::exception &e) {                                                                   \
            return send_error_json(ctx->writer, HTTP_STATUS_INTERNAL_SERVER_ERROR, e.what());                 \
        } catch (...) {                                                                                       \
            return send_error_json(ctx->writer, HTTP_STATUS_INTERNAL_SERVER_ERROR, "Unknown error occurred"); \
        }                                                                                                     \
    }
        service.GET("/health", SAFE_HANDLER(handle_health));
        if (params.llm_params.endpoint_metrics) {
            service.GET("/metrics", SAFE_HANDLER(handle_metrics));
        }
        service.GET("/v1/models", SAFE_HANDLER(handle_models));
        /* STABLE DIFFUSION */
        if (support_image()) {
            service.POST("/v1/images/{category}", SAFE_HANDLER(handle_images));
        }
        /* LLAMA */
        else {
            service.POST("/tokenize", SAFE_HANDLER(handle_tokenize));
            service.POST("/detokenize", SAFE_HANDLER(handle_detokenize));
            if (support_completion()) {
                service.POST("/v1/completions", SAFE_HANDLER(handle_legacy_completions));
                service.POST("/v1/chat/completions", SAFE_HANDLER(handle_chat_completions));
            }
            if (support_embedding()) {
                service.POST("/v1/embeddings", SAFE_HANDLER(handle_embeddings));
            }
            if (support_reranking()) {
                service.POST("/v1/rerank", SAFE_HANDLER(handle_rerank));
            }
        }
#undef SAFE_HANDLER
        // start loop
        processing_loop->start();

        // start server
        HttpServer server(&service);
        server.setHost(params.llm_params.hostname.c_str());
        server.setPort(params.llm_params.port);
        server.setThreadNum(params.llm_params.n_threads_http);
        SRV_INF("listening host = %s, port = %d\n", params.llm_params.hostname.c_str(), params.llm_params.port);
        server.run();

        processing_loop->stop();
        return server.stop();
    }

  private:
    //
    // Attributes
    //

    v2_httpserver_params params;
    httpserver_metrics metrics;
    std::unique_ptr<HThreadPool> processing_loop;

    // lora
    std::vector<common_adapter_lora_info> lora_adapters;

    /* STABLE DIFFUSION */

    // seq
    int32_t forwarded_seqs = 0;
    int32_t reversed_seqs  = 0;

    // model
    common_sd_init_result sd_init;
    v2_stablediffusion_context *sd_ctx = nullptr;

    /* LLAMA */

    // seq
    int32_t batched_seqs = 0;
    int32_t decoded_seqs = 0;

    // model
    common_init_result llm_init;
    llama_model *llm_model       = nullptr;
    llama_context *llm_ctx       = nullptr;
    const llama_vocab *llm_vocab = nullptr;
    llama_batch batch            = {};
    task_type batch_type         = TASK_UNKNOWN;

    // model addition
    bool cache_prompt = false;
    common_chat_templates_ptr chat_templates;

    // clip model
    clip_ctx *llm_ctx_clip = nullptr;

    // draft-model speculative decoding
    common_init_result llm_init_draft;
    llama_model *llm_model_draft       = nullptr;
    llama_context *llm_ctx_draft       = nullptr;
    const llama_vocab *llm_vocab_draft = nullptr;
    llama_batch batch_draft            = {};

    // model-free speculative decoding
    common_ngram_cache ngram_cache_static;
    common_ngram_cache ngram_cache_dynamic;

    // thread pool
    ggml_threadpool *threadpool       = nullptr;
    ggml_threadpool *threadpool_batch = nullptr;

    // tool calls
    bool support_tool_calls = false;
    // non-jinja tool calls
    llama_token tool_call_start_token              = LLAMA_TOKEN_NULL;
    std::vector<std::string> tool_call_start_words = {};
    size_t tool_call_start_words_longest_length    = 0;
    bool tool_call_start_trim                      = false;
    llama_token tool_call_end_token                = LLAMA_TOKEN_NULL;
    std::vector<std::string> tool_call_end_words   = {};
    bool tool_call_end_trim                        = false;

    // reasoning
    bool support_reasoning            = false;
    llama_token reasoning_start_token = LLAMA_TOKEN_NULL;
    llama_token reasoning_end_token   = LLAMA_TOKEN_NULL;

    static inline int32_t get_task_id() {
        thread_local static int32_t thread_id = -1;
        if (thread_id == -1) {
            static std::atomic<int32_t> next_thread_id{0};
            thread_id = next_thread_id++;
        }
        return thread_id;
    }

    inline bool support_tokenize() const {
        return llm_vocab != nullptr;
    }

    inline bool support_completion() const {
        // NB(thxCode): llama_causal_attn is a patch.
        return llm_ctx != nullptr && llama_causal_attn(llm_ctx) && !params.llm_params.reranking;
    }

    inline bool support_embedding() const {
        return llm_ctx != nullptr && params.llm_params.embedding;
    }

    inline bool support_reranking() const {
        // NB(thxCode): llama_causal_attn is a patch.
        return llm_ctx != nullptr && !llama_causal_attn(llm_ctx) && params.llm_params.reranking;
    }

    inline bool support_image() const {
        return sd_ctx != nullptr;
    }

    static inline bool is_mrope_model(llama_model *lm) {
        return lm != nullptr && llama_model_rope_type(lm) == LLAMA_ROPE_TYPE_MROPE;
    }

    //
    // Logics
    //

#if defined(WIN64) || defined(_WIN64) || defined(WIN32) || defined(_WIN32)
#define PIN_THREAD                                      \
    int32_t cpu = GetCurrentProcessorNumber();          \
    GROUP_AFFINITY groupAffinity;                       \
    ZeroMemory(&groupAffinity, sizeof(GROUP_AFFINITY)); \
    groupAffinity.Mask = 1ULL << cpu;                   \
    SetThreadGroupAffinity(GetCurrentThread(), &groupAffinity, NULL);
#elif defined(linux) || defined(__linux) || defined(__linux__)
#define PIN_THREAD                \
    int32_t cpu = sched_getcpu(); \
    cpu_set_t cpuset;             \
    CPU_ZERO(&cpuset);            \
    CPU_SET(cpu, &cpuset);        \
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#else
#define PIN_THREAD
#endif

    int process(const HttpContextPtr &ctx, const std::unique_ptr<btask> &&task_ptr) {
        PIN_THREAD;

        switch (task_ptr->get_type()) {
            case TASK_COMPLETIONS: {
                auto *task = dynamic_cast<completions_task *>(task_ptr.get());

                const char *rid       = task->req->get_rid();
                const req_type rtype  = task->req->get_type();
                const auto n_prompt   = int32_t(task->tokenized_prompts.size());
                const auto n_ctx      = int32_t(llama_n_ctx(llm_ctx));
                const bool need_mrope = is_mrope_model(llm_model);

                SRV_INFV(3, "rid %s | processing tokens: %d\n", rid, task->n_prefilling);

                auto append_decode_token = [&](int32_t tok_idx, llama_token tok) {
                    task->generated_tokens.push_back(tok);
                    task->n_decoded++;
                    task->n_decoding_budget--;

                    if (task->req->logprobs > 0) {
                        const std::vector<llama_token_data> cur = get_token_probabilities(llm_ctx, tok_idx);
                        const size_t n_vocab                    = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(llm_ctx)));
                        const size_t n_probs                    = task->req->logprobs;
                        // set probability for sampled token
                        for (size_t i = 0; i < n_vocab; i++) {
                            if (cur[i].id == tok) {
                                task->generated_probs.push_back(cur[i].p);
                                break;
                            }
                        }
                        // set probability for top n_probs tokens
                        task->generated_top_probs.emplace_back();
                        for (size_t i = 0; i < MIN(n_vocab, n_probs); i++) {
                            task->generated_top_probs[task->generated_top_probs.size() - 1].emplace_back(cur[i].id, cur[i].p);
                        }
                    }
                };

                task->t_start_prefill = ggml_time_us();

                int32_t seq_id  = task->get_tid();
                int32_t seq_pos = 0;
                do {
                    /**
                     * batch
                     */

                    // result
                    int32_t seq_batch_end = -1;

                    int32_t n_prompt_prefilled          = 0;
                    int32_t n_prompt_prefilled_text_pos = 0;
                    do {
                        // batched -> -1|seq_batch_end
                        std::future<int32_t> batched = processing_loop->commit([&]() {
                            if (batched_seqs <= decoded_seqs) {
                                batched_seqs = 0;
                                decoded_seqs = 0;
                                batch_type   = TASK_COMPLETIONS;

                                // apply lora adapters, only need to do it once per batch
                                if (!equal_lora(task->req->lora_adapters, lora_adapters)) {
                                    try {
                                        common_set_adapter_lora(llm_ctx, task->req->lora_adapters);
                                        lora_adapters = task->req->lora_adapters;
                                    } catch (const std::exception &e) {
                                        SRV_FUNC_ERR("process", "rid %s | prefilling, failed to apply lora %s\n", rid, e.what());
                                    }
                                }
                            } else if (batch_type != TASK_COMPLETIONS) {
                                SRV_FUNC_DBG("process", "rid %s | waiting previous batch finished: not completions batch\n", rid);
                                return -1;
                            } else if (!task->tokenized_prompts_include_images && batch.n_tokens > n_ctx) {
                                SRV_FUNC_DBG("process", "rid %s | waiting previous batch finished: not enough space to place all tokens\n", rid);
                                return -1;
                            } else if (task->tokenized_prompts_include_images && (task->n_prefilling + batch.n_tokens > n_ctx)) { // for vision, must place all tokens once
                                SRV_FUNC_DBG("process", "rid %s | waiting previous batch finished: not enough space to place all tokens\n", rid);
                                return -1;
                            } else if (!equal_lora(task->req->lora_adapters, lora_adapters)) {
                                SRV_FUNC_DBG("process", "rid %s | waiting previous batch finished: lora adapters not matched\n", rid);
                                return -1;
                            } else if (!params.llm_params.cont_batching) {
                                SRV_FUNC_DBG("process", "rid %s | waiting previous batch finished: not continue batching\n", rid);
                                return -1;
                            } else if (decoded_seqs > 0) {
                                SRV_FUNC_DBG("process", "rid %s | waiting previous batch finished: getting results\n", rid);
                                return -1;
                            }

                            /* prefill start */

                            if (task->n_prefilled < task->n_prefilling) {
                                if (task->n_prefilled == 0) {
                                    batched_seqs++; // increase to avoid entering the wrong task type

                                    if (!task->tokenized_prompts_include_images && cache_prompt) {
                                        // TODO: find the cache
                                        task->n_prefilled_cached = 0;
                                    }

                                    // mask kv cache
                                    llama_kv_self_seq_rm(llm_ctx, seq_id, seq_pos, -1);
                                    if (llm_ctx_draft) {
                                        llama_kv_self_seq_rm(llm_ctx_draft, seq_id, seq_pos, -1);
                                    }
                                }

                                do {
                                    if (std::holds_alternative<llama_tokens>(task->tokenized_prompts[n_prompt_prefilled])) {
                                        llama_tokens tokenized_text = std::get<llama_tokens>(task->tokenized_prompts[n_prompt_prefilled]);
                                        const auto n_text_pos       = int32_t(tokenized_text.size());
                                        for (; n_prompt_prefilled_text_pos < n_text_pos; n_prompt_prefilled_text_pos++) {
                                            if (1 + batch.n_tokens >= n_ctx) { // disallow batch's tokens size be equal to n_ctx
                                                SRV_FUNC_DBG("process", "rid %s | prefilling, not enough space to fill, waiting\n", rid);
                                                goto out_of_prefill;
                                            }
                                            const llama_token tok = tokenized_text[n_prompt_prefilled_text_pos];
                                            const bool need_embed = n_prompt_prefilled_text_pos + 1 == n_text_pos;
                                            if (need_mrope) {
                                                common_batch_add_with_mrope(batch, tok, seq_pos, n_text_pos, {seq_id}, need_embed);
                                                if (llm_ctx_draft != nullptr) {
                                                    common_batch_add_with_mrope(batch_draft, tok, seq_pos, n_text_pos, {seq_id}, need_embed);
                                                }
                                            } else {
                                                common_batch_add(batch, tok, seq_pos, {seq_id}, need_embed);
                                                if (llm_ctx_draft != nullptr) {
                                                    common_batch_add(batch_draft, tok, seq_pos, {seq_id}, need_embed);
                                                }
                                            }
                                            seq_pos++;
                                            task->n_prefilled++;
                                        }
                                        n_prompt_prefilled_text_pos = 0;
                                    } else {
                                        std::unique_ptr<llava_image_embed> tokenized_image = std::get<std::unique_ptr<llava_image_embed>>(std::move(task->tokenized_prompts[n_prompt_prefilled]));
                                        const int32_t n_image_pos                          = tokenized_image->n_image_pos;
                                        if (need_mrope) {

                                        } else {
                                        }
                                        task->n_prefilled += n_image_pos;
                                    }

                                    n_prompt_prefilled++;
                                } while (n_prompt_prefilled < n_prompt);

                            out_of_prefill:

                                if (n_prompt_prefilled == n_prompt) {
                                    if (task->n_decoded == 0) {
                                        if (batched_seqs == 1) {
                                            SRV_FUNC_DBG("process", "rid %s | prefilling\n", rid);
                                        } else {
                                            SRV_FUNC_DBG("process", "rid %s | prefilling in batch\n", rid);
                                        }
                                    }
                                    return batch.n_tokens - 1;
                                }

                                return -1;
                            }

                            /* decode next */

                            batched_seqs++; // increase to avoid entering the wrong task type

                            const llama_token tok = task->generated_tokens[task->generated_tokens.size() - 1];
                            if (need_mrope) {
                                common_batch_add_with_mrope(batch, tok, seq_pos, 1, {seq_id}, true);
                                if (llm_ctx_draft != nullptr) {
                                    common_batch_add_with_mrope(batch_draft, tok, seq_pos, 1, {seq_id}, true);
                                }
                            } else {
                                common_batch_add(batch, tok, seq_pos, {seq_id}, true);
                                if (llm_ctx_draft != nullptr) {
                                    common_batch_add(batch_draft, tok, seq_pos, {seq_id}, true);
                                }
                            }
                            seq_pos++;
                            if (batched_seqs == 1) {
                                SRV_FUNC_DBG("process", "rid %s | prefilling, decode next\n", rid);
                            } else {
                                SRV_FUNC_DBG("process", "rid %s | prefilling in batch, decode next\n", rid);
                            }
                            return batch.n_tokens - 1;
                        });

                        seq_batch_end = batched.get();
                        if (seq_batch_end == -1) {
                            std::this_thread::yield();
                            continue;
                        }

                        break;
                    } while (true);

                    /**
                     * decode
                     */

                    // decoded -> 0|-1|-2|...
                    std::future<int32_t> decoded = processing_loop->commit([&]() {
                        // only decode once
                        if (decoded_seqs == 0) {
                            SRV_FUNC_DBG("process", "rid %s | decoding\n", rid);
                            llama_set_embeddings(llm_ctx, false);
                            const int32_t decoded = llama_decode(llm_ctx, batch);
                            // TODO process -3
                            common_batch_clear(batch);
                            if (decoded != 0) {
                                decoded_seqs++; // increase to avoid deadlock
                                SRV_FUNC_ERR("process", "rid %s | decoding, failed to decode: decoded = %d\n", rid, decoded);
                                return decoded;
                            }
                        } else {
                            SRV_FUNC_DBG("process", "rid %s | decoding in batch\n", rid);
                        }

                        // sample
                        // TODO draft sample

                        const llama_token tok = common_sampler_sample(task->sampler, llm_ctx, seq_batch_end);
                        common_sampler_accept(task->sampler, tok, true);
                        append_decode_token(seq_batch_end, tok);

                        if (task->n_decoded == 1) {
                            task->t_prefilled    = double(ggml_time_us() - task->t_start_prefill) / 1.e3;
                            task->t_start_decode = ggml_time_us();
                        } else {
                            task->t_decoded = double(ggml_time_us() - task->t_start_decode) / 1.e3;
                        }

                        decoded_seqs++; // increase to avoid deadlock
                        return 0;
                    });
                    if (decoded.get() != 0) {
                        return send_error_json(ctx->writer, HTTP_STATUS_INTERNAL_SERVER_ERROR, "failed to decode");
                    }

                    // get token string
                    std::string token_str;
                    for (; task->n_decoded_processed < task->n_decoded; task->n_decoded_processed++) {
                        llama_token tok = task->generated_tokens[task->n_decoded_processed];
                        // accept special token
                        bool special = params.llm_params.special || task->req->sampling.preserved_tokens.find(tok) != task->req->sampling.preserved_tokens.end();
                        token_str += common_token_to_piece(llm_ctx, tok, special);
                        // check if the token is a reasoning token
                        if (support_reasoning) {
                            if (!task->reasoning_start_found) {
                                task->reasoning_start_found = tok == reasoning_start_token;
                            } else if (!task->reasoning_end_found) {
                                if (tok == reasoning_end_token) {
                                    task->reasoning_end_found = true;
                                } else {
                                    task->n_reasoning++;
                                }
                            } else {
                                task->reasoning_finished = true;
                            }
                        }
                    }

                    task->generated_text += token_str;

                    bool send_text = get_position_of_utf8(task->generated_text) == task->generated_text.size();

                    // check stop
                    //// check eog
                    if (llama_vocab_is_eog(llm_vocab, task->generated_tokens[task->generated_tokens.size() - 1])) {
                        SRV_DBG("rid %s | stopped by EOG\n", rid);
                        task->generated_finish_reason = "stop";
                    }
                    //// check budget
                    else if (task->n_decoding_budget <= 0) {
                        SRV_DBG("rid %s | stopped by length\n", rid);
                        task->generated_finish_reason = "length";
                    }
                    //// check stop word
                    else if (send_text) {
                        size_t stop_pos = std::string::npos;
                        for (const std::string &word : task->req->stop) {
                            size_t pos = task->generated_text.find(word, task->generated_text.size() - token_str.size());
                            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                                SRV_DBG("rid %s | stopped by word\n", rid);
                                task->generated_finish_reason = "stop";
                                stop_pos                      = pos;
                                break;
                            }
                        }
                        if (stop_pos != std::string::npos) {
                            task->generated_tokens.erase(task->generated_tokens.begin() + long(stop_pos), task->generated_tokens.end());
                        }
                    }

                    if (!task->generated_finish_reason.empty()) {
                        break;
                    }

                    // output
                    if (send_text && task->req->stream) {
                        int32_t sc = send_event_json(ctx->writer, task->to_json(llm_ctx));
                        if (sc != HTTP_STATUS_OK) {
                            SRV_ERR("rid %s | decoding, failed to send chunk, code = %d\n", rid, sc);
                            return sc;
                        }
                    } else if (!ctx->writer->isConnected()) {
                        SRV_ERR("rid %s | decoding, connection closed\n", rid);
                        return HTTP_STATUS_REQUEST_TIMEOUT;
                    }
                } while (true);

                metrics.on_tokens_prefilled(task->t_prefilled, task->n_prefilled);
                task->p_prefilled_tps = 1.e3 / task->t_prefilled * task->n_prefilled;
                metrics.on_tokens_decoded(task->t_decoded, task->n_decoded, task->n_drafted, task->n_drafted_accepted);
                task->p_decoded_tps = 1.e3 / task->t_decoded * task->n_decoded;
                task->p_drafted_apt = task->n_drafted == 0 ? 0.0 : double(task->n_drafted_accepted) / double(task->n_drafted) * 100.0;

                SRV_INF("rid %s | prefill_t = %d, prefill_tps = %.2f tps, ttft = %.2fms, decode_t = %d, decode_tps = %.2f tps, tpot = %.2fms, draft_t = %d, draft_apt = %.2f%%\n", rid, task->n_prefilled, task->p_prefilled_tps, task->t_prefilled, task->n_decoded, task->p_decoded_tps, task->t_decoded / double(task->n_decoded), task->n_drafted, task->p_drafted_apt);

                // output
                int32_t sc = send_json(ctx->writer, task->to_json(llm_ctx));
                if (sc != HTTP_STATUS_OK) {
                    SRV_ERR("rid %s | decoding, failed to send chunk, code = %d\n", rid, sc);
                }
                return sc;
            }
            case TASK_EMBEDDINGS: {
                auto *task = dynamic_cast<embeddings_task *>(task_ptr.get());

                const char *rid              = task->req->get_rid();
                const req_type rtype         = task->req->get_type();
                const auto n_seq             = int32_t(task->tokenized_inputs.size());
                const bool need_embed        = rtype == REQ_EMBED && llama_pooling_type(llm_ctx) == LLAMA_POOLING_TYPE_NONE;
                const auto n_ctx             = int32_t(llama_n_ctx(llm_ctx));
                const bool need_mrope        = is_mrope_model(llm_model);
                const bool is_embedding_only = !llama_causal_attn(llm_ctx);
                const int32_t n_embed        = llama_model_n_embd(llm_model);

                SRV_INFV(3, "rid %s | processing tokens: %d\n", rid, task->n_prefilling);

                // result
                task->embeds.reserve(n_seq);

                task->t_start_prefill = ggml_time_us();

                int32_t n_seq_prefilled = 0;
                do {
                    /**
                     * batch
                     */

                    // batched -> [{seq_id, seq_batch_end}*]
                    std::future<std::vector<std::pair<int32_t, int32_t>>> batched = processing_loop->commit([&]() {
                        std::vector<std::pair<int32_t, int32_t>> seq_batches;

                        if (batched_seqs <= decoded_seqs) {
                            batched_seqs = 0;
                            decoded_seqs = 0;
                            batch_type   = TASK_EMBEDDINGS;
                        } else if (batch_type != TASK_EMBEDDINGS) {
                            SRV_FUNC_DBG("process", "rid %s | waiting previous batch finished: not completions batch\n", rid);
                            return seq_batches;
                        } else if (batch.n_tokens > n_ctx) {
                            SRV_FUNC_DBG("process", "rid %s | waiting previous batch finished: not enough space to place all tokens\n", rid);
                            return seq_batches;
                        } else if (!params.llm_params.cont_batching || batched_seqs >= params.n_parallel) {
                            SRV_FUNC_DBG("process", "rid %s | waiting previous batch finished: not continue batching or full batch\n", rid);
                            return seq_batches;
                        } else if (decoded_seqs > 0) {
                            SRV_FUNC_DBG("process", "rid %s | waiting previous batch finished: getting results\n", rid);
                            return seq_batches;
                        }

                        for (; n_seq_prefilled < n_seq; n_seq_prefilled++) {
                            const auto n_pos = int32_t(task->tokenized_inputs[n_seq_prefilled].size());
                            if (n_pos + batch.n_tokens > n_ctx) { // allow batch's tokens size be equal to n_ctx
                                SRV_FUNC_DBG("process", "rid %s | prefilling, not enough space to fill, waiting\n", rid);
                                break;
                            }

                            int32_t seq_id = batched_seqs;
                            batched_seqs++; // increase to avoid entering the wrong task type

                            // clear kv cache
                            if (!is_embedding_only) {
                                llama_kv_self_seq_rm(llm_ctx, seq_id, 0, -1);
                            }

                            for (llama_pos seq_pos = 0; seq_pos < n_pos; seq_pos++) {
                                if (need_mrope) {
                                    common_batch_add_with_mrope(batch, task->tokenized_inputs[n_seq_prefilled][seq_pos], seq_pos, n_pos, {seq_id}, need_embed);
                                } else {
                                    common_batch_add(batch, task->tokenized_inputs[n_seq_prefilled][seq_pos], seq_pos, {seq_id}, need_embed);
                                }
                            }
                            task->n_prefilled += n_pos;
                            task->n_min_prefilled = task->n_min_prefilled == 0 ? n_pos : MIN(task->n_min_prefilled, n_pos);
                            task->n_max_prefilled = MAX(task->n_max_prefilled, n_pos);

                            seq_batches.emplace_back(seq_id, batch.n_tokens - 1);
                        }

                        if (batched_seqs == int32_t(seq_batches.size())) {
                            SRV_FUNC_DBG("process", "rid %s | prefilling\n", rid);
                        } else {
                            SRV_FUNC_DBG("process", "rid %s | prefilling in batch\n", rid);
                        }

                        return seq_batches;
                    });
                    std::vector<std::pair<int32_t, int32_t>> seq_batches          = batched.get();
                    if (seq_batches.empty()) {
                        std::this_thread::yield();
                        continue;
                    }

                    /**
                     * decode
                     */

                    // decoded -> 0|-1|-2|...
                    std::future<int32_t> decoded = processing_loop->commit([&]() {
                        // only decode once
                        if (decoded_seqs == 0) {
                            SRV_FUNC_DBG("process", "rid %s | embedding\n", rid);
                            llama_set_embeddings(llm_ctx, true);
                            const int32_t decoded = llama_decode(llm_ctx, batch);
                            common_batch_clear(batch);
                            if (decoded != 0) {
                                decoded_seqs += int32_t(seq_batches.size()); // increase to avoid deadlock
                                SRV_FUNC_ERR("process", "rid %s | embedding, failed to decode: result = %d\n", rid, decoded);
                                return decoded;
                            }
                        } else {
                            SRV_FUNC_DBG("process", "rid %s | embedding in batch\n", rid);
                        }

                        for (const std::pair<int32_t, int32_t> &seq_batch : seq_batches) {
                            const int32_t seq_id        = seq_batch.first;
                            const int32_t seq_batch_end = seq_batch.second;

                            if (rtype == REQ_EMBED) {
                                task->embeds.emplace_back(n_embed, 0.0f);
                            } else {
                                task->embeds.emplace_back(1, -1.e6);
                            }

                            const float *embed = llama_get_embeddings_seq(llm_ctx, seq_id);
                            if (embed == nullptr) {
                                embed = llama_get_embeddings_ith(llm_ctx, seq_batch_end);
                            }
                            if (embed == nullptr) {
                                SRV_FUNC_ERR("process", "rid %s | embedding, failed to decode: get embeddings\n", rid);
                                continue;
                            }

                            if (rtype == REQ_EMBED) {
                                common_embd_normalize(embed, task->embeds[task->embeds.size() - 1].data(), n_embed, 2);
                            } else {
                                task->embeds[task->embeds.size() - 1][0] = embed[0];
                            }
                        }

                        decoded_seqs += int32_t(seq_batches.size()); // increase to avoid deadlock
                        return 0;
                    });
                    if (decoded.get() != 0) {
                        return send_error_json(ctx->writer, HTTP_STATUS_INTERNAL_SERVER_ERROR, "failed to decode");
                    }
                } while (n_seq_prefilled < n_seq);

                task->t_prefilled = double(ggml_time_us() - task->t_start_prefill) / 1.e3;
                metrics.on_tokens_prefilled(task->t_prefilled, task->n_prefilled);
                task->p_prefilled_tps = 1.e3 / task->t_prefilled * task->n_prefilled;

                SRV_INF("rid %s | prefill_t = %d, prefill_tps = %.2f tps, ttft = %.2fms, total_t = %d, min_prompt_t = %d, max_prompt_t = %d\n", rid, task->n_prefilled, task->p_prefilled_tps, task->t_prefilled, task->n_prefilled, task->n_min_prefilled, task->n_max_prefilled);

                // output
                return send_json(ctx->writer, task->to_json());
            }
            case TASK_IMAGES: {
                auto *task = dynamic_cast<images_task *>(task_ptr.get());

                /**
                 * forward
                 */

                const char *rid      = task->req->get_rid();
                const req_type rtype = task->req->get_type();
                const int32_t n_seq  = task->req->get_n();

                // result
                std::vector<std::unique_ptr<v2_stablediffusion_sampling_stream>> streams;
                streams.reserve(n_seq);

                task->t_start_forward = ggml_time_us();

                int32_t n_seq_forwarded = 0;
                do {
                    // forwarded -> std::unique_ptr<v2_stablediffusion_sampling_stream>?
                    std::future<std::unique_ptr<v2_stablediffusion_sampling_stream>> forwarded = processing_loop->commit([&]() {
                        std::unique_ptr<v2_stablediffusion_sampling_stream> stream = nullptr;

                        if (forwarded_seqs <= reversed_seqs) {
                            forwarded_seqs = 0;
                            reversed_seqs  = 0;

                            // apply lora adapters, only need to do it once per batch
                            if (!equal_lora(task->req->lora_adapters, lora_adapters)) {
                                try {
                                    sd_ctx->apply_lora_adapters(task->req->lora_adapters);
                                    lora_adapters = task->req->lora_adapters;
                                } catch (const std::exception &e) {
                                    SRV_FUNC_ERR("process", "rid %s | forwarding diffusion, failed to apply lora %s\n", rid, e.what());
                                }
                            }
                        } else if (forwarded_seqs >= params.sd_params.max_batch_count) {
                            SRV_FUNC_DBG("process", "rid %s | waiting previous forwarding finished: exceeds max batch count\n", rid);
                            return stream;
                        } else if (!equal_lora(task->req->lora_adapters, lora_adapters)) {
                            SRV_FUNC_DBG("process", "rid %s | waiting previous forwarding finished: lora adapters not matched\n", rid);
                            return stream;
                        }

                        SRV_FUNC_DBG("process", "rid %s | forwarding diffusion\n", rid);

                        forwarded_seqs++; // avoid cleaning the batch

                        v2_stablediffusion_params_sampling sampling = task->req->sampling; // copy
                        sampling.seed += n_seq_forwarded;
                        stream = sd_ctx->generate_stream(task->req->get_prompt(), sampling);
                        return stream;
                    });
                    std::unique_ptr<v2_stablediffusion_sampling_stream> stream                 = forwarded.get();
                    if (stream == nullptr) {
                        std::this_thread::yield();
                        continue;
                    }
                    streams.push_back(std::move(stream));
                    SRV_INFV(3, "rid %s | forwarded diffusion, seq = %d\n", rid, n_seq_forwarded);

                    task->n_forward_steps++;
                    n_seq_forwarded++;
                } while (n_seq_forwarded < n_seq);

                task->t_forwarded = double(ggml_time_us() - task->t_start_forward) / 1.e3;
                metrics.on_image_forwarded(task->t_forwarded, task->n_forward_steps);
                task->p_forwarded_sps = 1.e3 / task->t_forwarded * task->n_forward_steps;

                /**
                 * reverse
                 */

                SRV_INFV(3, "rid %s | reversing diffusion\n", rid);

                const bool preview       = json_value(task->req->stream_options, "preview", false) || json_value(task->req->stream_options, "preview_faster", false);
                const bool chunk         = json_value(task->req->stream_options, "chunk", false);
                const int32_t chunk_size = json_value(task->req->stream_options, "chunk_size", 4096);
                auto send_chunk_json     = [&](const int32_t seq) {
                    if (!chunk || !preview) {
                        if (seq == n_seq - 1) {
                            return send_json(ctx->writer, task->to_json(seq));
                        }
                        return send_event_json(ctx->writer, task->to_json(seq));
                    }

                    std::string b64_json            = task->b64_jsons[seq];
                    size_t chunk_sent               = 0;
                    size_t chunk_send               = b64_json.size() / chunk_size + 1;
                    float chunk_send_progress       = 0.0f;
                    float chunk_send_progress_base  = float(task->progressed_steps[seq] - 1) / float(task->progress_steps[seq]);
                    float chunk_send_progress_scale = 1 / float(task->progress_steps[seq]);
                    while (!b64_json.empty()) {
                        chunk_sent++;
                        chunk_send_progress = chunk_send_progress_base + float(chunk_sent) / float(chunk_send) * chunk_send_progress_scale;
                        Json item           = {
                            {"index", seq},
                            {"progressed_steps", task->progressed_steps[seq]},
                            {"progress_steps", task->progress_steps[seq]},
                            {"progress", chunk_send_progress * 100},
                            {"object", "image.chunk"},
                            {"finish_reason", chunk_sent >= chunk_send ? "stop" : Json()},
                        };
                        item["b64_json"] = b64_json.substr(0, chunk_size);
                        Json resp        = {
                            {"created", std::time(nullptr)},
                            {"model", task->req->get_model()},
                            {"object", "list"},
                            {"data", Json::array({item})},
                        };
                        b64_json = b64_json.substr(chunk_size);
                        if (b64_json.empty() && seq == n_seq - 1 && json_value(task->req->stream_options, "include_usage", true)) {
                            resp["usage"] = {
                                {"time_to_process_ms", task->t_forwarded},
                                {"time_per_generation_ms", task->t_reversed / task->n_reverse_steps},
                                {"generation_per_second", task->p_reversed_sps},
                            };
                        }
                        int32_t sc = 0;
                        if (seq == n_seq - 1) {
                            sc = send_json(ctx->writer, resp);
                        } else {
                            sc = send_event_json(ctx->writer, resp);
                        }
                        if (sc != HTTP_STATUS_OK) {
                            return sc;
                        }
                    };
                    return int32_t(HTTP_STATUS_OK);
                };

                // result
                task->b64_jsons.resize(n_seq);
                task->progressed_steps.resize(n_seq);
                task->progress_steps.resize(n_seq);

                task->t_start_reverse = ggml_time_us();

                int32_t n_seq_reversed = 0;
                do {
                    v2_stablediffusion_sampling_stream *stream = streams[n_seq_reversed].get();

                    int32_t n_progressed_steps = 0;
                    int32_t n_progress_steps   = 0;

                    // reversed -> true|false
                    std::future<bool> reversed = processing_loop->commit([&]() {
                        bool incomplete = true;

                        uint64_t start_at = ggml_time_us();
                        incomplete        = sd_ctx->sample_stream(stream);
                        uint64_t rct      = ggml_time_us() - start_at;

                        std::pair<int32_t, int32_t> progress = sd_ctx->progress_stream(stream);
                        n_progressed_steps                   = progress.first;
                        n_progress_steps                     = progress.second;
                        SRV_FUNC_INFV(3, "process", "rid %s | reversed diffusion, seq = %d, progress = %03i/%03i, cost = %.2f%s\n", rid, n_seq_reversed, n_progressed_steps, n_progress_steps, double(rct) / (rct > 1.e6 ? 1.e6 : 1.e3), rct > 1.e6 ? "s" : "ms");
                        task->n_reverse_steps++;

                        task->progressed_steps[n_seq_reversed] = n_progressed_steps;
                        task->progress_steps[n_seq_reversed]   = n_progress_steps;
                        if (!incomplete) {
                            reversed_seqs++; // increase to avoid deadlock
                            auto generated_img              = sd_ctx->result_image_stream(stream);
                            std::string b64_json            = encode_base64(generated_img->data, generated_img->size);
                            task->b64_jsons[n_seq_reversed] = std::move(b64_json);
                        } else if (task->req->stream && preview) {
                            auto preview_img                = sd_ctx->preview_image_stream(stream, true);
                            std::string b64_json            = encode_base64(preview_img->data, preview_img->size);
                            task->b64_jsons[n_seq_reversed] = std::move(b64_json);
                        }

                        return incomplete;
                    });
                    bool incomplete            = reversed.get();

                    // output
                    if (task->req->stream && (incomplete || n_seq_reversed + 1 < n_seq)) {
                        int64_t t_tart_send = ggml_time_us();
                        int32_t sc          = send_chunk_json(n_seq_reversed);
                        task->t_start_reverse += (ggml_time_us() - t_tart_send); // exclude sending time, ugly, but works.
                        if (sc != HTTP_STATUS_OK) {
                            processing_loop->commit([&]() { reversed_seqs += n_seq - n_seq_reversed; }); // increase to avoid deadlock
                            SRV_ERR("rid %s | reversing diffusion, failed to send chunk, seq = %d, code = %d\n", rid, n_seq_reversed, sc);
                            return sc;
                        }
                    } else if (!ctx->writer->isConnected()) {
                        processing_loop->commit([&]() { reversed_seqs += n_seq - n_seq_reversed; }); // increase to avoid deadlock
                        SRV_ERR("rid %s | reversing diffusion, connection closed\n", rid);
                        return HTTP_STATUS_REQUEST_TIMEOUT;
                    }

                    if (!incomplete) {
                        n_seq_reversed++;
                    }
                } while (n_seq_reversed < n_seq);

                task->t_reversed = double(ggml_time_us() - task->t_start_reverse) / 1.e3;
                metrics.on_image_reversed(task->t_reversed, task->n_reverse_steps);
                task->p_reversed_sps = 1.e3 / task->t_reversed * task->n_reverse_steps;

                SRV_INF("rid %s | forward_s = %d, forward_sps = %.2f sps, reverse_s = %d, reverse_sps = %.2f sps\n", rid, task->n_forward_steps, task->p_forwarded_sps, task->n_reverse_steps, task->p_reversed_sps);

                // output
                if (task->req->stream) {
                    return send_chunk_json(n_seq - 1);
                }
                return send_json(ctx->writer, task->to_json(-1));
            }
            default: {
                throw std::runtime_error("unknown task type");
            }
        }

#undef ACQUIRE_GATE_KEY
#undef RELEASE_GATE_KEY
    }

    //
    // Processors
    //

    static int preprocessor(const HttpContextPtr &ctx) {
        std::string now = std::to_string(ggml_time_us());
        std::string rid = ctx->request->GetHeader(HEADER_X_REQUEST_ID, now);
        // set default headers
        ctx->response->SetHeader(HEADER_X_REQUEST_ID, rid);
        ctx->response->SetHeader(HEADER_X_REQUEST_ACCEPTED_AT, now);
        ctx->response->SetHeader(HEADER_SERVER, "llama-box/" + std::string(LLAMA_BOX_BUILD_VERSION));
        // log request
        if (ctx->request->path != "/health") {
            SRV_INF("rid %s | %4s %s %s\n",
                    rid.c_str(),
                    http_method_str(ctx->request->method), ctx->request->path.c_str(), ctx->request->client_addr.to_string().c_str());
        }
        return HTTP_STATUS_NEXT;
    }

    static int postprocessor(const HttpContextPtr &ctx) {
        // log response
        if (ctx->request->path != "/health") {
            std::string rid = ctx->response->GetHeader(HEADER_X_REQUEST_ID);
            uint64_t rct    = ggml_time_us() - std::stoull(ctx->response->GetHeader(HEADER_X_REQUEST_ACCEPTED_AT));
            SRV_INF("rid %s | %4s %s %s | status %d | cost %.2f%s | %s\n",
                    rid.c_str(),
                    http_method_str(ctx->request->method), ctx->request->path.c_str(), ctx->request->client_addr.to_string().c_str(),
                    ctx->writer->response->status_code,
                    double(rct) / (rct > 1.e6 ? 1.e6 : 1.e3), rct > 1.e6 ? "s" : "ms",
                    ctx->writer->isOpened() ? "opened" : "closed");
        }
        return HTTP_STATUS_NEXT;
    }

    //
    // Routes
    //

    int handle_health(const HttpContextPtr &ctx) {
        const Json resp = {
            {"status", "ok"},
        };
        return send_json(ctx->writer, resp);
    }

    int handle_metrics(const HttpContextPtr &ctx) {
        double t_image_forwarded_total           = metrics.t_image_forwarded_total.load();
        uint64_t n_image_steps_forwarded_total   = metrics.n_image_steps_forwarded_total.load();
        double t_image_reversed_total            = metrics.t_image_reversed_total.load();
        uint64_t n_image_steps_reversed_total    = metrics.n_image_steps_reversed_total.load();
        double t_tokens_prefilled_total          = metrics.t_tokens_prefilled_total.load();
        uint64_t n_tokens_prefilled_total        = metrics.n_tokens_prefilled_total.load();
        double t_tokens_decoded_total            = metrics.t_tokens_decoded_total.load();
        uint64_t n_tokens_decoded_total          = metrics.n_tokens_decoded_total.load();
        uint64_t n_tokens_drafted_total          = metrics.n_tokens_drafted_total.load();
        uint64_t n_tokens_drafted_accepted_total = metrics.n_tokens_drafted_accepted_total.load();

        const Json all_metrics_def = {
            {
                "counter",
                {
                    /* STABLE DIFFUSION */
                    {
                        {"name", "image_forward_total"},
                        {"help", "Number of image forwarded (steps) in diffusion processing."},
                        {"value", n_image_steps_forwarded_total},
                    },
                    {
                        {"name", "image_forward_seconds_total"},
                        {"help", "Image forward process time."},
                        {"value", t_image_forwarded_total / 1.e3},
                    },
                    {
                        {"name", "image_reverse_total"},
                        {"help", "Number of image reversed (steps) in diffusion processing."},
                        {"value", n_image_steps_reversed_total},
                    },
                    {
                        {"name", "image_reverse_seconds_total"},
                        {"help", "Image reverse process time."},
                        {"value", t_image_reversed_total / 1.e3},
                    },

                    /* LLAMA */

                    {
                        {"name", "tokens_prefill_total"},
                        {"help", "Number of prompt tokens processed."},
                        {"value", n_tokens_prefilled_total},
                    },
                    {
                        {"name", "tokens_prefill_seconds_total"},
                        {"help", "Prompt process time."},
                        {"value", t_tokens_prefilled_total / 1.e3},
                    },
                    {
                        {"name", "tokens_decode_total"},
                        {"help", "Number of generation tokens processed."},
                        {"value", n_tokens_decoded_total},
                    },
                    {
                        {"name", "tokens_decode_seconds_total"},
                        {"help", "Predict process time."},
                        {"value", t_tokens_decoded_total / 1.e3},
                    },
                    {
                        {"name", "tokens_drafted_total"},
                        {"help", "Number of speculative decoding tokens processed."},
                        {"value", n_tokens_drafted_total},
                    },
                    {
                        {"name", "tokens_drafted_accepted_total"},
                        {"help", "Number of speculative decoding tokens to be accepted."},
                        {"value", n_tokens_drafted_accepted_total},
                    },
                },
            },
            {
                "gauge",
                {
                    /* STABLE DIFFUSION */

                    {
                        {"name", "image_forward_steps_per_second"},
                        {"help", "Average image forwarded diffusion throughput in steps/s."},
                        {"value", n_image_steps_forwarded_total ? 1.e3 / double(t_image_forwarded_total) * double(n_image_steps_forwarded_total) : 0.},
                    },
                    {
                        {"name", "image_reverse_steps_per_second"},
                        {"help", "Average image reversed diffusion throughput in steps/s."},
                        {"value", n_image_steps_reversed_total ? 1.e3 / double(t_image_reversed_total) * double(n_image_steps_reversed_total) : 0.},
                    },

                    /* LLAMA */

                    {
                        {"name", "tokens_prefill_per_second"},
                        {"help", "Average prompt throughput in tokens/s."},
                        {"value", n_tokens_prefilled_total ? 1.e3 / double(t_tokens_prefilled_total) * double(n_tokens_prefilled_total) : 0.},
                    },
                    {
                        {"name", "tokens_decode_per_second"},
                        {"help", "Average generation throughput in tokens/s."},
                        {"value", n_tokens_decoded_total ? 1.e3 / double(t_tokens_decoded_total) * double(n_tokens_decoded_total) : 0.},
                    },
                    {
                        {"name", "kv_cache_usage_ratio"},
                        {"help", "KV-cache usage. 1 means 100 percent usage."},
                        {"value", support_completion() ? double(llama_kv_self_used_cells(llm_ctx)) / params.llm_params.n_ctx : 0},
                    },
                    {
                        {"name", "kv_cache_tokens"},
                        {"help", "KV-cache tokens."},
                        {"value", support_completion() ? llama_kv_self_n_tokens(llm_ctx) : 0},
                    },
                },
            },
        };

        std::stringstream metricss;
        for (const auto &el : all_metrics_def.items()) {
            const auto &type        = el.key();
            const auto &metrics_def = el.value();
            for (const auto &metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");
                const auto value       = metric_def.at("value");
                metricss << "# HELP llamabox:" << name << " " << help << "\n"
                         << "# TYPE llamabox:" << name << " " << type << "\n"
                         << "llamabox:" << name << " " << value << "\n";
            }
        }

        const std::string metrics_string = metricss.str();
        return send_string(ctx->writer, metrics_string, "text/plain; version=0.0.4");
    }

    int handle_tokenize(const HttpContextPtr &ctx) {
        if (!support_tokenize()) {
            return send_error_string(ctx->writer, HTTP_STATUS_FORBIDDEN, "You are not allowed to do tokenize from this model");
        }

        std::unique_ptr<tokenize_req> req = get_tokenize_req(ctx, params.llm_params);

        Json tokens_json = Json::array();
        {
            llama_tokens tokens = tokenize_prompt(llm_vocab, req->content, req->add_special, true);
            if (req->with_pieces) {
                for (const llama_token &id : tokens) {
                    std::string piece = common_token_to_piece(llm_ctx, id);
                    // if valid UTF-8, store as string
                    if (string_is_utf8(piece)) {
                        tokens_json.push_back({{"id", id}, {"piece", piece}});
                        continue;
                    }
                    // otherwise, store as array of byte values
                    Json piece_json = Json::array();
                    for (unsigned char c : piece) {
                        piece_json.push_back(static_cast<int>(c));
                    }
                    tokens_json.push_back({{"id", id}, {"piece", piece_json}});
                }
            } else {
                tokens_json = tokens;
            }
        }

        const Json resp = {
            {"model", req->model},
            {"tokens", tokens_json},
        };
        return send_json(ctx->writer, resp);
    }

    int handle_detokenize(const HttpContextPtr &ctx) {
        if (!support_tokenize()) {
            return send_error_string(ctx->writer, HTTP_STATUS_FORBIDDEN, "You are not allowed to do detokenize from this model");
        }

        std::unique_ptr<detokenize_req> req = get_detokenize_req(ctx, params.llm_params);

        const Json content_json = common_detokenize(llm_ctx, req->tokens, false);

        const Json resp = {
            {"model", req->model},
            {"content", content_json},
        };
        return send_json(ctx->writer, resp);
    }

    int handle_models(const HttpContextPtr &ctx) {
        Json metadata_json;
        if (support_image()) {
            std::pair<int, int> img_size = sd_ctx->get_default_image_size();
            metadata_json                = {
                {"n_slot", params.sd_params.n_parallel},
                {"seed", int32_t(params.sd_params.sampling.seed)},
                {"max_batch_count", params.sd_params.max_batch_count},
                {"max_height", params.sd_params.sampling.height},
                {"max_width", params.sd_params.sampling.width},
                {"default_height", MIN(img_size.first, params.sd_params.sampling.height)},
                {"default_width", MIN(img_size.second, params.sd_params.sampling.width)},
                {"guidance", params.sd_params.sampling.guidance},
                {"strength", params.sd_params.sampling.strength},
                {"sample_method", sd_sample_method_to_argument(params.sd_params.sampling.sample_method)},
                {"sampling_steps", params.sd_params.sampling.sampling_steps},
                {"cfg_scale", params.sd_params.sampling.cfg_scale},
                {"slg_scale", params.sd_params.sampling.slg_scale},
                {"slg_skip_layers", params.sd_params.sampling.slg_skip_layers},
                {"slg_start", params.sd_params.sampling.slg_start},
                {"slg_end", params.sd_params.sampling.slg_end},
                {"schedule_method", sd_schedule_to_argument(params.sd_params.sampling.schedule_method)},
                {"negative_prompt", params.sd_params.sampling.negative_prompt},
            };
        } else {
            metadata_json = {
                {"vocab_type", llama_vocab_type(llm_vocab)},
                {"n_vocab", llama_vocab_n_tokens(llm_vocab)},
                {"n_ctx_train", llama_model_n_ctx_train(llm_model)},
                {"n_embd", llama_model_n_embd(llm_model)},
                {"n_params", llama_model_n_params(llm_model)},
                {"size", llama_model_size(llm_model)},
                {"n_ctx", llama_n_ctx(llm_ctx)},
                {"n_slot", params.llm_params.n_parallel},
                {"n_slot_ctx", int32_t(llama_n_ctx(llm_ctx)) / params.llm_params.n_parallel},
                {"ctx_shift", params.llm_params.ctx_shift},
                {"seed", int32_t(params.llm_params.sampling.seed)},
                {"temperature", params.llm_params.sampling.temp},
                {"dynatemp_range", params.llm_params.sampling.dynatemp_range},
                {"dynatemp_exponent", params.llm_params.sampling.dynatemp_exponent},
                {"top_k", params.llm_params.sampling.top_k},
                {"top_p", params.llm_params.sampling.top_p},
                {"min_p", params.llm_params.sampling.min_p},
                {"top_n_sigma", params.llm_params.sampling.top_n_sigma},
                {"xtc_probability", params.llm_params.sampling.xtc_probability},
                {"xtc_threshold", params.llm_params.sampling.xtc_threshold},
                {"typical_p", params.llm_params.sampling.typ_p},
                {"repeat_last_n", params.llm_params.sampling.penalty_last_n},
                {"repeat_penalty", params.llm_params.sampling.penalty_repeat},
                {"presence_penalty", params.llm_params.sampling.penalty_present},
                {"frequency_penalty", params.llm_params.sampling.penalty_freq},
                {"dry_multiplier", params.llm_params.sampling.dry_multiplier},
                {"dry_base", params.llm_params.sampling.dry_base},
                {"dry_allowed_length", params.llm_params.sampling.dry_allowed_length},
                {"dry_penalty_last_n", params.llm_params.sampling.dry_penalty_last_n},
                {"dry_sequence_breakers", params.llm_params.sampling.dry_sequence_breakers},
                {"mirostat", params.llm_params.sampling.mirostat},
                {"mirostat_tau", params.llm_params.sampling.mirostat_tau},
                {"mirostat_eta", params.llm_params.sampling.mirostat_eta},
                {"support_vision", llm_ctx_clip != nullptr},
                {"support_speculative", llm_ctx_draft != nullptr},
                {"support_tool_calls", support_tool_calls},
                {"support_reasoning", support_reasoning},
            };
        }

        const Json resp = {
            {"object", "list"},
            {
                "data",
                {
                    {
                        {"id", support_image() ? params.sd_params.model_alias : params.llm_params.model_alias},
                        {"object", "model"},
                        {"created", std::time(nullptr)},
                        {"owned_by", "llama-box"},
                        {"meta", metadata_json},
                    },
                },
            },
        };
        return send_json(ctx->writer, resp);
    }

    int handle_legacy_completions(const HttpContextPtr &ctx) {
        if (!support_completion()) {
            return send_error_string(ctx->writer, HTTP_STATUS_FORBIDDEN, "You are not allowed to do completions from this model");
        }

        std::unique_ptr<RatelimitTokenBucket> token_bucket = nullptr;
        if (params.n_tps > 0) {
            std::string tps_str = ctx->request->GetHeader(HEADER_X_REQUEST_TOKENS_PER_SECOND, "");
            if (!tps_str.empty()) {
                int tps = params.n_tps;
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
                    return send_error_string(ctx->writer, HTTP_STATUS_GONE, "The request exceeds the maximum tokens per second");
                }
                token_bucket = std::make_unique<RatelimitTokenBucket>(tps, tps);
            }
        }

        std::unique_ptr<legacy_complete_req> req = get_legacy_complete_req(ctx, params, llm_ctx);

        const auto n_ctx = int32_t(llama_n_ctx(llm_ctx));

        int32_t n_prefilling = 0;

        std::vector<std::variant<llama_tokens, std::unique_ptr<llava_image_embed>>> tokenized_prompts;
        /* PLAIN TEXT */
        {
            llama_tokens tokenized_prompt = tokenize_prompt(llm_vocab, req->prompt, /* add_special= */ true, /* parse_special= */ true);
            n_prefilling                  = int32_t(tokenized_prompt.size());
            if (n_prefilling >= n_ctx && params.llm_params.ctx_shift) {
                SRV_WRN("rid %s | prompt tokens size exceeds the context size, force context shift\n", req->get_rid());
                tokenized_prompt.erase(tokenized_prompt.begin(), tokenized_prompt.end() - n_ctx + 1);
                n_prefilling = int32_t(tokenized_prompt.size());
            }
            tokenized_prompts.emplace_back(std::move(tokenized_prompt));
        }

        if (n_prefilling == 0) {
            return send_error_json(ctx->writer, HTTP_STATUS_BAD_REQUEST, "Illegal param: empty completions tokens");
        }

        int32_t n_decoding_budget = 0;
        if (params.llm_params.ctx_shift && req->max_tokens == -1) {
            n_decoding_budget = INT32_MAX;
        } else if (req->max_tokens == -1) {
            n_decoding_budget = n_ctx - n_prefilling;
        } else {
            n_decoding_budget = req->max_tokens - n_prefilling;
        }
        if (n_decoding_budget <= 0) {
            return send_error_json(ctx->writer, HTTP_STATUS_BAD_REQUEST, "Illegal param: \"prompt\" tokens size exceeds the context size");
        }

        common_sampler *sampler       = nullptr;
        common_sampler *sampler_draft = nullptr;
        {
            sampler = common_sampler_init(llm_model, req->sampling);
            if (sampler == nullptr) {
                return send_error_json(ctx->writer, HTTP_STATUS_BAD_REQUEST, "Illegal param: \"sampling\" is invalid");
            }
            if (llm_ctx_draft != nullptr) {
                common_params_sampling sampling_draft;
                sampling_draft.no_perf  = false;
                sampling_draft.top_k    = 10;
                sampling_draft.samplers = {
                    COMMON_SAMPLER_TYPE_TOP_K,
                };
                sampler_draft = common_sampler_init(llm_model, sampling_draft);
                if (sampler_draft == nullptr) {
                    common_sampler_free(sampler);
                    return send_error_json(ctx->writer, HTTP_STATUS_BAD_REQUEST, "Illegal param: \"sampling\" is invalid");
                }
            }
        }

        std::unique_ptr<completions_task> task = std::make_unique<completions_task>(get_task_id());
        task->token_bucket                     = std::move(token_bucket);
        task->tokenized_prompts                = std::move(tokenized_prompts);
        task->n_prefilling                     = n_prefilling;
        task->n_decoding_budget                = n_decoding_budget;
        task->sampler                          = sampler;
        task->sampler_draft                    = sampler_draft;
        task->cmpl_id                          = gen_completion_id();
        task->req                              = std::move(req);
        task->t_start_prefill                  = ggml_time_us();

        return process(ctx, std::move(task));
    }

    int handle_chat_completions(const HttpContextPtr &ctx) {
        if (!support_completion()) {
            return send_error_string(ctx->writer, HTTP_STATUS_FORBIDDEN, "You are not allowed to do chat operation from this model");
        }

        std::unique_ptr<RatelimitTokenBucket> token_bucket = nullptr;
        if (params.n_tps > 0) {
            std::string tps_str = ctx->request->GetHeader(HEADER_X_REQUEST_TOKENS_PER_SECOND, "");
            if (!tps_str.empty()) {
                int tps = params.n_tps;
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
                    return send_error_string(ctx->writer, HTTP_STATUS_GONE, "The request exceeds the maximum tokens per second");
                }
                token_bucket = std::make_unique<RatelimitTokenBucket>(tps, tps);
            }
        }

        std::unique_ptr<chat_complete_req> req = get_chat_complete_req(ctx, params, llm_ctx, support_tool_calls, chat_templates.get());

        const auto n_ctx = int32_t(llama_n_ctx(llm_ctx));

        int32_t n_prefilling = 0;

        std::vector<std::variant<llama_tokens, std::unique_ptr<llava_image_embed>>> tokenized_prompts;
        /* PLAIN TEXT */
        if (req->images.empty()) {
            llama_tokens tokenized_prompt = tokenize_prompt(llm_vocab, req->chat_params.prompt, /* add_special= */ true, /* parse_special= */ true);
            n_prefilling                  = int32_t(tokenized_prompt.size());
            if (n_prefilling >= n_ctx && params.llm_params.ctx_shift) {
                SRV_WRN("rid %s | prompt tokens size exceeds the context size, force context shift\n", req->get_rid());
                tokenized_prompt.erase(tokenized_prompt.begin(), tokenized_prompt.end() - n_ctx + 1);
                n_prefilling = int32_t(tokenized_prompt.size());
            }
            tokenized_prompts.emplace_back(std::move(tokenized_prompt));
        }
        /* VISION */
        else {
            const std::string image_sign = "<--IMAGE-->";

            std::string prompt = req->chat_params.prompt;

            size_t images_count = 0;
            size_t image_pos    = prompt.find(image_sign);
            bool add_bos        = true;
            while (image_pos != std::string::npos) {
                // process text
                if (const std::string text = prompt.substr(0, image_pos); !text.empty()) {
                    llama_tokens tokenized_text = common_tokenize(llm_vocab, text, /* add_special= */ add_bos, /* parse_special= */ true);
                    n_prefilling += int32_t(tokenized_text.size());
                    tokenized_prompts.emplace_back(std::move(tokenized_text));
                    add_bos = false;
                }

                // process image
                std::unique_ptr<llava_image_embed> image_embed = std::make_unique<llava_image_embed>();
                bool image_embed_result                        = llava_image_embed_make_with_clip_img(llm_ctx_clip, params.llm_params.cpuparams.n_threads, req->images[images_count].get(), &image_embed->embed, &image_embed->n_image_pos);
                if (!image_embed_result) {
                    return send_error_string(ctx->writer, HTTP_STATUS_INTERNAL_SERVER_ERROR, "Failed to embed the image");
                }
                n_prefilling += int32_t(image_embed->n_image_pos);
                req->images[images_count++] = nullptr; // release image asap
                // qwen2vl
                if (clip_is_qwen2vl(llm_ctx_clip)) {
                    // <|vision_start|>
                    llama_tokens tokenized_text = common_tokenize(llm_vocab, "<|vision_start|>", /* add_special= */ add_bos, /* parse_special= */ true);
                    n_prefilling += int32_t(tokenized_text.size());
                    tokenized_prompts.emplace_back(std::move(tokenized_text));
                    add_bos = false;
                    // <--IMAGE-->
                    tokenized_prompts.emplace_back(std::move(image_embed));
                    // <|vision_end|>
                    tokenized_text = common_tokenize(llm_vocab, "<|vision_end|>", /* add_special= */ false, /* parse_special= */ true);
                    n_prefilling += int32_t(tokenized_text.size());
                    tokenized_prompts.emplace_back(std::move(tokenized_text));
                }
                // minicpmv
                else if (clip_is_minicpmv(llm_ctx_clip) != 0) {
                    llava_image_embed *img_embd = image_embed.get();
                    int idx                     = 0;
                    auto slice_image_embed      = [&]() {
                        auto *embed = (float *)malloc(clip_embd_nbytes(llm_ctx_clip));
                        std::memcpy(embed, img_embd->embed + (idx++) * clip_n_patches(llm_ctx_clip) * clip_n_mmproj_embd(llm_ctx_clip), clip_embd_nbytes(llm_ctx_clip));

                        std::unique_ptr<llava_image_embed> slice_embed = std::make_unique<llava_image_embed>();
                        slice_embed->embed                             = embed;
                        slice_embed->n_image_pos                       = clip_n_patches(llm_ctx_clip);
                        return slice_embed;
                    };
                    // <image>
                    llama_tokens tokenized_text = common_tokenize(llm_vocab, "<image>", /* add_special= */ add_bos, /* parse_special= */ true);
                    n_prefilling += int32_t(tokenized_text.size());
                    tokenized_prompts.emplace_back(std::move(tokenized_text));
                    add_bos = false;
                    // lead patch <--IMAGE-->
                    tokenized_prompts.emplace_back(slice_image_embed());
                    // </image>
                    tokenized_text = common_tokenize(llm_vocab, "</image>", /* add_special= */ false, /* parse_special= */ true);
                    n_prefilling += int32_t(tokenized_text.size());
                    tokenized_prompts.emplace_back(std::move(tokenized_text));
                    const size_t n_img_embd = img_embd->n_image_pos / clip_n_patches(llm_ctx_clip);
                    if (n_img_embd > 1) {
                        const size_t n_img_embd_col = clip_uhd_num_image_embeds_col(llm_ctx_clip);
                        const int32_t version       = clip_is_minicpmv(llm_ctx_clip);
                        if (version < 3) {
                            // <slice>
                            tokenized_text = common_tokenize(llm_vocab, "<slice>", /* add_special= */ false, /* parse_special= */ true);
                            n_prefilling += int32_t(tokenized_text.size());
                            tokenized_prompts.emplace_back(std::move(tokenized_text));
                        }
                        std::string ifmt = "<slice>";
                        std::string ofmt = "</slice>";
                        if (version < 3) {
                            ifmt = "<image>";
                            ofmt = "</image>";
                        }
                        for (size_t i = 0; i < (n_img_embd - 1) / n_img_embd_col; ++i) {
                            for (size_t j = 0; j < n_img_embd_col; ++j) {
                                // <slice> | <image>
                                tokenized_text = common_tokenize(llm_vocab, ifmt, /* add_special= */ false, /* parse_special= */ true);
                                n_prefilling += int32_t(tokenized_text.size());
                                tokenized_prompts.emplace_back(std::move(tokenized_text));
                                // other patches <--IMAGE-->
                                tokenized_prompts.emplace_back(slice_image_embed());
                                // </slice> | </image>
                                tokenized_text = common_tokenize(llm_vocab, ofmt, /* add_special= */ false, /* parse_special= */ true);
                                n_prefilling += int32_t(tokenized_text.size());
                                tokenized_prompts.emplace_back(std::move(tokenized_text));
                            }
                        }
                        if (version < 3) {
                            // </slice>
                            tokenized_text = common_tokenize(llm_vocab, "</slice>", /* add_special= */ false, /* parse_special= */ true);
                            n_prefilling += int32_t(tokenized_text.size());
                            tokenized_prompts.emplace_back(std::move(tokenized_text));
                        }
                    }
                }
                // gemma3
                // NB(thxCode): clip_is_gemma3 is a patch.
                else if (clip_is_gemma3(llm_ctx_clip)) {
                    // <|start_of_image|>
                    llama_tokens tokenized_text = common_tokenize(llm_vocab, "<|start_of_image|>", /* add_special= */ add_bos, /* parse_special= */ true);
                    n_prefilling += int32_t(tokenized_text.size());
                    tokenized_prompts.emplace_back(std::move(tokenized_text));
                    add_bos = false;
                    // <--IMAGE-->
                    tokenized_prompts.emplace_back(std::move(image_embed));
                    // <|end_of_image|>
                    tokenized_text = common_tokenize(llm_vocab, "<|end_of_image|>", /* add_special= */ false, /* parse_special= */ true);
                    n_prefilling += int32_t(tokenized_text.size());
                    tokenized_prompts.emplace_back(std::move(tokenized_text));
                }
                // others
                else {
                    // <--IMAGE-->
                    tokenized_prompts.emplace_back(std::move(image_embed));
                }

                prompt    = prompt.substr(image_pos + image_sign.size());
                image_pos = prompt.find(image_sign);
            }
            // process remain text
            if (!prompt.empty()) {
                llama_tokens tokenized_text = common_tokenize(llm_vocab, prompt, add_bos, true);
                n_prefilling += int32_t(tokenized_text.size());
                tokenized_prompts.emplace_back(std::move(tokenized_text));
                add_bos = false;
            }
        }

        if (n_prefilling == 0) {
            return send_error_json(ctx->writer, HTTP_STATUS_BAD_REQUEST, "Illegal param: empty completions tokens");
        }

        bool tokenized_prompts_include_images = !req->images.empty();
        req->images.clear(); // release images asap

        bool tokenized_prompts_include_tools = !req->tools.empty();

        int32_t n_decoding_budget = 0;
        if (params.llm_params.ctx_shift && req->max_tokens == -1) {
            n_decoding_budget = INT32_MAX;
        } else if (req->max_tokens == -1) {
            n_decoding_budget = n_ctx - n_prefilling;
        } else {
            n_decoding_budget = req->max_tokens - n_prefilling;
        }
        if (n_decoding_budget <= 0) {
            return send_error_json(ctx->writer, HTTP_STATUS_BAD_REQUEST, "Illegal param: \"prompt\" tokens size exceeds the context size");
        }

        common_sampler *sampler       = nullptr;
        common_sampler *sampler_draft = nullptr;
        {
            sampler = common_sampler_init(llm_model, req->sampling);
            if (sampler == nullptr) {
                return send_error_json(ctx->writer, HTTP_STATUS_BAD_REQUEST, "Illegal param: \"sampling\" is invalid");
            }
            if (llm_ctx_draft != nullptr) {
                common_params_sampling sampling_draft;
                sampling_draft.no_perf  = false;
                sampling_draft.top_k    = 10;
                sampling_draft.samplers = {
                    COMMON_SAMPLER_TYPE_TOP_K,
                };
                sampler_draft = common_sampler_init(llm_model, sampling_draft);
                if (sampler_draft == nullptr) {
                    common_sampler_free(sampler);
                    return send_error_json(ctx->writer, HTTP_STATUS_BAD_REQUEST, "Illegal param: \"sampling\" is invalid");
                }
            }
        }

        std::unique_ptr<completions_task> task     = std::make_unique<completions_task>(get_task_id());
        task->token_bucket                         = std::move(token_bucket);
        task->tokenized_prompts                    = std::move(tokenized_prompts);
        task->tokenized_prompts_format             = req->chat_params.format;
        task->tokenized_prompts_include_images     = tokenized_prompts_include_images;
        task->tokenized_prompts_include_tools      = tokenized_prompts_include_tools;
        task->n_decoding_budget                    = n_decoding_budget;
        task->n_prefilling                         = n_prefilling;
        task->sampler                              = sampler;
        task->sampler_draft                        = sampler_draft;
        task->tool_call_start_token                = req->tool_call_start_token;
        task->tool_call_start_words                = std::move(req->tool_call_start_words);
        task->tool_call_start_words_longest_length = req->tool_call_start_words_longest_length;
        task->cmpl_id                              = gen_chat_completion_id();
        task->req                                  = std::move(req);
        task->t_start_prefill                      = ggml_time_us();

        return process(ctx, std::move(task));
    }

    int handle_embeddings(const HttpContextPtr &ctx) {
        if (!support_embedding()) {
            return send_error_string(ctx->writer, HTTP_STATUS_FORBIDDEN, "You are not allowed to do embedding from this model");
        }

        std::unique_ptr<embed_req> req = get_embed_req(ctx, params);

        const uint32_t n_ctx = llama_n_ctx(llm_ctx);

        int32_t n_prefilling = 0;

        std::vector<llama_tokens> tokenized_inputs = tokenize_prompts(llm_vocab, req->input, /* add_special= */ true, /* parse_special= */ true);
        for (llama_tokens &tokenized_input : tokenized_inputs) {
            if (tokenized_input.size() <= n_ctx) {
                n_prefilling += int32_t(tokenized_input.size());
                continue;
            }
            if (!params.force_context_shift) {
                return send_error_json(ctx->writer, HTTP_STATUS_BAD_REQUEST, "Illegal param: \"input\" tokens size exceeds the context size");
            }
            SRV_WRN("rid %s | input tokens size exceeds the context size, force context shift\n", req->get_rid());
            tokenized_input.erase(tokenized_input.begin(), tokenized_input.end() - n_ctx);
            n_prefilling += int32_t(n_ctx);
        }

        if (n_prefilling == 0) {
            return send_error_json(ctx->writer, HTTP_STATUS_BAD_REQUEST, "Illegal param: empty embedding tokens");
        }

        std::unique_ptr<embeddings_task> task = std::make_unique<embeddings_task>(get_task_id());
        task->tokenized_inputs                = std::move(tokenized_inputs);
        task->n_prefilling                    = n_prefilling;
        task->req                             = std::move(req);

        return process(ctx, std::move(task));
    }

    int handle_rerank(const HttpContextPtr &ctx) {
        if (!support_reranking()) {
            return send_error_string(ctx->writer, HTTP_STATUS_FORBIDDEN, "You are not allowed to do reranking from this model");
        }

        std::unique_ptr<rerank_req> req = get_rerank_req(ctx, params);

        const uint32_t n_ctx        = llama_n_ctx(llm_ctx);
        const llama_token tok_bos   = llama_vocab_bos(llm_vocab);
        const llama_token tok_eos   = llama_vocab_eos(llm_vocab);
        const llama_token tok_sep   = llama_vocab_sep(llm_vocab);
        const size_t n_tok_addition = 4;

        int32_t n_prefilling = 0;

        llama_tokens tokenized_query = tokenize_prompt(llm_vocab, req->query, /* add_special= */ false, /* parse_special= */ true);
        if (req->normalize && tokenized_query.size() * 2 + n_tok_addition > n_ctx) {
            return send_error_json(ctx->writer, HTTP_STATUS_BAD_REQUEST, R"(Illegal param: "query" length exceeds the context size, disable "normalize" to bypass this check)");
        }
        auto decorate = [&](const llama_tokens &tokenized_document) {
            const size_t n_tok = tokenized_query.size() + tokenized_document.size() + n_tok_addition;
            if (n_tok > n_ctx) {
                throw std::invalid_argument(R"(Illegal param: the sum of the lengths of "query" and "document" exceeds the context size)");
            }
            n_prefilling += int32_t(n_tok);
            // format input: [BOS]query[EOS][SEP]document[EOS]
            llama_tokens tokenized_input;
            tokenized_input.reserve(n_tok);
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
        for (const Json &document : req->documents) {
            llama_tokens tokenized_document = tokenize_prompt(llm_vocab, document, /* add_special= */ false, /* parse_special= */ true);
            tokenized_inputs.emplace_back(decorate(tokenized_document));
        }
        if (req->normalize) {
            tokenized_inputs.emplace_back(decorate(tokenized_query));
            // NB(thxCode): llama_vocab_unk is a patch.
            tokenized_inputs.emplace_back(decorate({llama_vocab_unk(llm_vocab)}));
        }

        if (n_prefilling == 0) {
            return send_error_json(ctx->writer, HTTP_STATUS_BAD_REQUEST, "Illegal param: empty reranking tokens");
        }

        std::unique_ptr<embeddings_task> task = std::make_unique<embeddings_task>(get_task_id());
        task->tokenized_inputs                = std::move(tokenized_inputs);
        task->n_prefilling                    = n_prefilling;
        task->req                             = std::move(req);

        return process(ctx, std::move(task));
    }

    int handle_images(const HttpContextPtr &ctx) {
        if (!support_image()) {
            return send_error_string(ctx->writer, HTTP_STATUS_FORBIDDEN, "You are not allowed to do image operation from this model");
        }

        const std::string category = ctx->request->GetParam("category");
        if (category != "generations" && category != "edits") {
            return send_error_string(ctx->writer, HTTP_STATUS_FORBIDDEN, "You are not allowed to do image operation from this model");
        }

        std::unique_ptr<images_task> task = std::make_unique<images_task>(get_task_id());
        if (category == "generations") {
            std::unique_ptr<image_generate_req> req = get_image_generate_req(ctx, params);
            task->req                               = std::move(req);
        } else {
            std::unique_ptr<image_edit_req> req = get_image_edit_req(ctx, params);
            task->req                           = std::move(req);
        }

        return process(ctx, std::move(task));
    }
};

static int start_httpserver(v2_httpserver_params &params) {
    httpserver srv(params);

    if (!srv.load()) {
        SRV_ERR("%s", "failed to load\n");
        return -1;
    }

    return srv.start();
}
