#pragma once

#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "llama.cpp/common/common.h"
#define JSON_ASSERT GGML_ASSERT
#include "llama.cpp/common/json.hpp"
#include "llama.cpp/common/log.h"
#include "llama.cpp/include/llama.h"

#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH 10485760
#include "llama.cpp/examples/server/httplib.h"

#define DEFAULT_OAICOMPAT_MODEL "gpt-3.5-turbo-0613"

#define SLT_INF(slot, fmt, ...)                                                              \
    LOG_INF("slot %25.*s: id %2d | task %d | " fmt, 25, __func__, (slot).id, (slot).id_task, \
            __VA_ARGS__)
#define SLT_WRN(slot, fmt, ...)                                                              \
    LOG_WRN("slot %25.*s: id %2d | task %d | " fmt, 25, __func__, (slot).id, (slot).id_task, \
            __VA_ARGS__)
#define SLT_ERR(slot, fmt, ...)                                                              \
    LOG_ERR("slot %25.*s: id %2d | task %d | " fmt, 25, __func__, (slot).id, (slot).id_task, \
            __VA_ARGS__)
#define SLT_DBG(slot, fmt, ...)                                                              \
    LOG_DBG("slot %25.*s: id %2d | task %d | " fmt, 25, __func__, (slot).id, (slot).id_task, \
            __VA_ARGS__)

#define SRV_INF(fmt, ...) LOG_INF("srv  %25.*s: " fmt, 25, __func__, __VA_ARGS__)
#define SRV_WRN(fmt, ...) LOG_WRN("srv  %25.*s: " fmt, 25, __func__, __VA_ARGS__)
#define SRV_ERR(fmt, ...) LOG_ERR("srv  %25.*s: " fmt, 25, __func__, __VA_ARGS__)
#define SRV_DBG(fmt, ...) LOG_DBG("srv  %25.*s: " fmt, 25, __func__, __VA_ARGS__)

#define QUE_INF(fmt, ...) LOG_INF("que  %25.*s: " fmt, 25, __func__, __VA_ARGS__)
#define QUE_WRN(fmt, ...) LOG_WRN("que  %25.*s: " fmt, 25, __func__, __VA_ARGS__)
#define QUE_ERR(fmt, ...) LOG_ERR("que  %25.*s: " fmt, 25, __func__, __VA_ARGS__)
#define QUE_DBG(fmt, ...) LOG_DBG("que  %25.*s: " fmt, 25, __func__, __VA_ARGS__)

using json = nlohmann::json;

// https://community.openai.com/t/openai-chat-list-of-error-codes-and-types/357791/11
enum error_type {
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_AUTHENTICATION,
    ERROR_TYPE_SERVER,
    ERROR_TYPE_NOT_FOUND,
    ERROR_TYPE_PERMISSION,
    ERROR_TYPE_UNAVAILABLE,   // custom error
    ERROR_TYPE_NOT_SUPPORTED, // custom error
};

template <typename T>
static T json_value(const json &body, const std::string &key, const T &default_value) {
    // Fallback null to default value
    if (body.contains(key) && !body.at(key).is_null()) {
        try {
            return body.at(key);
        } catch (NLOHMANN_JSON_NAMESPACE::detail::type_error const &) {
            SRV_WRN("wrong type supplied for parameter '%s'. Expected '%s', using default value\n",
                    key.c_str(), json(default_value).type_name());
            return default_value;
        }
    } else {
        return default_value;
    }
}

//
// chat template utils
//

// Format given chat. If tmpl is empty, we take the template from model metadata
inline std::string format_chat(const struct llama_model *model, const std::string &tmpl,
                               const std::vector<json> &messages) {
    std::vector<llama_chat_msg> chat;

    for (const auto &curr_msg : messages) {
        std::string role = json_value(curr_msg, "role", std::string(""));

        std::string content;
        if (curr_msg.contains("content")) {
            if (curr_msg["content"].is_string()) {
                content = curr_msg["content"].get<std::string>();
            } else if (curr_msg["content"].is_array()) {
                for (const json &part : curr_msg["content"]) {
                    if (part.contains("text")) {
                        content += "\n" + part["text"].get<std::string>();
                    }
                }
            } else {
                throw std::runtime_error("Invalid 'content' type (ref: "
                                         "https://github.com/ggerganov/llama.cpp/issues/8367)");
            }
        } else {
            throw std::runtime_error(
                "Missing 'content' (ref: https://github.com/ggerganov/llama.cpp/issues/8367)");
        }

        chat.push_back({role, content});
    }

    return llama_chat_apply_template(model, tmpl, chat, true);
}

//
// base64 utils (TODO: move to common in the future)
//

static const std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                        "abcdefghijklmnopqrstuvwxyz"
                                        "0123456789+/";

static inline bool is_base64(uint8_t c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

static inline std::vector<uint8_t> base64_decode(const std::string &encoded_string) {
    int i = 0;
    int j = 0;
    int in_ = 0;

    int in_len = encoded_string.size();

    uint8_t char_array_4[4];
    uint8_t char_array_3[3];

    std::vector<uint8_t> ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_];
        in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++) {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }

            char_array_3[0] = ((char_array_4[0]) << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; (i < 3); i++) {
                ret.push_back(char_array_3[i]);
            }

            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++) {
            char_array_4[j] = 0;
        }

        for (j = 0; j < 4; j++) {
            char_array_4[j] = base64_chars.find(char_array_4[j]);
        }

        char_array_3[0] = ((char_array_4[0]) << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (j = 0; j < i - 1; j++) {
            ret.push_back(char_array_3[j]);
        }
    }

    return ret;
}

//
// random string / id
//

static std::string random_string() {
    static const std::string str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

    std::random_device rd;
    std::mt19937 generator(rd());

    std::string result(32, ' ');

    for (int i = 0; i < 32; ++i) {
        result[i] = str[generator() % str.size()];
    }

    return result;
}

static std::string gen_chatcmplid() {
    return "chatcmpl-" + random_string();
}

static std::string gen_cmplid() {
    std::stringstream cmplid;
    cmplid << "cmpl-" << random_string();

    return cmplid.str();
}

//
// other common utils
//

static size_t common_part(const std::vector<llama_token> &a, const std::vector<llama_token> &b) {
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++) {
    }

    return i;
}

static size_t common_part(const std::string &a, const std::string &b) {
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++) {
    }

    return i;
}

static bool ends_with(const std::string &str, const std::string &suffix) {
    return str.size() >= suffix.size() &&
           0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

static size_t find_partial_stop_string(const std::string &stop, const std::string &text) {
    if (!text.empty() && !stop.empty()) {
        const char text_last_char = text.back();
        for (int64_t char_index = stop.size() - 1; char_index >= 0; char_index--) {
            if (stop[char_index] == text_last_char) {
                const std::string current_partial = stop.substr(0, char_index + 1);
                if (ends_with(text, current_partial)) {
                    return text.size() - char_index - 1;
                }
            }
        }
    }

    return std::string::npos;
}

static bool json_is_array_of_numbers(const json &data) {
    if (data.is_array()) {
        return std::all_of(data.begin(), data.end(), [](const json &e) { return e.is_number(); });
    }
    return false;
}

// format incomplete utf-8 multibyte character for output
static std::string tokens_to_output_formatted_string(const llama_context *ctx,
                                                     const llama_token token) {
    std::string out = token == -1 ? "" : llama_token_to_piece(ctx, token);

    // if the size is 1 and first bit is 1, meaning it's a partial character
    //   (size > 1 meaning it's already a known token)
    if (out.size() == 1 && (out[0] & 0x80) == 0x80) {
        std::stringstream ss;
        ss << std::hex << (out[0] & 0xff);
        std::string res(ss.str());
        out = "byte: \\x" + res;
    }

    return out;
}

struct completion_token_output {
    std::vector<llama_token> toks;
    std::string text_to_send;

    struct token_prob {
        llama_token tok;
        float prob;
    };

    std::vector<std::vector<token_prob>> probss;
};

// convert a vector of completion_token_output to json
static json probs_vector_to_json(const llama_context *ctx,
                                 const std::vector<completion_token_output> &probs,
                                 const bool oaicompat_completion = false,
                                 const bool oaicompat_completion_chat = false) {
    if (oaicompat_completion) {
        if (oaicompat_completion_chat) {
            json content = json::array();

            for (const auto &prob : probs) {
                const auto sz_toks = int32_t(prob.toks.size());
                for (int32_t i = 0; i < sz_toks; i++) {
                    const std::string token = tokens_to_output_formatted_string(ctx, prob.toks[i]);
                    float token_logprob = 1.0f;
                    std::vector<unsigned char> token_bytes(token.begin(), token.end());
                    json token_top_logprobs = json::array();
                    for (const auto &p : prob.probss[i]) {
                        const std::string p_token = tokens_to_output_formatted_string(ctx, p.tok);
                        float p_token_logprob = p.prob;
                        std::vector<unsigned char> p_token_bytes(p_token.begin(), p_token.end());
                        token_top_logprobs.push_back(json{
                            {"token", p_token},
                            {"logprob", p_token_logprob},
                            {"bytes", p_token_bytes},
                        });
                        if (p.tok == prob.toks[i]) {
                            token_logprob = p_token_logprob;
                        }
                    }

                    content.push_back(json{
                        {"token", token},
                        {"logprob", token_logprob},
                        {"bytes", token_bytes},
                        {"top_logprobs", token_top_logprobs},
                    });
                }
            }

            return json{{"content", content}};
        } else {
            json token_logprobs = json::array();
            json tokens = json::array();
            json top_logprobs = json::array();

            for (const auto &prob : probs) {
                const auto sz_toks = int32_t(prob.toks.size());
                for (int32_t i = 0; i < sz_toks; i++) {
                    const std::string token = tokens_to_output_formatted_string(ctx, prob.toks[i]);
                    float token_logprob = 1.0f;
                    json token_top_logprobs;
                    for (const auto &p : prob.probss[i]) {
                        const std::string p_token = tokens_to_output_formatted_string(ctx, p.tok);
                        float p_token_logprob = p.prob;
                        token_top_logprobs[p_token] = p_token_logprob;
                        if (p.tok == prob.toks[i]) {
                            token_logprob = p_token_logprob;
                        }
                    }

                    tokens.push_back(token);
                    token_logprobs.push_back(token_logprob);
                    top_logprobs.push_back(token_top_logprobs);
                }
            }

            return json{{"tokens", tokens},
                        {"token_logprobs", token_logprobs},
                        {"top_logprobs", top_logprobs}};
        }
    }

    json out = json::array();

    for (const auto &prob : probs) {
        const auto sz_toks = int32_t(prob.toks.size());
        for (int32_t i = 0; i < sz_toks; i++) {
            json probs_for_token = json::array();
            for (const auto &p : prob.probss[i]) {
                const std::string tok_str = tokens_to_output_formatted_string(ctx, p.tok);
                probs_for_token.push_back(json{
                    {"tok_str", tok_str},
                    {"prob", p.prob},
                });
            }

            const std::string tok_str = tokens_to_output_formatted_string(ctx, prob.toks[i]);
            out.push_back(json{
                {"content", tok_str},
                {"probs", probs_for_token},
            });
        }
    }

    return out;
}

static bool server_sent_event(httplib::DataSink &sink, const char *event, const json &data) {
    const std::string str = std::string(event) + ": " +
                            data.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
    return sink.write(str.c_str(), str.size());
}

//
// OAI utils
//

static json oaicompat_completion_request(const struct llama_model *model, const json &body,
                                         const std::string &chat_template) {
    // Print the request for debugging
    {
        json body_cp = body;
        if (body_cp.contains("messages")) {
            body_cp["messages"] = "[...]";
        } else if (body_cp.contains("prompt")) {
            body_cp["prompt"] = "...";
        }
        SRV_INF("params: %s\n",
                body_cp.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    }

    bool chat = !chat_template.empty();
    json llama_params;

    // Annotations for OAI compatibility
    llama_params["__oaicompat"] = true;
    llama_params["__oaicompat_completion"] = true;
    llama_params["__oaicompat_completion_chat"] = chat;
    llama_params["__oaicompat_completion_chat_vision"] = false;

    // Handle default field
    llama_params["model"] = json_value(body, "model", std::string(DEFAULT_OAICOMPAT_MODEL));
    llama_params["frequency_penalty"] = json_value(body, "frequency_penalty", 0.0f);
    llama_params["temperature"] = json_value(body, "temperature", 1.0f);
    llama_params["top_p"] = json_value(body, "top_p", 1.0f);

    // Handle "max_tokens" field
    llama_params["n_predict"] = json_value(body, "max_tokens", -1);

    // Apply chat template to the list of messages
    if (chat) {
        const json messages = body.at("messages");
        bool chat_vision = false;
        for (const json &msg : messages) {
            if (!msg.contains("content") || !msg.at("content").is_array()) {
                continue;
            }
            for (const json &part : msg.at("content")) {
                if (part.contains("type") && part.at("type") == "image_url") {
                    chat_vision = true;
                    break;
                }
            }
        }
        if (!chat_vision) {
            llama_params["prompt"] = format_chat(model, chat_template, messages);
        } else {
            llama_params["__oaicompat_completion_chat_vision"] = true;
            // Parse the vision messages,
            // see https://platform.openai.com/docs/guides/vision
            for (const json &msg : messages) {
                if (msg.contains("role") && msg.at("role") == "user") {
                    llama_params["prompt"] = msg.at("content");
                    break;
                }
            }
            if (!llama_params.contains("prompt")) {
                throw std::runtime_error(
                    R"(Illegal param: only "user" role is supported to request vision completion)");
            }
        }
    } else if (body.contains("prompt")) {
        llama_params["prompt"] = body.at("prompt");
    } else {
        throw std::runtime_error("Illegal param: missing required field: prompt");
    }

    // Handle "stop" field
    if (body.contains("stop") && body.at("stop").is_string()) {
        llama_params["stop"] = json::array({body.at("stop").get<std::string>()});
    } else {
        llama_params["stop"] = json_value(body, "stop", json::array());
    }

    // Handle "response_format" field
    if (body.contains("response_format")) {
        json response_format = json_value(body, "response_format", json::object());
        std::string response_type = json_value(response_format, "type", std::string());
        if (response_type == "json_object") {
            llama_params["json_schema"] = json_value(response_format, "schema", json::object());
        } else if (response_type == "json_schema") {
            json json_schema = json_value(response_format, "json_schema", json::object());
            llama_params["json_schema"] = json_value(json_schema, "schema", json::object());
        } else if (!response_type.empty() && response_type != "text") {
            throw std::runtime_error(
                R"(Illegal param: "response_format" must be one of "text" or "json_object", but got: )" +
                response_type);
        }
    }

    // Handle "n" field
    int n_choices = json_value(body, "n", 1);
    if (n_choices != 1) {
        throw std::runtime_error("Illegal param: only one completion choice is allowed");
    }

    // Handle "logprobs" field
    if (json_value(body, "logprobs", false)) {
        if (chat) {
            llama_params["n_probs"] = std::min(json_value(body, "top_logprobs", 2), 20);
        } else {
            llama_params["n_probs"] = std::min(json_value(body, "logprobs", 2), 5);
        }
    } else if (!body.contains("logprobs") && body.contains("top_logprobs")) {
        throw std::runtime_error(R"(Illegal param: "top_logprobs" requires "logprobs" to be set)");
    }

    // Params supported by OAI but unsupported by llama.cpp
    static const std::vector<std::string> unsupported_params{"tools", "tool_choice"};
    for (auto &param : unsupported_params) {
        if (body.contains(param)) {
            throw std::runtime_error("Unsupported param: " + param);
        }
    }

    // Copy remaining properties to llama_params
    // This allows user to use llama.cpp-specific params like "mirostat", "tfs_z",... via OAI
    // endpoint. See "launch_slot_with_task()" for a complete list of params supported by llama.cpp
    for (const auto &item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by
        // "max_tokens"
        const std::string &key = item.key();
        if (key == "messages" || (llama_params.contains(key) && key != "n_predict")) {
            continue;
        }
        llama_params[item.key()] = item.value();
    }

    // Handle "stream_options" field
    if (json_value(llama_params, "stream", false)) {
        if (!body.contains("stream_options")) {
            llama_params["stream_options"] = json{{"include_usage", true}};
        } else if (body.at("stream_options").is_object()) {
            if (!body.at("stream_options").contains("include_usage")) {
                llama_params["stream_options"]["include_usage"] = true;
            }
        } else {
            throw std::runtime_error("Illegal param: invalid type for \"stream_options\" field");
        }
    }

    return llama_params;
}

static json oaicompat_completion_response(const json &request, const json result,
                                          const std::string &completion_id, bool streaming = false,
                                          bool first = false) {
    bool stopped_word = json_value(result, "stopped_word", false);
    bool stopped_eos = json_value(result, "stopped_eos", false);
    bool stopped_limit = json_value(result, "stopped_limit", false);
    std::string content = json_value(result, "content", std::string(""));

    std::string finish_reason;
    if (stopped_word || stopped_eos) {
        finish_reason = "stop";
    }
    if (stopped_limit) {
        finish_reason = "length";
    }

    json res = json{
        {"id", completion_id},
        {"created", std::time(nullptr)},
        {"model", json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
    };

    bool chat = json_value(request, "__oaicompat_completion_chat", false);
    bool finish = !finish_reason.empty();
    json choice;
    if (chat) {
        // chat completion
        if (streaming) {
            res["object"] = "chat.completion.chunk";
            if (!finish && first) {
                choice = json{{"finish_reason", nullptr},
                              {"index", 0},
                              {"delta", json{{"role", "assistant"}}}};
            } else if (!finish) {
                choice = json{{"finish_reason", nullptr},
                              {"index", 0},
                              {"delta", json{{"content", content}}}};
            } else {
                // finished
                choice =
                    json{{"finish_reason", finish_reason}, {"index", 0}, {"delta", json::object()}};
            }
        } else {
            res["object"] = "chat.completion";
            if (!finish) {
                choice = json{{"finish_reason", nullptr},
                              {"index", 0},
                              {"message", json{{"content", content}, {"role", "assistant"}}}};
            } else {
                choice = json{{"finish_reason", finish_reason},
                              {"index", 0},
                              {"message", json{{"content", content}, {"role", "assistant"}}}};
            }
        }
    } else {
        // completion
        res["object"] = "text_completion";
        if (!finish) {
            choice = json{{"finish_reason", nullptr}, {"index", 0}, {"text", content}};
        } else {
            choice = json{{"finish_reason", finish_reason}, {"index", 0}, {"text", content}};
        }
    }
    bool logprobs = result.contains("completion_probabilities");
    if (!logprobs) {
        choice["logprobs"] = nullptr;
    } else {
        choice["logprobs"] = result.at("completion_probabilities");
    }
    res["choices"] = json::array({choice});

    // Add usage information
    bool include_usage = false;
    if (request.contains("stream_options")) {
        include_usage = json_value(request.at("stream_options"), "include_usage", false);
    }
    if (!streaming || (include_usage && !finish_reason.empty())) {
        int completion_tokens = json_value(result, "tokens_predicted", 0);
        int prompt_tokens = json_value(result, "tokens_evaluated", 0);
        json ts = json_value(result, "timings", json::object());
        double ttft = json_value(ts, "prompt_ms", 0.0);
        double tpot = json_value(ts, "predicted_per_token_ms", 0.0);
        double tps = json_value(ts, "predicted_per_second", 0.0);
        json usage = json{{"completion_tokens", completion_tokens},
                          {"prompt_tokens", prompt_tokens},
                          {"total_tokens", completion_tokens + prompt_tokens},
                          {"time_to_first_token_ms", ttft},
                          {"time_per_output_token_ms", tpot},
                          {"tokens_per_second", tps}};
        if (ts.contains("drafted_n")) {
            usage["draft_tokens"] = ts.at("drafted_n");
            usage["draft_tokens_acceptance"] = ts.at("drafted_accepted_p");
        }
        res["usage"] = usage;
    } else if (include_usage) {
        res["usage"] = nullptr;
    }

    return res;
}

static json oaicompat_embedding_request(const struct gpt_params &params, const json &body) {
    // Print the request for debugging
    {
        json body_cp = body;
        if (body_cp.at("input").is_string()) {
            body_cp["input"] = "...";
        } else {
            body_cp["input"] = "[...]";
        }
        SRV_INF("params: %s\n",
                body_cp.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    }

    json llama_params;

    // Annotations for OAI compatibility
    llama_params["__oaicompat"] = true;
    llama_params["__oaicompat_embedding"] = true;

    // Handle "model" field
    llama_params["model"] = json_value(body, "model", params.model_alias);

    // Handle "input" field
    llama_params["prompt"] = body.at("input");

    // Handle "encoding_format" field
    llama_params["encoding_format"] = json_value(body, "encoding_format", std::string("float"));

    return llama_params;
}

static json oaicompat_embedding_response(const json &request, const json &result) {
    json data = json::array();
    data.push_back(json{{"embedding", json_value(result, "embedding", json::array())},
                        {"index", 0},
                        {"object", "embedding"}});

    int num_prompt_tokens = json_value(result, "tokens_evaluated", 0);
    json res = json{
        {"model", json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
        {"object", "list"},
        {"usage", json{{"prompt_tokens", num_prompt_tokens}, {"total_tokens", num_prompt_tokens}}},
        {"data", data}};

    return res;
}

static json jinaaicompat_rerank_request(const struct gpt_params &params, const json &body) {
    // Print the request for debugging
    {
        json body_cp = body;
        if (body_cp.contains("query")) {
            body_cp["query"] = "...";
        }
        if (body_cp.contains("documents")) {
            body_cp["documents"] = "[...]";
        }
        SRV_INF("params: %s\n",
                body_cp.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    }

    json llama_params;

    // Annotations for OAI compatibility
    llama_params["__oaicompat"] = true;
    llama_params["__oaicompat_rerank"] = true;

    // Handle "model" field
    llama_params["model"] = json_value(body, "model", params.model_alias);

    // Handle "query" and "documents" fields
    json prompt = json::array();
    prompt.push_back(json_value(body, "query", std::string("")));
    for (const json &doc : body.at("documents")) {
        if (!doc.is_string()) {
            throw std::runtime_error("Illegal param: documents must be an array of strings");
        }
        prompt.push_back(doc.get<std::string>());
    }
    llama_params["prompt"] = prompt;

    // Handle "top_n" field
    size_t documents_size = body.at("documents").size();
    size_t top_n = json_value(body, "top_n", documents_size);
    if (top_n > documents_size) {
        top_n = documents_size;
    } else if (top_n <= 0) {
        throw std::runtime_error("Illegal param: top_n must be greater than 0");
    }
    llama_params["top_n"] = top_n;

    return llama_params;
}

static void jinaicompat_rerank_response_sort(json &result, int32_t low, int32_t high) {
    if (low >= high) {
        return;
    }

    json base = result[low];
    int i = low, j = high;
    while (i != j) {
        while (i < j && json_value(result[j], "score", 0.0) <= json_value(base, "score", 0.0))
            j--;
        while (i < j && json_value(result[i], "score", 0.0) >= json_value(base, "score", 0.0))
            i++;
        if (i < j) {
            json temp = result[i];
            result[i] = result[j];
            result[j] = temp;
        }
    }
    result[low] = result[i];
    result[i] = base;
    jinaicompat_rerank_response_sort(result, low, i - 1);
    jinaicompat_rerank_response_sort(result, i + 1, high);
}

static json jinaicompat_rerank_response(const json &request, json &result) {
    int32_t top_n = json_value(request, "top_n", 1);

    int num_prompt_tokens = 0;
    json prompt = request.at("prompt");
    json data = json::array();

    int32_t start = 0;
    auto end = int32_t(result.size() - 1);
    jinaicompat_rerank_response_sort(result, start, end);
    for (int32_t i = 0; i <= end; i++) {
        const json &ret = result[i];
        num_prompt_tokens += json_value(ret, "tokens_evaluated", 0);
        if (i < top_n) {
            const int32_t idx = json_value(ret, "index", 0);
            const double scr = json_value(ret, "score", 0.0);
            data.push_back(json{
                {"index", idx},
                {"document", {{"text", prompt[idx + 1]}}},
                {"relevance_score", scr},
            });
        }
    }

    json res = json{
        {"model", json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
        {"object", "list"},
        {"usage", json{{"prompt_tokens", num_prompt_tokens}, {"total_tokens", num_prompt_tokens}}},
        {"results", data}};

    return res;
}

static bool is_valid_utf8(const std::string &str) {
    const auto *bytes = reinterpret_cast<const unsigned char *>(str.data());
    const unsigned char *end = bytes + str.length();

    while (bytes < end) {
        if (*bytes <= 0x7F) {
            // 1-byte sequence (0xxxxxxx)
            bytes++;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // 2-byte sequence (110xxxxx 10xxxxxx)
            if (end - bytes < 2 || (bytes[1] & 0xC0) != 0x80)
                return false;
            bytes += 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // 3-byte sequence (1110xxxx 10xxxxxx 10xxxxxx)
            if (end - bytes < 3 || (bytes[1] & 0xC0) != 0x80 || (bytes[2] & 0xC0) != 0x80)
                return false;
            bytes += 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // 4-byte sequence (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
            if (end - bytes < 4 || (bytes[1] & 0xC0) != 0x80 || (bytes[2] & 0xC0) != 0x80 ||
                (bytes[3] & 0xC0) != 0x80)
                return false;
            bytes += 4;
        } else {
            // Invalid UTF-8 lead byte
            return false;
        }
    }

    return true;
}

static json format_error_response(const std::string &message, const enum error_type type) {
    std::string type_str;
    int code = 500;
    switch (type) {
    case ERROR_TYPE_INVALID_REQUEST:
        type_str = "invalid_request_error";
        code = 400;
        break;
    case ERROR_TYPE_AUTHENTICATION:
        type_str = "authentication_error";
        code = 401;
        break;
    case ERROR_TYPE_NOT_FOUND:
        type_str = "not_found_error";
        code = 404;
        break;
    case ERROR_TYPE_SERVER:
        type_str = "server_error";
        code = 500;
        break;
    case ERROR_TYPE_PERMISSION:
        type_str = "permission_error";
        code = 403;
        break;
    case ERROR_TYPE_NOT_SUPPORTED:
        type_str = "not_supported_error";
        code = 501;
        break;
    case ERROR_TYPE_UNAVAILABLE:
        type_str = "unavailable_error";
        code = 503;
        break;
    }
    return json{
        {"code", code},
        {"message", message},
        {"type", type_str},
    };
}
