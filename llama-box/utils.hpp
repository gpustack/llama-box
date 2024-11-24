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

#include "stablediffusion.hpp"

#define DEFAULT_OAICOMPAT_MODEL "gpt-3.5-turbo-0613"

#define SLT_INF(slot, fmt, ...) LOG_INF("slot %25.*s: id %2d | task %d | " fmt, 25, __func__, (slot).id, (slot).id_task, __VA_ARGS__)
#define SLT_WRN(slot, fmt, ...) LOG_WRN("slot %25.*s: id %2d | task %d | " fmt, 25, __func__, (slot).id, (slot).id_task, __VA_ARGS__)
#define SLT_ERR(slot, fmt, ...) LOG_ERR("slot %25.*s: id %2d | task %d | " fmt, 25, __func__, (slot).id, (slot).id_task, __VA_ARGS__)
#define SLT_DBG(slot, fmt, ...) LOG_DBG("slot %25.*s: id %2d | task %d | " fmt, 25, __func__, (slot).id, (slot).id_task, __VA_ARGS__)

#define SRV_INF(fmt, ...) LOG_INF("srv  %25.*s: " fmt, 25, __func__, __VA_ARGS__)
#define SRV_WRN(fmt, ...) LOG_WRN("srv  %25.*s: " fmt, 25, __func__, __VA_ARGS__)
#define SRV_ERR(fmt, ...) LOG_ERR("srv  %25.*s: " fmt, 25, __func__, __VA_ARGS__)
#define SRV_DBG(fmt, ...) LOG_DBG("srv  %25.*s: " fmt, 25, __func__, __VA_ARGS__)

#define QUE_INF(fmt, ...) LOG_INF("que  %25.*s: " fmt, 25, __func__, __VA_ARGS__)
#define QUE_WRN(fmt, ...) LOG_WRN("que  %25.*s: " fmt, 25, __func__, __VA_ARGS__)
#define QUE_ERR(fmt, ...) LOG_ERR("que  %25.*s: " fmt, 25, __func__, __VA_ARGS__)
#define QUE_DBG(fmt, ...) LOG_DBG("que  %25.*s: " fmt, 25, __func__, __VA_ARGS__)

using json         = nlohmann::json;
using llama_tokens = std::vector<llama_token>;

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
            SRV_WRN("wrong type supplied for parameter '%s'. Expected '%s', using default value\n", key.c_str(), json(default_value).type_name());
            return default_value;
        }
    } else {
        return default_value;
    }
}

//
// tokenizer and input processing utils
//

static bool json_is_array_of_numbers(const json &data) {
    if (data.is_array()) {
        return std::all_of(data.begin(), data.end(), [](const json &e) { return e.is_number_integer(); });
    }
    return false;
}

// is array having BOTH numbers & strings?
static bool json_is_array_of_mixed_numbers_strings(const json &data) {
    bool seen_string = false;
    bool seen_number = false;
    if (data.is_array()) {
        for (const auto &e : data) {
            seen_string |= e.is_string();
            seen_number |= e.is_number_integer();
            if (seen_number && seen_string) {
                return true;
            }
        }
    }
    return false;
}

static bool json_is_array_of_objects(const json &data) {
    if (data.is_array()) {
        return std::all_of(data.begin(), data.end(), [](const json &e) { return e.is_object(); });
    }
    return false;
}

/**
 * this handles 2 cases:
 * - only string, example: "string"
 * - mixed string and tokens, example: [12, 34, "string", 56, 78]
 */
static llama_tokens tokenize_mixed(const llama_context *ctx, const json &json_prompt, bool add_special, bool parse_special) {
    // If `add_bos` is true, we only add BOS, when json_prompt is a string,
    // or the first element of the json_prompt array is a string.
    llama_tokens prompt_tokens;

    if (json_prompt.is_array()) {
        bool first = true;
        for (const auto &jp : json_prompt) {
            if (jp.is_string()) {
                std::string s = jp.get<std::string>();
                llama_tokens p;
                if (first) {
                    p     = common_tokenize(ctx, s, add_special, parse_special);
                    first = false;
                } else {
                    p = common_tokenize(ctx, s, false, parse_special);
                }
                prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
            } else if (jp.is_number_integer()) {
                if (first) {
                    first = false;
                }
                prompt_tokens.push_back(jp.get<llama_token>());
            }
        }
    } else {
        std::string s = json_prompt.get<std::string>();
        prompt_tokens = common_tokenize(ctx, s, add_special, parse_special);
    }

    return prompt_tokens;
}

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * and multiple prompts (multi-tasks):
 * - "prompt": ["string1", "string2"]
 * - "prompt": ["string1", [12, 34, 56]]
 * - "prompt": [[12, 34, "string", 56, 78], [12, 34, 56]]
 */
static std::vector<llama_tokens> tokenize_input_prompts(llama_context *ctx, const json &json_prompt, bool add_special, bool parse_special) {
    std::vector<llama_tokens> result;
    if (json_prompt.is_string() || json_is_array_of_mixed_numbers_strings(json_prompt)) {
        // string or mixed
        result.push_back(tokenize_mixed(ctx, json_prompt, add_special, parse_special));
    } else if (json_is_array_of_numbers(json_prompt)) {
        // array of tokens
        result.push_back(json_prompt.get<llama_tokens>());
    } else if (json_prompt.is_array()) {
        // array of prompts
        result.reserve(json_prompt.size());
        for (const auto &p : json_prompt) {
            if (p.is_string() || json_is_array_of_mixed_numbers_strings(p)) {
                // string or mixed
                result.push_back(tokenize_mixed(ctx, p, add_special, parse_special));
            } else if (json_is_array_of_numbers(p)) {
                // array of tokens
                result.push_back(p.get<llama_tokens>());
            } else if (json_is_array_of_objects(p)) {
                // array of objects
                result.push_back(tokenize_mixed(ctx, p, add_special, parse_special));
            } else {
                throw std::runtime_error("element of \"prompt\" must be a string, a list of tokens, or a list of mixed strings & tokens");
            }
        }
    } else {
        throw std::runtime_error("\"prompt\" must be a string, a list of tokens, a list of mixed strings & tokens, or a list of prompts");
    }
    return result;
}

//
// template utils
//

// format rerank task: [BOS]query[EOS][SEP]doc[EOS]
static llama_tokens format_rerank(const struct llama_model *model, const llama_tokens &query, const llama_tokens &doc) {
    llama_tokens result;
    result.reserve(doc.size() + query.size() + 4);
    result.push_back(llama_token_bos(model));
    result.insert(result.end(), query.begin(), query.end());
    result.push_back(llama_token_eos(model));
    result.push_back(llama_token_sep(model));
    result.insert(result.end(), doc.begin(), doc.end());
    result.push_back(llama_token_eos(model));
    return result;
}

// format infill task
static llama_tokens format_infill(const llama_context *ctx, const json &input_prefix, const json &input_suffix, const json &input_extra,
                                  const int n_batch, const int n_predict, const int n_ctx, const bool spm_infill, const llama_tokens &tokens_prompt) {
    // TODO: optimize this block by reducing memory allocations and movement

    // use FIM repo-level pattern:
    // ref: https://arxiv.org/pdf/2409.12186
    //
    // [FIM_REP]myproject
    // [FIM_SEP]filename0
    // extra chunk 0
    // [FIM_SEP]filename1
    // extra chunk 1
    // ...
    // [FIM_SEP]filename
    // [FIM_PRE]prefix[FIM_SUF]suffix[FIM_MID]prompt
    //
    llama_tokens extra_tokens;
    extra_tokens.reserve(n_ctx);

    auto model         = llama_get_model(ctx);
    auto tokens_prefix = tokenize_mixed(ctx, input_prefix, false, false);
    auto tokens_suffix = tokenize_mixed(ctx, input_suffix, false, false);

    if (llama_token_fim_rep(model) != LLAMA_TOKEN_NULL) {
        // TODO: make project name an input
        static const auto k_fim_repo = common_tokenize(ctx, "myproject\n", false, false);

        extra_tokens.push_back(llama_token_fim_rep(model));
        extra_tokens.insert(extra_tokens.end(), k_fim_repo.begin(), k_fim_repo.end());
    }
    for (const auto &chunk : input_extra) {
        // { "text": string, "filename": string }
        const std::string text     = json_value(chunk, "text", std::string());
        const std::string filename = json_value(chunk, "filename", std::string("tmp"));

        if (llama_token_fim_sep(model) != LLAMA_TOKEN_NULL) {
            const auto k_fim_file = common_tokenize(ctx, filename + "\n", false, false);

            extra_tokens.insert(extra_tokens.end(), llama_token_fim_sep(model));
            extra_tokens.insert(extra_tokens.end(), k_fim_file.begin(), k_fim_file.end());
        } else {
            // chunk separator in binary form to avoid confusing the AI
            static const char k_chunk_prefix_str[]  = {0x0a, 0x0a, 0x2d, 0x2d, 0x2d, 0x20, 0x73, 0x6e, 0x69, 0x70,
                                                       0x70, 0x65, 0x74, 0x20, 0x2d, 0x2d, 0x2d, 0x0a, 0x0a, 0x00};
            static const auto k_chunk_prefix_tokens = common_tokenize(ctx, k_chunk_prefix_str, false, false);

            extra_tokens.insert(extra_tokens.end(), k_chunk_prefix_tokens.begin(), k_chunk_prefix_tokens.end());
        }

        const auto chunk_tokens = common_tokenize(ctx, text, false, false);
        extra_tokens.insert(extra_tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
    }

    if (llama_token_fim_sep(model) != LLAMA_TOKEN_NULL) {
        // TODO: current filename
        static const auto k_fim_file = common_tokenize(ctx, "filename\n", false, false);

        extra_tokens.insert(extra_tokens.end(), llama_token_fim_sep(model));
        extra_tokens.insert(extra_tokens.end(), k_fim_file.begin(), k_fim_file.end());
    }

    // for now pick FIM context to fit in a batch (ratio prefix:suffix = 3:1, TODO: configurable?)
    const int n_prefix_take = std::min<int>(tokens_prefix.size(), 3 * (n_batch / 4));
    const int n_suffix_take = std::min<int>(tokens_suffix.size(), std::max<int>(0, (n_batch / 4) - (2 + tokens_prompt.size())));
    SRV_DBG("n_prefix_take = %d, n_suffix_take = %d, total = %d\n", n_prefix_take, n_suffix_take, (n_prefix_take + n_suffix_take));

    // fill the rest of the context with extra chunks
    const int n_extra_take = std::min<int>(std::max<int>(0, n_ctx - (n_batch)-2 * n_predict), extra_tokens.size());

    tokens_prefix.erase(tokens_prefix.begin(), tokens_prefix.begin() + tokens_prefix.size() - n_prefix_take);
    tokens_suffix.resize(n_suffix_take);

    tokens_prefix.insert(tokens_prefix.begin(), llama_token_fim_pre(model));
    tokens_prefix.insert(tokens_prefix.end(), tokens_prompt.begin(), tokens_prompt.end());
    tokens_suffix.insert(tokens_suffix.begin(), llama_token_fim_suf(model));

    auto embd_inp = spm_infill ? tokens_suffix : tokens_prefix;
    auto embd_end = spm_infill ? tokens_prefix : tokens_suffix;

    if (llama_add_bos_token(model)) {
        embd_inp.insert(embd_inp.begin(), llama_token_bos(model));
    }

    SRV_DBG("extra: n_ctx = %d, n_extra_take = %d, n_extra = %d\n", n_ctx, n_extra_take, (int)extra_tokens.size());

    // put the extra context before the FIM prefix
    embd_inp.insert(embd_inp.begin(), extra_tokens.end() - n_extra_take, extra_tokens.end());

    embd_inp.insert(embd_inp.end(), embd_end.begin(), embd_end.end());
    embd_inp.push_back(llama_token_fim_mid(model));

    return embd_inp;
}

// Format given chat. If tmpl is empty, we take the template from model metadata
inline std::string format_chat(const struct llama_model *model, const std::string &tmpl, const std::vector<json> &messages) {
    std::vector<common_chat_msg> chat;

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
            throw std::runtime_error("Missing 'content' (ref: https://github.com/ggerganov/llama.cpp/issues/8367)");
        }

        chat.push_back({role, content});
    }

    return common_chat_apply_template(model, tmpl, chat, true);
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
    int i   = 0;
    int j   = 0;
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

static inline const std::string base64_encode(const unsigned char *input, size_t length) {
    std::string output;
    output.reserve(length);

    auto val = 0, valb = -6;
    for (size_t i = 0; i < length; ++i) {
        val = (val << 8) + static_cast<uint8_t>(input[i]);
        valb += 8;
        while (valb >= 0) {
            output.push_back(base64_chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }

    if (valb > -6) {
        output.push_back(base64_chars[((val << 8) >> valb) & 0x3F]);
    }

    while (output.size() % 4) {
        output.push_back('=');
    }

    return output;
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

static size_t longest_common_prefix(const std::vector<llama_token> &a, const std::vector<llama_token> &b) {
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++) {
    }

    return i;
}

static size_t longest_common_subsequence(const llama_tokens &a, const llama_tokens &b) {
    // check for empty sequences
    if (a.empty() || b.empty()) {
        return 0;
    }

    // get the lengths of the input sequences
    size_t a_len = a.size();
    size_t b_len = b.size();

    // initialize the maximum length of the longest common subsequence (LCS)
    size_t max_length = 0;

    // use two rows instead of a 2D matrix to optimize space
    std::vector<size_t> prev_row(b_len + 1, 0);
    std::vector<size_t> curr_row(b_len + 1, 0);

    // iterate through the elements of a
    for (size_t i = 1; i <= a_len; i++) {
        // iterate through the elements of b
        for (size_t j = 1; j <= b_len; j++) {
            // if elements at the current positions match
            if (a[i - 1] == b[j - 1]) {
                // if it's the first element of either sequences, set LCS length to 1
                if (i == 1 || j == 1) {
                    curr_row[j] = 1;
                } else {
                    // increment LCS length by 1 compared to the previous element
                    curr_row[j] = prev_row[j - 1] + 1;
                }

                // update max_length if necessary
                if (curr_row[j] > max_length) {
                    max_length = curr_row[j];
                }
            } else {
                // reset LCS length if elements don't match
                curr_row[j] = 0;
            }
        }

        // update the previous row for the next iteration
        prev_row = curr_row;
    }

    // return the maximum length of the LCS
    return max_length;
}

static bool ends_with(const std::string &str, const std::string &suffix) {
    return str.size() >= suffix.size() && 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
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

// format incomplete utf-8 multibyte character for output
static std::string tokens_to_output_formatted_string(const llama_context *ctx, const llama_token token) {
    std::string out = token == -1 ? "" : common_token_to_piece(ctx, token);

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
static json probs_vector_to_json(const llama_context *ctx, const std::vector<completion_token_output> &probs, const bool oaicompat_completion = false,
                                 const bool oaicompat_completion_chat = false) {
    if (oaicompat_completion) {
        if (oaicompat_completion_chat) {
            json content = json::array();

            for (const auto &prob : probs) {
                const auto sz_toks = int32_t(prob.toks.size());
                for (int32_t i = 0; i < sz_toks; i++) {
                    const std::string token = tokens_to_output_formatted_string(ctx, prob.toks[i]);
                    float token_logprob     = 1.0f;
                    std::vector<unsigned char> token_bytes(token.begin(), token.end());
                    json token_top_logprobs = json::array();
                    for (const auto &p : prob.probss[i]) {
                        const std::string p_token = tokens_to_output_formatted_string(ctx, p.tok);
                        float p_token_logprob     = p.prob;
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
            json tokens         = json::array();
            json top_logprobs   = json::array();

            for (const auto &prob : probs) {
                const auto sz_toks = int32_t(prob.toks.size());
                for (int32_t i = 0; i < sz_toks; i++) {
                    const std::string token = tokens_to_output_formatted_string(ctx, prob.toks[i]);
                    float token_logprob     = 1.0f;
                    json token_top_logprobs;
                    for (const auto &p : prob.probss[i]) {
                        const std::string p_token   = tokens_to_output_formatted_string(ctx, p.tok);
                        float p_token_logprob       = p.prob;
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

            return json{{"tokens", tokens}, {"token_logprobs", token_logprobs}, {"top_logprobs", top_logprobs}};
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
    const std::string str = std::string(event) + ": " + data.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
    return sink.write(str.c_str(), str.size());
}

//
// OAI utils
//

static json oaicompat_completions_request(const struct common_params &params, const json &body, const struct llama_model *model,
                                          const std::string &chat_template) {
    // Print the request for debugging
    {
        json body_cp = body;
        if (common_log_verbosity_thold < 2) {
            if (body_cp.contains("messages")) {
                body_cp["messages"] = "[...]";
            } else if (body_cp.contains("prompt")) {
                body_cp["prompt"] = "...";
            }
        }
        SRV_INF("params: %s\n", body_cp.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    }

    bool chat = !chat_template.empty();
    json llama_params;

    // Annotations for OAI compatibility
    llama_params["__oaicompat"]                        = true;
    llama_params["__oaicompat_completion"]             = true;
    llama_params["__oaicompat_completion_chat"]        = chat;
    llama_params["__oaicompat_completion_chat_vision"] = false;

    // Handle default field
    llama_params["model"]             = json_value(body, "model", params.model_alias);
    llama_params["frequency_penalty"] = json_value(body, "frequency_penalty", 0.0f);
    llama_params["temperature"]       = json_value(body, "temperature", 1.0f);
    llama_params["top_p"]             = json_value(body, "top_p", 1.0f);

    // Handle "max_tokens" field
    llama_params["n_predict"] = json_value(body, "max_tokens", -1);

    // Apply chat template to the list of messages
    if (chat) {
        const json messages = body.at("messages");
        bool chat_vision    = false;
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
                throw std::runtime_error(R"(Illegal param: only "user" role is supported to request vision completion)");
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
        json response_format      = json_value(body, "response_format", json::object());
        std::string response_type = json_value(response_format, "type", std::string());
        if (response_type == "json_object") {
            llama_params["json_schema"] = json_value(response_format, "schema", json::object());
        } else if (response_type == "json_schema") {
            json json_schema            = json_value(response_format, "json_schema", json::object());
            llama_params["json_schema"] = json_value(json_schema, "schema", json::object());
        } else if (!response_type.empty() && response_type != "text") {
            throw std::runtime_error(R"(Illegal param: "response_format" must be one of "text" or "json_object", but got: )" + response_type);
        }
    }

    // Handle "n" field
    int n_choices = json_value(body, "n", 1);
    if (n_choices != 1) {
        throw std::runtime_error("Illegal param: only one completion choice is allowed");
    }

    // Handle "logprobs" field
    if (chat) {
        if (!body.contains("logprobs") && body.contains("top_logprobs")) {
            throw std::runtime_error(R"(Illegal param: "top_logprobs" requires "logprobs" to be set)");
        }
        if (json_value(body, "logprobs", false)) {
            llama_params["n_probs"] = std::min(json_value(body, "top_logprobs", 1), 20);
        }
    } else {
        if (body.contains("logprobs")) {
            llama_params["n_probs"] = std::min(json_value(body, "logprobs", 1), 5);
        }
    }

    // Params supported by OAI but unsupported by llama.cpp
    static const std::vector<std::string> unsupported_params{"tools", "tool_choice"};
    for (auto &param : unsupported_params) {
        if (body.contains(param)) {
            throw std::runtime_error("Unsupported param: " + param);
        }
    }

    // Copy remaining properties to llama_params
    // This allows user to use llama.cpp-specific params like "mirostat", ... via OAI
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

static json oaicompat_completions_response(const json &request, const json &result, const std::string &completion_id, bool streaming = false,
                                           bool first = false) {
    json res = json{
        {"id", completion_id},
        {"created", std::time(nullptr)},
        {"model", json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
    };

    bool chat                      = json_value(request, "__oaicompat_completion_chat", false);
    int completion_tokens          = 0;
    int prompt_tokens              = 0;
    int drafted_tokens             = 0;
    double draft_tokens_acceptance = 0.0;
    double ttft                    = 0.0;
    double tpot                    = 0.0;
    double tps                     = 0.0;

    // Construct choices field
    json choices = json::array();
    if (first && streaming && chat) {
        res["object"] = "chat.completion.chunk";
        choices.push_back(json{
            {"finish_reason", nullptr},
            {"index", 0},
            {"delta", json{{"role", "assistant"}}},
        });
    }
    bool finish = false;
    for (const json &ret : result) {
        bool stopped_word  = json_value(ret, "stopped_word", false);
        bool stopped_eos   = json_value(ret, "stopped_eos", false);
        bool stopped_limit = json_value(ret, "stopped_limit", false);
        std::string finish_reason;
        if (stopped_word || stopped_eos) {
            finish_reason = "stop";
        }
        if (stopped_limit) {
            finish_reason = "length";
        }
        finish              = !finish_reason.empty();
        std::string content = json_value(ret, "content", std::string(""));
        int index           = json_value(ret, "index", 0);

        json choice;
        if (chat) {
            if (streaming) {
                res["object"] = "chat.completion.chunk";
                if (!finish) {
                    choice = json{
                        {"finish_reason", nullptr},
                        {"index", index},
                        {"delta", json{{"content", content}}},
                    };
                } else {
                    // finished
                    choice = json{
                        {"finish_reason", finish_reason},
                        {"index", index},
                        {"delta", json::object()},
                    };
                }
            } else {
                res["object"] = "chat.completion";
                if (!finish) {
                    choice = json{
                        {"finish_reason", nullptr},
                        {"index", index},
                        {"message", json{{"content", content}, {"role", "assistant"}}},
                    };
                } else {
                    choice = json{
                        {"finish_reason", finish_reason},
                        {"index", index},
                        {"message", json{{"content", content}, {"role", "assistant"}}},
                    };
                }
            }
        } else {
            res["object"] = "text_completion";
            if (!finish) {
                choice = json{
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"text", content},
                };
            } else {
                choice = json{
                    {"finish_reason", finish_reason},
                    {"index", index},
                    {"text", content},
                };
            }
        }

        bool logprobs = ret.contains("completion_probabilities");
        if (!logprobs) {
            choice["logprobs"] = nullptr;
        } else {
            choice["logprobs"] = ret.at("completion_probabilities");
        }

        choices.push_back(choice);

        completion_tokens += json_value(ret, "tokens_predicted", 0);
        prompt_tokens += json_value(ret, "tokens_evaluated", 0);
        json ts = json_value(ret, "timings", json::object());
        ttft += json_value(ts, "prompt_ms", 0.0);
        tpot += json_value(ts, "predicted_per_token_ms", 0.0);
        tps += json_value(ts, "predicted_per_second", 0.0);
        if (ts.contains("drafted_n")) {
            drafted_tokens += json_value(ts, "drafted_n", 0);
            draft_tokens_acceptance += json_value(ts, "drafted_accepted_p", 0.0);
        }
    }
    res["choices"] = choices;

    // Add usage field
    bool include_usage = false;
    if (request.contains("stream_options")) {
        include_usage = json_value(request.at("stream_options"), "include_usage", false);
    }
    if (!streaming || (include_usage && finish)) {
        const size_t result_size = result.size();
        ttft                     = ttft / result_size;
        tpot                     = tpot / result_size;
        tps                      = tps / result_size;
        json usage               = json{
                          {"completion_tokens", completion_tokens},
                          {"prompt_tokens", prompt_tokens},
                          {"total_tokens", completion_tokens + prompt_tokens},
                          {"time_to_first_token_ms", ttft},
                          {"time_per_output_token_ms", tpot},
                          {"tokens_per_second", tps},
        };
        if (drafted_tokens > 0) {
            draft_tokens_acceptance          = draft_tokens_acceptance / result_size;
            usage["draft_tokens"]            = drafted_tokens;
            usage["draft_tokens_acceptance"] = draft_tokens_acceptance;
        }
        res["usage"] = usage;
    } else if (include_usage) {
        res["usage"] = nullptr;
    }

    return res;
}

static json oaicompat_embeddings_request(const struct common_params &params, const json &body) {
    // Print the request for debugging
    {
        json body_cp = body;
        if (common_log_verbosity_thold < 2) {
            if (body_cp.at("input").is_string()) {
                body_cp["input"] = "...";
            } else {
                body_cp["input"] = "[...]";
            }
        }
        SRV_INF("params: %s\n", body_cp.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    }

    json llama_params;

    // Annotations for OAI compatibility
    llama_params["__oaicompat"]           = true;
    llama_params["__oaicompat_embedding"] = true;

    // Handle "model" field
    llama_params["model"] = json_value(body, "model", params.model_alias);

    // Handle "input" field
    llama_params["prompt"] = body.at("input");

    // Handle "encoding_format" field
    llama_params["encoding_format"] = json_value(body, "encoding_format", std::string("float"));

    return llama_params;
}

static json oaicompat_embeddings_response(const json &request, const json &result) {
    int num_prompt_tokens = 0;
    json data             = json::array();
    for (const auto &ret : result) {
        num_prompt_tokens += ret.contains("tokens_evaluated") ? ret.at("tokens_evaluated").get<int>() : 0;
        data.push_back(json{
            {"embedding", ret.at("embedding")},
            {"index", ret.at("index")},
            {"object", "embedding"},
        });
    }

    json res = json{
        {"created", std::time(nullptr)},
        {"model", json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
        {"object", "list"},
        {"data", data},
        {"usage", json{{"prompt_tokens", num_prompt_tokens}, {"total_tokens", num_prompt_tokens}}},
    };

    return res;
}

static json oaicompat_images_generations_request(const struct stablediffusion_params &params, const json &body) {
    // Print the request for debugging
    {
        json body_cp = body;
        if (common_log_verbosity_thold < 2) {
            body_cp["prompt"] = "...";
        }
        SRV_INF("params: %s\n", body_cp.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    }

    json llama_params;

    // Annotations for OAI compatibility
    llama_params["__oaicompat"]                = true;
    llama_params["__oaicompat_image"]          = true;
    llama_params["__oaicompat_image_generate"] = true;

    // Handle "model" field
    llama_params["model"] = json_value(body, "model", params.model_alias);

    // Handle "prompt" field
    llama_params["prompt"] = body.at("prompt");

    // Handle "n" field
    {
        auto batch_count = json_value(body, "n", 1);
        if (batch_count > params.max_batch_count) {
            throw std::runtime_error("Illegal param: n must be less than or equal to " + std::to_string(params.max_batch_count));
        }
        llama_params["batch_count"] = batch_count;
    }

    // Handle "sampler" and "cfg_scale" fields
    if (!body.contains("sampler")) {
        std::string quality = json_value(body, "quality", std::string("standard"));
        if (quality != "null" && quality != "hd" && quality != "standard") {
            throw std::runtime_error("Illegal param: quality must be one of 'hd' or 'standard'");
        }
        llama_params["sampler"]      = params.sampler;
        llama_params["sample_steps"] = params.sample_steps;
        llama_params["cfg_scale"]    = params.cfg_scale;
        if (quality == "hd") {
            llama_params["sample_steps"]    = params.sample_steps + 10;
            llama_params["negative_prompt"] = "low quality";
        }
        if (body.contains("style")) {
            std::string style = json_value(body, "style", std::string("vivid"));
            if (style != "vivid" && style != "natural") {
                throw std::runtime_error("Illegal param: style must be one of 'vivid' or 'natural'");
            }
            if (style == "vivid") {
                if (llama_params.contains("negative_prompt")) {
                    llama_params["negative_prompt"] += " and not vivid";
                } else {
                    llama_params["negative_prompt"] = "not vivid";
                }
            } else {
                if (llama_params.contains("negative_prompt")) {
                    llama_params["negative_prompt"] += " and unnatural";
                } else {
                    llama_params["negative_prompt"] = "unnatural";
                }
            }
        }
    } else {
        std::string sampler_str         = json_value(body, "sampler", std::string("euler_a"));
        llama_params["sampler"]         = sd_argument_to_sample_method(sampler_str.c_str());
        llama_params["cfg_scale"]       = json_value(body, "cfg_scale", params.cfg_scale);
        llama_params["sample_steps"]    = json_value(body, "sample_steps", params.sample_steps);
        llama_params["negative_prompt"] = json_value(body, "negative_prompt", std::string(""));
        if (body.contains("seed")) {
            llama_params["seed"] = body.at("seed");
        }
    }

    // Handle "size" field
    std::string size = json_value(body, "size", std::string("512x512"));
    {
        auto pos = size.find('x');
        if (pos == std::string::npos) {
            throw std::runtime_error("Illegal param: size must be in the format 'widthxheight'");
        }
        auto width  = std::stoi(size.substr(0, pos));
        auto height = std::stoi(size.substr(pos + 1));
        if (width < 256 || height < 256) {
            throw std::runtime_error("Illegal param: width and height must be at least 256");
        }
        if (width > params.max_width) {
            throw std::runtime_error("Illegal param: width must be at most " + std::to_string(params.max_width));
        }
        if (height > params.max_height) {
            throw std::runtime_error("Illegal param: height must be at most " + std::to_string(params.max_height));
        }
        llama_params["width"]  = width;
        llama_params["height"] = height;
    }

    // Handle "response_format" field
    std::string response_format = json_value(body, "response_format", std::string("b64_json"));
    if (response_format != "b64_json") {
        throw std::runtime_error("Illegal param: response_format must be 'b64_json'");
    }

    // Handle "stream" field
    if (json_value(body, "stream", false)) {
        llama_params["stream"] = true;
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

static json oaicompat_images_edits_request(const struct stablediffusion_params &params, const json &body) {
    // Print the request for debugging
    {
        json body_cp = body;
        if (common_log_verbosity_thold < 2) {
            body_cp["prompt"] = "...";
        }
        body_cp["image"] = "...";
        if (body_cp.contains("mask")) {
            body_cp["mask"] = "...";
        }
        SRV_INF("params: %s\n", body_cp.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    }

    json llama_params;

    // Annotations for OAI compatibility
    llama_params["__oaicompat"]            = true;
    llama_params["__oaicompat_image"]      = true;
    llama_params["__oaicompat_image_edit"] = true;

    // Handle "model" field
    llama_params["model"] = json_value(body, "model", params.model_alias);

    // Handle "image" field
    llama_params["image"] = body.at("image");

    // Handle "mask" field
    if (body.contains("mask")) {
        llama_params["mask"] = body.at("mask");
    }

    // Handle "prompt" field
    llama_params["prompt"] = body.at("prompt");

    // Handle "n" field
    {
        auto batch_count = json_value(body, "n", 1);
        if (batch_count > params.max_batch_count) {
            throw std::runtime_error("Illegal param: n must be less than or equal to " + std::to_string(params.max_batch_count));
        }
        llama_params["batch_count"] = batch_count;
    }

    // Handle "sampler" and "cfg_scale" fields
    if (!body.contains("sampler")) {
        std::string quality = json_value(body, "quality", std::string("standard"));
        if (quality != "null" && quality != "hd" && quality != "standard") {
            throw std::runtime_error("Illegal param: quality must be one of 'hd' or 'standard'");
        }
        llama_params["sampler"]      = params.sampler;
        llama_params["sample_steps"] = params.sample_steps;
        llama_params["cfg_scale"]    = params.cfg_scale;
        if (quality == "hd") {
            llama_params["sample_steps"]    = params.sample_steps + 10;
            llama_params["negative_prompt"] = "low quality";
        }
    } else {
        std::string sampler_str         = json_value(body, "sampler", std::string("euler_a"));
        llama_params["sampler"]         = sd_argument_to_sample_method(sampler_str.c_str());
        llama_params["cfg_scale"]       = json_value(body, "cfg_scale", params.cfg_scale);
        llama_params["sample_steps"]    = json_value(body, "sample_steps", params.sample_steps);
        llama_params["negative_prompt"] = json_value(body, "negative_prompt", std::string(""));
        if (body.contains("seed")) {
            llama_params["seed"] = body.at("seed");
        }
    }

    // Handle "size" field
    std::string size = json_value(body, "size", std::string("512x512"));
    {
        auto pos = size.find('x');
        if (pos == std::string::npos) {
            throw std::runtime_error("Illegal param: size must be in the format 'widthxheight'");
        }
        auto width  = std::stoi(size.substr(0, pos));
        auto height = std::stoi(size.substr(pos + 1));
        if (width < 256 || height < 256) {
            throw std::runtime_error("Illegal param: width and height must be at least 256");
        }
        if (width > params.max_width) {
            throw std::runtime_error("Illegal param: width must be at most " + std::to_string(params.max_width));
        }
        if (height > params.max_height) {
            throw std::runtime_error("Illegal param: height must be at most " + std::to_string(params.max_height));
        }
        llama_params["width"]  = width;
        llama_params["height"] = height;
    }

    // Handle "response_format" field
    std::string response_format = json_value(body, "response_format", std::string("b64_json"));
    if (response_format != "b64_json") {
        throw std::runtime_error("Illegal param: response_format must be 'b64_json'");
    }

    // Handle "stream" field
    if (json_value(body, "stream", false)) {
        llama_params["stream"] = true;
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

static json oaicompat_images_response(const json &request, const json &result, const bool streaming = false, const std::vector<json> &usages = {}) {
    json data = json::array();
    for (auto &ret : result) {
        json item = json{
            {"progress", ret.at("progress")},
            {"index", ret.at("index")},
        };
        if (streaming) {
            item["object"] = "image.chunk";
        } else {
            item["object"] = "image";
        }
        if (ret.contains("b64_json")) {
            item["b64_json"]      = ret.at("b64_json");
            item["finish_reason"] = "stop";
        } else {
            item["finish_reason"] = nullptr;
        }
        data.push_back(item);
    }

    json res = json{
        {"created", std::time(nullptr)},
        {"model", json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
        {"object", "list"},
        {"data", data},
    };

    // Add usage field
    bool include_usage = false;
    if (request.contains("stream_options")) {
        include_usage = json_value(request.at("stream_options"), "include_usage", false);
    }
    if (include_usage && !usages.empty()) {
        const size_t result_size = usages.size();
        double ttp               = 0.0;
        double tpg               = 0.0;
        double gps               = 0.0;
        for (const auto &usage : usages) {
            ttp += json_value(usage, "processing_ms", 0.0);
            tpg += json_value(usage, "generation_ms", 0.0);
            gps += json_value(usage, "generation_per_second", 0.0);
        }
        ttp          = ttp / result_size;
        tpg          = tpg / result_size;
        gps          = gps / result_size;
        res["usage"] = json{
            {"time_to_process_ms", ttp},
            {"time_per_generation_ms", tpg},
            {"generation_per_second", gps},
        };
    } else if (include_usage) {
        res["usage"] = nullptr;
    }

    return res;
}

static json jinaaicompat_rerank_request(const struct common_params &params, const json &body) {
    // Print the request for debugging
    {
        json body_cp = body;
        if (common_log_verbosity_thold < 2) {
            if (body_cp.contains("query")) {
                body_cp["query"] = "...";
            }
            if (body_cp.contains("documents")) {
                body_cp["documents"] = "[...]";
            }
        }
        SRV_INF("params: %s\n", body_cp.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    }

    json llama_params;

    // Annotations for OAI compatibility
    llama_params["__oaicompat"]        = true;
    llama_params["__oaicompat_rerank"] = true;

    // Handle "model" field
    llama_params["model"] = json_value(body, "model", params.model_alias);

    // Handle "query" and "documents" fields
    json prompt = json::array();
    prompt.push_back(json_value(body, "query", std::string("")));
    for (const json &doc : body.at("documents")) {
        if (doc.is_string()) {
            prompt.push_back(doc.get<std::string>());
        } else if (doc.is_object() && doc.contains("text")) {
            prompt.push_back(doc.at("text").get<std::string>());
        } else {
            throw std::runtime_error("Illegal param: documents must be an array of strings or objects with a 'text' field");
        }
    }
    llama_params["prompt"] = prompt;

    // Handle "top_n" field
    size_t documents_size = body.at("documents").size();
    size_t top_n          = json_value(body, "top_n", documents_size);
    if (top_n > documents_size) {
        top_n = documents_size;
    } else if (top_n <= 0) {
        throw std::runtime_error("Illegal param: top_n must be greater than 0");
    }
    llama_params["top_n"] = top_n;

    // Handle "return_documents" field
    bool return_documents            = json_value(body, "return_documents", true);
    llama_params["return_documents"] = return_documents;
    if (return_documents) {
        llama_params["__oaicompat_rerank_documents"] = body.at("documents");
    }

    // Handle "normalize" field
    llama_params["normalize"] = json_value(body, "normalize", true);

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
    result[i]   = base;
    jinaicompat_rerank_response_sort(result, low, i - 1);
    jinaicompat_rerank_response_sort(result, i + 1, high);
}

static json jinaicompat_rerank_response(const json &request, json &result) {
    json documents;
    int32_t top_n         = request.at("top_n");
    bool return_documents = request.at("return_documents");
    if (return_documents) {
        documents = request.at("__oaicompat_rerank_documents");
    }

    int num_prompt_tokens = 0;
    int32_t start         = 0;
    auto end              = int32_t(result.size() - 1);
    jinaicompat_rerank_response_sort(result, start, end);

    json data      = json::array();
    double scr_max = json_value(result[start], "score", 1e-6);
    double scr_min = json_value(result[end], "score", 1e-6);
    double scr_dst = scr_max - scr_min;
    double a = 0.01, b = 0.98;
    if (scr_dst < 1e-6) {
        scr_dst = scr_min;
        scr_min = 0.0;
        if (request.at("prompt")[0].get<std::string>() == documents[start].get<std::string>()) {
            a = 0;
            b = 1;
        }
    }
    const bool normalize = request.at("normalize").get<bool>();
    for (int32_t i = 0; i <= end && i < top_n; i++) {
        const json &ret = result[i];

        double scr = json_value(ret, "score", 1e-6);
        if (normalize) {
            scr = a + (scr - scr_min) * b / scr_dst;
        }

        int32_t tke = json_value(ret, "tokens_evaluated", 0);
        num_prompt_tokens += tke;

        int32_t idx = json_value(ret, "index", 0);
        json item   = json{
              {"index", idx},
              {"relevance_score", scr},
        };
        if (return_documents) {
            if (documents[idx].is_string()) {
                item["document"] = json{
                    {"text", documents[idx]},
                };
            } else {
                item["document"] = documents[idx];
            }
        }
        data.push_back(item);
    }

    json res = json{
        {"model", json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
        {"results", data},
        {"usage", json{{"prompt_tokens", num_prompt_tokens}, {"total_tokens", num_prompt_tokens}}},
    };

    return res;
}

static bool is_valid_utf8(const std::string &str) {
    const auto *bytes        = reinterpret_cast<const unsigned char *>(str.data());
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
            if (end - bytes < 4 || (bytes[1] & 0xC0) != 0x80 || (bytes[2] & 0xC0) != 0x80 || (bytes[3] & 0xC0) != 0x80)
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
            code     = 400;
            break;
        case ERROR_TYPE_AUTHENTICATION:
            type_str = "authentication_error";
            code     = 401;
            break;
        case ERROR_TYPE_NOT_FOUND:
            type_str = "not_found_error";
            code     = 404;
            break;
        case ERROR_TYPE_SERVER:
            type_str = "server_error";
            code     = 500;
            break;
        case ERROR_TYPE_PERMISSION:
            type_str = "permission_error";
            code     = 403;
            break;
        case ERROR_TYPE_NOT_SUPPORTED:
            type_str = "not_supported_error";
            code     = 501;
            break;
        case ERROR_TYPE_UNAVAILABLE:
            type_str = "unavailable_error";
            code     = 503;
            break;
    }
    return json{
        {"code", code},
        {"message", message},
        {"type", type_str},
    };
}
