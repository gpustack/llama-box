#pragma once

#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "llama.cpp/common/chat.h"
#include "llama.cpp/common/common.h"
#include "llama.cpp/common/log.h"
#include "llama.cpp/include/llama.h"

#define JSON_ASSERT GGML_ASSERT
#include "llama.cpp/common/json.hpp"

#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH 10485760
#define CPPHTTPLIB_TCP_NODELAY true
#include "llama.cpp/examples/server/httplib.h"

#include "stablediffusion.hpp"

#define DEFAULT_OAICOMPAT_MODEL "gpt-3.5-turbo"

#define SLT_INF(slot, fmt, ...)                                                                                                                 \
    if (common_log_verbosity_thold > 3) {                                                                                                       \
        if ((slot).rid.empty()) {                                                                                                               \
            LOG_INF("slt %25.*s: id %02d | task %d | " fmt, 25, __func__, (slot).id, (slot).id_task, __VA_ARGS__);                              \
        } else {                                                                                                                                \
            LOG_INF("slt %25.*s: rid %s | id %02d | task %d | " fmt, 25, __func__, (slot).rid.c_str(), (slot).id, (slot).id_task, __VA_ARGS__); \
        }                                                                                                                                       \
    }
#define SLT_WRN(slot, fmt, ...)                                                                                                             \
    if ((slot).rid.empty()) {                                                                                                               \
        LOG_WRN("slt %25.*s: id %02d | task %d | " fmt, 25, __func__, (slot).id, (slot).id_task, __VA_ARGS__);                              \
    } else {                                                                                                                                \
        LOG_WRN("slt %25.*s: rid %s | id %02d | task %d | " fmt, 25, __func__, (slot).rid.c_str(), (slot).id, (slot).id_task, __VA_ARGS__); \
    }
#define SLT_ERR(slot, fmt, ...)                                                                                                             \
    if ((slot).rid.empty()) {                                                                                                               \
        LOG_ERR("slt %25.*s: id %02d | task %d | " fmt, 25, __func__, (slot).id, (slot).id_task, __VA_ARGS__);                              \
    } else {                                                                                                                                \
        LOG_ERR("slt %25.*s: rid %s | id %02d | task %d | " fmt, 25, __func__, (slot).rid.c_str(), (slot).id, (slot).id_task, __VA_ARGS__); \
    }
#define SLT_DBG(slot, fmt, ...)                                                                                                                 \
    if (common_log_verbosity_thold > 4) {                                                                                                       \
        if ((slot).rid.empty()) {                                                                                                               \
            LOG_DBG("slt %25.*s: id %02d | task %d | " fmt, 25, __func__, (slot).id, (slot).id_task, __VA_ARGS__);                              \
        } else {                                                                                                                                \
            LOG_DBG("slt %25.*s: rid %s | id %02d | task %d | " fmt, 25, __func__, (slot).rid.c_str(), (slot).id, (slot).id_task, __VA_ARGS__); \
        }                                                                                                                                       \
    }

#define SRV_INF(fmt, ...) LOG_INF("srv %25.*s: " fmt, 25, __func__, __VA_ARGS__)
#define SRV_WRN(fmt, ...) LOG_WRN("srv %25.*s: " fmt, 25, __func__, __VA_ARGS__)
#define SRV_ERR(fmt, ...) LOG_ERR("srv %25.*s: " fmt, 25, __func__, __VA_ARGS__)
#define SRV_DBG(fmt, ...)                                       \
    if (common_log_verbosity_thold > 5) {                       \
        LOG_DBG("srv %25.*s: " fmt, 25, __func__, __VA_ARGS__); \
    }

#define QUE_INF(fmt, ...)                                       \
    if (common_log_verbosity_thold > 4) {                       \
        LOG_INF("que %25.*s: " fmt, 25, __func__, __VA_ARGS__); \
    }
#define QUE_WRN(fmt, ...) LOG_WRN("que %25.*s: " fmt, 25, __func__, __VA_ARGS__)
#define QUE_ERR(fmt, ...) LOG_ERR("que %25.*s: " fmt, 25, __func__, __VA_ARGS__)
#define QUE_DBG(fmt, ...)                                       \
    if (common_log_verbosity_thold > 5) {                       \
        LOG_DBG("que %25.*s: " fmt, 25, __func__, __VA_ARGS__); \
    }

using json = nlohmann::ordered_json;

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
static llama_tokens tokenize_mixed(const llama_vocab *vocab, const json &json_prompt, bool add_special, bool parse_special) {
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
                    p     = common_tokenize(vocab, s, add_special, parse_special);
                    first = false;
                } else {
                    p = common_tokenize(vocab, s, false, parse_special);
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
        prompt_tokens = common_tokenize(vocab, s, add_special, parse_special);
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
static std::vector<llama_tokens> tokenize_input_prompts(const llama_vocab *vocab, const json &json_prompt, bool add_special, bool parse_special) {
    std::vector<llama_tokens> result;
    if (json_prompt.is_string() || json_is_array_of_mixed_numbers_strings(json_prompt)) {
        // string or mixed
        result.push_back(tokenize_mixed(vocab, json_prompt, add_special, parse_special));
    } else if (json_is_array_of_numbers(json_prompt)) {
        // array of tokens
        result.push_back(json_prompt.get<llama_tokens>());
    } else if (json_prompt.is_array()) {
        // array of prompts
        result.reserve(json_prompt.size());
        for (const auto &p : json_prompt) {
            if (p.is_string() || json_is_array_of_mixed_numbers_strings(p)) {
                // string or mixed
                result.push_back(tokenize_mixed(vocab, p, add_special, parse_special));
            } else if (json_is_array_of_numbers(p)) {
                // array of tokens
                result.push_back(p.get<llama_tokens>());
            } else if (json_is_array_of_objects(p)) {
                // array of objects
                result.push_back(tokenize_mixed(vocab, p, add_special, parse_special));
            } else {
                throw std::runtime_error("Illegal param: \"prompt\" must be a string, a list of tokens, or a list of mixed strings & tokens");
            }
        }
    } else {
        throw std::runtime_error("Illegal param: \"prompt\" must be a string, a list of tokens, a list of mixed strings & tokens, or a list of prompts");
    }
    return result;
}

// return the last index of character that can form a valid string
// if the last character is potentially cut in half, return the index before the cut
// if validate_utf8(text) == text.size(), then the whole text is valid utf8
static size_t validate_utf8(const std::string &text) {
    size_t len = text.size();
    if (len == 0)
        return 0;

    // Check the last few bytes to see if a multi-byte character is cut off
    for (size_t i = 1; i <= 4 && i <= len; ++i) {
        unsigned char c = text[len - i];
        // Check for start of a multi-byte sequence from the end
        if ((c & 0xE0) == 0xC0) {
            // 2-byte character start: 110xxxxx
            // Needs at least 2 bytes
            if (i < 2)
                return len - i;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte character start: 1110xxxx
            // Needs at least 3 bytes
            if (i < 3)
                return len - i;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte character start: 11110xxx
            // Needs at least 4 bytes
            if (i < 4)
                return len - i;
        }
    }

    // If no cut-off multi-byte character is found, return full length
    return len;
}

//
// template utils
//

// format rerank task: [BOS]query[EOS][SEP]doc[EOS]
static llama_tokens format_rerank(const llama_vocab *vocab, const llama_tokens &query, const llama_tokens &doc) {
    llama_tokens result;
    result.reserve(doc.size() + query.size() + 4);
    result.push_back(llama_vocab_bos(vocab));
    result.insert(result.end(), query.begin(), query.end());
    result.push_back(llama_vocab_sep(vocab));
    result.insert(result.end(), doc.begin(), doc.end());
    result.push_back(llama_vocab_sep(vocab));
    result.push_back(llama_vocab_eos(vocab));
    return result;
}

// format infill task
static llama_tokens format_infill(const llama_vocab *vocab, const json &input_prefix, const json &input_suffix, const json &input_extra,
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

    auto tokens_prefix = tokenize_mixed(vocab, input_prefix, false, false);
    auto tokens_suffix = tokenize_mixed(vocab, input_suffix, false, false);

    if (llama_vocab_fim_rep(vocab) != LLAMA_TOKEN_NULL) {
        // TODO: make project name an input
        static const auto k_fim_repo = common_tokenize(vocab, "myproject\n", false, false);

        extra_tokens.push_back(llama_vocab_fim_rep(vocab));
        extra_tokens.insert(extra_tokens.end(), k_fim_repo.begin(), k_fim_repo.end());
    }
    for (const auto &chunk : input_extra) {
        // { "text": string, "filename": string }
        const std::string text     = json_value(chunk, "text", std::string());
        const std::string filename = json_value(chunk, "filename", std::string("tmp"));

        if (llama_vocab_fim_sep(vocab) != LLAMA_TOKEN_NULL) {
            const auto k_fim_file = common_tokenize(vocab, filename + "\n", false, false);

            extra_tokens.insert(extra_tokens.end(), llama_vocab_fim_sep(vocab));
            extra_tokens.insert(extra_tokens.end(), k_fim_file.begin(), k_fim_file.end());
        } else {
            // chunk separator in binary form to avoid confusing the AI
            static const char k_chunk_prefix_str[]  = {0x0a, 0x0a, 0x2d, 0x2d, 0x2d, 0x20, 0x73, 0x6e, 0x69, 0x70,
                                                       0x70, 0x65, 0x74, 0x20, 0x2d, 0x2d, 0x2d, 0x0a, 0x0a, 0x00};
            static const auto k_chunk_prefix_tokens = common_tokenize(vocab, k_chunk_prefix_str, false, false);

            extra_tokens.insert(extra_tokens.end(), k_chunk_prefix_tokens.begin(), k_chunk_prefix_tokens.end());
        }

        const auto chunk_tokens = common_tokenize(vocab, text, false, false);
        extra_tokens.insert(extra_tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
    }

    if (llama_vocab_fim_sep(vocab) != LLAMA_TOKEN_NULL) {
        // TODO: current filename
        static const auto k_fim_file = common_tokenize(vocab, "filename\n", false, false);

        extra_tokens.insert(extra_tokens.end(), llama_vocab_fim_sep(vocab));
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

    tokens_prefix.insert(tokens_prefix.begin(), llama_vocab_fim_pre(vocab));
    tokens_prefix.insert(tokens_prefix.end(), tokens_prompt.begin(), tokens_prompt.end());
    tokens_suffix.insert(tokens_suffix.begin(), llama_vocab_fim_pre(vocab));

    auto embd_inp = spm_infill ? tokens_suffix : tokens_prefix;
    auto embd_end = spm_infill ? tokens_prefix : tokens_suffix;

    if (llama_vocab_get_add_bos(vocab)) {
        embd_inp.insert(embd_inp.begin(), llama_vocab_bos(vocab));
    }

    SRV_DBG("extra: n_ctx = %d, n_extra_take = %d, n_extra = %d\n", n_ctx, n_extra_take, (int)extra_tokens.size());

    // put the extra context before the FIM prefix
    embd_inp.insert(embd_inp.begin(), extra_tokens.end() - n_extra_take, extra_tokens.end());

    embd_inp.insert(embd_inp.end(), embd_end.begin(), embd_end.end());
    embd_inp.push_back(llama_vocab_fim_mid(vocab));

    return embd_inp;
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
    return "cmpl-" + random_string();
}

static std::string gen_callid() {
    return "call-" + random_string();
}

//
// other common utils
//

static bool starts_with(const std::string &str, const std::string &prefix) {
    return str.size() >= prefix.size() && 0 == str.compare(0, prefix.size(), prefix);
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
    std::string out = token == LLAMA_TOKEN_NULL ? "" : common_token_to_piece(ctx, token);

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
    struct token_prob {
        llama_token tok;
        float prob;
    };

    std::vector<llama_token> toks;
    std::vector<float> probs;
    std::vector<std::vector<token_prob>> top_probs;
    std::string text_to_send;
};

// convert a vector of completion_token_output to json
static json probs_vector_to_json(const llama_context *ctx, const std::vector<completion_token_output> &probs, const bool oaicompat_completion = false, const bool oaicompat_completion_chat = false) {
    if (oaicompat_completion) {
        if (oaicompat_completion_chat) {
            json content = json::array();

            for (const auto &p : probs) {
                for (size_t i = 0; i < p.toks.size(); i++) {
                    const llama_token id    = p.toks[i];
                    const std::string token = tokens_to_output_formatted_string(ctx, id);
                    float token_logprob     = p.probs[i] == 0.0f ? std::numeric_limits<float>::lowest() : std::log(p.probs[i]);
                    std::vector<unsigned char> token_bytes(token.begin(), token.end());
                    json token_top_logprobs = json::array();
                    for (const auto &tp : p.top_probs[i]) {
                        const llama_token tp_id    = tp.tok;
                        const std::string tp_token = tokens_to_output_formatted_string(ctx, tp_id);
                        float tp_token_logprob     = tp.prob == 0.0f ? std::numeric_limits<float>::lowest() : std::log(tp.prob);
                        std::vector<unsigned char> tp_token_bytes(tp_token.begin(), tp_token.end());
                        token_top_logprobs.push_back(json{
                            {"token", tp_token},
                            {"logprob", tp_token_logprob},
                            {"bytes", tp_token_bytes},
                        });
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

            for (const auto &p : probs) {
                for (size_t i = 0; i < p.toks.size(); i++) {
                    const llama_token id    = p.toks[i];
                    const std::string token = tokens_to_output_formatted_string(ctx, id);
                    float token_logprob     = p.probs[i] == 0.0f ? std::numeric_limits<float>::lowest() : std::log(p.probs[i]);
                    json token_top_logprobs;
                    for (const auto &tp : p.top_probs[i]) {
                        const llama_token tp_id      = tp.tok;
                        const std::string tp_token   = tokens_to_output_formatted_string(ctx, tp_id);
                        float tp_token_logprob       = tp.prob == 0.0f ? std::numeric_limits<float>::lowest() : std::log(tp.prob);
                        token_top_logprobs[tp_token] = tp_token_logprob;
                    }

                    tokens.push_back(token);
                    token_logprobs.push_back(token_logprob);
                    top_logprobs.push_back(token_top_logprobs);
                }
            }

            return json{
                {"tokens", tokens},
                {"token_logprobs", token_logprobs},
                {"top_logprobs", top_logprobs},
            };
        }
    }

    json out = json::array();

    for (const auto &p : probs) {
        for (size_t i = 0; i < p.toks.size(); i++) {
            const llama_token id    = p.toks[i];
            const std::string token = tokens_to_output_formatted_string(ctx, id);
            float token_prob        = p.probs[i];
            std::vector<unsigned char> token_bytes(token.begin(), token.end());
            json token_top_probs = json::array();
            for (const auto &tp : p.top_probs[i]) {
                const llama_token tp_id    = tp.tok;
                const std::string tp_token = tokens_to_output_formatted_string(ctx, tp_id);
                float tp_token_prob        = tp.prob;
                std::vector<unsigned char> tp_token_bytes(tp_token.begin(), tp_token.end());
                token_top_probs.push_back(json{
                    {"id", tp_id},
                    {"token", tp_token},
                    {"prob", tp_token_prob},
                    {"bytes", tp_token_bytes},
                });
            }

            out.push_back(json{
                {"id", id},
                {"token", token},
                {"prob", token_prob},
                {"bytes", token_bytes},
                {"top_probs", token_top_probs},
            });
        }
    }

    return out;
}

static bool server_sent_event(httplib::DataSink &sink, const char *event, const json &data) {
    const std::string str = std::string(event) + ": " + data.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
    return sink.write(str.c_str(), str.size());
}

// thin wrapper around common_grammar_trigger with (de)serialization functions
struct server_grammar_trigger {
    common_grammar_trigger value;

    server_grammar_trigger() = default;

    server_grammar_trigger(const common_grammar_trigger &value)
        : value(value) {
    }

    server_grammar_trigger(const json &in) {
        value.type  = (common_grammar_trigger_type)in.at("type").get<int>();
        value.value = in.at("value").get<std::string>();
        if (value.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
            value.token = (llama_token)in.at("token").get<int>();
        }
    }

    json to_json() const {
        json out{
            {"type", (int)value.type},
            {"value", value.value},
        };
        if (value.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
            out["token"] = (int)value.token;
        }
        return out;
    }
};

//
// OAI utils
//

static json oaicompat_completions_request(const struct common_params &params, const std::string &rid, const json &body, const llama_model *model, const bool chat, const bool support_tool_calls, const bool use_jinja, const struct common_chat_templates *chat_tmpls) {
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
        SRV_INF("rid %s | %s\n", rid.c_str(), body_cp.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    }

    json llama_params;

    // Annotations for OAI compatibility
    llama_params["__oaicompat"]                        = true;
    llama_params["__oaicompat_completion"]             = true;
    llama_params["__oaicompat_completion_chat"]        = chat;
    llama_params["__oaicompat_completion_chat_tool"]   = false;
    llama_params["__oaicompat_completion_chat_vision"] = false;

    // Handle default field
    llama_params["model"]             = json_value(body, "model", params.model_alias);
    llama_params["frequency_penalty"] = json_value(body, "frequency_penalty", 0.0f);
    llama_params["presence_penalty"]  = json_value(body, "presence_penalty", 0.0f);
    llama_params["temperature"]       = json_value(body, "temperature", 1.0f);
    llama_params["top_p"]             = json_value(body, "top_p", 1.0f);

    // Handle "tools" and "tool_choice" field
    std::vector<common_chat_tool> chat_tools;
    common_chat_tool_choice chat_tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
    if (chat && support_tool_calls) {
        // "tools" and "functions", migrate "functions" to "tools"
        if (body.contains("tools") && !body.contains("functions")) {
            const json &tools = body.at("tools");
            if (!tools.is_array()) {
                throw std::runtime_error("Illegal param: \"tools\" must be an array");
            }
            for (const json &tool : tools) {
                if (!tool.contains("function")) {
                    continue;
                }
                const json &func = tool.at("function");
                if (!func.contains("name") || !func.at("name").is_string()) {
                    continue;
                }
                if (!func.contains("parameters") || !func.at("parameters").is_object()) {
                    continue;
                }
                std::string name        = func.at("name");
                std::string description = json_value(func, "description", std::string());
                std::string parameters  = func.at("parameters").dump(-1, ' ', false, json::error_handler_t::replace);
                chat_tools.push_back({name, description, parameters});
            }
        } else if (body.contains("functions")) {
            const json &functions = body.at("functions");
            if (!functions.is_array()) {
                throw std::runtime_error("Illegal param: \"functions\" must be an array");
            }
            for (const json &func : functions) {
                if (!func.contains("name") || !func.at("name").is_string()) {
                    continue;
                }
                if (!func.contains("parameters") || !func.at("parameters").is_object()) {
                    continue;
                }
                std::string name        = json_value(func, "name", std::string());
                std::string description = json_value(func, "description", std::string());
                std::string parameters  = func.at("parameters").dump(-1, ' ', false, json::error_handler_t::replace);
                chat_tools.push_back({name, description, parameters});
            }
        }
        if (!chat_tools.empty()) {
            chat_tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
            // "tool_choice" and "function_call", migrate "function_call" to "tool_choice"
            if (body.contains("tool_choice") && !body.contains("function_call")) {
                const json &tc = body.at("tool_choice");
                if (tc.is_object() && tc.contains("function")) {
                    const json &fc       = tc.at("function");
                    const std::string fn = json_value(fc, "name", std::string());
                    std::vector<common_chat_tool> available_chat_tools;
                    for (const common_chat_tool &t : chat_tools) {
                        if (t.name == fn) {
                            available_chat_tools.push_back(t);
                            chat_tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
                            break;
                        }
                    }
                    chat_tools = available_chat_tools;
                } else if (tc.is_string()) {
                    chat_tool_choice = common_chat_tool_choice_parse_oaicompat(tc.get<std::string>());
                } else {
                    throw std::runtime_error("Illegal param: \"tool_choice\" must be a string or an object");
                }
            } else if (body.contains("function_call")) {
                const json &fc = body.at("function_call");
                if (fc.is_object()) {
                    const std::string fn = json_value(fc, "name", std::string());
                    std::vector<common_chat_tool> available_chat_tools;
                    for (const common_chat_tool &t : chat_tools) {
                        if (t.name == fn) {
                            available_chat_tools.push_back(t);
                            chat_tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
                            break;
                        }
                    }
                    chat_tools = available_chat_tools;
                } else if (fc.is_string()) {
                    chat_tool_choice = common_chat_tool_choice_parse_oaicompat(fc.get<std::string>());
                } else {
                    throw std::runtime_error("Illegal param: \"function_call\" must be a string or an object");
                }
            }
        }
    }
    llama_params["__oaicompat_completion_chat_tool"] = !chat_tools.empty();

    // Handle "stop" field
    if (body.contains("stop") && body.at("stop").is_string()) {
        llama_params["stop"] = json::array({body.at("stop").get<std::string>()});
    } else {
        llama_params["stop"] = json_value(body, "stop", json::array());
    }

    // Handle "response_format" field
    json json_schema    = json_value(body, "json_schema", json());
    std::string grammar = json_value(body, "grammar", std::string());
    if (!json_schema.is_null() && !grammar.empty()) {
        throw std::runtime_error("Illegal param: use both \"json_schema\" and \"grammar\"");
    }
    if (body.contains("response_format")) {
        json response_format      = json_value(body, "response_format", json::object());
        std::string response_type = json_value(response_format, "type", std::string());
        if (response_type == "json_object") {
            json_schema = json_value(response_format, "schema", json::object());
        } else if (response_type == "json_schema") {
            json schema_wrapper = json_value(response_format, "json_schema", json::object());
            json_schema         = json_value(schema_wrapper, "schema", json::object());
        } else if (!response_type.empty() && response_type != "text") {
            throw std::runtime_error("Illegal param: \"response_format\" must be one of 'text' or 'json_object', but got: " + response_type);
        }
    }

    bool stream = json_value(body, "stream", false);

    // Apply chat template to the list of messages
    if (chat) {
        std::vector<common_chat_msg> chat_messages;
        json images = json::array();

        json messages = body.at("messages"); // copy message
        for (json &msg : messages) {
            std::string role = json_value(msg, "role", std::string(""));
            std::string content;
            if (msg.contains("content") && !msg.at("content").is_null()) {
                if (msg.at("content").is_string()) {
                    content = msg.at("content").get<std::string>();
                } else if (msg.at("content").is_array()) {
                    int32_t n_img = 0;
                    for (const json &part : msg.at("content")) {
                        if (part.contains("type") && part.at("type") == "image_url") {
                            // process image
                            llama_params["__oaicompat_completion_chat_vision"] = true;
                            std::string img                                    = json_value(part.at("image_url"), "url", std::string());
                            if (img.find("data:image/") != std::string::npos) {
                                const std::string split = "base64,";
                                const size_t idx        = img.find(split);
                                if (idx == std::string::npos) {
                                    throw std::runtime_error("Illegal param: \"image_url\" must be a valid base64-encoded image");
                                }
                                img = img.substr(idx + split.length());
                                if (img.empty()) {
                                    throw std::runtime_error("Illegal param: \"image_url\" is an empty image base64-encoded data");
                                }
                                try {
                                    const std::vector<uint8_t> img_buff = base64_decode(img);
                                    images.push_back(img_buff);
                                } catch (const std::exception &e) {
                                    throw std::runtime_error("Illegal param: \"image_url\" must be a valid base64-encoded image");
                                }
                            } else {
                                std::string host, path;
                                if (size_t pos = img.find("://"); pos == std::string::npos) {
                                    throw std::runtime_error("Illegal param: \"image_url\" must be a data URI or a valid URL");
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
                                httplib::Client cli(host);
                                cli.set_connection_timeout(15, 0);                      // 15 seconds
                                cli.set_read_timeout(300, 0);                           // 5 minutes
                                cli.set_keep_alive(false);                              // close connection after request
                                cli.set_follow_location(true);                          // follow redirects
                                cli.set_default_headers({{"User-Agent", "llama-box"}}); // set user-agent
                                cli.set_url_encode(true);                               // encode URL
                                cli.set_tcp_nodelay(true);                              // disable Nagle's algorithm
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
                                cli.enable_server_certificate_verification(false); // disable SSL verification
#endif
                                httplib::Result res = cli.Get(path);
                                if (!res || res->status != httplib::StatusCode::OK_200) {
                                    throw std::runtime_error("Illegal param: invalid \"image_url\", failed to fetch image from URL: " + img + ", status: " + std::to_string(res ? res->status : -1) + ", reason: " + (res ? res->reason : "unknown"));
                                }
                                const std::vector<uint8_t> img_buff(res->body.begin(), res->body.end());
                                images.push_back(img_buff);
                            }
                            n_img++;
                            continue;
                        }
                        if (part.contains("text")) {
                            if (!content.empty()) {
                                content += "\n";
                            }
                            for (int i = 0; i < n_img; i++) {
                                content += "<image>\n";
                            }
                            content += part.at("text").get<std::string>();
                            n_img = 0;
                        }
                    }
                    for (int i = 0; i < n_img; i++) {
                        content += "\n<image>";
                    }
                } else {
                    throw std::runtime_error("Illegal param: invalid \"content\"");
                }
                msg["content"] = content; // updated
                chat_messages.push_back({role, content, {}, {}, "", "", ""});
            } else if (msg.contains("tool_calls") && !msg.at("tool_calls").is_null()) {
                if (msg.at("tool_calls").is_array()) {
                    std::vector<common_chat_tool_call> chat_tcs;
                    for (const json &part : msg.at("tool_calls")) {
                        common_chat_tool_call chat_tc;
                        if (!part.contains("type") || part.at("type") != "function") {
                            continue;
                        }
                        if (!part.contains("function") || !part.at("function").is_object()) {
                            continue;
                        }
                        const json &func = part.at("function");
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
                    chat_messages.push_back({role, "", {}, chat_tcs, "", "", ""});
                } else {
                    throw std::runtime_error("Illegal param: invalid \"tool_calls\"");
                }
            } else {
                throw std::runtime_error("Illegal param: missing 'content' or 'tool_calls' in \"messages\" item");
            }
        }
        llama_params["multi_modal_data"] = json{{"images", images}};

        std::string prompt;
        {
            common_chat_templates_inputs inputs;
            inputs.messages              = chat_messages;
            inputs.tools                 = chat_tools;
            inputs.tool_choice           = chat_tool_choice;
            inputs.json_schema           = json_schema.is_null() ? "" : json_schema.dump();
            inputs.grammar               = grammar;
            inputs.add_generation_prompt = json_value(body, "add_generation_prompt", true);
            inputs.use_jinja             = use_jinja;
            inputs.parallel_tool_calls   = support_tool_calls && json_value(body, "parallel_tool_calls", true);
            if (!chat_tools.empty() && chat_tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && !grammar.empty()) {
                throw std::runtime_error("Illegal param: cannot use custom \"grammar\" constraints with tools");
            }
            common_chat_params chat_params                     = common_chat_templates_apply2(model, chat_tmpls, inputs);
            llama_params["__oaicompat_completion_chat_format"] = chat_params.format;
            llama_params["parallel_tool_calls"]                = inputs.parallel_tool_calls;
            llama_params["grammar"]                            = chat_params.grammar;
            llama_params["grammar_lazy"]                       = chat_params.grammar_lazy;
            auto grammar_triggers                              = json::array();
            for (const common_grammar_trigger &trigger : chat_params.grammar_triggers) {
                server_grammar_trigger ct(trigger);
                grammar_triggers.push_back(ct.to_json());
            }
            if (!grammar_triggers.empty()) {
                llama_params["grammar_triggers"] = grammar_triggers;
            }
            if (!chat_params.preserved_tokens.empty()) {
                llama_params["preserved_tokens"] = chat_params.preserved_tokens;
            }
            for (const auto &stop : chat_params.additional_stops) {
                llama_params["stop"].push_back(stop);
            }
            prompt = chat_params.prompt;
        }
        llama_params["prompt"] = prompt;
        if (common_log_verbosity_thold > 2) {
            SRV_INF("rid %s | formatted prompt\n%s\n", rid.c_str(), prompt.c_str());
        }
    } else if (body.contains("prompt")) {
        llama_params["prompt"] = body.at("prompt");
    } else {
        throw std::runtime_error("Illegal param: missing \"prompt\"");
    }

    // Handle "max_tokens" field
    if (body.contains("max_completion_tokens")) {
        llama_params["n_predict"] = json_value(body, "max_completion_tokens", -1);
    } else {
        llama_params["n_predict"] = json_value(body, "max_tokens", -1);
    }

    // Handle "n" field
    int n_choices = json_value(body, "n", 1);
    if (n_choices != 1) {
        throw std::runtime_error("Illegal param: \"n\" must be 1");
    }

    // Handle "logprobs" field
    if (chat) {
        if (!body.contains("logprobs") && body.contains("top_logprobs")) {
            throw std::runtime_error("Illegal param: \"top_logprobs\" requires \"logprobs\" to be set");
        }
        if (json_value(body, "logprobs", false)) {
            llama_params["n_probs"] = std::min(json_value(body, "top_logprobs", 1), 20);
        }
    } else {
        if (body.contains("logprobs")) {
            llama_params["n_probs"] = std::min(json_value(body, "logprobs", 1), 5);
        }
    }

    // Copy remaining properties to llama_params
    // This allows user to use llama.cpp-specific params like "mirostat", ... via OAI
    // endpoint. See "launch_slot_with_task()" for a complete list of params supported by llama.cpp
    for (const auto &item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by
        // "max_tokens"
        const std::string &key = item.key();
        if (key == "messages" || key == "tools" || key == "tool_choice" || (llama_params.contains(key) && key != "n_predict")) {
            continue;
        }
        llama_params[item.key()] = item.value();
    }

    // Handle "stream_options" field
    if (stream) {
        if (!body.contains("stream_options")) {
            llama_params["stream_options"] = json{{"include_usage", true}};
        } else if (body.at("stream_options").is_object()) {
            if (!body.at("stream_options").contains("include_usage")) {
                llama_params["stream_options"]["include_usage"] = true;
            }
        } else {
            throw std::runtime_error("Illegal param: invalid \"stream_options\"");
        }
    }

    return llama_params;
}

static json oaicompat_completions_response(const std::string &rid, const json &request, const json &result, const std::string &completion_id, bool streaming = false, bool first = false) {
    json res = json{
        {"id", completion_id},
        {"created", std::time(nullptr)},
        {"model", json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
    };

    bool chat                              = json_value(request, "__oaicompat_completion_chat", false);
    int prompt_tokens                      = 0;
    int prompt_cached_tokens               = 0;
    int completion_tokens                  = 0;
    int completion_drafted_tokens          = 0;
    int completion_drafted_accepted_tokens = 0;
    int completion_reasoning_tokens        = 0;
    double ttft                            = 0.0;
    double tpot                            = 0.0;
    double pps                             = 0.0;
    double tps                             = 0.0;
    double dta                             = 0.0;

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
        std::string finish_reason = json_value(ret, "stop_type", std::string(""));
        std::string content       = json_value(ret, "content", std::string(""));
        json tool_calls           = json_value(ret, "tool_calls", json::array());
        int index                 = json_value(ret, "index", 0);
        finish                    = !finish_reason.empty();

        json choice = json{{"index", index}};
        if (!finish) {
            choice["finish_reason"] = nullptr;
        } else {
            choice["finish_reason"] = finish_reason;
        }
        if (chat) {
            json delta_message = json{{"content", content}};
            if (!tool_calls.empty()) {
                if (content.empty()) {
                    delta_message["content"] = nullptr;
                }
                delta_message["tool_calls"] = tool_calls;
            }
            if (streaming) {
                res["object"]   = "chat.completion.chunk";
                choice["delta"] = delta_message;
            } else {
                res["object"]         = "chat.completion";
                delta_message["role"] = "assistant";
                choice["message"]     = delta_message;
            }
        } else {
            res["object"]  = "text_completion";
            choice["text"] = content;
        }

        bool logprobs = ret.contains("completion_probabilities");
        if (!logprobs) {
            choice["logprobs"] = nullptr;
        } else {
            choice["logprobs"] = ret.at("completion_probabilities");
        }

        choices.push_back(choice);

        prompt_tokens += json_value(ret, "tokens_evaluated", 0);
        prompt_cached_tokens += json_value(ret, "tokens_evaluated_cached", 0);
        completion_tokens += json_value(ret, "tokens_predicted", 0);
        completion_drafted_tokens += json_value(ret, "tokens_drafted", 0);
        completion_drafted_accepted_tokens += json_value(ret, "tokens_drafted_accepted", 0);
        completion_reasoning_tokens += json_value(ret, "tokens_reasoning", 0);
        {
            json ts = json_value(ret, "timings", json::object());
            ttft += json_value(ts, "prompt_ms", 0.0);
            tpot += json_value(ts, "predicted_per_token_ms", 0.0);
            pps += json_value(ts, "prompt_per_second", 0.0);
            tps += json_value(ts, "predicted_per_second", 0.0);
        }
    }
    res["choices"] = choices;

    // Add usage field
    bool include_usage = false;
    if (request.contains("stream_options")) {
        include_usage = json_value(request.at("stream_options"), "include_usage", false);
    }
    if (!streaming || include_usage) {
        res["usage"] = nullptr;
        if (finish) {
            const auto rs = double(result.size());
            ttft          = ttft / rs;
            tpot          = tpot / rs;
            pps           = pps / rs;
            tps           = tps / rs;

            json usage = json{
                {"prompt_tokens", prompt_tokens},
                {"completion_tokens", completion_tokens},
                {"total_tokens", completion_tokens + prompt_tokens},
                {
                    "prompt_tokens_details",
                    {
                        {"cached_tokens", prompt_cached_tokens},
                    },
                },
                {
                    "completion_tokens_details",
                    {
                        {"reasoning_tokens", completion_reasoning_tokens},
                        {"accepted_prediction_tokens", completion_drafted_accepted_tokens},
                        {"rejected_prediction_tokens", completion_drafted_tokens - completion_drafted_accepted_tokens},
                    },
                },
            };
            // additional details for usage
            usage["time_to_first_token_ms"]   = ttft;
            usage["time_per_output_token_ms"] = tpot;
            usage["prompt_tokens_per_second"] = pps;
            usage["tokens_per_second"]        = tps;
            if (completion_drafted_tokens > 0) {
                dta                              = float(completion_drafted_accepted_tokens) / float(completion_drafted_tokens);
                usage["draft_tokens"]            = completion_drafted_tokens;
                usage["draft_tokens_acceptance"] = dta;
            }

            res["usage"] = usage;
            SRV_INF("rid %s | prompt_tokens: %d, prompt_cached_tokens: %d, completion_tokens: %d, completion_reasoning_tokens: %d, completion_draft_tokens: %d, ttft: %.2fms, tpot: %.2fms, pps: %.2f, tps: %.2f, dta: %.2f%%\n", rid.c_str(), prompt_tokens, prompt_cached_tokens, completion_tokens, completion_reasoning_tokens, completion_drafted_tokens, ttft, tpot, pps, tps, dta * 100);
        }
    }

    return res;
}

static json oaicompat_embeddings_request(const struct common_params &params, const std::string &rid, const json &body) {
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
        SRV_INF("rid %s | %s\n", rid.c_str(), body_cp.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
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
    if (body.contains("encoding_format")) {
        std::string encoding_format = body.at("encoding_format").get<std::string>();
        if (encoding_format != "float" && encoding_format != "base64") {
            throw std::runtime_error("Illegal param: \"encoding_format\" must be one of 'float' or 'base64'");
        }
        llama_params["encoding_format"] = encoding_format;
    } else {
        llama_params["encoding_format"] = "float";
    }

    return llama_params;
}

static json oaicompat_embeddings_response(const std::string &rid, const json &request, const json &result) {
    const bool use_base64 = json_value(request, "encoding_format", std::string("float")) == "base64";

    int total_tokens      = 0;
    int min_prompt_tokens = 0;
    int max_prompt_tokens = 0;
    json data             = json::array();
    for (const auto &ret : result) {
        int tke = ret.contains("tokens_evaluated") ? ret.at("tokens_evaluated").get<int>() : 0;
        total_tokens += tke;
        min_prompt_tokens = min_prompt_tokens == 0 ? tke : std::min(min_prompt_tokens, tke);
        max_prompt_tokens = std::max(max_prompt_tokens, tke);

        json item = json{
            {"index", ret.at("index")},
            {"object", "embedding"},
        };
        if (!use_base64) {
            item["embedding"] = ret.at("embedding");
        } else {
            const std::vector<float> embedding = ret.at("embedding").get<std::vector<float>>();
            item["embedding"]                  = base64_encode(reinterpret_cast<const unsigned char *>(embedding.data()), embedding.size());
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
    {
        json usage = json{
            {"prompt_tokens", total_tokens},
            {"total_tokens", total_tokens},
            {"min_prompt_tokens", min_prompt_tokens},
            {"max_prompt_tokens", max_prompt_tokens},
        };

        res["usage"] = usage;
        SRV_INF("rid %s | total_tokens: %d, min_prompt_tokens: %d, max_prompt_tokens: %d\n", rid.c_str(), total_tokens, min_prompt_tokens, max_prompt_tokens);
    }

    return res;
}

static json oaicompat_images_generations_request(const struct stablediffusion_params &params, const std::string rid, const json &body) {
    // Print the request for debugging
    {
        json body_cp = body;
        if (common_log_verbosity_thold < 2) {
            body_cp["prompt"] = "...";
        }
        SRV_INF("rid %s | %s\n", rid.c_str(), body_cp.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    }

    // Handle "response_format" field
    std::string response_format = json_value(body, "response_format", std::string("b64_json"));
    if (response_format != "b64_json") {
        throw std::runtime_error("Illegal param: \"response_format\" must be 'b64_json'");
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
            throw std::runtime_error("Illegal param: \"n\" must be less than or equal to " + std::to_string(params.max_batch_count));
        }
        llama_params["batch_count"] = batch_count;
    }

    // Handle "sampler" and "cfg_scale" fields
    if (!body.contains("sampler") && !body.contains("sample_method")) {
        std::string quality = json_value(body, "quality", std::string("standard"));
        if (quality != "null" && quality != "hd" && quality != "standard") {
            throw std::runtime_error("Illegal param: \"quality\" must be one of 'hd' or 'standard'");
        }
        llama_params["sample_method"]   = params.sampling.sample_method;
        llama_params["sampling_steps"]  = params.sampling.sampling_steps;
        llama_params["schedule_method"] = params.sampling.schedule_method;
        llama_params["cfg_scale"]       = params.sampling.cfg_scale;
        if (quality == "hd") {
            llama_params["sampling_steps"]  = params.sampling.sampling_steps + 2;
            llama_params["negative_prompt"] = "low quality";
        }
        if (body.contains("style")) {
            std::string style = json_value(body, "style", std::string("vivid"));
            if (style != "vivid" && style != "natural") {
                throw std::runtime_error("Illegal param: \"style\" must be one of 'vivid' or 'natural'");
            }
            if (style == "vivid") {
                if (llama_params.contains("negative_prompt")) {
                    llama_params["negative_prompt"] = "low quality, not vivid";
                } else {
                    llama_params["negative_prompt"] = "not vivid";
                }
            } else {
                if (llama_params.contains("negative_prompt")) {
                    llama_params["negative_prompt"] = "low quality, unnatural";
                } else {
                    llama_params["negative_prompt"] = "unnatural";
                }
            }
        }
    } else {
        std::string sample_method_str = "euler_a";
        if (body.contains("sample_method")) {
            sample_method_str = body.at("sample_method").get<std::string>();
        } else if (body.contains("sampler")) {
            sample_method_str = body.at("sampler").get<std::string>();
        }
        llama_params["sample_method"] = sd_argument_to_sample_method(sample_method_str.c_str());
        int sampling_steps            = 10;
        if (body.contains("sampling_steps")) {
            sampling_steps = json_value(body, "sampling_steps", 10);
        } else if (body.contains("sample_steps")) {
            sampling_steps = json_value(body, "sample_steps", 10);
        }
        llama_params["sampling_steps"]  = sampling_steps;
        std::string schedule_method_str = "default";
        if (body.contains("schedule_method")) {
            schedule_method_str = body.at("schedule_method").get<std::string>();
        } else if (body.contains("scheduler")) {
            schedule_method_str = body.at("scheduler").get<std::string>();
        } else if (body.contains("schedule")) {
            schedule_method_str = body.at("schedule").get<std::string>();
        }
        llama_params["schedule_method"] = sd_argument_to_schedule(schedule_method_str.c_str());

        // Copy remaining properties to llama_params
        // This allows user to use stable-diffusion.cpp-specific params like "slg_scale", ... via OAI
        // endpoint. See "launch_slot_with_task()" for a complete list of params supported by stable-diffusion.cpp
        for (const auto &item : body.items()) {
            const std::string &key = item.key();
            if (key == "n" ||
                key == "size" ||
                key == "sampler" ||
                key == "sample_steps" ||
                key == "schedule" || key == "scheduler" ||
                llama_params.contains(key)) {
                continue;
            }
            llama_params[item.key()] = item.value();
        }
    }

    // Handle "size" field
    std::string size = json_value(body, "size", std::string("512x512"));
    {
        auto pos = size.find('x');
        if (pos == std::string::npos) {
            throw std::runtime_error("Illegal param: \"size\" must be in the format '{width}x{height}'");
        }
        auto width  = std::stoi(size.substr(0, pos));
        auto height = std::stoi(size.substr(pos + 1));
        if (width < 256 || height < 256) {
            throw std::runtime_error("Illegal param: width and height of \"size\" must be at least 256");
        }
        if (width > params.sampling.width) {
            throw std::runtime_error("Illegal param: width of \"size\" must be at most " + std::to_string(params.sampling.width));
        }
        if (height > params.sampling.height) {
            throw std::runtime_error("Illegal param: height of \"size\" must be at most " + std::to_string(params.sampling.height));
        }
        if (width % 64 != 0 || height % 64 != 0) {
            throw std::runtime_error("Illegal param: width and height of \"size\" must be multiples of 64");
        }
        llama_params["width"]  = width;
        llama_params["height"] = height;
    }

    // Handle "stream" & "stream_options" field
    // "stream_options": {"include_usage": bool, "chunk_result": bool, "chunk_size": int, "preview": bool}
    if (json_value(body, "stream", false)) {
        llama_params["stream"] = true;
        if (!body.contains("stream_options")) {
            llama_params["stream_options"] = json{{"include_usage", true}};
        } else {
            if (body.at("stream_options").is_object()) {
                llama_params["stream_options"] = body.at("stream_options");
                if (!llama_params["stream_options"].contains("include_usage")) {
                    llama_params["stream_options"]["include_usage"] = true;
                }
            } else {
                throw std::runtime_error("Illegal param: invalid \"stream_options\"");
            }
        }
    }

    return llama_params;
}

static json oaicompat_images_edits_request(const struct stablediffusion_params &params, const std::string rid, const json &body) {
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
        if (body_cp.contains("control")) {
            body_cp["control"] = "...";
        }
        SRV_INF("rid %s | %s\n", rid.c_str(), body_cp.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    }

    // Handle "response_format" field
    std::string response_format = json_value(body, "response_format", std::string("b64_json"));
    if (response_format != "b64_json") {
        throw std::runtime_error("Illegal param: \"response_format\" must be 'b64_json'");
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
            throw std::runtime_error("Illegal param: \"n\" must be less than or equal to " + std::to_string(params.max_batch_count));
        }
        llama_params["batch_count"] = batch_count;
    }

    // Handle "sampler" and "cfg_scale" fields
    if (!body.contains("sampler") && !body.contains("sample_method")) {
        std::string quality = json_value(body, "quality", std::string("standard"));
        if (quality != "null" && quality != "hd" && quality != "standard") {
            throw std::runtime_error("Illegal param: \"quality\" must be one of 'hd' or 'standard'");
        }
        llama_params["sample_method"]   = params.sampling.sample_method;
        llama_params["sampling_steps"]  = params.sampling.sampling_steps;
        llama_params["schedule_method"] = params.sampling.schedule_method;
        llama_params["cfg_scale"]       = params.sampling.cfg_scale;
        if (quality == "hd") {
            llama_params["sampling_steps"]  = params.sampling.sampling_steps + 2;
            llama_params["negative_prompt"] = "low quality";
        }
    } else {
        std::string sample_method_str = "euler_a";
        if (body.contains("sample_method")) {
            sample_method_str = body.at("sample_method").get<std::string>();
        } else if (body.contains("sampler")) {
            sample_method_str = body.at("sampler").get<std::string>();
        }
        llama_params["sample_method"] = sd_argument_to_sample_method(sample_method_str.c_str());
        int sampling_steps            = 10;
        if (body.contains("sampling_steps")) {
            sampling_steps = json_value(body, "sampling_steps", 10);
        } else if (body.contains("sample_steps")) {
            sampling_steps = json_value(body, "sample_steps", 10);
        }
        llama_params["sampling_steps"]  = sampling_steps;
        std::string schedule_method_str = "default";
        if (body.contains("schedule_method")) {
            schedule_method_str = body.at("schedule_method").get<std::string>();
        } else if (body.contains("scheduler")) {
            schedule_method_str = body.at("scheduler").get<std::string>();
        } else if (body.contains("schedule")) {
            schedule_method_str = body.at("schedule").get<std::string>();
        }
        llama_params["schedule_method"] = sd_argument_to_schedule(schedule_method_str.c_str());

        // Copy remaining properties to llama_params
        // This allows user to use stable-diffusion.cpp-specific params like "slg_scale", ... via OAI
        // endpoint. See "launch_slot_with_task()" for a complete list of params supported by stable-diffusion.cpp
        for (const auto &item : body.items()) {
            const std::string &key = item.key();
            if (key == "n" ||
                key == "size" ||
                key == "sampler" ||
                key == "sample_steps" ||
                key == "schedule" || key == "scheduler" ||
                llama_params.contains(key)) {
                continue;
            }
            llama_params[item.key()] = item.value();
        }
    }

    // Handle "size" field
    std::string size = json_value(body, "size", std::string("512x512"));
    {
        auto pos = size.find('x');
        if (pos == std::string::npos) {
            throw std::runtime_error("Illegal param: \"size\" must be in the format '{width}x{height}'");
        }
        auto width  = std::stoi(size.substr(0, pos));
        auto height = std::stoi(size.substr(pos + 1));
        if (width < 256 || height < 256) {
            throw std::runtime_error("Illegal param: width and height of \"size\" must be at least 256");
        }
        if (width > params.sampling.width) {
            throw std::runtime_error("Illegal param: width of \"size\" must be at most " + std::to_string(params.sampling.width));
        }
        if (height > params.sampling.height) {
            throw std::runtime_error("Illegal param: height of \"size\" must be at most " + std::to_string(params.sampling.height));
        }
        llama_params["width"]  = width;
        llama_params["height"] = height;
    }

    // Handle "stream" & "stream_options" field
    // "stream_options": {"include_usage": bool, "chunk_result": bool, "chunk_size": int, "preview": bool}
    if (json_value(body, "stream", false)) {
        llama_params["stream"] = true;
        if (!body.contains("stream_options")) {
            llama_params["stream_options"] = json{{"include_usage", true}};
        } else {
            if (body.at("stream_options").is_object()) {
                llama_params["stream_options"] = body.at("stream_options");
                if (!llama_params["stream_options"].contains("include_usage")) {
                    llama_params["stream_options"]["include_usage"] = true;
                }
            } else {
                throw std::runtime_error("Illegal param: invalid \"stream_options\"");
            }
        }
    }

    return llama_params;
}

static json oaicompat_images_response(const std::string &rid, const json &request, const json &result, const bool streaming = false, const bool stop = false, const std::vector<json> &usages = {}) {
    json data = json::array();
    for (auto &ret : result) {
        json item = json{
            {"progressed_steps", ret.at("progressed_steps")},
            {"progress", ret.at("progress")},
            {"index", ret.at("index")},
        };
        if (streaming) {
            item["object"] = "image.chunk";
        } else {
            item["object"] = "image";
        }
        if (ret.contains("b64_json")) {
            item["b64_json"] = ret.at("b64_json");
            if (stop) {
                item["finish_reason"] = "stop";
            }
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
    if (include_usage) {
        res["usage"] = nullptr;
        if (!usages.empty()) {
            double ttp = 0.0;
            double tpg = 0.0;
            double gps = 0.0;

            const auto rs = double(usages.size());
            for (const auto &usage : usages) {
                ttp += json_value(usage, "processing_ms", 0.0);
                tpg += json_value(usage, "generation_ms", 0.0);
                gps += json_value(usage, "generation_per_second", 0.0);
            }
            ttp = ttp / rs;
            tpg = tpg / rs;
            gps = gps / rs;

            json usage = json{
                {"time_to_process_ms", ttp},
                {"time_per_generation_ms", tpg},
                {"generation_per_second", gps},
            };

            res["usage"] = usage;
            SRV_INF("rid %s | ttp: %.2fms, tpg: %.2fms, gps: %.2f\n", rid.c_str(), ttp, tpg, gps);
        }
    }

    return res;
}

static json jinaaicompat_rerank_request(const struct common_params &params, const std::string &rid, const json &body, const llama_vocab *vocab) {
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
        SRV_INF("rid %s | %s\n", rid.c_str(), body_cp.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    }

    json llama_params;

    // Annotations for OAI compatibility
    llama_params["__oaicompat"]        = true;
    llama_params["__oaicompat_rerank"] = true;

    // Handle "model" field
    llama_params["model"] = json_value(body, "model", params.model_alias);

    // Handle "query" and "documents" fields
    json prompt       = json::array();
    std::string query = json_value(body, "query", std::string(""));
    prompt.push_back(query);
    for (const json &doc : body.at("documents")) {
        if (doc.is_string()) {
            prompt.push_back(doc.get<std::string>());
        } else if (doc.is_object() && doc.contains("text")) {
            prompt.push_back(doc.at("text").get<std::string>());
        } else {
            throw std::runtime_error("Illegal param: \"documents\" must be an array of strings or objects with 'text'");
        }
    }
    // Add the query again for reranking
    prompt.push_back(query);
    // Add an empty string for reranking
    // NB(thxCode): llama_vocab_unk is a patch.
    prompt.push_back(common_token_to_piece(vocab, llama_vocab_unk(vocab), false));
    llama_params["prompt"] = prompt;

    // Handle "top_n" field
    size_t documents_size = body.at("documents").size();
    size_t top_n          = json_value(body, "top_n", documents_size);
    if (top_n > documents_size) {
        top_n = documents_size;
    } else if (top_n <= 0) {
        throw std::runtime_error("Illegal param: \"top_n\" must be greater than 0");
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

static json jinaicompat_rerank_response(const std::string &rid, const json &request, json &result) {
    json documents;
    int32_t top_n         = request.at("top_n");
    bool return_documents = request.at("return_documents");
    if (return_documents) {
        documents = request.at("__oaicompat_rerank_documents");
    }

    int total_tokens      = 0;
    int min_prompt_tokens = 0;
    int max_prompt_tokens = 0;
    int32_t start         = 0;
    auto end              = int32_t(result.size() - 3);
    jinaicompat_rerank_response_sort(result, start, end);

    json data      = json::array();
    double scr_max = std::max(json_value(result[result.size() - 2], "score", 1e-6), json_value(result[start], "score", 1e-6));
    double scr_min = std::min(json_value(result[result.size() - 1], "score", 1e-6), json_value(result[end], "score", 1e-6));
    double scr_dst = scr_max - scr_min;
    double a = 0.001, b = 0.998;
    if (scr_dst < 1e-6 || request.at("prompt")[0].get<std::string>() == documents[json_value(result[start], "index", 0)].get<std::string>()) {
        scr_dst = scr_max;
        scr_min = 0.0f;
        a = 0, b = 1;
    }
    const bool normalize = request.at("normalize").get<bool>();
    for (int32_t i = start; i <= end && i < top_n; i++) {
        const json &ret = result[i];

        double scr = json_value(ret, "score", 1e-6);
        if (normalize) {
            scr = a + (scr - scr_min) * b / scr_dst;
            if (scr > 0.99999) {
                scr = 1;
            }
        }

        int32_t tke = json_value(ret, "tokens_evaluated", 0);
        total_tokens += tke;
        min_prompt_tokens = min_prompt_tokens == 0 ? tke : std::min(min_prompt_tokens, tke);
        max_prompt_tokens = std::max(max_prompt_tokens, tke);

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
    };

    // Add usage field
    {
        json usage = json{
            {"prompt_tokens", total_tokens},
            {"total_tokens", total_tokens},
            {"min_prompt_tokens", min_prompt_tokens},
            {"max_prompt_tokens", max_prompt_tokens},
        };

        res["usage"] = usage;
        SRV_INF("rid %s | total_tokens: %d, min_prompt_tokens: %d, max_prompt_tokens: %d\n", rid.c_str(), total_tokens, min_prompt_tokens, max_prompt_tokens);
    }

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

static std::vector<llama_token_data> get_token_probabilities(llama_context *ctx, int idx) {
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

static bool are_lora_equal(
    const std::vector<common_adapter_lora_info> &l1,
    const std::vector<common_adapter_lora_info> &l2) {
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

// parse lora config from JSON request, returned a copy of lora_base with updated scale
static std::vector<common_adapter_lora_info> parse_lora_request(
    const std::vector<common_adapter_lora_info> &lora_base,
    const json &data) {
    std::vector<common_adapter_lora_info> lora(lora_base);
    int max_idx = lora.size();

    // clear existing value
    for (auto &entry : lora) {
        entry.scale = 0.0f;
    }

    // set value
    for (const auto &entry : data) {
        int id      = json_value(entry, "id", -1);
        float scale = json_value(entry, "scale", 0.0f);
        if (0 <= id && id < max_idx) {
            lora[id].scale = scale;
        } else {
            throw std::runtime_error("invalid adapter id");
        }
    }

    return lora;
}

struct llava_text_token_batch_wrapper {
    std::vector<llama_pos> pos;
    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id> seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t> logits;
    llama_batch batch;

    llava_text_token_batch_wrapper(llama_token *token, int32_t n_tokens, llama_pos pos_0, llama_seq_id seq_id) {
        pos.resize(n_tokens);
        n_seq_id.resize(n_tokens);
        seq_ids.resize(n_tokens + 1);
        logits.resize(n_tokens);
        seq_id_0.resize(1);
        seq_id_0[0]       = seq_id;
        seq_ids[n_tokens] = nullptr;
        batch             = {
            /*n_tokens       =*/n_tokens,
            /*tokens         =*/token,
            /*embd           =*/nullptr,
            /*pos            =*/pos.data(),
            /*n_seq_id       =*/n_seq_id.data(),
            /*seq_id         =*/seq_ids.data(),
            /*logits         =*/logits.data(),
        };
        for (int i = 0; i < n_tokens; i++) {
            batch.pos[i]      = pos_0 + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i]   = seq_id_0.data();
            batch.logits[i]   = false;
        }
    }
};

struct llava_image_embed_batch_wrapper {
    std::vector<llama_pos> pos;
    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id> seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t> logits;
    llama_batch batch;

    llava_image_embed_batch_wrapper(float *embd, int32_t n_tokens, llama_pos pos_0, llama_seq_id seq_id) {
        pos.resize(n_tokens);
        n_seq_id.resize(n_tokens);
        seq_ids.resize(n_tokens + 1);
        logits.resize(n_tokens);
        seq_id_0.resize(1);
        seq_id_0[0]       = seq_id;
        seq_ids[n_tokens] = nullptr;
        batch             = {
            /*n_tokens       =*/n_tokens,
            /*tokens         =*/nullptr,
            /*embd           =*/embd,
            /*pos            =*/pos.data(),
            /*n_seq_id       =*/n_seq_id.data(),
            /*seq_id         =*/seq_ids.data(),
            /*logits         =*/logits.data(),
        };
        for (int i = 0; i < n_tokens; i++) {
            batch.pos[i]      = pos_0 + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i]   = seq_id_0.data();
            batch.logits[i]   = false;
        }
    }
};

struct qwen2vl_text_token_batch_wrapper {
    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id> seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t> logits;
    llama_batch batch;

    qwen2vl_text_token_batch_wrapper(llama_token *token, int32_t n_tokens, llama_pos *pos, llama_seq_id seq_id) {
        n_seq_id.resize(n_tokens);
        seq_ids.resize(n_tokens + 1);
        logits.resize(n_tokens);
        seq_id_0.resize(1);
        seq_id_0[0]       = seq_id;
        seq_ids[n_tokens] = nullptr;
        batch             = {
            /*n_tokens       =*/n_tokens,
            /*tokens         =*/token,
            /*embd           =*/nullptr,
            /*pos            =*/pos,
            /*n_seq_id       =*/n_seq_id.data(),
            /*seq_id         =*/seq_ids.data(),
            /*logits         =*/logits.data(),
        };
        for (int i = 0; i < n_tokens; i++) {
            batch.n_seq_id[i] = 1;
            batch.seq_id[i]   = seq_id_0.data();
            batch.logits[i]   = false;
        }
    }
};

struct qwen2vl_image_embed_batch_wrapper {
    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id> seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t> logits;
    llama_batch batch;

    qwen2vl_image_embed_batch_wrapper(float *embd, int32_t n_tokens, llama_pos *pos, llama_seq_id seq_id) {
        n_seq_id.resize(n_tokens);
        seq_ids.resize(n_tokens + 1);
        logits.resize(n_tokens);
        seq_id_0.resize(1);
        seq_id_0[0]       = seq_id;
        seq_ids[n_tokens] = nullptr;
        batch             = {
            /*n_tokens       =*/n_tokens,
            /*tokens         =*/nullptr,
            /*embd           =*/embd,
            /*pos            =*/pos,
            /*n_seq_id       =*/n_seq_id.data(),
            /*seq_id         =*/seq_ids.data(),
            /*logits         =*/logits.data(),
        };
        for (int i = 0; i < n_tokens; i++) {
            batch.n_seq_id[i] = 1;
            batch.seq_id[i]   = seq_id_0.data();
            batch.logits[i]   = false;
        }
    }
};

void common_batch_add_with_mrope(
    struct llama_batch &batch,
    llama_token id,
    llama_pos st_pos_id,
    int32_t n_eval,
    const std::vector<llama_seq_id> &seq_ids,
    bool logits) {
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