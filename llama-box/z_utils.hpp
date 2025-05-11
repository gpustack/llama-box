#pragma once

// heads
#include <chrono>
#include <queue>
#include <random>
#include <utility>

#define JSON_ASSERT GGML_ASSERT
#include "llama.cpp/common/json.hpp"
#include "llama.cpp/common/log.h"

// defines

#define SRV_INF(fmt, ...) LOG_INF("srv %25.*s: " fmt, 25, __func__, __VA_ARGS__)
#define SRV_INFV(v, fmt, ...)                                   \
    if (common_log_verbosity_thold >= v) {                      \
        LOG_INF("srv %25.*s: " fmt, 25, __func__, __VA_ARGS__); \
    }
#define SRV_WRN(fmt, ...) LOG_WRN("srv %25.*s: " fmt, 25, __func__, __VA_ARGS__)
#define SRV_ERR(fmt, ...) LOG_ERR("srv %25.*s: " fmt, 25, __func__, __VA_ARGS__)
#define SRV_DBG(fmt, ...)                                       \
    if (common_log_verbosity_thold > 5) {                       \
        LOG_DBG("srv %25.*s: " fmt, 25, __func__, __VA_ARGS__); \
    }

#define SRV_FUNC_INF(func, fmt, ...) LOG_INF("srv %25.*s: " fmt, 25, func, __VA_ARGS__)
#define SRV_FUNC_INFV(v, func, fmt, ...)                    \
    if (common_log_verbosity_thold >= v) {                  \
        LOG_INF("srv %25.*s: " fmt, 25, func, __VA_ARGS__); \
    }
#define SRV_FUNC_WRN(func, fmt, ...) LOG_WRN("srv %25.*s: " fmt, 25, func, __VA_ARGS__)
#define SRV_FUNC_ERR(func, fmt, ...) LOG_ERR("srv %25.*s: " fmt, 25, func, __VA_ARGS__)
#define SRV_FUNC_DBG(func, fmt, ...)                        \
    if (common_log_verbosity_thold > 5) {                   \
        LOG_DBG("srv %25.*s: " fmt, 25, func, __VA_ARGS__); \
    }

template <typename F> class ScopeGuard {
  public:
    explicit ScopeGuard(F && f) : func_(std::forward<F>(f)) {}

    ~ScopeGuard() {
        if (active_) {
            func_();
        }
    }

    ScopeGuard(const ScopeGuard &)             = delete;
    ScopeGuard & operator=(const ScopeGuard &) = delete;

    void dismiss() { active_ = false; }

  private:
    F    func_;
    bool active_ = true;
};

#define DEFER(...)        auto CONCAT(_defer_, __LINE__) = ScopeGuard([&]() { __VA_ARGS__; })
#define CONCAT_IMPL(a, b) a##b
#define CONCAT(a, b)      CONCAT_IMPL(a, b)

struct RatelimitTokenBucket {
    explicit RatelimitTokenBucket(int32_t capacity, int32_t rate) :
        capacity(capacity),
        rate(rate),
        tokens_remain(capacity),
        last_time(std::chrono::steady_clock::now()) {}

    // try_acquire tokens, return true if success
    bool try_acquire() {
        int tokens = 1;
        if (this->tokens_remain < 1) {
            refill();
            if (this->tokens_remain < 1) {
                return false;
            }
        }
        this->tokens_remain -= 1;
        return true;
    }

  private:
    int                                   capacity;
    int                                   rate;
    int                                   tokens_remain;
    std::chrono::steady_clock::time_point last_time;

    void refill() {
        const auto now     = std::chrono::steady_clock::now();
        auto       elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
        if (elapsed < 1000) {
            elapsed = 0;
        }
        int new_tokens = (int(elapsed) / 1000) * rate;
        if (new_tokens > 0) {
            tokens_remain = std::min(capacity, tokens_remain + new_tokens);
            last_time     = now;
        }
    }
};

struct ParallelControlTokenBucket {
    explicit ParallelControlTokenBucket(int32_t capacity) {
        for (int i = 0; i < capacity; i++) {
            tokens_q.push(i);
            tokens_s.push_back(i);
        }
    }

    // acquire token, await if not enough
    int32_t acquire() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]() { return !tokens_q.empty(); });
        int32_t token = tokens_q.front();
        tokens_q.pop();
        tokens_s[token] = -1;
        return token;
    }

    // release token
    void release(int32_t token) {
        std::lock_guard<std::mutex> lock(mtx);
        if (token < int32_t(tokens_s.size()) && tokens_s[token] == -1) {
            tokens_s[token] = token;
            tokens_q.push(token);
        }
        cv.notify_one();
    }

  private:
    std::mutex              mtx;
    std::condition_variable cv;
    std::queue<int32_t>     tokens_q;
    std::vector<int32_t>    tokens_s;
};

// externs

extern const char * LLAMA_BOX_COMMIT;
extern int          LLAMA_BOX_BUILD_NUMBER;
extern const char * LLAMA_BOX_BUILD_VERSION;
extern const char * LLAMA_BOX_BUILD_COMPILER;
extern const char * LLAMA_BOX_BUILD_TARGET;
extern const char * LLAMA_CPP_COMMIT;
extern int          LLAMA_CPP_BUILD_NUMBER;
extern const char * STABLE_DIFFUSION_CPP_COMMIT;
extern int          STABLE_DIFFUSION_CPP_BUILD_NUMBER;
extern const char * CONCURRENT_QUEUE_COMMIT;
extern int          CONCURRENT_QUEUE_BUILD_NUMBER;
extern const char * READER_WRITER_QUEUE_COMMIT;
extern int          READER_WRITER_QUEUE_BUILD_NUMBER;

// utils

static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

static inline bool char_is_base64(uint8_t c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

static inline std::vector<uint8_t> decode_base64(const std::string & encoded_string) {
    int i   = 0;
    int j   = 0;
    int in_ = 0;

    int in_len = int(encoded_string.size());

    uint8_t char_array_4[4];
    uint8_t char_array_3[3];

    std::vector<uint8_t> ret;

    while (in_len-- && (encoded_string[in_] != '=') && char_is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_];
        in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++) {
                char_array_4[i] = base64_chars.find(char(char_array_4[i]));
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
            char_array_4[j] = base64_chars.find(char(char_array_4[j]));
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

static inline std::string encode_base64(const unsigned char * input, size_t length) {
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

static inline bool string_is_utf8(const std::string & str) {
    const auto *          bytes = reinterpret_cast<const unsigned char *>(str.data());
    const unsigned char * end   = bytes + str.length();

    while (bytes < end) {
        if (*bytes <= 0x7F) {
            // 1-byte sequence (0xxxxxxx)
            bytes++;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // 2-byte sequence (110xxxxx 10xxxxxx)
            if (end - bytes < 2 || (bytes[1] & 0xC0) != 0x80) {
                return false;
            }
            bytes += 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // 3-byte sequence (1110xxxx 10xxxxxx 10xxxxxx)
            if (end - bytes < 3 || (bytes[1] & 0xC0) != 0x80 || (bytes[2] & 0xC0) != 0x80) {
                return false;
            }
            bytes += 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // 4-byte sequence (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
            if (end - bytes < 4 || (bytes[1] & 0xC0) != 0x80 || (bytes[2] & 0xC0) != 0x80 ||
                (bytes[3] & 0xC0) != 0x80) {
                return false;
            }
            bytes += 4;
        } else {
            // Invalid UTF-8 lead byte
            return false;
        }
    }

    return true;
}

// get_position_of_utf8 returns the last index of character that can form a valid string
// if the last character is potentially cut in half, return the index before the cut
// if get_position_of_utf8(text) == text.size(), then the whole text is valid utf8
static inline size_t get_position_of_utf8(const std::string & text) {
    size_t len = text.size();
    if (len == 0) {
        return 0;
    }

    // Check the last few bytes to see if a multi-byte character is cut off
    for (size_t i = 1; i <= 4 && i <= len; ++i) {
        unsigned char c = text[len - i];
        // Check for start of a multi-byte sequence from the end
        if ((c & 0xE0) == 0xC0) {
            // 2-byte character start: 110xxxxx
            // Needs at least 2 bytes
            if (i < 2) {
                return len - i;
            }
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte character start: 1110xxxx
            // Needs at least 3 bytes
            if (i < 3) {
                return len - i;
            }
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte character start: 11110xxx
            // Needs at least 4 bytes
            if (i < 4) {
                return len - i;
            }
        }
    }

    // If no cut-off multi-byte character is found, return full length
    return len;
}

template <typename T>
static inline T json_value(const nlohmann::json & body, const std::string & key, const T & default_value) {
    // Fallback null to default value
    if (body.contains(key) && !body.at(key).is_null()) {
        try {
            return body.at(key);
        } catch (const NLOHMANN_JSON_NAMESPACE::detail::type_error &) {
            SRV_WRN("wrong type supplied for parameter '%s', expected '%s', using default value\n", key.c_str(),
                    nlohmann::json(default_value).type_name());
            return default_value;
        }
    } else {
        return default_value;
    }
}

static inline bool json_is_array_or_string(const nlohmann::json & data) {
    return data.is_string() || data.is_array();
}

static inline bool json_is_array_of_numbers(const nlohmann::json & data) {
    if (data.is_array()) {
        return std::all_of(data.begin(), data.end(), [](const nlohmann::json & e) { return e.is_number_integer(); });
    }
    return false;
}

static inline bool json_is_array_of_mixed_numbers_strings(const nlohmann::json & data) {
    bool seen_string = false;
    bool seen_number = false;
    if (data.is_array()) {
        for (const auto & e : data) {
            seen_string |= e.is_string();
            seen_number |= e.is_number_integer();
            if (seen_number && seen_string) {
                return true;
            }
        }
    }
    return false;
}

static inline bool json_is_array_of_objects(const nlohmann::json & data) {
    if (data.is_array()) {
        return std::all_of(data.begin(), data.end(), [](const nlohmann::json & e) { return e.is_object(); });
    }
    return false;
}

/**
 * this handles 2 cases:
 * - only string, example: "string"
 * - mixed string and tokens, example: [12, 34, "string", 56, 78]
 */
static inline llama_tokens tokenize_prompt(const llama_vocab * vocab, const nlohmann::json & json_prompt,
                                           bool add_special, bool parse_special) {
    llama_tokens result;
    if (json_prompt.is_array()) {
        bool first = true;
        for (const auto & jp : json_prompt) {
            if (jp.is_string()) {
                std::string  s = jp.get<std::string>();
                llama_tokens p;
                if (first) {
                    p     = common_tokenize(vocab, s, add_special, parse_special);
                    first = false;
                } else {
                    p = common_tokenize(vocab, s, false, parse_special);
                }
                result.insert(result.end(), p.begin(), p.end());
            } else if (jp.is_number_integer()) {
                if (first) {
                    first = false;
                }
                result.push_back(jp.get<llama_token>());
            }
        }
    } else if (json_prompt.is_string()) {
        std::string s = json_prompt.get<std::string>();
        result        = common_tokenize(vocab, s, add_special, parse_special);
    } else {
        throw std::runtime_error(
            "Illegal param: content must be a string, a list of tokens, a list of mixed strings & tokens");
    }
    return result;
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
static inline std::vector<llama_tokens> tokenize_prompts(const llama_vocab * vocab, const nlohmann::json & json_prompt,
                                                         bool add_special, bool parse_special) {
    std::vector<llama_tokens> result;
    if (json_prompt.is_string() || json_is_array_of_mixed_numbers_strings(json_prompt)) {
        // string or mixed
        result.push_back(tokenize_prompt(vocab, json_prompt, add_special, parse_special));
    } else if (json_is_array_of_numbers(json_prompt)) {
        // array of tokens
        result.push_back(json_prompt.get<llama_tokens>());
    } else if (json_prompt.is_array()) {
        // array of prompts
        result.reserve(json_prompt.size());
        for (const auto & p : json_prompt) {
            if (p.is_string() || json_is_array_of_mixed_numbers_strings(p)) {
                // string or mixed
                result.push_back(tokenize_prompt(vocab, p, add_special, parse_special));
            } else if (json_is_array_of_numbers(p)) {
                // array of tokens
                result.push_back(p.get<llama_tokens>());
            } else if (json_is_array_of_objects(p)) {
                // array of objects
                result.push_back(tokenize_prompt(vocab, p, add_special, parse_special));
            } else {
                throw std::runtime_error(
                    "Illegal param: content must be a string, a list of tokens, or a list of mixed strings & tokens");
            }
        }
    } else {
        throw std::runtime_error(
            "Illegal param: content must be a string, a list of tokens, a list of mixed strings & tokens");
    }
    return result;
}

static inline std::string random_string() {
    static const std::string str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

    std::random_device rd;
    std::mt19937       generator(rd());

    std::string result(32, ' ');

    for (int i = 0; i < 32; ++i) {
        result[i] = str[generator() % str.size()];
    }

    return result;
}

static inline std::string tokens_to_output_formatted_string(const llama_context * ctx, const llama_token token) {
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

struct longest_common_prefix {
    int32_t s = 0;
    size_t  l = 0;
};

static inline std::unique_ptr<longest_common_prefix> find_longest_common_prefix(const llama_tokens & target,
                                                                                const llama_tokens & match) {
    const size_t tsz = target.size();
    const size_t msz = match.size();
    if (tsz == 0 || msz == 0) {
        return nullptr;
    }

    size_t s = 0, e = 0;
    for (; s < tsz; s++) {
        e = s;
        for (; e < tsz && e - s < msz && target[e] == match[e - s]; e++) {
        }
        if (e > s) {
            break;
        }
    }

    if (e == s) {
        return nullptr;
    }

    return std::make_unique<longest_common_prefix>(longest_common_prefix{ int32_t(s), e - s });
}

// Computes FNV-1a hash of the data
static std::string hash_fnv(const uint8_t * data, size_t len) {
    constexpr uint64_t FNV_OFFSET_BASIS = 0xcbf29ce484222325ULL;
    constexpr uint64_t FNV_PRIME        = 0x100000001b3ULL;

    uint64_t hash = FNV_OFFSET_BASIS;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= FNV_PRIME;
    }

    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(16) << hash;
    return ss.str();
}
