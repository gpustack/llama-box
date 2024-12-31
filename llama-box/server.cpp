#include <atomic>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <memory>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "llama.cpp/common/common.h"
#include "llama.cpp/common/json-schema-to-grammar.h"
#include "llama.cpp/common/log.h"
#include "llama.cpp/common/ngram-cache.h"
#include "llama.cpp/common/sampling.h"
#include "llama.cpp/common/speculative.h"
#include "llama.cpp/examples/llava/clip.h"
#include "llama.cpp/examples/llava/llava.h"
#include "llama.cpp/ggml/include/ggml.h"
#include "llama.cpp/include/llama.h"

#include "param.hpp"
#include "ratelimiter.hpp"
#include "rpcserver.hpp"
#include "utils.hpp"

// mime type for sending response
#define MIMETYPE_TEXT "text/plian; charset=utf-8"
#define MIMETYPE_JSON "application/json; charset=utf-8"
#define HEADER_REQUEST_ID "X-Request-ID"
#define HEADER_REQUEST_ACCEPTED_AT "X-Request-Accepted-At"
#define HEADER_REQUEST_TOKENS_PER_SECOND "X-Request-Tokens-Per-Second"

using json = nlohmann::json;

enum tool_call_parser_type {
    TOOL_CALL_PARSER_TYPE_NONE,
    TOOL_CALL_PARSER_TYPE_STRING,
    TOOL_CALL_PARSER_TYPE_TOKEN,
};

enum stop_type {
    STOP_TYPE_NONE,
    STOP_TYPE_EOS,
    STOP_TYPE_WORD,
    STOP_TYPE_LIMIT,
    STOP_TYPE_TOOL,
};

static inline std::string format_stop_type(const enum stop_type type) {
    switch (type) {
        case STOP_TYPE_EOS:
        case STOP_TYPE_WORD:
            return "stop";
        case STOP_TYPE_LIMIT:
            return "length";
        case STOP_TYPE_TOOL:
            return "tool_calls";
        default:
            return "";
    }
}

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

static inline json format_error_response(const std::string &message, const enum error_type type) {
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

enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_STARTED,
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

enum server_state {
    SERVER_STATE_LOADING_MODEL, // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,         // Server is ready and model is loaded
};

enum server_task_type {
    SERVER_TASK_TYPE_COMPLETION,
    SERVER_TASK_TYPE_EMBEDDING,
    SERVER_TASK_TYPE_RERANK,
    SERVER_TASK_TYPE_INFILL,
    SERVER_TASK_TYPE_IMAGE,
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
    SERVER_TASK_TYPE_METRICS,
    SERVER_TASK_TYPE_SLOT_SAVE,
    SERVER_TASK_TYPE_SLOT_RESTORE,
    SERVER_TASK_TYPE_SLOT_ERASE,
    SERVER_TASK_TYPE_SET_LORA,
};

struct server_task {
    server_task_type type;

    std::string rid;
    int id        = -1; // to be filled by server_queue
    int id_target = -1;

    llama_tokens prompt_tokens;
    json data;

    int tps = 0;

    server_task(server_task_type type)
        : type(type) {
    }
};

struct server_task_result {
    int id = -1;

    json data;

    bool stop  = false;
    bool error = false;
};

struct slot_params {
    bool stream = true;

    /* STABLE DIFFUSION */

    bool stream_preview_faster = false;
    bool stream_preview        = false;

    struct stablediffusion_params_sampling sd_params;

    /* LLAMA */

    bool return_tokens       = false;
    int32_t n_predict        = -1; // new tokens to predict
    int32_t n_indent         = 0;  // mininum line indentation for the generated text in number of whitespace characters
    int32_t n_keep           = 0;  // number of tokens to keep from initial prompt
    int32_t n_discard        = 0;  // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int64_t t_max_prompt_ms  = -1; // TODO: implement
    int64_t t_max_predict_ms = -1; // if positive, limit the generation phase to this time limit

    struct common_params_sampling llm_params;

    std::vector<std::string> antiprompt;
};

struct server_slot {
    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;
    std::function<void(int)> callback_on_release;

    server_task_type task_type = SERVER_TASK_TYPE_COMPLETION;
    int id;
    std::string rid;
    int id_task      = -1;
    size_t index     = 0;
    slot_state state = SLOT_STATE_IDLE;
    struct slot_params params;

    llama_tokens prompt_tokens;
    std::string prompt_string;
    json prompt_multi_modal_data;

    /* STABLE DIFFUSION */

    bool oaicompat_image          = false;
    bool oaicompat_image_generate = false;
    bool oaicompat_image_edit     = false;

    // sampling
    stablediffusion_sampling_stream *sdsstream = nullptr;

    // state
    int32_t n_image_generated_steps = 0;
    int64_t t_start_process_image;
    int64_t t_start_generate_image;
    double t_image_processing; // ms
    double t_image_generation; // ms

    /* LLAMA */

    bool oaicompat_completion             = false;
    bool oaicompat_completion_chat        = false;
    bool oaicompat_completion_chat_tool   = false;
    bool oaicompat_completion_chat_vision = false;

    // generating
    int32_t n_ctx                     = 0; // context size per slot
    int32_t n_past                    = 0;
    int32_t n_past_mmd                = 0;
    int32_t n_decoded                 = 0;
    int32_t n_remaining               = -1;
    int32_t i_batch                   = -1;
    int32_t n_predict                 = -1; // TODO: disambiguate from params.n_predict
    int32_t n_prompt_tokens           = 0;
    int32_t n_prompt_tokens_processed = 0;
    size_t last_nl_pos                = 0;

    // output
    std::string generated_text;
    llama_tokens generated_tokens;
    llama_tokens cache_tokens;
    std::vector<completion_token_output> generated_token_probs;
    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;
    stop_type stop;
    std::string stopping_word;

    // sampling
    json json_schema;
    struct common_sampler *smpl = nullptr;
    llama_tokens sampled;

    // stats
    size_t n_sent_text        = 0; // number of sent text character
    size_t n_sent_token_probs = 0;
    int64_t t_start_process_prompt;
    int64_t t_start_generation;
    double t_prompt_processing; // ms
    double t_token_generation;  // ms

    // speculative
    int32_t n_drafted          = 0;
    int32_t n_drafted_accepted = 0;
    llama_tokens sampled_draft;
    // draft-model speculative decoding
    struct common_sampler *smpl_draft = nullptr;
    // model-free speculative decoding
    int32_t lookup_ngram_min = 0;
    common_ngram_cache ctx_ngram_cache;

    // tps rate limit
    token_bucket *token_bkt = nullptr;

    // mrope position
    llama_pos st_pos_id = 0;

    // tool calls
    tool_call_parser_type tool_call_parser = TOOL_CALL_PARSER_TYPE_NONE;
    bool parallel_tool_calls               = true;
    std::vector<json> generated_tool_calls;

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        prompt_tokens.clear();
        prompt_string = "";
        prompt_multi_modal_data.clear();

        /* STABLE DIFFUSION */

        if (oaicompat_image) {
            if (params.sd_params.init_img_buffer != nullptr) {
                stbi_image_free(params.sd_params.init_img_buffer);
                params.sd_params.init_img_buffer = nullptr;
            }
            if (params.sd_params.control_img_buffer != nullptr) {
                stbi_image_free(params.sd_params.control_img_buffer);
                params.sd_params.control_img_buffer = nullptr;
            }
            if (params.sd_params.mask_img_buffer != nullptr) {
                stbi_image_free(params.sd_params.mask_img_buffer);
                params.sd_params.mask_img_buffer = nullptr;
            }
            if (sdsstream != nullptr) {
                sd_sampling_stream_free(sdsstream->stream);
                delete sdsstream;
                sdsstream = nullptr;
            }
            return;
        }

        /* LLAMA */

        n_prompt_tokens           = 0;
        n_prompt_tokens_processed = 0;
        last_nl_pos               = 0;

        generated_text = "";
        generated_text.clear();

        has_new_line       = false;
        truncated          = false;
        stop               = STOP_TYPE_NONE;
        stopping_word      = "";
        n_past             = 0;
        n_past_mmd         = 0;
        n_sent_text        = 0;
        n_sent_token_probs = 0;

        if (smpl != nullptr) {
            common_sampler_free(smpl);
            smpl = nullptr;
        }

        sampled.clear();
        generated_token_probs.clear();

        if (token_bkt != nullptr) {
            delete token_bkt;
            token_bkt = nullptr;
        }

        n_drafted          = 0;
        n_drafted_accepted = 0;
        sampled_draft.clear();

        if (smpl_draft != nullptr) {
            common_sampler_free(smpl_draft);
            smpl_draft = nullptr;
        }

        lookup_ngram_min = 0;

        st_pos_id = 0;

        tool_call_parser    = TOOL_CALL_PARSER_TYPE_NONE;
        parallel_tool_calls = true;
        generated_tool_calls.clear();
    }

    bool is_non_causal() const {
        return task_type == SERVER_TASK_TYPE_EMBEDDING || task_type == SERVER_TASK_TYPE_RERANK;
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool has_budget(const common_params &global_params) {
        if (params.n_predict == -1 && global_params.n_predict == -1) {
            return true; // limitless
        }

        n_remaining = -1;

        if (params.n_predict != -1) {
            n_remaining = params.n_predict - n_decoded;
        } else if (global_params.n_predict != -1) {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0; // no budget
    }

    void add_token(const completion_token_output &token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }
        generated_token_probs.push_back(token);
    }

    void release() {
        if (is_processing()) {
            /* STABLE DIFFUSION */

            if (oaicompat_image) {
                SLT_INF(*this, "%s", "stop processing\n");

                t_last_used        = ggml_time_us();
                t_image_generation = double(ggml_time_us() - t_start_generate_image) / 1e3;
                state              = SLOT_STATE_IDLE;
                callback_on_release(id);
                return;
            }

            /* LLAMA */

            SLT_INF(*this, "stop processing: n_past = %d, truncated = %d\n", n_past, truncated);

            t_last_used        = ggml_time_us();
            t_token_generation = double(ggml_time_us() - t_start_generation) / 1e3;
            state              = SLOT_STATE_IDLE;
            callback_on_release(id);
        }
    }

    json get_formated_timings() const {
        /* STABLE DIFFUSION */

        if (oaicompat_image) {
            return json{
                {"processing_ms", t_image_processing},
                {"generation_n", n_image_generated_steps},
                {"generation_ms", t_image_generation},
                {"generation_per_step_ms", t_image_generation / n_image_generated_steps},
                {"generation_per_second", 1e3 / t_image_generation * n_image_generated_steps},
            };
        }

        /* LLAMA */

        json ret = json{
            {"prompt_n", n_prompt_tokens_processed},
            {"prompt_ms", t_prompt_processing},
            {"prompt_per_token_ms", t_prompt_processing / n_prompt_tokens_processed},
            {"prompt_per_second", 1e3 / t_prompt_processing * n_prompt_tokens_processed},

            {"predicted_n", n_decoded},
            {"predicted_ms", t_token_generation},
            {"predicted_per_token_ms", t_token_generation / n_decoded},
            {"predicted_per_second", 1e3 / t_token_generation * n_decoded},
        };

        if (n_drafted > 0) {
            const int32_t n_decoded_with_drafted = n_decoded + n_drafted - n_drafted_accepted;
            ret["predicted_per_token_ms"]        = t_token_generation / n_decoded_with_drafted;
            ret["predicted_per_second"]          = 1e3 / t_token_generation * n_decoded_with_drafted;
            ret["drafted_n"]                     = n_drafted;
            ret["drafted_accepted_n"]            = n_drafted_accepted;
            ret["drafted_accepted_p"]            = float(n_drafted_accepted) / float(n_drafted);
        }

        return ret;
    }

    size_t find_stopping_strings(const std::string &text, const size_t last_token_size, bool is_full_stop) {
        size_t stop_pos = std::string::npos;

        for (const std::string &word : params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                pos = find_partial_stop_string(word, text);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void push_token_into_result(llama_context *llm_ctx, int32_t tok_idx, llama_token tok, completion_token_output &result) {
        if (lookup_ngram_min > 0) {
            prompt_tokens.push_back(tok);
            common_ngram_cache_update(ctx_ngram_cache, lookup_ngram_min, LLAMA_NGRAM_MAX, prompt_tokens, 1, false);
        }

        result.toks.push_back(tok);

        if (params.llm_params.n_probs > 0) {
            const std::vector<llama_token_data> cur = get_token_probabilities(llm_ctx, tok_idx);
            const size_t n_vocab                    = llama_n_vocab(llama_get_model(llm_ctx));
            const size_t n_probs                    = params.llm_params.n_probs;
            // set probability for sampled token
            for (size_t i = 0; i < n_vocab; i++) {
                if (cur[i].id == tok) {
                    result.probs.push_back(cur[i].p);
                    break;
                }
            }
            // set probability for top n_probs tokens
            result.top_probs.emplace_back();
            for (size_t i = 0; i < std::min(n_vocab, n_probs); i++) {
                result.top_probs[result.top_probs.size() - 1].push_back({cur[i].id, cur[i].p});
            }
        }
    }
};

struct server_metrics {
    int64_t t_start = 0;

    /* STABLE DIFFUSION */

    uint64_t t_image_processing_total      = 0;
    uint64_t t_image_generation_total      = 0;
    uint64_t n_image_generated_steps_total = 0;

    uint64_t t_image_processing      = 0;
    uint64_t t_image_generation      = 0;
    uint64_t n_image_generated_steps = 0;

    /* LLAMA */

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;
    uint64_t n_tokens_drafted_total          = 0;
    uint64_t n_tokens_drafted_accepted_total = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;
    uint64_t n_decode_total            = 0;
    uint64_t n_busy_slots_total        = 0;
    uint64_t n_tokens_predicted        = 0;
    uint64_t t_tokens_generation       = 0;
    uint64_t n_tokens_drafted          = 0;
    uint64_t n_tokens_drafted_accepted = 0;

    void init() {
        t_start = ggml_time_us();
    }

    void on_prefilled(const server_slot &slot) {
        t_image_processing += uint64_t(slot.t_image_processing);
        t_image_processing_total += uint64_t(slot.t_image_processing);

        n_prompt_tokens_processed += slot.n_prompt_tokens_processed;
        n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
        t_prompt_processing += uint64_t(slot.t_prompt_processing);
        t_prompt_processing_total += uint64_t(slot.t_prompt_processing);
    }

    void on_decoded(const std::vector<server_slot> &slots) {
        n_decode_total++;
        for (const auto &slot : slots) {
            if (slot.is_processing()) {
                n_busy_slots_total++;
            }
        }
    }

    void on_finished(const server_slot &slot) {
        n_image_generated_steps += slot.n_image_generated_steps;
        n_image_generated_steps_total += slot.n_image_generated_steps;
        t_image_generation += uint64_t(slot.t_image_generation);
        t_image_generation_total += uint64_t(slot.t_image_generation);

        n_tokens_predicted += slot.n_decoded;
        n_tokens_predicted_total += slot.n_decoded;
        t_tokens_generation += uint64_t(slot.t_token_generation);
        t_tokens_generation_total += uint64_t(slot.t_token_generation);
        n_tokens_drafted += slot.n_drafted;
        n_tokens_drafted_total += slot.n_drafted;
        n_tokens_drafted_accepted += slot.n_drafted_accepted;
        n_tokens_drafted_accepted_total += slot.n_drafted_accepted;
    }

    void reset_bucket() {
        t_image_processing        = 0;
        n_prompt_tokens_processed = 0;
        t_image_generation        = 0;

        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
        n_tokens_drafted          = 0;
        n_tokens_drafted_accepted = 0;
    }
};

struct server_task_queue {
    int id = 0;
    bool running;

    // queues
    std::deque<server_task> queue_tasks;
    std::deque<server_task> queue_tasks_deferred;

    std::mutex mutex_tasks;
    std::condition_variable condition_tasks;

    // callback functions
    std::function<void(server_task)> callback_new_task;
    std::function<void(void)> callback_update_slots;

    // Add a new task to the end of the queue
    int post(server_task task, bool front = false) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        if (task.id == -1) {
            task.id = id++;
        }
        QUE_DBG("new task, id = %d, front = %d\n", task.id, front);
        if (front) {
            queue_tasks.push_front(std::move(task));
        } else {
            queue_tasks.push_back(std::move(task));
        }
        condition_tasks.notify_one();
        return task.id;
    }

    // Add multiple tasks to the end of the queue
    std::unordered_set<int> post(std::vector<server_task> &tasks, bool front = false) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        std::unordered_set<int> ids(tasks.size());
        for (auto &task : tasks) {
            if (task.id == -1) {
                task.id = id++;
            }
            QUE_DBG("new task, id = %d/%d, front = %d\n", task.id, (int)tasks.size(), front);
            if (front) {
                queue_tasks.push_front(std::move(task));
            } else {
                queue_tasks.push_back(std::move(task));
            }
            ids.insert(task.id);
        }
        condition_tasks.notify_one();
        return ids;
    }

    // Add a new task, but defer until one slot is available
    void defer(server_task task) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        QUE_DBG("defer task, id = %d\n", task.id);
        queue_tasks_deferred.push_back(std::move(task));
        condition_tasks.notify_one();
    }

    // Get the next id for creating a new task
    int get_new_id() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        int new_id = id++;
        return new_id;
    }

    // Register function to process a new task
    void on_new_task(std::function<void(server_task)> callback) {
        callback_new_task = std::move(callback);
    }

    // Register the function to be called when all slots data is ready to be
    // processed
    void on_update_slots(std::function<void(void)> callback) {
        callback_update_slots = std::move(callback);
    }

    // Call when the state of one slot is changed, it will move one task from deferred to main queue
    void pop_deferred_task() {
        // move deferred tasks back to main loop
        std::unique_lock<std::mutex> lock(mutex_tasks);
        if (!queue_tasks_deferred.empty()) {
            queue_tasks.emplace_back(std::move(queue_tasks_deferred.front()));
            queue_tasks_deferred.pop_front();
        }
        condition_tasks.notify_one();
    }

    // End the start_loop routine
    void terminate() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        running = false;
        condition_tasks.notify_all();
    }

    /**
     * Main loop consists of these steps:
     * - Wait until a new task arrives
     * - Process the task (i.e. maybe copy data into slot)
     * - Check if multitask is finished
     * - Update all slots
     */
    void start_loop() {
        running = true;

        while (true) {
            QUE_DBG("%s", "processing new tasks\n");

            while (true) {
                std::unique_lock<std::mutex> lock(mutex_tasks);
                if (queue_tasks.empty()) {
                    lock.unlock();
                    break;
                }
                server_task task = queue_tasks.front();
                queue_tasks.pop_front();
                lock.unlock();

                QUE_DBG("processing task, id = %d\n", task.id);
                callback_new_task(std::move(task));
            }

            // all tasks in the current loop is processed, slots data is now ready
            QUE_DBG("%s", "update slots\n");
            callback_update_slots();

            QUE_DBG("%s", "waiting for new tasks\n");
            {
                std::unique_lock<std::mutex> lock(mutex_tasks);
                if (queue_tasks.empty()) {
                    if (!running) {
                        QUE_DBG("%s", "terminate\n");
                        return;
                    }
                    condition_tasks.wait(lock, [&] { return (!queue_tasks.empty() || !running); });
                }
            }
        }
    }
};

struct server_task_result_queue {
    typedef std::function<void(int, int, server_task_result &)> callback_multitask_t;
    callback_multitask_t callback_update_multitask;

    // for keeping track of all tasks waiting for the result
    std::unordered_set<int> waiting_task_ids;

    // the main result queue
    std::vector<server_task_result> queue_results;

    std::mutex mutex_results;
    std::condition_variable condition_results;

    // add the id_task to the list of tasks waiting for response
    void add_waiting_task_id(int id_task) {
        std::unique_lock<std::mutex> lock(mutex_results);
        SRV_DBG("add task %d to waiting list. current waiting = %d (before add)\n", id_task, (int)waiting_task_ids.size());
        waiting_task_ids.insert(id_task);
    }

    // add all tasks to the list of tasks waiting for response
    void add_waiting_task_ids(const std::unordered_set<int> &id_tasks) {
        std::unique_lock<std::mutex> lock(mutex_results);
        for (const auto &id_task : id_tasks) {
            SRV_DBG("add task %d to waiting list. current waiting = %d (before add)\n", id_task, (int)waiting_task_ids.size());
            waiting_task_ids.insert(id_task);
        }
    }

    // when the request is finished, we can remove task associated with it
    void remove_waiting_task_id(int id_task) {
        std::unique_lock<std::mutex> lock(mutex_results);
        SRV_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int)waiting_task_ids.size());
        waiting_task_ids.erase(id_task);
    }

    // remove tasks from the list of tasks waiting for response
    void remove_waiting_task_ids(const std::unordered_set<int> &id_tasks) {
        std::unique_lock<std::mutex> lock(mutex_results);
        for (const auto &id_task : id_tasks) {
            SRV_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int)waiting_task_ids.size());
            waiting_task_ids.erase(id_task);
        }
    }

    // this function blocks the thread until there is a response for the id_task
    server_task_result recv(int id_task) {
        std::unordered_set<int> id_tasks = {id_task};
        return recv(id_tasks);
    }

    // this function blocks the thread until there is a response for one of the id_tasks
    server_task_result recv(const std::unordered_set<int> &id_tasks) {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_results);
            condition_results.wait(lock, [&] { return !queue_results.empty(); });

            for (int i = 0; i < (int)queue_results.size(); i++) {
                if (id_tasks.find(queue_results[i].id) != id_tasks.end()) {
                    server_task_result res = queue_results[i];
                    queue_results.erase(queue_results.begin() + i);
                    return res;
                }
            }
        }

        // should never reach here
    }

    // Send a new result to a waiting id_task
    void send(server_task_result &result) {
        std::unique_lock<std::mutex> lock(mutex_results);
        SRV_DBG("task %d | sending result\n", result.id);
        for (const auto &id_task : waiting_task_ids) {
            if (result.id == id_task) {
                SRV_DBG("task %d | moved to result queue\n", result.id);
                queue_results.push_back(std::move(result));
                condition_results.notify_all();
                return;
            }
        }
    }
};

struct server_context {
    llama_box_params params;
    std::vector<common_lora_adapter_container> lora_adapters;

    server_task_queue queue_tasks;
    server_task_result_queue queue_results;

    server_metrics metrics;

    std::vector<server_slot> slots;
    json default_generation_settings_for_props;
    llama_batch batch = {};

    /* STABLE DIFFUSION */

    stablediffusion_params sd_params;
    stablediffusion_context *sd_ctx = nullptr;

    /* LLAMA */

    common_params llm_params;
    llama_model *llm_model = nullptr;
    llama_context *llm_ctx = nullptr;
    clip_ctx *llm_ctx_clip = nullptr;

    bool cache_prompt; // remember the prompt to avoid reprocessing all prompt

    // draft-model speculative decoding
    llama_batch batch_draft;
    llama_model *llm_model_draft = nullptr;
    llama_context *llm_ctx_draft = nullptr;
    // model-free speculative decoding
    common_ngram_cache ngram_cache_static;
    common_ngram_cache ngram_cache_dynamic;

    // thread pool
    ggml_threadpool *threadpool       = nullptr;
    ggml_threadpool *threadpool_batch = nullptr;

    // tool calls
    std::string chat_template_alias         = "chatml";
    bool support_tool_calls                 = false;
    bool tool_call_id_generate              = true;
    std::string tool_call_start             = "";
    llama_token tool_call_start_tok         = LLAMA_TOKEN_NULL;
    bool tool_call_start_trim               = true;
    std::vector<std::string> tool_call_ends = {};
    llama_token tool_call_end_tok           = LLAMA_TOKEN_NULL;
    bool tool_call_end_trim                 = true;

    ~server_context() {
        lora_adapters.clear();

        for (server_slot &slot : slots) {
            if (slot.smpl != nullptr) {
                common_sampler_free(slot.smpl);
            }
            if (slot.smpl_draft != nullptr) {
                common_sampler_free(slot.smpl_draft);
            }
            delete slot.token_bkt;
        }
        slots.clear();

        llama_batch_free(batch);
        if (llm_ctx != nullptr) {
            llama_detach_threadpool(llm_ctx);
            llama_free_model(llm_model);
            llama_free(llm_ctx);
        }
        if (llm_ctx_clip != nullptr) {
            clip_free(llm_ctx_clip);
        }

        llama_batch_free(batch_draft);
        if (llm_ctx_draft != nullptr) {
            llama_detach_threadpool(llm_ctx_draft);
            llama_free_model(llm_model_draft);
            llama_free(llm_ctx_draft);
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
    }

    bool load_model(llama_box_params &params_) {
        SRV_INF("loading model '%s'\n", llm_params.model.c_str());

        params     = params_;
        llm_params = params.llm_params;
        sd_params  = params.sd_params;

        /* STABLE DIFFUSION */

        // load stable diffusion model
        if (params.endpoint_images) {
            sd_ctx = common_sd_init_from_params(sd_params);
            if (sd_ctx == nullptr) {
                SRV_ERR("failed to load stable diffusion model, '%s'\n", sd_params.model.c_str());
                return false;
            }

            if (sd_params.model_alias.empty()) {
                if (sd_params.model.find_last_of('/') != std::string::npos) {
                    sd_params.model_alias = sd_params.model.substr(sd_params.model.find_last_of('/') + 1);
                } else if (sd_params.model.find_last_of('\\') != std::string::npos) {
                    sd_params.model_alias = sd_params.model.substr(sd_params.model.find_last_of('\\') + 1);
                } else {
                    sd_params.model_alias = sd_params.model;
                }
            }
            if (sd_params.sampling.strength <= 0.0f) {
                sd_params.sampling.strength = sd_ctx->get_default_strength();
            }
            if (sd_params.sampling.sample_method >= N_SAMPLE_METHODS) {
                sd_params.sampling.sample_method = sd_ctx->get_default_sample_method();
            }
            if (sd_params.sampling.sampling_steps <= 0) {
                sd_params.sampling.sampling_steps = sd_ctx->get_default_sampling_steps();
            }
            if (sd_params.sampling.cfg_scale <= 0.0f) {
                sd_params.sampling.cfg_scale = sd_ctx->get_default_cfg_scale();
            }
            if (sd_params.sampling.slg_scale <= 0.0f) {
                sd_params.sampling.slg_scale = sd_ctx->get_default_slg_scale();
            }
            for (const auto &la : sd_params.lora_adapters) {
                common_lora_adapter_container loaded_la;
                loaded_la.path  = la.path;
                loaded_la.scale = la.scale;
                lora_adapters.push_back(loaded_la);
            }

            SRV_INF("seed: %d, flash attn: %s, guidance: %f, strength: %f, sample method: %s, sampling steps: %d, cfg scale: %.2f, slg scale: %.2f, schedule method: %s\n",
                    sd_params.seed,
                    sd_params.flash_attn ? "true" : "false",
                    sd_params.sampling.guidance,
                    sd_params.sampling.strength,
                    sd_sample_method_to_argument(sd_params.sampling.sample_method),
                    sd_params.sampling.sampling_steps,
                    sd_params.sampling.cfg_scale,
                    sd_params.sampling.slg_scale,
                    sd_schedule_to_argument(sd_params.sampling.schedule_method));

            return true;
        }

        /* LLAMA */

        if (llm_params.model_alias.empty()) {
            if (llm_params.model.find_last_of('/') != std::string::npos) {
                llm_params.model_alias = llm_params.model.substr(llm_params.model.find_last_of('/') + 1);
            } else if (llm_params.model.find_last_of('\\') != std::string::npos) {
                llm_params.model_alias = llm_params.model.substr(llm_params.model.find_last_of('\\') + 1);
            } else {
                llm_params.model_alias = llm_params.model;
            }
        }

        // load multimodal projection model
        if (!llm_params.mmproj.empty()) {
            if (llm_params.n_ctx < 2048) {
                SRV_WRN("%s", "n_ctx is too small for multimodal projection, setting to 2048\n");
                llm_params.n_ctx = 2048;
            }
            // NB(thxCode): clip_model_load is a patch.
            llm_ctx_clip = clip_model_load(llm_params.mmproj.c_str(), /* verbosity */ common_log_verbosity_thold, llm_params.n_gpu_layers, params_.max_image_size);
            if (llm_ctx_clip == nullptr) {
                SRV_ERR("failed to load multimodal project model, '%s'\n", llm_params.mmproj.c_str());
                return false;
            }
        }

        // load the draft model if needed
        if (!llm_params.speculative.model.empty() && llm_params.speculative.n_max > 0) {
            SRV_INF("loading draft model '%s'\n", llm_params.speculative.model.c_str());

            common_params llm_params_draft   = llm_params;
            llm_params_draft.model           = llm_params.speculative.model;
            llm_params_draft.n_gpu_layers    = llm_params.speculative.n_gpu_layers;
            llm_params_draft.cpuparams       = llm_params.speculative.cpuparams;
            llm_params_draft.cpuparams_batch = llm_params.speculative.cpuparams_batch;
            llm_params_draft.cache_type_k    = GGML_TYPE_F16;
            llm_params_draft.cache_type_v    = GGML_TYPE_F16;
            llm_params_draft.warmup          = false;
            common_init_result ir            = common_init_from_params(llm_params_draft);
            llm_model_draft                  = ir.model;
            llm_ctx_draft                    = ir.context;
            if (llm_model_draft == nullptr) {
                SRV_ERR("failed to load draft model, '%s'\n", llm_params.speculative.model.c_str());
                return false;
            }
        }

        // load the ngram cache if needed
        if (params.lookup_ngram_min > 0) {
            if (!llm_params.lookup_cache_static.empty()) {
                try {
                    ngram_cache_static = common_ngram_cache_load(llm_params.lookup_cache_static);
                } catch (std::ifstream::failure const &) {
                    SRV_ERR("failed to load static ngram cache, '%s'\n", llm_params.lookup_cache_static.c_str());
                    return false;
                }
            }
            if (!llm_params.lookup_cache_dynamic.empty()) {
                try {
                    ngram_cache_dynamic = common_ngram_cache_load(llm_params.lookup_cache_dynamic);
                } catch (std::ifstream::failure const &) {
                    // NOP
                }
            }
        }

        common_init_result ir = common_init_from_params(llm_params);
        llm_model             = ir.model;
        llm_ctx               = ir.context;
        lora_adapters         = ir.lora_adapters;
        if (llm_model == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", llm_params.model.c_str());
            return false;
        }

        // check multimodal projection model compatibility
        if (llm_ctx_clip != nullptr) {
            const int n_embd_clip = clip_n_mmproj_embd(llm_ctx_clip);
            const int n_embd      = llama_n_embd(llm_model);
            if (n_embd_clip != n_embd) {
                SRV_ERR("multimodal projector embedding length is not equal to the model, n_embd_clip = %d, n_embd = %d\n", n_embd_clip, n_embd);
                return false;
            }
        }

        // check draft model compatibility if needed
        if (llm_ctx_draft != nullptr) {
            const bool vocab_type_draft = llama_vocab_type(llm_model_draft);
            const bool vocab_type       = llama_vocab_type(llm_model);
            if (vocab_type_draft != vocab_type) {
                SRV_ERR("draft model vocabulary type is not equal to the model, vocab_type_draft = %d, vocab_type = %d\n", vocab_type_draft, vocab_type);
                return false;
            }

            if (llama_add_bos_token(llm_model_draft) != llama_add_bos_token(llm_model) ||
                llama_add_eos_token(llm_model_draft) != llama_add_eos_token(llm_model) ||
                llama_token_bos(llm_model_draft) != llama_token_bos(llm_model) ||
                llama_token_eos(llm_model_draft) != llama_token_eos(llm_model)) {
                SRV_ERR("%s", "draft model special tokens are not equal to the model\n");
                return false;
            }
        }

        cache_prompt = (params.cache_prompt || params.lookup_ngram_min > 0 || llm_ctx_draft != nullptr) && llm_ctx_clip == nullptr;
        SRV_INF("prompt caching %s\n", cache_prompt ? "enabled" : "disabled");

        // sample tokens per second
        if (params.n_tps < 0) {
            SRV_INF("%s", "sampling tokens per second, this will take some time...\n");
            const int32_t n_check            = std::min(int32_t(llama_n_ctx(llm_ctx)), llm_params.n_ubatch);
            llama_tokens check_prompt_tokens = {llama_token_bos(llm_model)};
            common_sampler *check_smpl       = common_sampler_init(llm_model, llm_params.sampling);
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
                if (llama_token_is_eog(llm_model, id)) {
                    break;
                }
                common_sampler_accept(check_smpl, id, false);
                check_prompt_tokens.push_back(id);
            }
            params.n_tps = ceil(1e3 / (double(ggml_time_us() - t_start_decoding) / 1e3) * n_check_decoded);
            common_sampler_free(check_smpl);
            llama_kv_cache_clear(llm_ctx);
            llama_synchronize(llm_ctx);
            llama_perf_context_reset(llm_ctx);
            SRV_INF("sampled tokens per second, tps = %d\n", params.n_tps);
        }

        // thread pool
        {
            struct ggml_threadpool_params tpp = ggml_threadpool_params_from_cpu_params(llm_params.cpuparams);
            threadpool                        = ggml_threadpool_new(&tpp);
            if (!threadpool) {
                SRV_ERR("threadpool create failed : n_threads %d\n", tpp.n_threads);
                return false;
            }

            struct ggml_threadpool_params tpp_batch = ggml_threadpool_params_from_cpu_params(llm_params.cpuparams_batch);
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

        // chat template
        {
            if (llm_params.enable_chat_template) {
                // if a custom chat template is not supplied, we will use the one that comes
                // with the model (if any)
                bool builtin = false;
                if (llm_params.chat_template.empty()) {
                    llm_params.chat_template = load_chat_template();
                    builtin                  = true;
                }
                if (llm_params.chat_template.size() <= 20) {
                    for (char &c : llm_params.chat_template) {
                        c = char(std::tolower(c));
                    }
                }
                // NB(thxCode): llama_chat_template_alias is a patch.
                chat_template_alias = llama_chat_template_alias(llm_params.chat_template.c_str());
                if (chat_template_alias == "chatml") {
                    // <tool_call>
                    // {"name":"","arguments":{}}
                    // </tool_call>
                    support_tool_calls   = true;
                    tool_call_start      = "<tool_call>";
                    tool_call_start_tok  = LLAMA_TOKEN_NULL;
                    tool_call_start_trim = true;
                    tool_call_ends       = {"</tool_call>", "</tool_call>\n"};
                    tool_call_end_tok    = LLAMA_TOKEN_NULL;
                    tool_call_end_trim   = true;
                } else if (starts_with(chat_template_alias, "mistral-")) {
                    // [TOOL_CALLS][{"name":"","arguments":{}}]
                    support_tool_calls   = true;
                    tool_call_start      = "";
                    tool_call_start_tok  = 9; // [TOOL_CALLS]
                    tool_call_start_trim = false;
                    tool_call_ends       = {};
                    tool_call_end_tok    = llama_token_eos(llm_model);
                    tool_call_end_trim   = false;
                } else if (chat_template_alias == "llama3") {
                    // {"name":"","arguments":{}}
                    support_tool_calls   = true;
                    tool_call_start      = "{\"";
                    tool_call_start_tok  = LLAMA_TOKEN_NULL;
                    tool_call_start_trim = false;
                    tool_call_ends       = {"}}", "}} "};
                    tool_call_end_tok    = LLAMA_TOKEN_NULL;
                    tool_call_end_trim   = false;
                } else if (chat_template_alias == "granite") {
                    // <tool_call>[{"name":"","arguments":{}}]
                    support_tool_calls   = true;
                    tool_call_start      = "<tool_call>";
                    tool_call_start_tok  = LLAMA_TOKEN_NULL;
                    tool_call_start_trim = true;
                    tool_call_ends       = {"}]", "}] "};
                    tool_call_end_tok    = LLAMA_TOKEN_NULL;
                    tool_call_end_trim   = false;
                }
                SRV_INF("chat template, built_in: %s, alias: %s, tool call: %s, example:\n%s\n",
                        builtin ? "true" : "false",
                        chat_template_alias.c_str(),
                        support_tool_calls ? "supported" : "unsupported",
                        common_chat_format_example(llm_model, llm_params.chat_template).c_str());
            } else {
                SRV_INF("%s", "chat template is disabled\n");
            }
        }

        return true;
    }

    std::string load_chat_template() const {
        std::string tkey = "tokenizer.chat_template";
        int32_t tlen     = llama_model_meta_val_str(llm_model, tkey.c_str(), nullptr, 0);
        if (tlen > 0) {
            std::vector<char> tval(tlen + 1, 0);
            if (llama_model_meta_val_str(llm_model, tkey.c_str(), tval.data(), tlen + 1) == tlen) {
                return {tval.data(), (unsigned long)tlen};
            }
        }
        return "chatml"; // see llama_chat_apply_template_internal
    }

    bool init() {
        SRV_INF("initializing slots, n_slots = %d\n", llm_params.n_parallel);

        const int32_t n_ctx_slot = llm_ctx ? int32_t(llama_n_ctx(llm_ctx)) / llm_params.n_parallel : 0;
        for (int i = 0; i < llm_params.n_parallel; i++) {
            server_slot slot;

            slot.id                = i;
            slot.n_ctx             = n_ctx_slot;
            slot.n_predict         = llm_params.n_predict;
            slot.params.llm_params = llm_params.sampling;
            slot.params.sd_params  = sd_params.sampling;

            SLT_INF(slot, "new slot n_ctx_slot = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int) { queue_tasks.pop_deferred_task(); };

            slot.reset();

            slots.push_back(slot);
        }

        default_generation_settings_for_props         = get_formated_generation(slots.front());
        default_generation_settings_for_props["seed"] = -1;

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel
        // tokens note that n_batch can be > n_ctx (e.g. for non-causal
        // attention models such as BERT where the KV cache is not used)
        if (llm_ctx != nullptr) {
            auto n_batch = int32_t(llama_n_batch(llm_ctx));
            batch        = llama_batch_init(std::max(n_batch, llm_params.n_parallel), 0, 1);
            if (llm_ctx_draft != nullptr) {
                batch_draft = llama_batch_init(std::max(n_batch, llm_params.n_parallel), 0, 1);
            }
        }

        metrics.init();
        return true;
    }

    void clean(httplib::Server &svr) {
        svr.stop();
        if (params.lookup_ngram_min > 0 && !llm_params.lookup_cache_dynamic.empty()) {
            common_ngram_cache_save(ngram_cache_dynamic, llm_params.lookup_cache_dynamic);
        }
        llama_backend_free();
    }

    server_slot *get_slot_by_id(int id) {
        for (server_slot &slot : slots) {
            if (slot.id == id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot *get_available_slot(const server_task &task) {
        server_slot *ret = nullptr;

        // find the slot that has at least n% prompt similarity
        if (llm_params.slot_prompt_similarity != 0.0f) {
            int lcs_len      = 0;
            float similarity = 0;

            for (server_slot &slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // skip the slot if it does not contain cached tokens
                if (slot.cache_tokens.empty()) {
                    continue;
                }

                // length of the Longest Common Subsequence between the current slot's prompt and the input prompt
                int cur_lcs_len = int32_t(common_lcs(slot.cache_tokens, task.prompt_tokens));

                // fraction of the common subsequence length compared to the current slot's prompt length
                float cur_similarity = static_cast<float>(cur_lcs_len) / static_cast<float>(slot.cache_tokens.size());

                // select the current slot if the criteria match
                if (cur_lcs_len > lcs_len && cur_similarity > llm_params.slot_prompt_similarity) {
                    lcs_len    = cur_lcs_len;
                    similarity = cur_similarity;
                    ret        = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_DBG(*ret, "selected slot by lcs similarity, lcs_len = %d, similarity = %f\n", lcs_len, similarity);
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = ggml_time_us();
            for (server_slot &slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // select the current slot if the criteria match
                if (slot.t_last_used < t_last) {
                    t_last = slot.t_last_used;
                    ret    = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_DBG(*ret, "selected slot by lru, t_last = %" PRId64 "\n", t_last);
            }
        }

        return ret;
    }

    bool launch_slot_with_task(server_slot &slot, const server_task &task) {
        // sampling parameter defaults are loaded from the global server context (but individual requests can still override them)
        slot_params defaults;
        defaults.llm_params = llm_params.sampling;
        defaults.sd_params  = sd_params.sampling;

        const json &data = task.data;

        /* STABLE DIFFUSION */

        if (sd_ctx != nullptr) {
            slot.oaicompat_image          = true;
            slot.oaicompat_image_generate = json_value(data, "__oaicompat_image_generate", false);
            slot.oaicompat_image_edit     = json_value(data, "__oaicompat_image_edit", false);

            slot.params.stream = json_value(data, "stream", false);
            if (data.contains("stream_options")) {
                slot.params.stream_preview        = json_value(data.at("stream_options"), "preview", false);
                slot.params.stream_preview_faster = json_value(data.at("stream_options"), "preview_faster", false);
            }

            slot.params.sd_params                  = defaults.sd_params;
            slot.params.sd_params.seed             = json_value(data, "seed", sd_params.seed);
            slot.params.sd_params.height           = json_value(data, "height", defaults.sd_params.height);
            slot.params.sd_params.width            = json_value(data, "width", defaults.sd_params.width);
            slot.params.sd_params.guidance         = json_value(data, "guidance", defaults.sd_params.guidance);
            slot.params.sd_params.strength         = json_value(data, "strength", defaults.sd_params.strength);
            slot.params.sd_params.sample_method    = json_value(data, "sample_method", defaults.sd_params.sample_method);
            slot.params.sd_params.sampling_steps   = json_value(data, "sampling_steps", defaults.sd_params.sampling_steps);
            slot.params.sd_params.cfg_scale        = json_value(data, "cfg_scale", defaults.sd_params.cfg_scale);
            slot.params.sd_params.slg_scale        = json_value(data, "slg_scale", defaults.sd_params.slg_scale);
            slot.params.sd_params.slg_skip_layers  = json_value(data, "slg_skip_layers", defaults.sd_params.slg_skip_layers);
            slot.params.sd_params.slg_start        = json_value(data, "slg_start", defaults.sd_params.slg_start);
            slot.params.sd_params.slg_end          = json_value(data, "slg_end", defaults.sd_params.slg_end);
            slot.params.sd_params.schedule_method  = json_value(data, "schedule_method", defaults.sd_params.schedule_method);
            slot.params.sd_params.control_strength = json_value(data, "control_strength", defaults.sd_params.control_strength);
            slot.params.sd_params.control_canny    = json_value(data, "control_strength", defaults.sd_params.control_canny);
            slot.params.sd_params.negative_prompt  = json_value(data, "negative_prompt", std::string(""));

            // get prompt
            slot.prompt_string = json_value(data, "prompt", std::string(""));

            // get image
            if (slot.oaicompat_image_edit) {
                uint8_t *control_img_buffer = nullptr;
                if (data.contains("control")) {
                    int cc           = 0;
                    int cw           = 0;
                    int ch           = 0;
                    auto control_img = data.at("control").get<std::string>();
                    SLT_INF(slot, "loading control: %zu\n", control_img.length());
                    control_img_buffer = stbi_load_from_memory((const stbi_uc *)control_img.c_str(), (int)control_img.length(), &cw, &ch, &cc, 3);
                    if (control_img_buffer == nullptr) {
                        auto reason = stbi_failure_reason();
                        SLT_ERR(slot, "failed to load control: %s\n", reason);
                        send_error(task, "failed to load control", ERROR_TYPE_INVALID_REQUEST);
                        return false;
                    }
                    if (cw <= 0 || ch <= 0) {
                        send_error(task, "control width and height cannot be zero", ERROR_TYPE_INVALID_REQUEST);
                        return false;
                    }
                    SLT_WRN(slot, "control changes width and height from %dx%d to %dx%d\n", slot.params.sd_params.width, slot.params.sd_params.height, cw, ch);
                    slot.params.sd_params.height = ch;
                    slot.params.sd_params.width  = cw;
                }
#define free_images_0                        \
    if (control_img_buffer != nullptr) {     \
        stbi_image_free(control_img_buffer); \
    }
                uint8_t *init_img_buffer = nullptr;
                int iw                   = 0;
                int ih                   = 0;
                int ic                   = 0;
                auto init_img            = data.at("image").get<std::string>();
                SLT_INF(slot, "loading image: %zu\n", init_img.length());
                init_img_buffer = stbi_load_from_memory((const stbi_uc *)init_img.c_str(), (int)init_img.length(), &iw, &ih, &ic, 3);
                if (init_img_buffer == nullptr) {
                    free_images_0;
                    auto reason = stbi_failure_reason();
                    SLT_ERR(slot, "failed to load image: %s\n", reason);
                    send_error(task, "failed to load image", ERROR_TYPE_INVALID_REQUEST);
                    return false;
                }
#define free_images_1 \
    free_images_0;    \
    stbi_image_free(init_img_buffer);
                if (ic < 3) {
                    free_images_1;
                    send_error(task, "image must be at least 3 channels", ERROR_TYPE_INVALID_REQUEST);
                    return false;
                }
                if (iw <= 0 || ih <= 0) {
                    free_images_1;
                    send_error(task, "image width and height cannot be zero", ERROR_TYPE_INVALID_REQUEST);
                    return false;
                }
                if (iw != slot.params.sd_params.width || ih != slot.params.sd_params.height) {
                    LOG_INF("image dimensions do not match, resizing image\n");
                    int rw                     = slot.params.sd_params.width;
                    int rh                     = slot.params.sd_params.height;
                    auto *resized_image_buffer = (uint8_t *)malloc(rw * rh * 3);
                    if (resized_image_buffer == nullptr) {
                        free_images_1;
                        send_error(task, "failed to create resized image buffer", ERROR_TYPE_INVALID_REQUEST);
                        return false;
                    }
                    if (!stbir_resize(init_img_buffer, iw, ih, 0,
                                      resized_image_buffer, rw, rh, 0, STBIR_TYPE_UINT8,
                                      3 /*RGB channel*/, STBIR_ALPHA_CHANNEL_NONE, 0,
                                      STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
                                      STBIR_FILTER_BOX, STBIR_FILTER_BOX,
                                      STBIR_COLORSPACE_SRGB, nullptr)) {
                        auto reason = stbi_failure_reason();
                        SLT_ERR(slot, "failed to resize image: %s\n", reason);
                        free_images_1;
                        send_error(task, "failed to resize image", ERROR_TYPE_INVALID_REQUEST);
                        return false;
                    }
                    stbi_image_free(init_img_buffer);
                    init_img_buffer = resized_image_buffer;
                }

                uint8_t *mask_img_buffer = nullptr;
                if (data.contains("mask")) {
                    int mw        = 0;
                    int mh        = 0;
                    int mc        = 0;
                    auto mask_img = data.at("mask").get<std::string>();
                    SLT_INF(slot, "loading mask: %zu\n", mask_img.length());
                    mask_img_buffer = stbi_load_from_memory((const stbi_uc *)mask_img.c_str(), (int)mask_img.length(), &mw, &mh, &mc, 1);
                    if (mask_img_buffer == nullptr) {
                        free_images_1;
                        auto reason = stbi_failure_reason();
                        SLT_ERR(slot, "failed to load mask: %s\n", reason);
                        send_error(task, "failed to load mask", ERROR_TYPE_INVALID_REQUEST);
                        return false;
                    }
#define free_images_2 \
    free_images_1;    \
    stbi_image_free(mask_img_buffer);
                    if (mc < 1) {
                        free_images_2;
                        send_error(task, "mask must be at least 1 channels", ERROR_TYPE_INVALID_REQUEST);
                        return false;
                    }
                    if (mw <= 0 || mh <= 0) {
                        free_images_2;
                        send_error(task, "mask width and height cannot be zero", ERROR_TYPE_INVALID_REQUEST);
                        return false;
                    }
                    if (mw != slot.params.sd_params.width || mh != slot.params.sd_params.height) {
                        LOG_INF("mask dimensions do not match, resizing image\n");
                        int rw                    = slot.params.sd_params.width;
                        int rh                    = slot.params.sd_params.height;
                        auto *resized_mask_buffer = (uint8_t *)malloc(rw * rh * 1);
                        if (resized_mask_buffer == nullptr) {
                            free_images_2;
                            send_error(task, "failed to create resized mask buffer", ERROR_TYPE_INVALID_REQUEST);
                            return false;
                        }
                        if (!stbir_resize(mask_img_buffer, mw, mh, 0,
                                          resized_mask_buffer, rw, rh, 0, STBIR_TYPE_UINT8,
                                          1 /*RGB channel*/, STBIR_ALPHA_CHANNEL_NONE, 0,
                                          STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
                                          STBIR_FILTER_BOX, STBIR_FILTER_BOX,
                                          STBIR_COLORSPACE_SRGB, nullptr)) {
                            auto reason = stbi_failure_reason();
                            SLT_ERR(slot, "failed to resize mask: %s\n", reason);
                            free_images_2;
                            send_error(task, "failed to resize mask", ERROR_TYPE_INVALID_REQUEST);
                            return false;
                        }
                        stbi_image_free(mask_img_buffer);
                        mask_img_buffer = resized_mask_buffer;
                    }
                } else {
                    mask_img_buffer = (uint8_t *)malloc(slot.params.sd_params.width * slot.params.sd_params.height * 1);
                    if (mask_img_buffer == nullptr) {
                        free_images_1;
                        send_error(task, "failed to create mask buffer", ERROR_TYPE_INVALID_REQUEST);
                        return false;
                    }
                    memset(mask_img_buffer, 255, slot.params.sd_params.width * slot.params.sd_params.height * 1);
                }

                slot.params.sd_params.control_img_buffer = control_img_buffer;
                slot.params.sd_params.init_img_buffer    = init_img_buffer;
                slot.params.sd_params.mask_img_buffer    = mask_img_buffer;
#undef free_images_0
#undef free_images_1
#undef free_images_2
            }

            slot.state = SLOT_STATE_STARTED;

            SLT_INF(slot, "processing task, seed: %d, guidance: %f, strength: %f, sample method: %s, sampling steps: %d, cfg scale: %.2f, slg scale: %.2f, schedule method: %s\n",
                    slot.params.sd_params.seed,
                    slot.params.sd_params.guidance,
                    slot.params.sd_params.strength,
                    sd_sample_method_to_argument(slot.params.sd_params.sample_method),
                    slot.params.sd_params.sampling_steps,
                    slot.params.sd_params.cfg_scale,
                    slot.params.sd_params.slg_scale,
                    sd_schedule_to_argument(slot.params.sd_params.schedule_method));

            return true;
        }

        /* LLAMA */

        slot.oaicompat_completion             = json_value(data, "__oaicompat_completion", false);
        slot.oaicompat_completion_chat        = json_value(data, "__oaicompat_completion_chat", false);
        slot.oaicompat_completion_chat_tool   = json_value(data, "__oaicompat_completion_chat_tool", false) && support_tool_calls;
        slot.oaicompat_completion_chat_vision = json_value(data, "__oaicompat_completion_chat_vision", false) && llm_ctx_clip != nullptr;

        slot.params.stream = json_value(data, "stream", false);

        slot.params.return_tokens = json_value(data, "return_tokens", false);
        slot.params.n_predict     = json_value(data, "n_predict", llm_params.n_predict);
        slot.params.n_indent      = json_value(data, "n_indent", 0);
        slot.params.n_keep        = json_value(data, "n_keep", llm_params.n_keep);
        slot.params.n_discard     = json_value(data, "n_discard", 0);
        // slot.params.t_max_prompt_ms    = json_value(data, "t_max_prompt_ms",   -1); // TODO: implement
        slot.params.t_max_predict_ms = json_value(data, "t_max_predict_ms", -1);

        slot.params.llm_params                    = defaults.llm_params;
        slot.params.llm_params.top_k              = json_value(data, "top_k", defaults.llm_params.top_k);
        slot.params.llm_params.top_p              = json_value(data, "top_p", defaults.llm_params.top_p);
        slot.params.llm_params.min_p              = json_value(data, "min_p", defaults.llm_params.min_p);
        slot.params.llm_params.xtc_probability    = json_value(data, "xtc_probability", defaults.llm_params.xtc_probability);
        slot.params.llm_params.xtc_threshold      = json_value(data, "xtc_threshold", defaults.llm_params.xtc_threshold);
        slot.params.llm_params.typ_p              = json_value(data, "typical_p", defaults.llm_params.typ_p);
        slot.params.llm_params.temp               = json_value(data, "temperature", defaults.llm_params.temp);
        slot.params.llm_params.dynatemp_range     = json_value(data, "dynatemp_range", defaults.llm_params.dynatemp_range);
        slot.params.llm_params.dynatemp_exponent  = json_value(data, "dynatemp_exponent", defaults.llm_params.dynatemp_exponent);
        slot.params.llm_params.penalty_last_n     = json_value(data, "repeat_last_n", defaults.llm_params.penalty_last_n);
        slot.params.llm_params.penalty_repeat     = json_value(data, "repeat_penalty", defaults.llm_params.penalty_repeat);
        slot.params.llm_params.penalty_freq       = json_value(data, "frequency_penalty", defaults.llm_params.penalty_freq);
        slot.params.llm_params.penalty_present    = json_value(data, "presence_penalty", defaults.llm_params.penalty_present);
        slot.params.llm_params.dry_multiplier     = json_value(data, "dry_multiplier", defaults.llm_params.dry_multiplier);
        slot.params.llm_params.dry_base           = json_value(data, "dry_base", defaults.llm_params.dry_base);
        slot.params.llm_params.dry_allowed_length = json_value(data, "dry_allowed_length", defaults.llm_params.dry_allowed_length);
        slot.params.llm_params.dry_penalty_last_n = json_value(data, "dry_penalty_last_n", defaults.llm_params.dry_penalty_last_n);
        slot.params.llm_params.mirostat           = json_value(data, "mirostat", defaults.llm_params.mirostat);
        slot.params.llm_params.mirostat_tau       = json_value(data, "mirostat_tau", defaults.llm_params.mirostat_tau);
        slot.params.llm_params.mirostat_eta       = json_value(data, "mirostat_eta", defaults.llm_params.mirostat_eta);
        slot.params.llm_params.seed               = json_value(data, "seed", defaults.llm_params.seed);
        slot.params.llm_params.n_probs            = json_value(data, "n_probs", defaults.llm_params.n_probs);
        slot.params.llm_params.min_keep           = json_value(data, "min_keep", defaults.llm_params.min_keep);

        // process "llm_params" parameters
        if (slot.params.llm_params.penalty_last_n < -1) {
            send_error(task, "Illegal param: repeat_last_n must be >= -1", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        if (slot.params.llm_params.dry_penalty_last_n < -1) {
            send_error(task, "Illegal param: dry_penalty_last_n must be >= -1", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        if (slot.params.llm_params.penalty_last_n == -1) {
            // note: should be the slot's context and not the full context, but it's ok
            slot.params.llm_params.penalty_last_n = llama_n_ctx(llm_ctx);
        }
        if (slot.params.llm_params.dry_penalty_last_n == -1) {
            slot.params.llm_params.dry_penalty_last_n = llama_n_ctx(llm_ctx);
        }
        if (slot.params.llm_params.dry_base < 1.0f) {
            slot.params.llm_params.dry_base = defaults.llm_params.dry_base;
        }

        // process "json_schema" and "grammar"
        if (data.contains("json_schema") && !data.at("json_schema").is_null() && data.contains("grammar") && !data.at("grammar").is_null()) {
            send_error(task, R"(either "json_schema" or "grammar" can be specified, but not both)", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        if (data.contains("json_schema") && !data.contains("grammar")) {
            try {
                auto schema                    = json_value(data, "json_schema", json::object());
                slot.params.llm_params.grammar = json_schema_to_grammar(schema);
            } catch (const std::exception &e) {
                send_error(task, std::string("\"json_schema\": ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
        } else {
            slot.params.llm_params.grammar = json_value(data, "grammar", defaults.llm_params.grammar);
        }

        // process "n_predict"
        if (slot.n_predict > 0 && slot.params.n_predict > slot.n_predict) {
            // Might be better to reject the request with a 400 ?
            slot.params.n_predict = slot.n_predict;
            SLT_WRN(slot, "n_predict = %d exceeds server configuration, setting to %d", slot.n_predict, slot.n_predict);
        }

        // get prompt
        {
            if (slot.oaicompat_completion_chat_vision) {
                slot.prompt_string           = data.at("prompt").get<std::string>();
                slot.prompt_multi_modal_data = data.at("multi_modal_data");
            } else {
                slot.prompt_tokens = task.prompt_tokens;
            }
        }

        {
            slot.params.llm_params.logit_bias.clear();

            if (json_value(data, "ignore_eos", false) && llama_token_eos(llm_model) != LLAMA_TOKEN_NULL) {
                slot.params.llm_params.logit_bias.push_back({llama_token_eos(llm_model), -INFINITY});
            }

            const auto &logit_bias = data.find("logit_bias");
            if (logit_bias != data.end() && logit_bias->is_array()) {
                const int n_vocab = llama_n_vocab(llm_model);
                for (const auto &el : *logit_bias) {
                    // TODO: we may want to throw errors here, in case "el" is
                    // incorrect
                    if (el.is_array() && el.size() == 2) {
                        float bias;
                        if (el[1].is_number()) {
                            bias = el[1].get<float>();
                        } else if (el[1].is_boolean() && !el[1].get<bool>()) {
                            bias = -INFINITY;
                        } else {
                            continue;
                        }

                        if (el[0].is_number_integer()) {
                            llama_token tok = el[0].get<llama_token>();
                            if (tok >= 0 && tok < n_vocab) {
                                slot.params.llm_params.logit_bias.push_back({tok, bias});
                            }
                        } else if (el[0].is_string()) {
                            auto toks = common_tokenize(llm_model, el[0].get<std::string>(), false);
                            for (auto tok : toks) {
                                slot.params.llm_params.logit_bias.push_back({tok, bias});
                            }
                        }
                    }
                }
            }
        }

        {
            slot.params.antiprompt.clear();

            const auto &stop = data.find("stop");
            if (stop != data.end() && stop->is_array()) {
                for (const auto &word : *stop) {
                    if (!word.empty()) {
                        slot.params.antiprompt.push_back(word);
                    }
                }
            }
        }

        {
            const auto &samplers = data.find("samplers");
            if (samplers != data.end()) {
                if (samplers->is_array()) {
                    std::vector<std::string> sampler_names;
                    for (const auto &name : *samplers) {
                        if (name.is_string()) {
                            sampler_names.emplace_back(name);
                        }
                    }
                    slot.params.llm_params.samplers = common_sampler_types_from_names(sampler_names, false);
                } else if (samplers->is_string()) {
                    std::string sampler_string;
                    for (const auto &name : *samplers) {
                        sampler_string += name;
                    }
                    slot.params.llm_params.samplers = common_sampler_types_from_chars(sampler_string);
                }
            } else {
                slot.params.llm_params.samplers = llm_params.sampling.samplers;
            }
        }

        {
            if (slot.smpl != nullptr) {
                common_sampler_free(slot.smpl);
            }

            slot.smpl = common_sampler_init(llm_model, slot.params.llm_params);
            if (slot.smpl == nullptr) {
                // for now, the only error that may happen here is invalid
                // grammar
                send_error(task, "failed to parse grammar", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            if (llm_ctx_draft != nullptr) {
                if (slot.smpl_draft != nullptr) {
                    common_sampler_free(slot.smpl_draft);
                }

                struct common_params_sampling params_draft = slot.params.llm_params;
                params_draft.top_k                         = 10;
                params_draft.samplers                      = {
                    COMMON_SAMPLER_TYPE_TOP_K,
                };
                slot.smpl_draft = common_sampler_init(llm_model_draft, params_draft);
                if (slot.smpl_draft == nullptr) {
                    // for now, the only error that may happen here is invalid grammar
                    send_error(task, "failed to parse grammar", ERROR_TYPE_INVALID_REQUEST);
                    return false;
                }
            }
        }

        {
            if (slot.token_bkt != nullptr) {
                delete slot.token_bkt;
                slot.token_bkt = nullptr;
            }
            if (task.tps > 0) {
                slot.token_bkt = new token_bucket(task.tps, task.tps);
                if (slot.token_bkt == nullptr) {
                    send_error(task, "failed to create token bucket", ERROR_TYPE_SERVER);
                    return false;
                }
            }
        }

        if (params.lookup_ngram_min > 0) {
            slot.lookup_ngram_min = params.lookup_ngram_min;
            if (!slot.ctx_ngram_cache.empty()) {
                common_ngram_cache_merge(ngram_cache_dynamic, slot.ctx_ngram_cache);
                slot.ctx_ngram_cache.clear();
                const auto sz_ngram_cache = int32_t(ngram_cache_dynamic.size());
                if (sz_ngram_cache >= 100000) {
                    if (!llm_params.lookup_cache_dynamic.empty()) {
                        common_ngram_cache_save(ngram_cache_dynamic, llm_params.lookup_cache_dynamic);
                    }
                    // shuffle the keys to avoid always using the same ngrams
                    std::vector<common_ngram> keys;
                    for (const auto &pair : ngram_cache_dynamic) {
                        keys.push_back(pair.first);
                    }
                    std::random_device rd;
                    std::mt19937 g(rd());
                    std::shuffle(keys.begin(), keys.end(), g);
                    // erase to avoid memory increase
                    const auto sz_ngram_cache_evicts = sz_ngram_cache - 80000;
                    for (int32_t i = 0; i < sz_ngram_cache_evicts; ++i) {
                        ngram_cache_dynamic.erase(keys[i]);
                    }
                }
            }
        }

        {
            slot.parallel_tool_calls = json_value(data, "parallel_tool_calls", true);
        }

        slot.state = SLOT_STATE_STARTED;

        SLT_INF(slot, "processing task, max_tps = %s\n", slot.token_bkt ? std::to_string(slot.token_bkt->capacity).c_str() : "N/A");

        return true;
    }

    bool process_token(completion_token_output &result, server_slot &slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        slot.sampled.clear();
        std::string token_str;
        for (const llama_token &tok : result.toks) {
            token_str += common_token_to_piece(llm_ctx, tok, llm_params.special);
            slot.sampled.push_back(tok);
        }

        // search stop word and delete it
        slot.generated_text += token_str;
        if (slot.params.return_tokens) {
            for (const llama_token &tok : result.toks) {
                slot.generated_tokens.push_back(tok);
            }
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text             = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(slot.generated_text.begin() + long(pos) + long(stop_pos), slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token) {
                stop_pos  = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            if (send_text && slot.oaicompat_completion_chat_tool) {
                if (slot.tool_call_parser == TOOL_CALL_PARSER_TYPE_NONE) {
                    if (!tool_call_start.empty()) {
                        if (str_test.length() < tool_call_start.length()) {
                            send_text = false;
                            if (tool_call_start_tok != LLAMA_TOKEN_NULL) {
                                for (const llama_token &tok : result.toks) {
                                    if (tok == tool_call_start_tok) {
                                        slot.tool_call_parser = TOOL_CALL_PARSER_TYPE_TOKEN;
                                        break;
                                    }
                                }
                            }
                        } else if (starts_with(str_test, tool_call_start)) {
                            send_text             = false;
                            slot.tool_call_parser = TOOL_CALL_PARSER_TYPE_STRING;
                        }
                    } else if (tool_call_start_tok != LLAMA_TOKEN_NULL) {
                        for (const llama_token &tok : result.toks) {
                            if (tok == tool_call_start_tok) {
                                slot.tool_call_parser = TOOL_CALL_PARSER_TYPE_TOKEN;
                                break;
                            }
                        }
                    }
                } else if (slot.tool_call_parser != TOOL_CALL_PARSER_TYPE_NONE) {
                    send_text = false;
                    std::string functions_str;
                    if (slot.tool_call_parser == TOOL_CALL_PARSER_TYPE_STRING) {
                        for (const auto &tool_call_end : tool_call_ends) {
                            if (!ends_with(str_test, tool_call_end)) {
                                continue;
                            }
                            functions_str = str_test;
                            if (tool_call_start_trim) {
                                functions_str = functions_str.substr(tool_call_start.length());
                            }
                            if (tool_call_end_trim) {
                                functions_str = functions_str.substr(0, functions_str.length() - tool_call_end.length());
                            }
                            break;
                        }
                    } else if (slot.tool_call_parser == TOOL_CALL_PARSER_TYPE_TOKEN && tool_call_end_tok != LLAMA_TOKEN_NULL) {
                        for (auto i = int(result.toks.size()) - 1; i >= 0; --i) {
                            if (result.toks[i] == tool_call_end_tok) {
                                functions_str = str_test;
                                break;
                            }
                        }
                    }
                    if (!functions_str.empty()) {
                        slot.tool_call_parser = TOOL_CALL_PARSER_TYPE_NONE;
                        try {
                            auto append_tool_calls = [&](json &function) {
                                if (!function.is_object()) {
                                    throw std::runtime_error("function is an object");
                                }
                                if (!function.contains("name")) {
                                    throw std::runtime_error("function does not contain \"name\" field");
                                }
                                if (!function.contains("arguments")) {
                                    throw std::runtime_error("function does not contain \"arguments\" field");
                                }
                                if (!function.at("arguments").is_string()) {
                                    function["arguments"] = function.at("arguments").dump(-1, ' ', false, json::error_handler_t::replace);
                                }
                                json tool_call = json{
                                    {"type", "function"},
                                    {"function", function},
                                };
                                if (tool_call_id_generate) {
                                    tool_call["id"] = gen_callid();
                                }
                                slot.generated_tool_calls.push_back(tool_call);
                            };
                            json functions = json::parse(functions_str);
                            if (functions.is_array()) {
                                for (auto &function : functions) {
                                    append_tool_calls(function);
                                }
                            } else {
                                append_tool_calls(functions);
                            }
                            if (!slot.parallel_tool_calls) {
                                slot.stop           = STOP_TYPE_TOOL;
                                slot.has_next_token = false;
                            }
                            // TODO(thxCode): duplicate code for now, will refactor later
                            result.text_to_send = slot.generated_text.substr(pos, stop_pos);
                            slot.n_sent_text += result.text_to_send.size();
                        } catch (const std::exception &e) {
                            SLT_ERR(slot, "failed to parse tool call: %s, fallback\n", e.what());
                        }
                    }
                }
            }

            // check if the last token is EOG
            if (!send_text && slot.oaicompat_completion_chat_tool && slot.generated_tool_calls.empty()) {
                send_text = llama_token_is_eog(llm_model, result.toks[result.toks.size() - 1]);
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, stop_pos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            }

            slot.add_token(result);
            if (send_text && slot.params.stream) {
                send_partial_completion(slot, result);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // check the limits
        if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(llm_params)) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_decoded = %d, n_predict = %d\n", slot.n_decoded, slot.params.n_predict);
        }

        if (slot.has_new_line) {
            // if we have already seen a new line, we stop after a certain time limit
            if (slot.params.t_max_predict_ms > 0 && (ggml_time_us() - slot.t_start_generation > 1000.0f * slot.params.t_max_predict_ms)) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_decoded = %d, t_max_predict_ms = %d ms\n", slot.n_decoded, (int)slot.params.t_max_predict_ms);
            }

            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_decoded = %d, n_indent = %d\n", slot.n_decoded, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);
                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;
        }

        // if context shift is disabled, we stop when it reaches the context limit
        if (!llm_params.ctx_shift && slot.n_past >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, n_past = %d, n_prompt_tokens = %d, n_decoded = %d, n_ctx = %d\n",
                    slot.n_past, slot.n_prompt_tokens, slot.n_decoded, slot.n_ctx);
        }

        // check the EOT
        if (llama_token_is_eog(llm_model, result.toks[result.toks.size() - 1])) {
            slot.stop           = slot.generated_tool_calls.empty() ? STOP_TYPE_EOS : STOP_TYPE_TOOL;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        int32_t n_ctx_train = llama_n_ctx_train(llm_model);
        if (slot.params.n_predict < 1 && slot.n_predict < 1 && slot.n_prompt_tokens + slot.n_decoded >= n_ctx_train) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false; // stop prediction

            SLT_WRN(slot,
                    "n_predict (%d) is set for infinite generation, "
                    "limiting generated tokens to n_ctx_train (%d) to avoid EOS-less generation infinite loop\n",
                    slot.params.n_predict, n_ctx_train);
        }

        return slot.has_next_token; // continue
    }

    json get_formated_generation(const server_slot &slot) const {
        std::vector<std::string> samplers;
        samplers.reserve(slot.params.llm_params.samplers.size());
        for (const auto &sampler : slot.params.llm_params.samplers) {
            samplers.emplace_back(common_sampler_type_to_str(sampler));
        }

        /* STABLE DIFFUSION */

        if (sd_ctx != nullptr) {
            return json{
                {"model", sd_params.model_alias},
                {"seed", sd_params.seed},
                {"seed_cur", slot.params.sd_params.seed},
                {"max_batch_count", sd_params.max_batch_count},
                {"max_height", sd_params.sampling.height},
                {"max_width", sd_params.sampling.width},
                {"height", slot.params.sd_params.height},
                {"width", slot.params.sd_params.width},
                {"guidance", slot.params.sd_params.guidance},
                {"sample_method", sd_sample_method_to_argument(slot.params.sd_params.sample_method)},
                {"sampling_steps", slot.params.sd_params.sampling_steps},
                {"cfg_scale", slot.params.sd_params.cfg_scale},
                {"slg_scale", slot.params.sd_params.slg_scale},
                {"slg_skip_layers", slot.params.sd_params.slg_skip_layers},
                {"slg_start", slot.params.sd_params.slg_start},
                {"slg_end", slot.params.sd_params.slg_end},
                {"schedule_method", sd_schedule_to_argument(slot.params.sd_params.schedule_method)},
                {"clip_l_model", sd_params.clip_l_model},
                {"clip_g_model", sd_params.clip_g_model},
                {"t5xxl_model", sd_params.t5xxl_model},
                {"vae_model", sd_params.vae_model},
                {"vae_tiling", sd_params.vae_tiling},
                {"taesd_model", sd_params.taesd_model},
                {"upscale_model", sd_params.upscale_model},
                {"upscale_repeats", sd_params.upscale_repeats},
                {"control_net_model", sd_params.control_net_model},
                {"control_strength", slot.params.sd_params.control_strength},
                {"control_canny", slot.params.sd_params.control_canny},
            };
        }

        /* LLAMA */

        return json{
            {"n_ctx", slot.n_ctx},
            {"n_predict", slot.n_predict}, // Server configured n_predict
            {"model", llm_params.model_alias},
            {"seed", slot.params.llm_params.seed},
            {"seed_cur", slot.smpl ? common_sampler_get_seed(slot.smpl) : 0},
            {"temperature", slot.params.llm_params.temp},
            {"dynatemp_range", slot.params.llm_params.dynatemp_range},
            {"dynatemp_exponent", slot.params.llm_params.dynatemp_exponent},
            {"top_k", slot.params.llm_params.top_k},
            {"top_p", slot.params.llm_params.top_p},
            {"min_p", slot.params.llm_params.min_p},
            {"xtc_probability", slot.params.llm_params.xtc_probability},
            {"xtc_threshold", slot.params.llm_params.xtc_threshold},
            {"typical_p", slot.params.llm_params.typ_p},
            {"repeat_last_n", slot.params.llm_params.penalty_last_n},
            {"repeat_penalty", slot.params.llm_params.penalty_repeat},
            {"presence_penalty", slot.params.llm_params.penalty_present},
            {"frequency_penalty", slot.params.llm_params.penalty_freq},
            {"mirostat", slot.params.llm_params.mirostat},
            {"mirostat_tau", slot.params.llm_params.mirostat_tau},
            {"mirostat_eta", slot.params.llm_params.mirostat_eta},
            {"stop", slot.params.antiprompt},
            {"max_tokens", slot.params.n_predict}, // User configured n_predict
            {"n_keep", slot.params.n_keep},
            {"n_discard", slot.params.n_discard},
            {"ignore_eos", slot.params.llm_params.ignore_eos},
            {"stream", slot.params.stream},
            {"n_probs", slot.params.llm_params.n_probs},
            {"min_keep", slot.params.llm_params.min_keep},
            {"grammar", slot.params.llm_params.grammar},
            {"samplers", samplers},
            {"speculative", llm_ctx_draft != nullptr},
        };
    }

    void send_error(const server_task &task, const std::string &error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot &slot, const std::string &error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.id_task, error, type);
    }

    void send_error(const int id_task, const std::string &error, const enum error_type type = ERROR_TYPE_SERVER) {
        SRV_ERR("task %d | error = %s\n", id_task, error.c_str());

        server_task_result res;
        res.id    = id_task;
        res.stop  = false;
        res.error = true;
        res.data  = format_error_response(error, type);

        queue_results.send(res);
    }

    void send_partial_completion(server_slot &slot, completion_token_output tkn) {
        server_task_result res;
        res.id    = slot.id_task;
        res.error = false;
        res.stop  = false;
        res.data  = json{
             {"content", tkn.text_to_send},
             {"tokens", tkn.toks},
             {"id_slot", slot.id},
             {"index", slot.index},
             {"stop", false},
             {"model", llm_params.model_alias},
        };

        if (slot.params.llm_params.n_probs > 0) {
            const llama_tokens to_send_toks = common_tokenize(llm_ctx, tkn.text_to_send, false);
            const size_t probs_pos          = std::min(slot.n_sent_token_probs, slot.generated_token_probs.size());
            const size_t probs_stop_pos     = std::min(slot.n_sent_token_probs + to_send_toks.size(), slot.generated_token_probs.size());

            std::vector<completion_token_output> probs_output;
            if (probs_pos < probs_stop_pos) {
                probs_output = std::vector<completion_token_output>(slot.generated_token_probs.begin() + long(probs_pos),
                                                                    slot.generated_token_probs.begin() + long(probs_stop_pos));
            }
            slot.n_sent_token_probs = probs_stop_pos;

            res.data["completion_probabilities"] = probs_vector_to_json(llm_ctx, probs_output, slot.oaicompat_completion, slot.oaicompat_completion_chat);
        }

        queue_results.send(res);
    }

    void send_completion(const server_slot &slot) {
        server_task_result res;
        res.id    = slot.id_task;
        res.error = false;
        res.stop  = true;
        res.data  = json{
             {"id_slot", slot.id},
             {"index", slot.index},
             {"stop", true},
             {"model", llm_params.model_alias},
             {"tokens_predicted", slot.n_decoded},
             {"tokens_evaluated", slot.n_prompt_tokens},
             {"tokens_cached", slot.n_past},
             {"generation_settings", get_formated_generation(slot)},
             {"has_new_line", slot.has_new_line},
             {"truncated", slot.truncated},
             {"stop_type", format_stop_type(slot.stop)},
             {"stopping_word", slot.stopping_word},
             {"timings", slot.get_formated_timings()},
        };
        if (slot.generated_tool_calls.empty()) {
            res.data["content"] = !slot.params.stream ? slot.generated_text : "";
            res.data["tokens"]  = !slot.params.stream ? slot.generated_tokens : llama_tokens{};
        } else {
            res.data["tool_calls"] = slot.generated_tool_calls;
        }

        if (!slot.params.stream && slot.params.llm_params.n_probs > 0) {
            std::vector<completion_token_output> probs;
            if (!slot.params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(llm_ctx, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                probs              = std::vector<completion_token_output>(slot.generated_token_probs.begin(), slot.generated_token_probs.end() - int(safe_offset));
            } else {
                probs = std::vector<completion_token_output>(slot.generated_token_probs.begin(), slot.generated_token_probs.end());
            }

            res.data["completion_probabilities"] = probs_vector_to_json(llm_ctx, probs, slot.oaicompat_completion, slot.oaicompat_completion_chat);
        }

        queue_results.send(res);
    }

    void send_embedding(const server_slot &slot, const llama_batch &batch_view) {
        server_task_result res;
        res.id    = slot.id_task;
        res.error = false;
        res.stop  = true;

        const int n_embd = llama_n_embd(llm_model);

        std::vector<float> embedding(n_embd, 0.0f);

        for (int i = batch_view.n_tokens - 1; i >= 0; i--) {
            if (!batch_view.logits[i] || batch_view.seq_id[i][0] != slot.id) {
                continue;
            }

            const float *embd = llama_get_embeddings_seq(llm_ctx, batch_view.seq_id[i][0]);
            if (embd == nullptr) {
                embd = llama_get_embeddings_ith(llm_ctx, i);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);
                break;
            }

            // normalize only when there is pooling
            common_embd_normalize(embd, embedding.data(), n_embd, 2);
            break;
        }

        res.data = json{
            {"index", slot.index},
            {"embedding", embedding},
            {"tokens_evaluated", slot.n_prompt_tokens},
        };

        queue_results.send(res);
    }

    void send_rerank(const server_slot &slot, const llama_batch &batch_view) {
        server_task_result res;
        res.id    = slot.id_task;
        res.error = false;
        res.stop  = true;

        float score = -1e6;

        for (int i = batch_view.n_tokens - 1; i >= 0; i--) {
            if (!batch_view.logits[i] || batch_view.seq_id[i][0] != slot.id) {
                continue;
            }

            const float *embd = llama_get_embeddings_seq(llm_ctx, batch_view.seq_id[i][0]);
            if (embd == nullptr) {
                embd = llama_get_embeddings_ith(llm_ctx, i);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch_view.token[i], batch_view.seq_id[i][0]);
                break;
            }

            score = embd[0];
            break;
        }

        res.data = json{
            {"index", slot.index},
            {"score", score},
            {"tokens_evaluated", slot.n_prompt_tokens},
        };

        queue_results.send(res);
    }

    void send_image(const server_slot &slot, const int32_t progressed_steps, const int32_t progress_steps, const stablediffusion_generated_image &generated_image) {
        server_task_result res;
        res.id    = slot.id_task;
        res.error = false;
        res.stop  = progressed_steps == progress_steps;
        res.data  = json{
             {"index", slot.index},
             {"progressed_steps", progressed_steps},
             {"progress_steps", progress_steps},
             {"progress", float(progressed_steps) / float(progress_steps) * 100},
             {"stop", res.stop},
             {"model", sd_params.model_alias},
        };
        if (generated_image.data != nullptr) {
            res.data["b64_json"] = base64_encode(generated_image.data, generated_image.size);
        }
        if (res.stop) {
            res.data["timings"] = slot.get_formated_timings();
        }

        queue_results.send(res);
    }

    //
    // Functions to create new task(s) and receive result(s)
    //

    // break the input "prompt" into multiple tasks if needed, then format and tokenize the input prompt(s)
    std::vector<server_task> create_tasks_inference(const std::string rid, json data, const server_task_type task_type, int tps = 0) {
        std::vector<server_task> tasks;
        auto create_task = [&](json &task_data, llama_tokens &prompt_tokens, int tps = 0) {
            server_task task(task_type);
            task.rid           = rid;
            task.data          = task_data;
            task.prompt_tokens = std::move(prompt_tokens);
            task.tps           = tps;
            tasks.push_back(std::move(task));
        };

        /* STABLE DIFFUSION */

        bool image = json_value(data, "__oaicompat_image", false);
        if (image) {
            llama_tokens empty_tokens;
            int batch_count = json_value(data, "batch_count", 1);
            SRV_DBG("creating multi-image tasks, batch_count = %d\n", batch_count);
            for (int i = 0; i < batch_count; i++) {
                data["index"] = i;
                data["seed"]  = json_value(data, "seed", int64_t(sd_params.seed)) + i;
                create_task(data, empty_tokens, tps);
            }

            return tasks;
        }

        /* LLAMA */

        bool chat_vision = json_value(data, "__oaicompat_completion_chat_vision", false);
        if (!chat_vision) {
            // because llama_tokenize api is thread-safe, we can tokenize the prompt from HTTP thread
            bool add_special                            = task_type != SERVER_TASK_TYPE_INFILL && task_type != SERVER_TASK_TYPE_RERANK;
            std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(llm_ctx, data.at("prompt"), add_special, true);
            switch (task_type) {
                case SERVER_TASK_TYPE_INFILL: {
                    SRV_DBG("creating infill tasks, n_prompts = %d\n", (int)tokenized_prompts.size());
                    for (size_t i = 0; i < tokenized_prompts.size(); i++) {
                        data["index"] = i;
                        auto tokens   = format_infill(
                            llm_ctx,
                            data.at("input_prefix"),
                            data.at("input_suffix"),
                            data.at("input_extra"),
                            llm_params.n_batch,
                            llm_params.n_predict,
                            slots[0].n_ctx, // TODO: there should be a better way
                            llm_params.spm_infill,
                            tokenized_prompts[i]);
                        create_task(data, tokens, tps);
                    }
                } break;
                case SERVER_TASK_TYPE_RERANK: {
                    // prompts[0] is the question
                    // prompts[length-2] is the question too, used for reranking
                    // prompts[length-1] is unknown token, used fro reranking
                    // the rest are the answers/documents
                    GGML_ASSERT(tokenized_prompts.size() > 1);
                    SRV_DBG("creating rerank tasks, n_prompts = %d\n", (int)tokenized_prompts.size() - 3);
                    for (size_t i = 1; i < tokenized_prompts.size(); i++) {
                        data["index"] = i - 1;
                        auto tokens   = format_rerank(llm_model, tokenized_prompts[0], tokenized_prompts[i]);
                        create_task(data, tokens, tps);
                    }
                } break;
                default: {
                    SRV_DBG("creating multi-prompt tasks, n_prompts = %d\n", (int)tokenized_prompts.size());
                    for (size_t i = 0; i < tokenized_prompts.size(); i++) {
                        data["index"] = i;
                        create_task(data, tokenized_prompts[i], tps);
                    }
                }
            }
        } else {
            llama_tokens empty_tokens;
            create_task(data, empty_tokens, tps);
        }

        return tasks;
    }

    void cancel_tasks(const std::unordered_set<int> &id_tasks) {
        std::vector<server_task> cancel_tasks;
        cancel_tasks.reserve(id_tasks.size());
        for (const auto &id_task : id_tasks) {
            server_task task(SERVER_TASK_TYPE_CANCEL);
            task.id_target = id_task;
            cancel_tasks.push_back(task);
        }
        queue_results.remove_waiting_task_ids(id_tasks);
        // push to beginning of the queue, so it has highest priority
        queue_tasks.post(cancel_tasks, true);
    }

    // receive the results from task(s) created by create_tasks_inference
    void receive_multi_results(
        const std::unordered_set<int> &id_tasks,
        const std::function<void(std::vector<server_task_result> &)> &result_handler,
        const std::function<void(json)> &error_handler) {
        std::vector<server_task_result> results(id_tasks.size());
        for (size_t i = 0; i < id_tasks.size(); i++) {
            server_task_result result = queue_results.recv(id_tasks);
            if (result.error) {
                error_handler(result.data);
                cancel_tasks(id_tasks);
                return;
            }

            const size_t idx = result.data["index"];
            GGML_ASSERT(idx < results.size() && "index out of range");
            results[idx] = result;
        }
        result_handler(results);
    }

    // receive the results from task(s) created by create_tasks_inference, in stream mode
    void receive_multi_results_stream(
        const std::unordered_set<int> &id_tasks,
        const std::function<bool(server_task_result &)> &result_handler,
        const std::function<void(json)> &error_handler) {
        size_t n_finished = 0;
        while (true) {
            server_task_result result = queue_results.recv(id_tasks);
            if (result.error) {
                error_handler(result.data);
                cancel_tasks(id_tasks);
                break;
            }

            if (!result_handler(result)) {
                cancel_tasks(id_tasks);
                break;
            }

            if (result.stop) {
                if (++n_finished == id_tasks.size()) {
                    break;
                }
            }
        }
    }

    //
    // Functions to process the task
    //

    void process_single_task(server_task task) {
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
            case SERVER_TASK_TYPE_IMAGE: {
                const int id_slot = json_value(task.data, "id_slot", -1);

                server_slot *slot = id_slot != -1 ? get_slot_by_id(id_slot) : get_available_slot(task);

                if (slot == nullptr) {
                    // if no slot is available, we defer this task for
                    // processing later
                    SRV_DBG("no slot is available, defer task, id_task = %d\n", task.id);
                    queue_tasks.defer(task);
                    break;
                }
                if (slot->is_processing()) {
                    // if requested slot is unavailable, we defer this task for
                    // processing later
                    SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                    queue_tasks.defer(task);
                    break;
                }
                if ((task.type == SERVER_TASK_TYPE_EMBEDDING || task.type == SERVER_TASK_TYPE_RERANK) &&
                    int(task.prompt_tokens.size()) < slot->id) {
                    // if requested slot is unsatisfied, we defer this task for
                    // processing later
                    SRV_DBG("no slot is unsatisfied, defer task, id_task = %d\n", task.id);
                    queue_tasks.defer(task);
                    // trigger next iteration
                    queue_tasks.post(server_task(SERVER_TASK_TYPE_NEXT_RESPONSE));
                    break;
                }

                slot->reset();

                slot->rid       = task.rid;
                slot->id_task   = task.id;
                slot->task_type = task.type;
                slot->index     = json_value(task.data, "index", 0);
                // slot->prompt_tokens = task.prompt_tokens; // NB(thxCode): prompt_tokens will be processed in launch_slot_with_task

                if (!launch_slot_with_task(*slot, task)) {
                    SRV_ERR("failed to launch slot with task, id_task = %d\n", task.id);
                    break;
                }
            } break;
            case SERVER_TASK_TYPE_CANCEL: {
                // release slot linked with the task id
                for (auto &slot : slots) {
                    if (slot.id_task == task.id_target) {
                        slot.release();
                        break;
                    }
                }
            } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE: {
                // do nothing
            } break;
            case SERVER_TASK_TYPE_METRICS: {
                json slots_data = json::array();

                int n_idle_slots       = 0;
                int n_processing_slots = 0;

                for (server_slot &slot : slots) {
                    json slot_data             = get_formated_generation(slot);
                    slot_data["id"]            = slot.id;
                    slot_data["id_task"]       = slot.id_task;
                    slot_data["is_processing"] = slot.is_processing();
                    slot_data["state"]         = slot.state;
                    slot_data["next_token"]    = {
                        {"has_next_token", slot.has_next_token},
                        {"has_new_line", slot.has_new_line},
                        {"n_remain", slot.n_remaining},
                        {"n_decoded", slot.n_decoded},
                        {"stopping_word", slot.stopping_word},
                    };

                    if (slot.is_processing()) {
                        n_processing_slots++;
                    } else {
                        n_idle_slots++;
                    }

                    slots_data.push_back(slot_data);
                }
                SRV_DBG("n_idle_slots = %d, n_processing_slots = %d\n", n_idle_slots, n_processing_slots);

                server_task_result res;
                res.id    = task.id;
                res.stop  = true;
                res.error = false;
                res.data  = {
                    {"idle", n_idle_slots},
                    {"processing", n_processing_slots},
                    {"deferred", queue_tasks.queue_tasks_deferred.size()},
                    {"t_start", metrics.t_start},

                    /* STABLE DIFFUSION */

                    {"n_image_generated_steps_total", metrics.n_image_generated_steps_total},
                    {"t_image_processing_total", metrics.t_image_processing_total},
                    {"t_image_generation_total", metrics.t_image_generation_total},
                    {"n_image_generated_steps", metrics.n_image_generated_steps},
                    {"t_image_processing", metrics.t_image_processing},
                    {"t_image_generation", metrics.t_image_generation},

                    /* LLAMA */

                    {"n_prompt_tokens_processed_total", metrics.n_prompt_tokens_processed_total},
                    {"t_tokens_generation_total", metrics.t_tokens_generation_total},
                    {"n_tokens_predicted_total", metrics.n_tokens_predicted_total},
                    {"t_prompt_processing_total", metrics.t_prompt_processing_total},
                    {"n_tokens_drafted_total", metrics.n_tokens_drafted_total},
                    {"n_tokens_drafted_accepted_total", metrics.n_tokens_drafted_accepted_total},
                    {"n_prompt_tokens_processed", metrics.n_prompt_tokens_processed},
                    {"t_prompt_processing", metrics.t_prompt_processing},
                    {"n_tokens_predicted", metrics.n_tokens_predicted},
                    {"t_tokens_generation", metrics.t_tokens_generation},
                    {"n_decode_total", metrics.n_decode_total},
                    {"n_busy_slots_total", metrics.n_busy_slots_total},
                    {"kv_cache_tokens_count", llm_ctx ? llama_get_kv_cache_token_count(llm_ctx) : 0},
                    {"kv_cache_used_cells", llm_ctx ? llama_get_kv_cache_used_cells(llm_ctx) : 0},
                    {"slots", slots_data},
                };

                if (json_value(task.data, "reset_bucket", false)) {
                    metrics.reset_bucket();
                }
                queue_results.send(res);
            } break;
            case SERVER_TASK_TYPE_SLOT_SAVE: {
                int id_slot       = task.data.at("id_slot");
                server_slot *slot = get_slot_by_id(id_slot);
                GGML_ASSERT(slot != nullptr);

                if (slot->is_processing()) {
                    // if requested slot is unavailable, we defer this task for
                    // processing later
                    queue_tasks.defer(task);
                    break;
                }

                const size_t token_count = slot->cache_tokens.size();
                const int64_t t_start    = ggml_time_us();

                std::string filename = task.data.at("filename");
                std::string filepath = task.data.at("filepath");

                const size_t nwrite = llama_state_seq_save_file(llm_ctx, filepath.c_str(), slot->id, slot->cache_tokens.data(), token_count);

                const int64_t t_end    = ggml_time_us();
                const double t_save_ms = double(t_end - t_start) / 1000.0;

                server_task_result result;
                result.id    = task.id;
                result.stop  = true;
                result.error = false;
                result.data  = json{{"id_slot", id_slot},
                                    {"filename", filename},
                                    {"n_saved", token_count}, // tokens saved
                                    {"n_written", nwrite},    // bytes written
                                    {"timings", {{"save_ms", t_save_ms}}}};
                queue_results.send(result);
            } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE: {
                int id_slot       = task.data.at("id_slot");
                server_slot *slot = get_slot_by_id(id_slot);
                GGML_ASSERT(slot != nullptr);

                if (slot->is_processing()) {
                    // if requested slot is unavailable, we defer this task for
                    // processing later
                    queue_tasks.defer(task);
                    break;
                }

                const int64_t t_start = ggml_time_us();

                std::string filename = task.data.at("filename");
                std::string filepath = task.data.at("filepath");

                slot->cache_tokens.resize(slot->n_ctx);
                size_t token_count = 0;
                size_t nread =
                    llama_state_seq_load_file(llm_ctx, filepath.c_str(), slot->id, slot->cache_tokens.data(), slot->cache_tokens.size(), &token_count);
                if (nread == 0) {
                    slot->cache_tokens.resize(0);
                    send_error(task, "unable to restore slot, no available space in KV cache or invalid slot save file", ERROR_TYPE_INVALID_REQUEST);
                    break;
                }
                slot->cache_tokens.resize(token_count);

                const int64_t t_end       = ggml_time_us();
                const double t_restore_ms = double(t_end - t_start) / 1000.0;

                server_task_result result;
                result.id    = task.id;
                result.stop  = true;
                result.error = false;
                result.data  = json{{"id_slot", id_slot},
                                    {"filename", filename},
                                    {"n_restored", token_count}, // tokens restored
                                    {"n_read", nread},           // bytes read
                                    {"timings", {{"restore_ms", t_restore_ms}}}};
                queue_results.send(result);
            } break;
            case SERVER_TASK_TYPE_SLOT_ERASE: {
                int id_slot       = task.data.at("id_slot");
                server_slot *slot = get_slot_by_id(id_slot);
                GGML_ASSERT(slot != nullptr);

                if (slot->is_processing()) {
                    // if requested slot is unavailable, we defer this task for
                    // processing later
                    queue_tasks.defer(task);
                    break;
                }

                // Erase token cache
                const size_t n_erased = slot->cache_tokens.size();
                llama_kv_cache_seq_rm(llm_ctx, slot->id, -1, -1);
                if (llm_ctx_draft != nullptr) {
                    llama_kv_cache_seq_rm(llm_ctx_draft, slot->id, -1, -1);
                }
                slot->cache_tokens.clear();

                server_task_result result;
                result.id    = task.id;
                result.stop  = true;
                result.error = false;
                result.data  = json{{"id_slot", id_slot}, {"n_erased", n_erased}};
                queue_results.send(result);
            } break;
            case SERVER_TASK_TYPE_SET_LORA: {
                if (sd_ctx != nullptr) {
                    std::vector<sd_lora_adapter_container_t> sd_lora_adapters;
                    for (auto &lora_adapter : lora_adapters) {
                        sd_lora_adapters.push_back({lora_adapter.path.c_str(), lora_adapter.scale});
                    }
                    sd_ctx->apply_lora_adpters(sd_lora_adapters);
                } else {
                    common_lora_adapters_apply(llm_ctx, lora_adapters);
                }

                server_task_result result;
                result.id    = task.id;
                result.stop  = true;
                result.error = false;
                result.data  = json{{"success", true}};
                queue_results.send(result);
            } break;
        }
    }

    void update_slots() {
        // check if all slots are idle
        {
            bool all_idle = true;
            for (auto &slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_DBG("%s", "all slots are idle\n");
                if (!cache_prompt) {
                    if (llm_ctx != nullptr) {
                        llama_kv_cache_clear(llm_ctx);
                    }
                    if (llm_ctx_draft != nullptr) {
                        llama_kv_cache_clear(llm_ctx_draft);
                    }
                }
                return;
            }
        }

        // trigger next iteration
        {
            server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
            queue_tasks.post(task);
        }

        /* STABLE DIFFUSION */

        if (sd_ctx != nullptr) {
            for (server_slot &slot : slots) {
                if (slot.state != SLOT_STATE_STARTED) {
                    continue;
                }

                slot.state                   = SLOT_STATE_PROCESSING_PROMPT;
                slot.n_image_generated_steps = 0;
                slot.t_start_process_image   = ggml_time_us();
                slot.t_start_generate_image  = 0;

                SLT_DBG(slot, "%s", "creating image generation stream\n");
                slot.sdsstream = sd_ctx->generate_stream(slot.prompt_string.c_str(), slot.params.sd_params);
                if (slot.sdsstream == nullptr) {
                    slot.release();
                    send_error(slot, "failed to create image generation stream", ERROR_TYPE_SERVER);
                    continue;
                }

                slot.state                  = SLOT_STATE_GENERATING;
                slot.t_start_generate_image = ggml_time_us();
                slot.t_image_processing     = double(slot.t_start_generate_image - slot.t_start_process_image) / 1e3;
                metrics.on_prefilled(slot);

                SLT_INF(slot, "created image generation stream, %.2fs\n", slot.t_image_processing / 1e3);
            }

            for (server_slot &slot : slots) {
                if (slot.state != SLOT_STATE_GENERATING) {
                    continue;
                }

                stablediffusion_generated_image generated_image{};

                SLT_DBG(slot, "%s", "sampling image\n");
                size_t t0     = ggml_time_us();
                bool goahead  = sd_ctx->sample_stream(slot.sdsstream);
                size_t t1     = ggml_time_us();
                auto progress = sd_ctx->progress_stream(slot.sdsstream);
                SLT_INF(slot, "sampled image %03i/%03i %.2fs/it\n", progress.first, progress.second, (t1 - t0) / 1e6);
                if (goahead) {
                    if (slot.params.stream) {
                        if (slot.params.stream_preview_faster) {
                            generated_image = sd_ctx->preview_image_stream(slot.sdsstream, true);
                        } else if (slot.params.stream_preview) {
                            generated_image = sd_ctx->preview_image_stream(slot.sdsstream);
                        }
                        send_image(slot, progress.first, progress.second + 1, generated_image);
                    }
                    continue;
                }

                slot.n_image_generated_steps = progress.second;
                generated_image              = sd_ctx->result_image_stream(slot.sdsstream);
                if (generated_image.data == nullptr) {
                    slot.release();
                    send_error(slot, "failed to get result image from generation stream", ERROR_TYPE_SERVER);
                    continue;
                }
                slot.release();
                send_image(slot, progress.first + 1, progress.second + 1, generated_image);

                stbi_image_free(generated_image.data);
            }

            return;
        }

        /* LLAMA */

        // apply context-shift if needed
        // TODO: simplify and improve
        for (server_slot &slot : slots) {
            if (slot.is_processing() && slot.n_past + 1 >= slot.n_ctx) {
                if (!llm_params.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in
                    // process_token()
                    slot.release();
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    continue;
                }

                // Shift context
                const int n_keep    = slot.params.n_keep + llama_add_bos_token(llm_model);
                const int n_left    = slot.n_past - n_keep;
                const int n_discard = slot.params.n_discard ? slot.params.n_discard : (n_left / 2);

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                llama_kv_cache_seq_rm(llm_ctx, slot.id, n_keep, n_keep + n_discard);
                llama_kv_cache_seq_add(llm_ctx, slot.id, n_keep + n_discard, slot.n_past, -n_discard);
                if (llm_ctx_draft != nullptr) {
                    llama_kv_cache_seq_rm(llm_ctx_draft, slot.id, n_keep, n_keep + n_discard);
                    llama_kv_cache_seq_add(llm_ctx_draft, slot.id, n_keep + n_discard, slot.n_past, -n_discard);
                }

                if (cache_prompt && !slot.oaicompat_completion_chat_vision) {
                    for (size_t i = n_keep + n_discard; i < slot.cache_tokens.size(); i++) {
                        slot.cache_tokens[i - n_discard] = slot.cache_tokens[i];
                    }

                    slot.cache_tokens.resize(slot.cache_tokens.size() - n_discard);
                }

                slot.n_past -= n_discard;

                slot.truncated = true;
            }
        }

        // start populating the batch for this iteration
        common_batch_clear(batch);
        if (llm_ctx_draft) {
            common_batch_clear(batch_draft);
        }

        // first, add sampled tokens from any ongoing sequences
        for (auto &slot : slots) {
            if (slot.state != SLOT_STATE_GENERATING) {
                continue;
            }

            if (slot.token_bkt && !slot.token_bkt->acquire()) {
                continue;
            }

            slot.i_batch = batch.n_tokens;
            if (need_mrope()) {
                int32_t st_pos_id = slot.st_pos_id;
                st_pos_id += slot.n_drafted_accepted;
                common_batch_add_with_mrope(batch, slot.sampled[slot.sampled.size() - 1], st_pos_id, 1, {slot.id}, true);
                if (!slot.sampled_draft.empty()) {
                    for (const llama_token &tok : slot.sampled_draft) {
                        common_batch_add_with_mrope(batch, tok, st_pos_id + 1, 1, {slot.id}, true);
                        st_pos_id += 1;
                    }
                }
                slot.st_pos_id++;
            } else {
                int32_t slot_npast = slot.n_past;
                slot_npast += slot.n_drafted_accepted;
                common_batch_add(batch, slot.sampled[slot.sampled.size() - 1], slot_npast, {slot.id}, true);
                if (!slot.sampled_draft.empty()) {
                    for (const llama_token &tok : slot.sampled_draft) {
                        common_batch_add(batch, tok, slot_npast + 1, {slot.id}, true);
                        slot_npast += 1;
                    }
                }
            }
            slot.n_past++;

            if (cache_prompt && !slot.oaicompat_completion_chat_vision) {
                for (const llama_token &tok : slot.sampled) {
                    slot.cache_tokens.push_back(tok);
                }
            }

            SLT_DBG(slot, "slot decode token, n_ctx = %d, n_past = %d, n_cache_tokens = %d, truncated = %d\n",
                    slot.n_ctx, slot.n_past, (int)slot.cache_tokens.size(), slot.truncated);
        }

        // process in chunks of params.n_batch
        auto n_batch  = int32_t(llama_n_batch(llm_ctx));
        auto n_ubatch = int32_t(llama_n_ubatch(llm_ctx));

        // track if this is an embedding or non-embedding batch
        // if we've added sampled tokens above, we are in non-embedding mode
        // -1: none, 0: non-embedding, 1: embedding
        // TODO: make enum
        int32_t batch_type = batch.n_tokens > 0 ? 0 : -1;

        // next, batch any pending prompts without exceeding n_batch
        if (llm_params.cont_batching || batch.n_tokens == 0) {
            for (auto &slot : slots) {
                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    auto &prompt_tokens = slot.prompt_tokens;

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        // empty prompt passed -> release the slot and send empty response
                        if (!slot.oaicompat_completion_chat_vision && prompt_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");
                            slot.release();
                            switch (slot.task_type) {
                                case SERVER_TASK_TYPE_EMBEDDING:
                                    send_error(slot, "empty input", ERROR_TYPE_INVALID_REQUEST);
                                    break;
                                case SERVER_TASK_TYPE_RERANK:
                                    send_error(slot, "empty query", ERROR_TYPE_INVALID_REQUEST);
                                    break;
                                default:
                                    send_completion(slot);
                            }
                            continue;
                        }

                        slot.t_start_process_prompt = ggml_time_us();
                        slot.t_start_generation     = 0;
                        slot.n_past                 = 0;
                        slot.n_past_mmd             = 0;
                        slot.st_pos_id              = 0;
                        slot.n_prompt_tokens        = int32_t(prompt_tokens.size());
                        slot.state                  = SLOT_STATE_PROCESSING_PROMPT;

                        if (slot.oaicompat_completion_chat_vision && !preprocess_multi_modal_data(slot, n_batch)) {
                            slot.release();
                            send_error(slot, "failed to preprocess multi-modal images", ERROR_TYPE_SERVER);
                            continue;
                        }

                        SLT_INF(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, n_prompt_tokens = %d\n", slot.n_ctx, slot.params.n_keep, slot.n_prompt_tokens);

                        if (slot.is_non_causal()) {
                            if (slot.n_prompt_tokens > n_ubatch) {
                                slot.release();
                                send_error(slot, "input is too large to process, please increase the physical batch size", ERROR_TYPE_SERVER);
                                continue;
                            }

                            if (slot.n_prompt_tokens > slot.n_ctx) {
                                slot.release();
                                send_error(slot, "input is larger than the max context size. skipping", ERROR_TYPE_INVALID_REQUEST);
                                continue;
                            }
                        } else {
                            if (!llm_params.ctx_shift) {
                                // if context shift is disabled, we make sure prompt size is smaller than KV size
                                // TODO: there should be a separate parameter that control prompt truncation
                                //       context shift should be applied only during the generation phase
                                if (slot.n_prompt_tokens >= slot.n_ctx) {
                                    slot.release();
                                    send_error(slot, "the request exceeds the available context size. try increasing the context size or enable context shift", ERROR_TYPE_INVALID_REQUEST);
                                    continue;
                                }
                            }

                            if (slot.params.n_keep < 0) {
                                slot.params.n_keep = slot.n_prompt_tokens;
                            }
                            slot.params.n_keep = std::min(slot.n_ctx - 4, slot.params.n_keep);

                            // if input prompt is too big, truncate it
                            if (slot.n_prompt_tokens >= slot.n_ctx) {
                                const int n_left = slot.n_ctx - slot.params.n_keep;

                                const int n_block_size  = n_left / 2;
                                const int erased_blocks = (slot.n_prompt_tokens - slot.params.n_keep - n_block_size) / n_block_size;

                                llama_tokens new_tokens(
                                    prompt_tokens.begin(),
                                    prompt_tokens.begin() + slot.params.n_keep);

                                new_tokens.insert(
                                    new_tokens.end(),
                                    prompt_tokens.begin() + slot.params.n_keep + erased_blocks * n_block_size,
                                    prompt_tokens.end());

                                prompt_tokens = std::move(new_tokens);

                                slot.truncated       = true;
                                slot.n_prompt_tokens = int32_t(prompt_tokens.size());

                                SLT_WRN(slot, "input truncated, n_ctx = %d, n_keep = %d, n_left = %d, n_prompt_tokens = %d\n", slot.n_ctx, slot.params.n_keep, n_left, slot.n_prompt_tokens);

                                GGML_ASSERT(slot.n_prompt_tokens < slot.n_ctx);
                            }

                            if (cache_prompt && !slot.oaicompat_completion_chat_vision && !slot.cache_tokens.empty()) {
                                // reuse any previously computed tokens that are
                                // common with the new prompt
                                slot.n_past    = int32_t(common_lcp(slot.cache_tokens, prompt_tokens));
                                slot.st_pos_id = slot.n_past;

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (llm_params.n_cache_reuse > 0 && slot.n_past > 0) {
                                    size_t head_c = slot.n_past; // cache
                                    size_t head_p = slot.n_past; // current prompt

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, slot.n_past = %d\n", llm_params.n_cache_reuse, slot.n_past);

                                    while (head_c < slot.cache_tokens.size() && head_p < prompt_tokens.size()) {
                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.cache_tokens.size() && head_p + n_match < prompt_tokens.size() &&
                                               slot.cache_tokens[head_c + n_match] == prompt_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t)llm_params.n_cache_reuse) {
                                            SLT_INF(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match,
                                                    head_c, head_c + n_match, head_p, head_p + n_match);
                                            // for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //     SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(llm_ctx,
                                            //     prompt_tokens[i]).c_str());
                                            // }

                                            const int64_t kv_shift = (int64_t)head_p - (int64_t)head_c;

                                            llama_kv_cache_seq_rm(llm_ctx, slot.id, head_p, head_c);
                                            llama_kv_cache_seq_add(llm_ctx, slot.id, head_c, -1, kv_shift);
                                            if (llm_ctx_draft != nullptr) {
                                                llama_kv_cache_seq_rm(llm_ctx_draft, slot.id, head_p, head_c);
                                                llama_kv_cache_seq_add(llm_ctx_draft, slot.id, head_c, -1, kv_shift);
                                            }

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.cache_tokens[head_p + i] = slot.cache_tokens[head_c + i];
                                                slot.n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new slot.n_past = %d\n", slot.n_past);
                                }
                            }
                        }

                        if (slot.n_past == slot.n_prompt_tokens && slot.n_past > 0) {
                            // we have to evaluate at least 1 token to generate logits.
                            SLT_DBG(slot, "need to evaluate at least 1 token to generate logits, n_past = %d, n_prompt_tokens = %d\n", slot.n_past, slot.n_prompt_tokens);

                            slot.n_past--;
                            slot.st_pos_id--;
                        }

                        slot.n_prompt_tokens_processed = 0;
                    }

                    // non-causal tasks require to fit the entire prompt in the physical batch
                    if (slot.is_non_causal()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.n_tokens + slot.n_prompt_tokens > n_batch) {
                            continue;
                        }
                    }

                    // check that we are in the right batch_type, if not defer the slot
                    const int32_t slot_type = slot.is_non_causal() ? 1 : 0;
                    if (batch_type == -1) {
                        batch_type = slot_type;
                    } else if (batch_type != slot_type) {
                        continue;
                    }

                    // keep only the common part
                    int32_t slot_npast = slot.n_past;
                    if (!llama_kv_cache_seq_rm(llm_ctx, slot.id, slot_npast, -1)) {
                        // could not partially delete (likely using a on-Transformer model)
                        llama_kv_cache_seq_rm(llm_ctx, slot.id, -1, -1);
                        // there is no common part left
                        slot.n_past     = 0;
                        slot.n_past_mmd = 0;
                        slot.st_pos_id  = 0;
                    }
                    if (llm_ctx_draft != nullptr) {
                        if (!llama_kv_cache_seq_rm(llm_ctx_draft, slot.id, slot_npast, -1)) {
                            llama_kv_cache_seq_rm(llm_ctx_draft, slot.id, -1, -1);
                        }
                    }
                    SLT_DBG(slot, "kv cache rm [%d, end)\n", slot_npast);

                    // remove the non-common part from the cache
                    if (!slot.oaicompat_completion_chat_vision) {
                        slot.cache_tokens.resize(slot.n_past);
                    }

                    // add prompt tokens for processing in the current batch
                    const int32_t n_eval = std::min(slot.n_prompt_tokens - slot.n_past, n_batch - batch.n_tokens);
                    const bool need_embd = slot.task_type == SERVER_TASK_TYPE_EMBEDDING && llama_pooling_type(llm_ctx) == LLAMA_POOLING_TYPE_NONE;
                    while (slot.n_past < slot.n_prompt_tokens && batch.n_tokens < n_batch) {
                        const int32_t idx = slot.n_past - slot.n_past_mmd;

                        if (need_mrope()) {
                            common_batch_add_with_mrope(batch, prompt_tokens[idx], slot.st_pos_id, n_eval, {slot.id}, need_embd);
                            if (llm_ctx_draft != nullptr) {
                                common_batch_add_with_mrope(batch_draft, prompt_tokens[idx], slot.st_pos_id, n_eval, {slot.id}, need_embd);
                            }
                            slot.st_pos_id++;
                        } else {
                            common_batch_add(batch, prompt_tokens[idx], slot.n_past, {slot.id}, need_embd);
                            if (llm_ctx_draft != nullptr) {
                                common_batch_add(batch_draft, prompt_tokens[idx], slot.n_past, {slot.id}, need_embd);
                            }
                        }

                        if (cache_prompt && !slot.oaicompat_completion_chat_vision) {
                            slot.cache_tokens.push_back(prompt_tokens[idx]);
                        }

                        slot.n_prompt_tokens_processed++;
                        slot.n_past++;
                    }

                    SLT_INF(slot, "prompt processing, n_past = %d, n_tokens = %d, n_prompt_tokens = %d, n_preprocessed_tokens = %d\n", slot.n_past, batch.n_tokens, slot.n_prompt_tokens, slot.n_prompt_tokens - slot.n_prompt_tokens_processed);

                    // entire prompt has been processed
                    if (slot.n_past == slot.n_prompt_tokens) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.n_tokens > 0);

                        common_sampler_reset(slot.smpl);
                        if (llm_ctx_draft != nullptr) {
                            common_sampler_reset(slot.smpl_draft);
                        }

                        // Process all prompt tokens through sampler system
                        for (const llama_token &token : prompt_tokens) {
                            common_sampler_accept(slot.smpl, token, false);
                            if (llm_ctx_draft != nullptr) {
                                common_sampler_accept(slot.smpl_draft, token, false);
                            }
                        }

                        // extract the logits only for the last token
                        batch.logits[batch.n_tokens - 1] = true;

                        slot.n_decoded = 0;
                        slot.i_batch   = batch.n_tokens - 1;

                        SLT_INF(slot, "prompt done, n_past = %d, n_tokens = %d\n", slot.n_past, batch.n_tokens);
                    }
                }

                if (batch.n_tokens >= n_batch) {
                    break;
                }
            }
        }

        if (batch.n_tokens == 0) {
            SRV_DBG("%s", "no tokens to decode\n");
            return;
        }

        SRV_DBG("decoding batch, n_tokens = %d\n", batch.n_tokens);

        // make sure we're in the right embedding mode
        llama_set_embeddings(llm_ctx, batch_type == 1);

        // process the created batch of tokens
        for (int32_t i = 0; i < batch.n_tokens; i += n_batch) {
            const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

            // clang-format off
            llama_batch batch_view = {
                n_tokens,
                batch.token + i,
                nullptr,
                batch.pos + (need_mrope() ? 4 * i : i),
                batch.n_seq_id + i,
                batch.seq_id + i,
                batch.logits + i,
            };
            // clang-format on

            const int ret = llama_decode(llm_ctx, batch_view);
            metrics.on_decoded(slots);
            if (ret != 0) {
                if (n_batch == 1 || ret < 0) {
                    // if you get here, it means the KV cache is full - try
                    // increasing it via the context size
                    SRV_ERR("failed to decode the batch: KV cache is full - try increasing it via the context size, i = %d, n_batch = %d, ret = %d\n", i, n_batch, ret);

                    for (auto &slot : slots) {
                        slot.release();
                        send_error(slot, "input prompt is too big compared to KV size, please increase KV size");
                    }
                    break; // break loop of n_batch
                }

                // retry with half the batch size to try to find a free slot in
                // the KV cache
                n_batch /= 2;
                i -= n_batch;

                SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size - try increasing it via the context size or enable defragmentation, i = %d, n_batch = %d, ret = %d\n", i, n_batch, ret);

                continue; // continue loop of n_batch
            }
            if (llm_ctx_draft != nullptr && batch_draft.n_tokens > 0) {
                const int32_t n_draft_tokens = std::min(n_batch, batch_draft.n_tokens - i);

                // clang-format off
                llama_batch batch_draft_view = {
                    n_draft_tokens,
                    batch_draft.token + i,
                    nullptr,
                    batch_draft.pos + (need_mrope() ? 4 * i : i),
                    batch_draft.n_seq_id + i,
                    batch_draft.seq_id + i,
                    batch_draft.logits + i,
                };
                // clang-format on

                const int ret_draft = llama_decode(llm_ctx_draft, batch_draft_view);
                GGML_ASSERT(ret_draft == 0);
            }

            for (auto &slot : slots) {
                if (slot.i_batch < (int)i || slot.i_batch >= (int)(i + n_tokens)) {
                    continue; // continue loop of slots
                }

                if (slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task_type == SERVER_TASK_TYPE_EMBEDDING) {
                        // prompt evaluated for embedding
                        send_embedding(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    if (slot.task_type == SERVER_TASK_TYPE_RERANK) {
                        send_rerank(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    // prompt evaluated for next-token prediction
                    slot.state = SLOT_STATE_GENERATING;
                } else if (slot.state != SLOT_STATE_GENERATING) {
                    continue; // continue loop of slots
                }

                completion_token_output result;
                if (!slot.sampled_draft.empty()) {
                    auto sz_draft = int32_t(slot.sampled_draft.size());
                    // +1 to allow for the last token to be generated
                    for (int32_t j = 0; j < sz_draft + 1; ++j) {
                        // greedy verification only
                        bool accept     = false;
                        int32_t tok_idx = slot.i_batch - i + j;
                        llama_token tok = common_sampler_sample(slot.smpl, llm_ctx, tok_idx);
                        common_sampler_accept(slot.smpl, tok, true);
                        if (j < sz_draft && tok == slot.sampled_draft[j]) {
                            accept = true;
                        }
                        slot.n_decoded += 1;
                        if (!accept) {
                            break;
                        }
                        slot.push_token_into_result(llm_ctx, tok_idx, tok, result);
                        slot.n_drafted_accepted += 1;
                    }
                } else {
                    int32_t tok_idx = slot.i_batch - i;
                    llama_token tok = common_sampler_sample(slot.smpl, llm_ctx, tok_idx);
                    common_sampler_accept(slot.smpl, tok, true);
                    slot.push_token_into_result(llm_ctx, tok_idx, tok, result);
                    slot.n_decoded += 1;
                }

                if (slot.n_decoded == 1) {
                    slot.t_start_generation  = ggml_time_us();
                    slot.t_prompt_processing = double(slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                    metrics.on_prefilled(slot);
                }

                if (llm_ctx_draft != nullptr) {
                    llama_pos pos = slot.n_past + slot.n_drafted_accepted;
                    llama_kv_cache_seq_rm(llm_ctx_draft, slot.id, pos, -1);

                    slot.sampled_draft.clear();

                    common_batch_clear(batch_draft);
                    common_batch_add(batch_draft, result.toks[result.toks.size() - 1], pos, {slot.id}, true);
                    if (llama_decode(llm_ctx_draft, batch_draft)) {
                        slot.release();
                        send_error(slot, "failed to draft decode", ERROR_TYPE_SERVER);
                        continue; // continue loop of slots
                    }
                    slot.n_drafted += 1;

                    for (int32_t j = 0; j < llm_params.speculative.n_max; ++j) {
                        llama_token tok = common_sampler_sample(slot.smpl_draft, llm_ctx_draft, 0, true);

                        const llama_token_data_array *cur_p = common_sampler_get_candidates(slot.smpl_draft);
                        if (cur_p->data[0].p < llm_params.speculative.p_min) {
                            break;
                        }

                        slot.sampled_draft.push_back(tok);
                        common_sampler_accept(slot.smpl_draft, tok, true);
                        if (llama_token_is_eog(llm_model_draft, tok)) {
                            break;
                        }
                        common_batch_clear(batch_draft);
                        common_batch_add(batch_draft, tok, pos + 1 + j, {slot.id}, true);
                        if (llama_decode(llm_ctx_draft, batch_draft)) {
                            break;
                        }
                        slot.n_drafted += 1;
                    }

                    // ignore small drafts
                    if (int32_t(slot.sampled_draft.size()) < llm_params.speculative.n_min) {
                        slot.sampled_draft.clear();
                    }
                } else if (params.lookup_ngram_min > 0) {
                    llama_pos pos = slot.n_past + slot.n_drafted_accepted;
                    llama_kv_cache_seq_rm(llm_ctx, slot.id, pos, -1);

                    slot.sampled_draft.clear();

                    slot.sampled_draft.push_back(result.toks[result.toks.size() - 1]);
                    common_ngram_cache_draft(slot.prompt_tokens, slot.sampled_draft, llm_params.speculative.n_max, params.lookup_ngram_min, LLAMA_NGRAM_MAX,
                                             slot.ctx_ngram_cache, ngram_cache_dynamic, ngram_cache_static);
                    slot.n_drafted += int32_t(slot.sampled_draft.size()) - 1;

                    slot.sampled_draft.erase(slot.sampled_draft.begin());
                }

                if (!process_token(result, slot)) {
                    // release slot because of stop condition
                    slot.release();
                    send_completion(slot);
                    metrics.on_finished(slot);
                }

                slot.i_batch = -1;
            }
        }
    }

    bool preprocess_multi_modal_data_text(server_slot &slot, int32_t n_batch, const std::string text, const bool add_bos) const {
        llama_tokens tokens = common_tokenize(llm_ctx, text, add_bos, true);
        const auto n_tokens = int32_t(tokens.size());
        SLT_INF(slot, "processing text tokens: %d\n", n_tokens);
        if (common_log_verbosity_thold > 2) {
            for (const llama_token &token : tokens) {
                SLT_INF(slot, "processing text tokens | %6d -> '%s'\n", token, common_token_to_piece(llm_ctx, token).c_str());
            }
            SLT_INF(slot, "%s", "processing text tokens\n");
        }

        if (clip_is_qwen2vl(llm_ctx_clip)) {
            for (int32_t j = 0; j < n_tokens; j += n_batch) {
                int32_t n_eval = std::min(n_batch, int32_t(n_tokens - j));
                std::vector<llama_pos> batch_txt_mrope_pos;
                {
                    batch_txt_mrope_pos.resize(n_eval * 4);
                    std::fill(batch_txt_mrope_pos.begin(), batch_txt_mrope_pos.end(), 0);
                    for (int i = 0; i < n_eval * 3; i++) {
                        batch_txt_mrope_pos[i] = slot.st_pos_id + (i % n_eval);
                    }
                }
                qwen2vl_text_token_batch_wrapper batch_txt = qwen2vl_text_token_batch_wrapper((tokens.data() + j), n_eval, batch_txt_mrope_pos.data(), slot.id);
                if (llama_decode(llm_ctx, batch_txt.batch)) {
                    SLT_ERR(slot, "%s", "failed to decode text");
                    return false;
                }
                slot.n_past += n_eval;
                slot.st_pos_id += n_eval;
                slot.n_prompt_tokens += n_eval;
                slot.n_prompt_tokens_processed += n_eval;
            }
            slot.prompt_tokens.insert(slot.prompt_tokens.end(), tokens.begin(), tokens.end());
            return true;
        }

        for (int32_t j = 0; j < n_tokens; j += n_batch) {
            int32_t n_eval                           = std::min(n_batch, int32_t(n_tokens - j));
            llava_text_token_batch_wrapper batch_txt = llava_text_token_batch_wrapper((tokens.data() + j), n_eval, slot.n_past, slot.id);
            if (llama_decode(llm_ctx, batch_txt.batch)) {
                SLT_ERR(slot, "%s", "failed to decode text");
                return false;
            }
            slot.n_past += n_eval;
            slot.n_prompt_tokens += n_eval;
            slot.n_prompt_tokens_processed += n_eval;
        }
        slot.prompt_tokens.insert(slot.prompt_tokens.end(), tokens.begin(), tokens.end());
        return true;
    }

    bool preprocess_multi_modal_data_image(server_slot &slot, int32_t n_batch, const llava_image_embed *img_embd) const {
        const int32_t n_embd = llama_n_embd(llama_get_model(llm_ctx));
        SLT_INF(slot, "processing image tokens: %d\n", img_embd->n_image_pos);

        if (clip_is_qwen2vl(llm_ctx_clip)) {
            if (!preprocess_multi_modal_data_text(slot, n_batch, std::string("<|vision_start|>"), false)) {
                return false;
            }

            auto qwen2vl_decode_img_embd = [&](const llava_image_embed *img_embd, const std::vector<llama_pos> &img_mrope_pos) {
                for (int32_t j = 0; j < img_embd->n_image_pos; j += n_batch) {
                    const int32_t n_eval = std::min(n_batch, img_embd->n_image_pos - j);
                    std::vector<llama_pos> batch_img_mrope_pos;
                    {
                        batch_img_mrope_pos.resize(img_mrope_pos.size());
                        std::fill(batch_img_mrope_pos.begin(), batch_img_mrope_pos.end(), 0);
                        memcpy(batch_img_mrope_pos.data(), &img_mrope_pos[j], n_eval * sizeof(llama_pos));
                        memcpy(&batch_img_mrope_pos[n_eval * 1], &img_mrope_pos[img_embd->n_image_pos * 1 + j], n_eval * sizeof(llama_pos));
                        memcpy(&batch_img_mrope_pos[n_eval * 2], &img_mrope_pos[img_embd->n_image_pos * 2 + j], n_eval * sizeof(llama_pos));
                        memcpy(&batch_img_mrope_pos[n_eval * 3], &img_mrope_pos[img_embd->n_image_pos * 3 + j], n_eval * sizeof(llama_pos));
                    }
                    qwen2vl_image_embed_batch_wrapper batch_img = qwen2vl_image_embed_batch_wrapper((img_embd->embed + j * n_embd), n_eval, batch_img_mrope_pos.data(), slot.id);
                    if (llama_decode(llm_ctx, batch_img.batch)) {
                        return false;
                    }
                    slot.n_past += n_eval;
                    slot.n_past_mmd += n_eval;
                    slot.n_prompt_tokens += n_eval;
                    slot.n_prompt_tokens_processed += n_eval;
                }
                return true;
            };

            std::vector<llama_pos> img_mrope_pos;
            {
                struct clip_image_size *img_size = clip_get_load_image_size(llm_ctx_clip);
                const int32_t ps                 = clip_patch_size(llm_ctx_clip) * 2;
                const int ph                     = img_size->height / ps + (img_size->height % ps > 0);
                const int pw                     = img_size->width / ps + (img_size->width % ps > 0);
                img_mrope_pos.resize(img_embd->n_image_pos * 4);
                for (int32_t y = 0; y < ph; y++) {
                    for (int32_t x = 0; x < pw; x++) {
                        int i                                        = y * pw + x;
                        img_mrope_pos[i]                             = slot.st_pos_id;
                        img_mrope_pos[i + img_embd->n_image_pos * 1] = slot.st_pos_id + y;
                        img_mrope_pos[i + img_embd->n_image_pos * 2] = slot.st_pos_id + x;
                        img_mrope_pos[i + img_embd->n_image_pos * 3] = 0;
                    }
                }
                slot.st_pos_id += std::max(ph, pw);
            }
            if (!qwen2vl_decode_img_embd(img_embd, img_mrope_pos)) {
                SLT_ERR(slot, "%s", "failed to decode image");
                return false;
            }

            if (!preprocess_multi_modal_data_text(slot, n_batch, std::string("<|vision_end|>"), false)) {
                return false;
            }
            return true;
        }

        auto llava_decode_img_embd = [&](const llava_image_embed *img_embd) {
            for (int32_t j = 0; j < img_embd->n_image_pos; j += n_batch) {
                const int32_t n_eval                      = std::min(n_batch, img_embd->n_image_pos - j);
                llava_image_embed_batch_wrapper batch_img = llava_image_embed_batch_wrapper((img_embd->embed + j * n_embd), n_eval, slot.n_past, slot.id);
                if (llama_decode(llm_ctx, batch_img.batch)) {
                    return false;
                }
                slot.n_past += n_eval;
                slot.n_past_mmd += n_eval;
                slot.n_prompt_tokens += n_eval;
                slot.n_prompt_tokens_processed += n_eval;
            }
            return true;
        };

        if (clip_is_minicpmv(llm_ctx_clip) != 0) {
            int idx                         = 0;
            auto slice_minicpmv_image_embed = [&](const llava_image_embed *img_embd) {
                auto *embed = (float *)malloc(clip_embd_nbytes(llm_ctx_clip));
                std::memcpy(embed, img_embd->embed + (idx++) * clip_n_patches(llm_ctx_clip) * clip_n_mmproj_embd(llm_ctx_clip), clip_embd_nbytes(llm_ctx_clip));

                auto *slice_embed        = (llava_image_embed *)malloc(sizeof(llava_image_embed));
                slice_embed->embed       = embed;
                slice_embed->n_image_pos = clip_n_patches(llm_ctx_clip);
                return slice_embed;
            };

            size_t n_img_embd = img_embd->n_image_pos / clip_n_patches(llm_ctx_clip);
            if (!preprocess_multi_modal_data_text(slot, n_batch, std::string("<image>"), false)) {
                return false;
            }
            llava_image_embed *img_embd_sliced = slice_minicpmv_image_embed(img_embd);
            bool decoded                       = llava_decode_img_embd(img_embd_sliced);
            llava_image_embed_free(img_embd_sliced);
            if (!decoded) {
                SLT_ERR(slot, "%s", "failed to decode sliced image");
                return false;
            }
            if (!preprocess_multi_modal_data_text(slot, n_batch, std::string("</image>"), false)) {
                return false;
            }
            if (n_img_embd > 1) {
                size_t n_img_embd_col = clip_uhd_num_image_embeds_col(llm_ctx_clip);
                if (!preprocess_multi_modal_data_text(slot, n_batch, std::string("<slice>"), false)) {
                    return false;
                }
                for (size_t i = 0; i < (n_img_embd - 1) / n_img_embd_col; ++i) {
                    for (size_t j = 0; j < n_img_embd_col; ++j) {
                        if (!preprocess_multi_modal_data_text(slot, n_batch, std::string("<image>"), false)) {
                            return false;
                        }
                        img_embd_sliced = slice_minicpmv_image_embed(img_embd);
                        decoded         = llava_decode_img_embd(img_embd_sliced);
                        llava_image_embed_free(img_embd_sliced);
                        if (!decoded) {
                            SLT_ERR(slot, "%s", "failed to decode sliced image");
                            return false;
                        }
                        if (!preprocess_multi_modal_data_text(slot, n_batch, std::string("</image>"), false)) {
                            return false;
                        }
                    }
                }
                if (!preprocess_multi_modal_data_text(slot, n_batch, std::string("</slice>"), false)) {
                    return false;
                }
            }
            return true;
        }

        if (!llava_decode_img_embd(img_embd)) {
            SLT_ERR(slot, "%s", "failed to decode image");
            return false;
        }
        return true;
    }

    bool preprocess_multi_modal_data(server_slot &slot, int32_t n_batch) const {
        // remove previous memory
        llama_kv_cache_seq_rm(llm_ctx, slot.id, -1, -1);
        if (llm_ctx_draft != nullptr) {
            llama_kv_cache_seq_rm(llm_ctx_draft, slot.id, -1, -1);
        }

        const std::string image_sign = "<image>";
        const json images_json       = slot.prompt_multi_modal_data.at("images");

        std::string prompt_string = slot.prompt_string;
        size_t images_count       = 0;
        size_t image_pos          = prompt_string.find(image_sign);
        bool add_bos              = true;
        while (image_pos != std::string::npos) {
            // process text
            const std::string text = prompt_string.substr(0, image_pos);
            if (!preprocess_multi_modal_data_text(slot, n_batch, text, add_bos)) {
                return false;
            }
            add_bos = false;

            // process image
            uint8_t *dt = nullptr;
            int w       = 0;
            int h       = 0;
            int c       = 0;
            {
                const std::string img               = images_json.at(images_count++).get<std::string>();
                const std::vector<uint8_t> img_buff = base64_decode(img);

                dt = stbi_load_from_memory((const stbi_uc *)img_buff.data(), (int)img_buff.size(), &w, &h, &c, 3);
                if (dt == nullptr) {
                    auto reason = stbi_failure_reason();
                    SLT_ERR(slot, "failed to load image: %s\n", reason);
                    return false;
                }

                int m = std::max(w, h);
                if (params.max_image_size > 0 && m > params.max_image_size) {
                    SLT_INF(slot, "image dimensions exceeded the maximum size: %d, resizing image\n", params.max_image_size);
                    float nr  = float(params.max_image_size) / float(m);
                    int nw    = std::max(int(std::ceil(float(w) * nr)), 1);
                    int nh    = std::max(int(std::ceil(float(h) * nr)), 1);
                    auto *ndt = (uint8_t *)malloc(nw * nh * c);
                    if (ndt == nullptr) {
                        SLT_ERR(slot, "%s", "failed to resize image: allocate new buffer\n");
                        stbi_image_free(dt);
                        return false;
                    }
                    bool resized = stbir_resize(
                        dt, w, h, 0,
                        ndt, nw, nh, 0, STBIR_TYPE_UINT8,
                        c /*RGB channel*/, STBIR_ALPHA_CHANNEL_NONE, 0,
                        STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
                        STBIR_FILTER_BOX, STBIR_FILTER_BOX,
                        STBIR_COLORSPACE_SRGB, nullptr);
                    stbi_image_free(dt);
                    if (!resized) {
                        auto reason = stbi_failure_reason();
                        SLT_ERR(slot, "failed to resize image: %s\n", reason);
                        return false;
                    }
                    SLT_INF(slot, "image resized from %dx%d to %dx%d\n", w, h, nw, nh);
                    dt = ndt;
                    w  = nw;
                    h  = nh;
                }
            }
            // NB(thxCode): llava_image_embed_make_with_data is a patch.
            llava_image_embed *img_embd = llava_image_embed_make_with_data(llm_ctx_clip, llm_params.cpuparams.n_threads, dt, w, h, c);
            if (!img_embd) {
                SLT_ERR(slot, "%s", "failed to embed image\n");
                return false;
            }
            bool processed = preprocess_multi_modal_data_image(slot, n_batch, img_embd);
            llava_image_embed_free(img_embd);
            if (!processed) {
                return false;
            }

            prompt_string = prompt_string.substr(image_pos + image_sign.length());
            image_pos     = prompt_string.find(image_sign);
        }

        // process remain text
        llama_tokens tokens = common_tokenize(llm_ctx, prompt_string, add_bos, true);
        SLT_INF(slot, "processing text tokens: %zu\n", tokens.size());
        if (common_log_verbosity_thold > 2) {
            for (const llama_token &token : tokens) {
                SLT_INF(slot, "processing text tokens | %6d -> '%s'\n", token, common_token_to_piece(llm_ctx, token).c_str());
            }
            SLT_INF(slot, "%s", "processing text tokens\n");
        }
        slot.prompt_tokens.insert(slot.prompt_tokens.end(), tokens.begin(), tokens.end());
        slot.n_prompt_tokens += int32_t(tokens.size());

        return true;
    }

    json model_meta() const {
        /* STABLE DIFFUSION */

        if (sd_ctx != nullptr) {
            return json{
                {"max_batch_count", sd_params.max_batch_count},
                {"max_height", sd_params.sampling.height},
                {"max_width", sd_params.sampling.width},
                {"guidance", sd_params.sampling.guidance},
                {"strength", sd_params.sampling.strength},
                {"sample_method", sd_sample_method_to_argument(sd_params.sampling.sample_method)},
                {"sampling_steps", sd_params.sampling.sampling_steps},
                {"cfg_scale", sd_params.sampling.cfg_scale},
                {"slg_scale", sd_params.sampling.slg_scale},
                {"slg_skip_layers", sd_params.sampling.slg_skip_layers},
                {"slg_start", sd_params.sampling.slg_start},
                {"slg_end", sd_params.sampling.slg_end},
                {"schedule_method", sd_schedule_to_argument(sd_params.sampling.schedule_method)},
                {"negative_prompt", sd_params.sampling.negative_prompt},
                {"n_slot", llm_params.n_parallel},
            };
        }

        /* LLAMA */

        return json{
            {"vocab_type", llama_vocab_type(llm_model)},
            {"n_vocab", llama_n_vocab(llm_model)},
            {"n_ctx_train", llama_n_ctx_train(llm_model)},
            {"n_embd", llama_n_embd(llm_model)},
            {"n_params", llama_model_n_params(llm_model)},
            {"size", llama_model_size(llm_model)},
            {"n_ctx", llama_n_ctx(llm_ctx)},
            {"n_slot", llm_params.n_parallel},
            {"support_vision", llm_ctx_clip != nullptr},
            {"support_speculative", llm_ctx_draft != nullptr},
            {"support_tool_calls", support_tool_calls},
        };
    }

    //
    // Functions to distinguish
    //

    bool support_completion() const {
        return llm_ctx != nullptr;
    }

    bool support_completion_only() const {
        // NB(thxCode): llama_supports_embedding_only is a patch.
        return llm_ctx != nullptr && !llama_supports_embedding_only(llm_ctx);
    }

    bool support_embedding() const {
        return llm_ctx != nullptr;
    }

    bool support_embedding_only() const {
        // NB(thxCode): llama_supports_embedding_only is a patch.
        return llm_ctx != nullptr && llama_supports_embedding_only(llm_ctx);
    }

    bool support_image() const {
        return sd_ctx != nullptr;
    }

    bool need_mrope() const {
        // NB(thxCode): llama_model_needs_mrope is a patch.
        return llm_model != nullptr && llama_model_needs_mrope(llm_model);
    }
};

static void log_server_request(const httplib::Request &req, httplib::Response &res) {
    if (req.path == "/v1/health") {
        return;
    }

    std::string rid = req.get_header_value(HEADER_REQUEST_ID);
    if (rid.empty()) {
        rid = std::to_string(ggml_time_us());
    }
    res.set_header(HEADER_REQUEST_ID, rid);
    res.set_header(HEADER_REQUEST_ACCEPTED_AT, std::to_string(ggml_time_us()));
    SRV_INF("rid %s | %s %s %s:%d\n",
            rid.c_str(), req.method.c_str(), req.path.c_str(), req.remote_addr.c_str(), req.remote_port);
}

static void log_server_response(const httplib::Request &req, const httplib::Response &res) {
    if (req.path == "/v1/health") {
        return;
    }

    std::string rid = res.get_header_value(HEADER_REQUEST_ID);
    uint64_t rst    = res.get_header_value_u64(HEADER_REQUEST_ACCEPTED_AT);
    SRV_INF("rid %s | %s %s %s:%d | status %d | cost %.2fs\n",
            rid.c_str(), req.method.c_str(), req.path.c_str(), req.remote_addr.c_str(), req.remote_port, res.status, (ggml_time_us() - rst) / 1e6);
}

static void log_server_exception(const httplib::Request &req, httplib::Response &res, const std::string &err_msg) {
    if (req.path == "/v1/health") {
        return;
    }

    std::string rid = res.get_header_value(HEADER_REQUEST_ID);
    if (rid.empty()) {
        rid = req.get_header_value(HEADER_REQUEST_ID);
    }

    SRV_ERR("rid %s | exception = %s\n", rid.c_str(), err_msg.c_str());
}

std::function<void(int)> shutdown_handler;
std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C
        // twice this is for better developer experience, we can remove when the
        // server is stable enough
        SRV_WRN("%s", "received second interrupt, terminating immediately\n");
        exit(1);
    }

    shutdown_handler(signal);
}

int main(int argc, char **argv) {
    common_log_set_prefix(common_log_main(), true);
    common_log_set_timestamps(common_log_main(), true);
    llama_log_set(
        [](ggml_log_level level, const char *text, void * /*user_data*/) {
            if (LOG_DEFAULT_LLAMA <= common_log_verbosity_thold) {
                common_log_add(common_log_main(), level, "%s", text);
            }
        },
        nullptr);
    sd_log_set(
        [](sd_log_level_t level, const char *text, void * /*user_data*/) {
            if (LOG_DEFAULT_LLAMA <= common_log_verbosity_thold) {
                common_log_add(common_log_main(), sd_log_level_to_ggml_log_level(level), "%s", text);
            }
        },
        nullptr);
    sd_progress_set(
        [](int step, int steps, float time, void * /*user_data*/) {
            // nothing to do
        },
        nullptr);

    llama_box_params params;
    if (!llama_box_params_parse(argc, argv, params)) {
        llama_box_params_print_usage(argc, argv, params);
        return 1;
    }
    common_params &llm_params = params.llm_params;

    // print arguments
    LOG_INF("\n");
    std::ostringstream argss;
    for (int i = 0; i < argc; i++) {
        argss << argv[i];
        if (i < argc - 1) {
            argss << " ";
        }
    }
    LOG_INF("arguments  : %s\n", argss.str().c_str());
    LOG_INF("version    : %s (%s)\n", LLAMA_BOX_BUILD_VERSION, LLAMA_BOX_COMMIT);
    LOG_INF("compiler   : %s\n", LLAMA_BOX_BUILD_COMPILER);
    LOG_INF("target     : %s\n", LLAMA_BOX_BUILD_TARGET);
    LOG_INF("vendor     : llama.cpp %s (%d), stable-diffusion.cpp %s (%d)\n", LLAMA_CPP_COMMIT, LLAMA_CPP_BUILD_NUMBER, STABLE_DIFFUSION_CPP_COMMIT, STABLE_DIFFUSION_CPP_BUILD_NUMBER);
    LOG_INF("%s\n", common_params_get_system_info(llm_params).c_str());
    LOG_INF("\n");

    //
    // serve as rpc server
    //

    if (params.rpc_params.port > 0) {
        llama_numa_init(llm_params.numa);

        rpcserver_params &rpc_params = params.rpc_params;
        return rpcserver_start(rpc_params);
    }

    //
    // serve as server
    //

    auto res_error = [](httplib::Response &res, json data) {
        json final_response{
            {"error", data},
            {"detail", json_value(data, "message", std::string("Unknown Error"))},
        };
        res.set_content(final_response.dump(-1, ' ', false, json::error_handler_t::replace), MIMETYPE_JSON);
        res.status = json_value(data, "code", httplib::StatusCode::InternalServerError_500);
    };
    auto res_ok = [](httplib::Response &res, json data) {
        res.set_content(data.dump(-1, ' ', false, json::error_handler_t::replace), MIMETYPE_JSON);
        res.status = 200;
    };

    server_context ctx_server;

    llama_numa_init(llm_params.numa);
    llama_backend_init();

    httplib::Server svr;
    std::atomic<server_state> state{SERVER_STATE_LOADING_MODEL};

    svr.set_read_timeout(llm_params.timeout_read);
    svr.set_write_timeout(llm_params.timeout_write);
    svr.set_payload_max_length(CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH);
    svr.set_idle_interval(params.conn_idle);
    svr.set_keep_alive_timeout(params.conn_keepalive);
    svr.set_default_headers({{"Server", "llama-box/" + std::string(LLAMA_BOX_BUILD_VERSION)}});
    svr.set_logger(log_server_response);
    svr.set_exception_handler([&](const httplib::Request &req, httplib::Response &res, const std::exception_ptr &ep) {
        error_type err_type = ERROR_TYPE_SERVER;
        std::string message;
        try {
            std::rethrow_exception(ep);
        } catch (std::runtime_error &e) {
            err_type = ERROR_TYPE_INVALID_REQUEST;
            message  = e.what();
        } catch (std::exception &e) {
            message = e.what();
        } catch (...) {
            message = "Unknown exception";
        }
        log_server_exception(req, res, message);

        res_error(res, format_error_response(message, err_type));
    });
    svr.set_error_handler([&](const httplib::Request &, httplib::Response &res) {
        if (res.status == 404) {
            res_error(res, format_error_response("Not Found", ERROR_TYPE_NOT_FOUND));
        }
        // for other error codes, we skip processing here because it's
        // already done by res_error()
    });
    svr.set_pre_routing_handler([&](const httplib::Request &req, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        // if this is OPTIONS request, skip validation because browsers don't include Authorization header
        if (req.method == "OPTIONS") {
            res.set_header("Access-Control-Allow-Credentials", "true");
            res.set_header("Access-Control-Allow-Methods", "GET, POST");
            res.set_header("Access-Control-Allow-Headers", "*");
            res.set_content("", "text/html");                 // blank response, no data
            return httplib::Server::HandlerResponse::Handled; // skip further processing
        }
        // if the server is still loading the model, skip processing
        if (state.load() == SERVER_STATE_LOADING_MODEL) {
            res_error(res, format_error_response("Loading model", ERROR_TYPE_UNAVAILABLE));
            return httplib::Server::HandlerResponse::Handled;
        }
        log_server_request(req, res);
        return httplib::Server::HandlerResponse::Unhandled;
    });

    //
    // Handlers
    //

    const auto handle_health = [&](const httplib::Request &, httplib::Response &res) {
        // error and loading states are handled by middleware
        const json response{
            {"status", "ok"},
        };
        res_ok(res, response);
    };

    const auto handle_metrics = [&](const httplib::Request &req, httplib::Response &res) {
        // construct task
        server_task task(SERVER_TASK_TYPE_METRICS);
        task.rid = res.get_header_value(HEADER_REQUEST_ID);
        task.data.push_back({{"reset_bucket", true}});

        // post task
        task.id = ctx_server.queue_tasks.post(task, true); // high-priority task
        ctx_server.queue_results.add_waiting_task_id(task.id);

        // get result
        server_task_result result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        std::stringstream metrics;
        {
            json data = result.data;

            /* STABLE DIFFUSION */

            uint64_t t_image_processing_total      = data.at("t_image_processing_total");
            uint64_t t_image_generation_total      = data.at("t_image_generation_total");
            uint64_t n_image_generated_steps_total = data.at("n_image_generated_steps_total");
            uint64_t t_image_processing            = data.at("t_image_processing");
            uint64_t t_image_generation            = data.at("t_image_generation");
            uint64_t n_image_generated_steps       = data.at("n_image_generated_steps");

            /* LLAMA */

            uint64_t n_prompt_tokens_processed_total = data.at("n_prompt_tokens_processed_total");
            uint64_t t_prompt_processing_total       = data.at("t_prompt_processing_total");
            uint64_t n_tokens_predicted_total        = data.at("n_tokens_predicted_total");
            uint64_t t_tokens_generation_total       = data.at("t_tokens_generation_total");
            uint64_t n_tokens_drafted_total          = data.at("n_tokens_drafted_total");
            uint64_t n_tokens_drafted_accepted_total = data.at("n_tokens_drafted_accepted_total");
            uint64_t n_decode_total                  = data.at("n_decode_total");
            uint64_t n_busy_slots_total              = data.at("n_busy_slots_total");
            uint64_t n_prompt_tokens_processed       = data.at("n_prompt_tokens_processed");
            uint64_t t_prompt_processing             = data.at("t_prompt_processing");
            uint64_t n_tokens_predicted              = data.at("n_tokens_predicted");
            uint64_t t_tokens_generation             = data.at("t_tokens_generation");
            int32_t kv_cache_used_cells              = data.at("kv_cache_used_cells");
            uint64_t kv_cache_tokens_count           = data.at("kv_cache_tokens_count");
            uint64_t processing                      = data.at("processing");
            uint64_t deferred                        = data.at("deferred");

            // metrics definition:
            // https://prometheus.io/docs/practices/naming/#metric-names
            json all_metrics_def = json{
                {
                    "counter",
                    {
                        /* STABLE DIFFUSION */

                        {
                            {"name", "image_process_seconds_total"},
                            {"help", "Image process time."},
                            {"value", double(t_image_processing_total) / 1.e3},
                        },
                        {
                            {"name", "image_generate_seconds_total"},
                            {"help", "Image generate time."},
                            {"value", double(t_image_generation_total) / 1.e3},
                        },
                        {
                            {"name", "image_generate_steps_total"},
                            {"help", "Number of image generate steps."},
                            {"value", n_image_generated_steps_total},
                        },

                        /* LLAMA */

                        {
                            {"name", "prompt_tokens_total"},
                            {"help", "Number of prompt tokens processed."},
                            {"value", n_prompt_tokens_processed_total},
                        },
                        {
                            {"name", "prompt_seconds_total"},
                            {"help", "Prompt process time."},
                            {"value", double(t_prompt_processing_total) / 1.e3},
                        },
                        {
                            {"name", "tokens_predicted_total"},
                            {"help", "Number of generation tokens processed."},
                            {"value", n_tokens_predicted_total},
                        },
                        {
                            {"name", "tokens_predicted_seconds_total"},
                            {"help", "Predict process time."},
                            {"value", double(t_tokens_generation_total) / 1.e3},
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
                        {
                            {"name", "n_decode_total"},
                            {"help", "Total number of llama_decode() calls."},
                            {"value", n_decode_total},
                        },
                        {
                            {"name", "n_busy_slots_per_decode"},
                            {"help", "Average number of busy slots per llama_decode() call."},
                            {"value", (float)n_busy_slots_total / (float)n_decode_total},
                        },
                    },
                },
                {
                    "gauge",
                    {
                        /* STABLE DIFFUSION */

                        {
                            {"name", "image_steps_seconds"},
                            {"help", "Average image generation throughput in steps/s."},
                            {"value", n_image_generated_steps ? 1.e3 / double(t_image_generation * n_image_generated_steps) : 0.},
                        },

                        /* LLAMA */

                        {
                            {"name", "prompt_tokens_seconds"},
                            {"help", "Average prompt throughput in tokens/s."},
                            {"value", n_prompt_tokens_processed ? 1.e3 / double(t_prompt_processing * n_prompt_tokens_processed) : 0.},
                        },
                        {
                            {"name", "predicted_tokens_seconds"},
                            {"help", "Average generation throughput in tokens/s."},
                            {"value", n_tokens_predicted ? 1.e3 / double(t_tokens_generation * n_tokens_predicted) : 0.},
                        },
                        {
                            {"name", "kv_cache_usage_ratio"},
                            {"help", "KV-cache usage. 1 means 100 percent usage."},
                            {"value", 1. * kv_cache_used_cells / ctx_server.llm_params.n_ctx},
                        },
                        {
                            {"name", "kv_cache_tokens"},
                            {"help", "KV-cache tokens."},
                            {"value", kv_cache_tokens_count},
                        },
                        {
                            {"name", "requests_processing"},
                            {"help", "Number of request processing."},
                            {"value", processing},
                        },
                        {
                            {"name", "requests_deferred"},
                            {"help", "Number of request deferred."},
                            {"value", deferred},
                        },
                    },
                },
            };

            for (const auto &el : all_metrics_def.items()) {
                const auto &type        = el.key();
                const auto &metrics_def = el.value();

                for (const auto &metric_def : metrics_def) {
                    const std::string name = metric_def.at("name");
                    const std::string help = metric_def.at("help");

                    auto value = json_value(metric_def, "value", 0.);
                    metrics << "# HELP llamabox:" << name << " " << help << "\n"
                            << "# TYPE llamabox:" << name << " " << type << "\n"
                            << "llamabox:" << name << " " << value << "\n";
                }
            }
        }

        const std::string response = metrics.str();
        res.set_content(response, "text/plain; version=0.0.4");
    };

    const auto handle_props = [&](const httplib::Request &, httplib::Response &res) {
        const json response{
            {"default_generation_settings", ctx_server.default_generation_settings_for_props},
            {"total_slots", ctx_server.llm_params.n_parallel},
            {"chat_template", ctx_server.llm_params.chat_template.c_str()},
        };
        res_ok(res, response);
    };

    const auto handle_tokenize = [&](const httplib::Request &req, httplib::Response &res) {
        if (!ctx_server.support_completion()) {
            res.status = httplib::StatusCode::Forbidden_403;
            res.set_content("You are not allowed to do tokenize from this model", MIMETYPE_TEXT);
            return;
        }

        const json request = json::parse(req.body);

        if (!request.contains("content")) {
            res_error(res, format_error_response("\"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        json tokens_json = json::array();

        const bool add_special = json_value(request, "add_special", false);
        const bool with_pieces = json_value(request, "with_pieces", false);
        llama_tokens tokens    = tokenize_mixed(ctx_server.llm_ctx, request.at("content"), add_special, true);
        if (with_pieces) {
            for (const llama_token &token : tokens) {
                json piece_json;

                std::string piece = common_token_to_piece(ctx_server.llm_ctx, token);
                if (is_valid_utf8(piece)) {
                    // If valid UTF-8, store as string
                    piece_json = piece;
                } else {
                    // Otherwise, store as array of byte values
                    piece_json = json::array();
                    for (unsigned char c : piece) {
                        piece_json.push_back(static_cast<int>(c));
                    }
                }

                tokens_json.push_back({{"id", token}, {"piece", piece_json}});
            }
        } else {
            tokens_json = tokens;
        }

        const json response{
            {"tokens", tokens_json},
        };
        res_ok(res, response);
    };

    const auto handle_detokenize = [&](const httplib::Request &req, httplib::Response &res) {
        if (!ctx_server.support_completion()) {
            res.status = httplib::StatusCode::Forbidden_403;
            res.set_content("You are not allowed to do detokenize from this model", MIMETYPE_TEXT);
            return;
        }

        const json request = json::parse(req.body);

        if (!request.contains("tokens")) {
            res_error(res, format_error_response("\"tokens\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        const std::string content = common_detokenize(ctx_server.llm_ctx, request.at("tokens"), false);

        const json response{
            {"content", content},
        };
        res_ok(res, response);
    };

    const auto handle_slots_list = [&](const httplib::Request &req, httplib::Response &res) {
        // construct task
        server_task task(SERVER_TASK_TYPE_METRICS);
        task.rid = res.get_header_value(HEADER_REQUEST_ID);

        // post task
        task.id = ctx_server.queue_tasks.post(task, true); // high-priority task
        ctx_server.queue_results.add_waiting_task_id(task.id);

        // get result
        server_task_result result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        const int n_idle_slots = result.data.at("idle");
        if (req.has_param("fail_on_no_slot")) {
            if (n_idle_slots == 0) {
                res_error(res, format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return;
            }
        }

        const json response = result.data.at("slots");
        res_ok(res, response);
    };

    const auto handle_slots_save = [&](const httplib::Request &req, httplib::Response &res, int id_slot) {
        json request = json::parse(req.body);

        std::string filename = request.at("filename");
        if (!fs_validate_filename(filename)) {
            res_error(res, format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        std::string filepath = ctx_server.llm_params.slot_save_path + filename;

        // construct task
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.rid  = res.get_header_value(HEADER_REQUEST_ID);
        task.data = {{"id_slot", id_slot}, {"filename", filename}, {"filepath", filepath}};

        // post task
        task.id = ctx_server.queue_tasks.post(task);
        ctx_server.queue_results.add_waiting_task_id(task.id);

        // get result
        server_task_result result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        if (result.error) {
            res_error(res, result.data);
            return;
        }

        const json response = result.data;
        res_ok(res, response);
    };

    const auto handle_slots_restore = [&](const httplib::Request &req, httplib::Response &res, int id_slot) {
        json request = json::parse(req.body);

        std::string filename = request.at("filename");
        if (!fs_validate_filename(filename)) {
            res_error(res, format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        std::string filepath = ctx_server.llm_params.slot_save_path + filename;

        // construct task
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.rid  = res.get_header_value(HEADER_REQUEST_ID);
        task.data = {{"id_slot", id_slot}, {"filename", filename}, {"filepath", filepath}};

        // post task
        task.id = ctx_server.queue_tasks.post(task);
        ctx_server.queue_results.add_waiting_task_id(task.id);

        // get result
        server_task_result result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        if (result.error) {
            res_error(res, result.data);
            return;
        }

        const json response = result.data;
        res_ok(res, response);
    };

    const auto handle_slots_erase = [&](const httplib::Request &req, httplib::Response &res, int id_slot) {
        // construct task
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.rid  = res.get_header_value(HEADER_REQUEST_ID);
        task.data = {{"id_slot", id_slot}};

        // post task
        task.id = ctx_server.queue_tasks.post(task);
        ctx_server.queue_results.add_waiting_task_id(task.id);

        // get result
        server_task_result result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        if (result.error) {
            res_error(res, result.data);
            return;
        }

        const json response = result.data;
        res_ok(res, response);
    };

    const auto handle_slots_action = [&](const httplib::Request &req, httplib::Response &res) {
        int id_slot = -1;
        {
            const std::string id_slot_str = req.path_params.at("id_slot");
            if (!id_slot_str.empty()) {
                try {
                    id_slot = std::stoi(id_slot_str);
                } catch (const std::exception &) {
                    id_slot = -1;
                }
            }
        }
        if (id_slot < 0 || id_slot >= ctx_server.llm_params.n_parallel) {
            res_error(res, format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        // forward
        const std::string action = req.get_param_value("action");
        if (action == "save") {
            handle_slots_save(req, res, id_slot);
        } else if (action == "restore") {
            handle_slots_restore(req, res, id_slot);
        } else if (action == "erase") {
            handle_slots_erase(req, res, id_slot);
        } else {
            res_error(res, format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        }
    };

    const auto handle_lora_adapters_list = [&](const httplib::Request &, httplib::Response &res) {
        json response = json::array();
        for (size_t i = 0; i < ctx_server.lora_adapters.size(); ++i) {
            auto &la = ctx_server.lora_adapters[i];
            response.push_back({
                {"id", i},
                {"path", la.path},
                {"scale", la.scale},
            });
        }
        res_ok(res, response);
    };

    const auto handle_lora_adapters_apply = [&](const httplib::Request &req, httplib::Response &res) {
        const std::vector<json> request = json::parse(req.body);

        for (common_lora_adapter_container &la : ctx_server.lora_adapters) {
            la.scale = 0.0f;
        }
        auto max_idx = int32_t(ctx_server.lora_adapters.size());
        for (json part : request) {
            int id      = part.at("id");
            float scale = part.at("scale");
            if (0 <= id && id < max_idx) {
                ctx_server.lora_adapters[id].scale = scale;
                continue;
            }
            throw std::runtime_error("invalid adapter id");
        }

        // construct task
        server_task task(SERVER_TASK_TYPE_SET_LORA);
        task.rid = res.get_header_value(HEADER_REQUEST_ID);

        // post task
        task.id = ctx_server.queue_tasks.post(task);
        ctx_server.queue_results.add_waiting_task_id(task.id);

        // get result
        server_task_result result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        const json response = result.data;
        res_ok(res, response);
    };

    const auto handle_infill = [&](const httplib::Request &req, httplib::Response &res) {
        if (!ctx_server.support_completion_only()) {
            res.status = httplib::StatusCode::Forbidden_403;
            res.set_content("You are not allowed to do infill from this model", MIMETYPE_TEXT);
            return;
        }

        // check model compatibility
        std::string err;
        if (llama_token_fim_pre(ctx_server.llm_model) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_token_fim_suf(ctx_server.llm_model) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_token_fim_mid(ctx_server.llm_model) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res_error(res, format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        int tps = 0;
        {
            const std::string tps_s = req.get_header_value(HEADER_REQUEST_TOKENS_PER_SECOND);
            if (!tps_s.empty()) {
                try {
                    tps = std::stoi(tps_s);
                } catch (const std::exception &) {
                    tps = ctx_server.params.n_tps;
                }
            }
            if (tps > ctx_server.params.n_tps) {
                // if the request exceeds the maximum tokens per second, return 410 Gone
                if (ctx_server.params.n_tps > 0) {
                    res.status = httplib::StatusCode::Gone_410;
                    res.set_content("This request exceeds the maximum tokens per second", MIMETYPE_TEXT);
                    return;
                }
                // if the server is not limited by tokens per second, set tps to 0
                tps = 0;
            }
        }

        json request          = json::parse(req.body);
        const std::string rid = res.get_header_value(HEADER_REQUEST_ID);

        if (!request.contains("input_prefix")) {
            res_error(res, format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }
        if (!request.contains("input_suffix")) {
            res_error(res, format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }
        if (request.contains("input_extra") && !request.at("input_extra").is_array()) {
            res_error(res, format_error_response(R"("input_extra" must be an array of {"filename": string, "text": string})", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        {
            json input_extra = json_value(request, "input_extra", json::array());
            for (const auto &chunk : input_extra) {
                // { "text": string, "filename": string }
                if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                    res_error(res,
                              format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                    return;
                }
                // filename is optional
                if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                    res_error(res, format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                    return;
                }
            }
            request["input_extra"] = input_extra; // default to empty array if it's not exist
        }

        // construct task
        std::vector<server_task> tasks = ctx_server.create_tasks_inference(rid, request, SERVER_TASK_TYPE_INFILL, tps);

        // post task
        std::unordered_set<int> task_ids = ctx_server.queue_tasks.post(tasks);
        ctx_server.queue_results.add_waiting_task_ids(task_ids);

        // get result: process non-streaming requests
        if (!json_value(request, "stream", false)) {
            ctx_server.receive_multi_results(
                task_ids,
                [&](const std::vector<server_task_result> &results) {
                    json response       = results[0].data;
                    std::string tps_str = std::to_string(json_value(response.at("timings"), "predicted_per_second", double(tps)));
                    if (results.size() > 1) {
                        response = json::array({response});
                        for (size_t i = 1; i < results.size(); ++i) {
                            response.push_back(results[i].data);
                        }
                    }
                    res.set_header("X-Response-Tokens-Per-Second", tps_str);
                    res_ok(res, response);
                },
                [&](const json &error_data) {
                    res_error(res, error_data);
                });

            ctx_server.cancel_tasks(task_ids);
            return;
        }

        // get result: process streaming requests
        const auto on_chunk = [&ctx_server, task_ids, tps](size_t, httplib::DataSink &sink) {
            ctx_server.receive_multi_results_stream(
                task_ids,
                [&](const server_task_result &result) -> bool {
                    json response = result.data;
                    if (!server_sent_event(sink, "data", response)) {
                        LOG_ERR("%s", "srv             handle_infill: failed to send chunk data\n");
                        sink.done();
                        return false;
                    }
                    if (result.stop) {
                        std::string tps_str = std::to_string(json_value(result.data.at("timings"), "predicted_per_second", double(tps)));
                        sink.done_with_trailer({{"X-Response-Tokens-Per-Second", tps_str}});
                    }
                    return true;
                },
                [&](const json &error_data) {
                    server_sent_event(sink, "error", error_data);
                    sink.done();
                });

            return false;
        };
        const auto on_complete = [&ctx_server, task_ids](bool) {
            ctx_server.cancel_tasks(task_ids);
        };

        res.set_header("Cache-Control", "no-cache, no-store, no-transform");
        res.set_header("Connection", "keep-alive");
        res.set_header("Trailer", "X-Response-Tokens-Per-Second");
        res.set_chunked_content_provider("text/event-stream; charset=utf-8", on_chunk, on_complete);
    };

    const auto handle_models = [&](const httplib::Request &, httplib::Response &res) {
        json response{
            {"object", "list"},
            {
                "data",
                {
                    {
                        {"id", ctx_server.sd_ctx != nullptr ? ctx_server.sd_params.model_alias : ctx_server.llm_params.model_alias},
                        {"object", "model"},
                        {"created", std::time(nullptr)},
                        {"owned_by", "llama-box"},
                        {"meta", ctx_server.model_meta()},
                    },
                },
            },
        };
        res_ok(res, response);
    };

    const auto handle_completions = [&](const httplib::Request &req, httplib::Response &res) {
        if (!ctx_server.support_completion_only()) {
            res.status = httplib::StatusCode::Forbidden_403;
            res.set_content("You are not allowed to do completion from this model", MIMETYPE_TEXT);
            return;
        }

        int tps = 0;
        {
            const std::string tps_s = req.get_header_value(HEADER_REQUEST_TOKENS_PER_SECOND);
            if (!tps_s.empty()) {
                try {
                    tps = std::stoi(tps_s);
                } catch (const std::exception &) {
                    tps = ctx_server.params.n_tps;
                }
            }
            if (tps > ctx_server.params.n_tps) {
                // if the request exceeds the maximum tokens per second, return 410 Gone
                if (ctx_server.params.n_tps > 0) {
                    res.status = httplib::StatusCode::Gone_410;
                    res.set_content("This request exceeds the maximum tokens per second", MIMETYPE_TEXT);
                    return;
                }
                // if the server is not limited by tokens per second, set tps to 0
                tps = 0;
            }
        }

        json request                    = json::parse(req.body);
        const std::string rid           = res.get_header_value(HEADER_REQUEST_ID);
        const std::string completion_id = gen_cmplid();

        bool oaicompat = req.path == "/v1/completions";
        if (!request.contains("prompt")) {
            res_error(res, format_error_response("\"prompt\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        if (oaicompat) {
            request = oaicompat_completions_request(ctx_server.llm_params, rid, request, ctx_server.llm_model, false, false);
        }

        // construct task
        std::vector<server_task> tasks = ctx_server.create_tasks_inference(rid, request, SERVER_TASK_TYPE_COMPLETION, tps);

        // post task
        std::unordered_set<int> task_ids = ctx_server.queue_tasks.post(tasks);
        ctx_server.queue_results.add_waiting_task_ids(task_ids);

        // get result: process non-streaming requests
        if (!json_value(request, "stream", false)) {
            ctx_server.receive_multi_results(
                task_ids,
                [&](std::vector<server_task_result> &results) {
                    if (!oaicompat) {
                        json response       = results[0].data;
                        std::string tps_str = std::to_string(json_value(response.at("timings"), "predicted_per_second", double(tps)));
                        if (results.size() > 1) {
                            response = json::array({response});
                            for (size_t i = 1; i < results.size(); ++i) {
                                response.push_back(results[i].data);
                            }
                        }
                        res.set_header("X-Response-Tokens-Per-Second", tps_str);
                        res_ok(res, response);
                        return;
                    }

                    json response = json::array();
                    for (const server_task_result &ret : results) {
                        response.push_back(ret.data);
                    }
                    response            = oaicompat_completions_response(request, response, completion_id);
                    std::string tps_str = std::to_string(json_value(response.at("usage"), "tokens_per_second", double(tps)));
                    res.set_header("X-Response-Tokens-Per-Second", tps_str);
                    res_ok(res, response);
                },
                [&](const json &error_data) {
                    res_error(res, error_data);
                });

            ctx_server.cancel_tasks(task_ids);
            return;
        }

        // get result: process streaming requests
        const auto on_chunk = [&ctx_server, task_ids, oaicompat, request, rid, completion_id, tps](size_t, httplib::DataSink &sink) {
            ctx_server.receive_multi_results_stream(
                task_ids,
                [&](const server_task_result &result) -> bool {
                    json response = result.data;
                    if (oaicompat) {
                        response = oaicompat_completions_response(request, json::array({response}), completion_id, true);
                    }
                    if (!server_sent_event(sink, "data", response)) {
                        LOG_WRN("srv        handle_completions: rid %s | failed to send chunk data\n", rid.c_str());
                        sink.done();
                        return false;
                    }
                    if (result.stop) {
                        if (oaicompat) {
                            const std::string done = "data: [DONE] \n\n";
                            if (!sink.write(done.c_str(), done.size())) {
                                LOG_WRN("srv        handle_completions: rid %s | failed to send chunk data\n", rid.c_str());
                                sink.done();
                                return false;
                            }
                        }
                        std::string tps_str = std::to_string(json_value(result.data.at("timings"), "predicted_per_second", double(tps)));
                        sink.done_with_trailer({{"X-Response-Tokens-Per-Second", tps_str}});
                    }
                    return true;
                },
                [&](const json &error_data) {
                    server_sent_event(sink, "error", error_data);
                    sink.done();
                });

            return false;
        };
        const auto on_complete = [&ctx_server, task_ids](bool) {
            ctx_server.cancel_tasks(task_ids);
        };

        res.set_header("Cache-Control", "no-cache, no-store, no-transform");
        res.set_header("Connection", "keep-alive");
        res.set_header("Trailer", "X-Response-Tokens-Per-Second");
        res.set_chunked_content_provider("text/event-stream", on_chunk, on_complete);
    };

    const auto handle_chat_completions = [&](const httplib::Request &req, httplib::Response &res) {
        if (!ctx_server.support_completion_only()) {
            res.status = httplib::StatusCode::Forbidden_403;
            res.set_content("You are not allowed to do completion from this model", MIMETYPE_TEXT);
            return;
        }

        int tps = 0;
        {
            const std::string tps_s = req.get_header_value(HEADER_REQUEST_TOKENS_PER_SECOND);
            if (!tps_s.empty()) {
                try {
                    tps = std::stoi(tps_s);
                } catch (const std::exception &) {
                    tps = ctx_server.params.n_tps;
                }
            }
            if (tps > ctx_server.params.n_tps) {
                // if the request exceeds the maximum tokens per second, return 410 Gone
                if (ctx_server.params.n_tps > 0) {
                    res.status = httplib::StatusCode::Gone_410;
                    res.set_content("This request exceeds the maximum tokens per second", MIMETYPE_TEXT);
                    return;
                }
                // if the server is not limited by tokens per second, set tps to 0
                tps = 0;
            }
        }

        json request                    = json::parse(req.body);
        const std::string rid           = res.get_header_value(HEADER_REQUEST_ID);
        const std::string completion_id = gen_chatcmplid();

        if (!request.contains("messages") || !request.at("messages").is_array()) {
            res_error(res, format_error_response("\"messages\" must be provided and must be an array", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        request = oaicompat_completions_request(ctx_server.llm_params, rid, request, ctx_server.llm_model, true, ctx_server.support_tool_calls);

        // construct task
        std::vector<server_task> tasks = ctx_server.create_tasks_inference(rid, request, SERVER_TASK_TYPE_COMPLETION, tps);

        // post task
        const std::unordered_set<int> task_ids = ctx_server.queue_tasks.post(tasks);
        ctx_server.queue_results.add_waiting_task_ids(task_ids);

        // get result: process non-streaming requests
        if (!json_value(request, "stream", false)) {
            ctx_server.receive_multi_results(
                task_ids,
                [&, tps](std::vector<server_task_result> &results) {
                    json response = json::array();
                    for (const server_task_result &ret : results) {
                        response.push_back(ret.data);
                    }

                    response            = oaicompat_completions_response(request, response, completion_id);
                    std::string tps_str = std::to_string(json_value(response.at("usage"), "tokens_per_second", double(tps)));
                    res.set_header("X-Response-Tokens-Per-Second", tps_str);
                    res_ok(res, response);
                },
                [&](const json &error_data) {
                    res_error(res, error_data);
                });

            ctx_server.cancel_tasks(task_ids);
            return;
        }

        // process streaming requests
        const auto on_chunk = [&ctx_server, task_ids, request, rid, completion_id, tps](size_t, httplib::DataSink &sink) {
            bool first = true;
            ctx_server.receive_multi_results_stream(
                task_ids,
                [&](const server_task_result &result) -> bool {
                    if (first) {
                        first         = false;
                        json response = oaicompat_completions_response(request, json::array(), completion_id, true, true);
                        if (!server_sent_event(sink, "data", response)) {
                            LOG_WRN("srv   handle_chat_completions: rid %s | failed to send chunk data\n", rid.c_str());
                            sink.done();
                            return false;
                        }
                    }

                    json response = oaicompat_completions_response(request, json::array({result.data}), completion_id, true);
                    if (!server_sent_event(sink, "data", response)) {
                        LOG_WRN("srv   handle_chat_completions: rid %s | failed to send chunk data\n", rid.c_str());
                        sink.done();
                        return false;
                    }

                    if (result.stop) {
                        const std::string done = "data: [DONE] \n\n";
                        if (!sink.write(done.c_str(), done.size())) {
                            LOG_WRN("srv   handle_chat_completions: rid %s | failed to send chunk data\n", rid.c_str());
                            sink.done();
                            return false;
                        }

                        std::string tps_str = std::to_string(json_value(result.data.at("timings"), "predicted_per_second", double(tps)));
                        sink.done_with_trailer({{"X-Response-Tokens-Per-Second", tps_str}});
                    }
                    return true;
                },
                [&](const json &error_data) {
                    server_sent_event(sink, "error", error_data);
                    sink.done();
                });

            return false;
        };
        const auto on_complete = [&ctx_server, task_ids](bool) {
            ctx_server.cancel_tasks(task_ids);
        };

        res.set_header("Cache-Control", "no-cache, no-store, no-transform");
        res.set_header("Connection", "keep-alive");
        res.set_header("Trailer", "X-Response-Tokens-Per-Second");
        res.set_chunked_content_provider("text/event-stream", on_chunk, on_complete);
    };

    const auto handle_embeddings = [&](const httplib::Request &req, httplib::Response &res) {
        if (!ctx_server.support_embedding()) {
            res.status = httplib::StatusCode::Forbidden_403;
            res.set_content("You are not allowed to do embedding from this model", MIMETYPE_TEXT);
            return;
        }

        json request          = json::parse(req.body);
        const std::string rid = res.get_header_value(HEADER_REQUEST_ID);

        if (!request.contains("input")) {
            res_error(res, format_error_response("\"input\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        request = oaicompat_embeddings_request(ctx_server.llm_params, rid, request);

        // construct task
        std::vector<server_task> tasks = ctx_server.create_tasks_inference(rid, request, SERVER_TASK_TYPE_EMBEDDING);

        // post task
        std::unordered_set<int> task_ids = ctx_server.queue_tasks.post(tasks);
        ctx_server.queue_results.add_waiting_task_ids(task_ids);

        // get result
        ctx_server.receive_multi_results(
            task_ids,
            [&](std::vector<server_task_result> &results) {
                json response = json::array();
                for (const server_task_result &ret : results) {
                    response.push_back(ret.data);
                }

                response = oaicompat_embeddings_response(request, response);
                res_ok(res, response);
            },
            [&](const json &error_data) {
                res_error(res, error_data);
            });

        ctx_server.cancel_tasks(task_ids);
    };

    const auto handle_rerank = [&](const httplib::Request &req, httplib::Response &res) {
        if (!ctx_server.support_embedding_only()) {
            res.status = httplib::StatusCode::Forbidden_403;
            res.set_content("You are not allowed to do reranking from this model", MIMETYPE_TEXT);
            return;
        }

        json request          = json::parse(req.body);
        const std::string rid = res.get_header_value(HEADER_REQUEST_ID);

        if (!request.contains("query")) {
            res_error(res, format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        if (!request.at("query").is_string()) {
            res_error(res, format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        if (!request.contains("documents")) {
            res_error(res, format_error_response("\"documents\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        if (!request.at("documents").is_array()) {
            res_error(res, format_error_response("\"documents\" must be a array", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        if (request.at("documents").empty()) {
            res_error(res, format_error_response("\"documents\" must not be empty", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        request = jinaaicompat_rerank_request(ctx_server.llm_params, request, ctx_server.llm_ctx);

        // construct task
        std::vector<server_task> tasks = ctx_server.create_tasks_inference(rid, request, SERVER_TASK_TYPE_RERANK);

        // post task
        std::unordered_set<int> task_ids = ctx_server.queue_tasks.post(tasks);
        ctx_server.queue_results.add_waiting_task_ids(task_ids);

        // get result
        ctx_server.receive_multi_results(
            task_ids,
            [&](std::vector<server_task_result> &results) {
                json response = json::array();
                for (const server_task_result &ret : results) {
                    response.push_back(ret.data);
                }

                response = jinaicompat_rerank_response(request, response);
                res_ok(res, response);
            },
            [&](const json &error_data) {
                res_error(res, error_data);
            });

        ctx_server.cancel_tasks(task_ids);
    };

    const auto handle_images = [&](const httplib::Request &req, httplib::Response &res) {
        if (!ctx_server.support_image()) {
            res.status = httplib::StatusCode::Forbidden_403;
            res.set_content("You are not allowed to do operation from this model", MIMETYPE_TEXT);
            return;
        }

        json request;
        const std::string rid = res.get_header_value(HEADER_REQUEST_ID);

        bool generations = req.path == "/v1/images/generations";
        if (generations) {
            request = json::parse(req.body);
            if (!request.contains("prompt")) {
                res_error(res, format_error_response("\"prompt\" must be provided", ERROR_TYPE_INVALID_REQUEST));
                return;
            }
            request = oaicompat_images_generations_request(ctx_server.sd_params, rid, request);
        } else {
            if (!req.is_multipart_form_data()) {
                res_error(res, format_error_response("Request must be multipart/form-data", ERROR_TYPE_INVALID_REQUEST));
                return;
            }
            request = json{};
            if (!req.has_file("image")) {
                res_error(res, format_error_response("\"image\" must be provided", ERROR_TYPE_INVALID_REQUEST));
                return;
            } else {
                request["image"] = req.get_file_value("image").content;
            }
            if (!req.has_file("prompt")) {
                res_error(res, format_error_response("\"prompt\" must be provided", ERROR_TYPE_INVALID_REQUEST));
                return;
            } else {
                request["prompt"] = req.get_file_value("prompt").content;
            }
            if (req.has_file("mask")) {
                request["mask"] = req.get_file_value("mask").content;
            }
            if (req.has_file("control")) {
                request["control"] = req.get_file_value("control").content;
            }
            if (req.has_file("model")) {
                request["model"] = req.get_file_value("model").content;
            }
            if (req.has_file("size")) {
                request["size"] = req.get_file_value("size").content;
            }
            if (req.has_file("n")) {
                try {
                    request["n"] = std::stoi(req.get_file_value("n").content);
                } catch (const std::exception &) {
                    res_error(res, format_error_response("\"n\" must be an integer", ERROR_TYPE_INVALID_REQUEST));
                    return;
                }
            }
            if (req.has_file("guidance")) {
                try {
                    request["guidance"] = std::stof(req.get_file_value("guidance").content);
                } catch (const std::exception &) {
                    res_error(res, format_error_response("\"guidance\" must be a float", ERROR_TYPE_INVALID_REQUEST));
                    return;
                }
            }
            if (req.has_file("strength")) {
                try {
                    request["strength"] = std::stof(req.get_file_value("strength").content);
                } catch (const std::exception &) {
                    res_error(res, format_error_response("\"strength\" must be a float", ERROR_TYPE_INVALID_REQUEST));
                    return;
                }
            }
            if (req.has_file("sample_method") || req.has_file("sampler")) {
                if (req.has_file("sample_method")) {
                    request["sample_method"] = req.get_file_value("sample_method").content;
                } else {
                    request["sample_method"] = req.get_file_value("sampler").content;
                }
                if (req.has_file("seed")) {
                    try {
                        request["seed"] = std::stoi(req.get_file_value("seed").content);
                    } catch (const std::exception &) {
                        res_error(res, format_error_response("\"seed\" must be an integer", ERROR_TYPE_INVALID_REQUEST));
                        return;
                    }
                }
                try {
                    if (req.has_file("sampling_steps")) {
                        request["sampling_steps"] = std::stoi(req.get_file_value("sampling_steps").content);
                    } else if (req.has_file("sample_steps")) {
                        request["sampling_steps"] = std::stoi(req.get_file_value("sample_steps").content);
                    }
                } catch (const std::exception &) {
                    res_error(res, format_error_response("\"sampling_steps\" must be an integer", ERROR_TYPE_INVALID_REQUEST));
                    return;
                }
                if (req.has_file("cfg_scale")) {
                    try {
                        request["cfg_scale"] = std::stof(req.get_file_value("cfg_scale").content);
                    } catch (const std::exception &) {
                        res_error(res, format_error_response("\"cfg_scale\" must be a float", ERROR_TYPE_INVALID_REQUEST));
                        return;
                    }
                }
                if (req.has_file("slg_scale")) {
                    try {
                        request["slg_scale"] = std::stof(req.get_file_value("slg_scale").content);
                    } catch (const std::exception &) {
                        res_error(res, format_error_response("\"slg_scale\" must be a float", ERROR_TYPE_INVALID_REQUEST));
                        return;
                    }
                }
                // TODO slg_skip_layers
                if (req.has_file("slg_start")) {
                    try {
                        request["slg_start"] = std::stof(req.get_file_value("slg_start").content);
                    } catch (const std::exception &) {
                        res_error(res, format_error_response("\"slg_start\" must be a float", ERROR_TYPE_INVALID_REQUEST));
                        return;
                    }
                }
                if (req.has_file("slg_end")) {
                    try {
                        request["slg_end"] = std::stof(req.get_file_value("slg_end").content);
                    } catch (const std::exception &) {
                        res_error(res, format_error_response("\"slg_end\" must be a float", ERROR_TYPE_INVALID_REQUEST));
                        return;
                    }
                }
                if (req.has_file("schedule_method")) {
                    request["schedule_method"] = req.get_file_value("schedule_method").content;
                } else if (req.has_file("schedule")) {
                    request["schedule_method"] = req.get_file_value("schedule").content;
                }
                if (req.has_file("control_strength")) {
                    try {
                        request["control_strength"] = std::stof(req.get_file_value("control_strength").content);
                    } catch (const std::exception &) {
                        res_error(res, format_error_response("\"control_strength\" must be a float", ERROR_TYPE_INVALID_REQUEST));
                        return;
                    }
                }
                if (req.has_file("control_canny")) {
                    request["control_canny"] = req.get_file_value("control_canny").content == "true";
                }
                if (req.has_file("negative_sampler")) {
                    request["negative_sampler"] = req.get_file_value("negative_sampler").content;
                }
            }
            if (req.has_file("stream")) {
                request["stream"] = req.get_file_value("stream").content == "true";
                if (req.has_file("stream_options_include_usage")) {
                    request["stream_options"] = json{{"include_usage", req.get_file_value("stream_options_include_usage").content == "true"}};
                } else {
                    request["stream_options"] = json{{"include_usage", true}};
                }
            }
            if (json_value(request["stream_options"], "include_usage", false)) {
                if (req.has_file("stream_options_chunk_result")) {
                    request["stream_options"]["chunk_result"] = req.get_file_value("stream_options_chunk_result").content == "true";
                }
                if (req.has_file("stream_options_chunk_size")) {
                    try {
                        request["stream_options"]["chunk_size"] = std::stoi(req.get_file_value("stream_options_chunk_size").content);
                    } catch (const std::exception &) {
                        res_error(res, format_error_response("\"stream_options_chunk_size\" must be an integer", ERROR_TYPE_INVALID_REQUEST));
                        return;
                    }
                }
                if (req.has_file("stream_options_preview")) {
                    request["stream_options"]["preview"] = req.get_file_value("stream_options_preview").content == "true";
                }
                if (req.has_file("stream_options_preview_faster")) {
                    request["stream_options"]["preview_faster"] = req.get_file_value("stream_options_preview_faster").content == "true";
                }
            }
            request = oaicompat_images_edits_request(ctx_server.sd_params, request);
        }

        // construct task
        std::vector<server_task> tasks = ctx_server.create_tasks_inference(rid, request, SERVER_TASK_TYPE_IMAGE);

        // post task
        std::unordered_set<int> task_ids = ctx_server.queue_tasks.post(tasks);
        ctx_server.queue_results.add_waiting_task_ids(task_ids);

        // get result: process non-streaming requests
        if (!json_value(request, "stream", false)) {
            std::vector<json> usages;
            ctx_server.receive_multi_results(
                task_ids,
                [&](std::vector<server_task_result> &results) {
                    json response = json::array();
                    for (const server_task_result &ret : results) {
                        response.push_back(ret.data);
                        if (ret.stop) {
                            usages.push_back(ret.data.at("timings"));
                        }
                    }

                    response = oaicompat_images_response(request, response, false, true, usages);
                    res_ok(res, response);
                },
                [&](const json &error_data) {
                    res_error(res, error_data);
                });

            ctx_server.cancel_tasks(task_ids);
            return;
        }

        // get result: process streaming requests
        const auto on_chunk = [&ctx_server, task_ids, request, rid](size_t, httplib::DataSink &sink) {
            bool chunk_result = false;
            size_t chunk_size = 4096;
            if (request.contains("stream_options")) {
                chunk_result = json_value(request.at("stream_options"), "chunk_result", false);
                chunk_size   = json_value(request.at("stream_options"), "chunk_size", 4096);
                if (chunk_size < 1024) {
                    chunk_size = 1024;
                }
            }
            std::vector<json> usages;
            std::vector<bool> stops(task_ids.size(), false);
            ctx_server.receive_multi_results_stream(
                task_ids,
                [&](const server_task_result &result) -> bool {
                    if (result.stop) {
                        usages.push_back(result.data.at("timings"));
                        stops[json_value(result.data, "index", 0)] = true;
                    }
                    const bool all_stops = std::all_of(stops.begin(), stops.end(), [](bool stop) { return stop; });
                    if (!result.data.contains("b64_json") || !chunk_result) {
                        json response = oaicompat_images_response(request, json::array({result.data}), true, result.stop, all_stops ? usages : std::vector<json>());
                        if (!server_sent_event(sink, "data", response)) {
                            LOG_WRN("srv             handle_images: rid %s | failed to send chunk data\n", rid.c_str());
                            sink.done();
                            return false;
                        }
                    } else {
                        // split the huge result into chunks
                        std::string b64_json     = result.data.at("b64_json").get<std::string>();
                        int32_t progressed_steps = result.data.at("progressed_steps").get<int32_t>();
                        int32_t progress_steps   = result.data.at("progress_steps").get<int32_t>();

                        json chunk_index                = result.data.at("index");
                        json chunk_model                = result.data.at("model");
                        size_t chunk_sent               = 0;
                        size_t chunk_send               = b64_json.size() / chunk_size + 1;
                        float chunk_send_progress       = 0.0f;
                        float chunk_send_progress_base  = float(progressed_steps - 1) / float(progress_steps);
                        float chunk_send_progress_scale = 1 / float(progress_steps);
                        bool chunk_stop                 = false;
                        while (!chunk_stop) {
                            chunk_sent++;
                            chunk_send_progress = chunk_send_progress_base + float(chunk_sent) / float(chunk_send) * chunk_send_progress_scale;
                            chunk_stop          = chunk_sent == chunk_send;
                            json chunk_data     = {
                                {"index", chunk_index},
                                {"progressed_steps", progressed_steps},
                                {"progress_steps", progress_steps},
                                {"progress", chunk_send_progress * 100},
                                {"stop", chunk_stop},
                                {"model", chunk_model},
                            };
                            if (chunk_stop) {
                                chunk_data["b64_json"] = b64_json;
                            } else {
                                chunk_data["b64_json"] = b64_json.substr(0, chunk_size);
                                b64_json               = b64_json.substr(chunk_size);
                            }
                            json response = oaicompat_images_response(request, json::array({chunk_data}), true, chunk_stop && result.stop, chunk_stop && result.stop ? usages : std::vector<json>());
                            if (!server_sent_event(sink, "data", response)) {
                                LOG_WRN("srv             handle_images: rid %s | failed to send chunk data\n", rid.c_str());
                                sink.done();
                                return false;
                            }
                        }
                    }
                    if (all_stops) {
                        const std::string done = "data: [DONE] \n\n";
                        if (!sink.write(done.c_str(), done.size())) {
                            LOG_WRN("srv             handle_images: rid %s | failed to send chunk data\n", rid.c_str());
                            sink.done();
                            return false;
                        }
                        sink.done();
                    }
                    return true;
                },
                [&](const json &error_data) {
                    server_sent_event(sink, "error", error_data);
                    sink.done();
                });

            return false;
        };
        const auto on_complete = [&ctx_server, task_ids](bool) {
            ctx_server.cancel_tasks(task_ids);
        };

        res.set_header("Cache-Control", "no-cache, no-store, no-transform");
        res.set_header("Connection", "keep-alive");
        res.set_chunked_content_provider("text/event-stream", on_chunk, on_complete);
    };

    //
    // Router
    //

    svr.Get("/health", handle_health);
    if (llm_params.endpoint_metrics) {
        svr.Get("/metrics", handle_metrics);
    }
    svr.Get("/props", handle_props);
    svr.Post("/tokenize", handle_tokenize);
    svr.Post("/detokenize", handle_detokenize);
    if (llm_params.endpoint_slots) {
        svr.Get("/slots", handle_slots_list);
        if (!llm_params.slot_save_path.empty()) {
            // only enable slot operate endpoint if slot_save_path is set
            svr.Post("/slots/:id_slot", handle_slots_action);
        }
    }
    if (!llm_params.lora_adapters.empty()) {
        svr.Get("/lora-adapters", handle_lora_adapters_list);
        if (llm_params.lora_init_without_apply) {
            // only enable lora adapters apply endpoint if lora_init_without_apply is set
            svr.Post("/lora-adapters", handle_lora_adapters_apply);
        }
    }
    if (params.endpoint_infill) {
        svr.Post("/infill", handle_infill);
    }
    svr.Post("/completion", handle_completions);
    svr.Get("/v1/models", handle_models);
    svr.Post("/v1/completions", handle_completions);
    svr.Post("/v1/chat/completions", handle_chat_completions);
    if (llm_params.embedding) {
        svr.Post("/v1/embeddings", handle_embeddings);
    }
    if (llm_params.reranking) {
        svr.Post("/v1/rerank", handle_rerank);
    }
    if (params.endpoint_images) {
        svr.Post("/v1/images/generations", handle_images);
        svr.Post("/v1/images/edits", handle_images);
    }

    //
    // Start
    //

    // +2 threads for monitoring endpoints: /metrics and /slots
    const int32_t n_threads_http_addition = 2;
    int32_t n_threads_http                = llm_params.n_threads_http;
    if (n_threads_http < 1) {
        n_threads_http = llm_params.n_parallel + n_threads_http_addition;
    }
    svr.new_task_queue = [&n_threads_http] { return new httplib::ThreadPool(n_threads_http); };

    // bind HTTP listen port
    bool was_bound = false;
    if (llm_params.port == 0) {
        int bound_port = svr.bind_to_any_port(llm_params.hostname);
        if ((was_bound = (bound_port >= 0))) {
            llm_params.port = bound_port;
        }
    } else {
        was_bound = svr.bind_to_port(llm_params.hostname, llm_params.port);
    }
    if (!was_bound) {
        SRV_ERR("%s", "existing due to listening error\n");
        ctx_server.clean(svr);
        return 1;
    }
    SRV_INF("listening, hostname = %s, port = %d, n_threads = %d + %d\n",
            llm_params.hostname.c_str(), llm_params.port, n_threads_http, n_threads_http_addition);

    // run the HTTP server in a thread
    std::thread t([&]() { svr.listen_after_bind(); });
    svr.wait_until_ready();

    // load the model
    SRV_INF("%s", "loading model\n");
    if (!ctx_server.load_model(params)) {
        ctx_server.clean(svr);
        t.join();
        SRV_ERR("%s", "exiting due to model loading error\n");
        return 1;
    }

    // init server
    SRV_INF("%s", "initializing server\n");
    if (!ctx_server.init()) {
        ctx_server.clean(svr);
        t.join();
        SRV_ERR("%s", "exiting due to server initializing error\n");
        return 1;
    }
    state.store(SERVER_STATE_READY);

    ctx_server.queue_tasks.on_new_task(
        std::bind(&server_context::process_single_task, &ctx_server, std::placeholders::_1));
    ctx_server.queue_tasks.on_update_slots(
        std::bind(&server_context::update_slots, &ctx_server));
    shutdown_handler = [&](int) {
        ctx_server.queue_tasks.terminate();
    };

    SRV_INF("%s", "starting server\n");
    ctx_server.queue_tasks.start_loop();

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    struct sigaction sigint_action{};
    sigint_action.sa_handler = signal_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, nullptr);
    sigaction(SIGTERM, &sigint_action, nullptr);
#elif defined(_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    SRV_INF("%s", "stopping server\n");
    ctx_server.clean(svr);
    t.join();

    return 0;
}
