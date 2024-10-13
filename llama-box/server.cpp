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
#include "llama.cpp/ggml/include/ggml.h"
#include "llama.cpp/include/llama.h"

#include "llama.cpp/examples/llava/clip.h"
#include "llama.cpp/examples/llava/llava.h"

#include "param.hpp"
#include "ratelimiter.hpp"
#include "rpcserver.hpp"
#include "utils.hpp"

// mime type for sending response
#define MIMETYPE_JSON "application/json; charset=utf-8"

using json = nlohmann::json;

enum stop_type {
    STOP_TYPE_FULL,
    STOP_TYPE_PARTIAL,
};

enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

enum server_state {
    SERVER_STATE_LOADING_MODEL, // Server is starting up, model not fully
                                // loaded yet
    SERVER_STATE_READY,         // Server is ready and model is loaded
};

enum server_task_type {
    SERVER_TASK_TYPE_COMPLETION,
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
    SERVER_TASK_TYPE_METRICS,
    SERVER_TASK_TYPE_SLOT_SAVE,
    SERVER_TASK_TYPE_SLOT_RESTORE,
    SERVER_TASK_TYPE_SLOT_ERASE,
    SERVER_TASK_TYPE_SET_LORA,
};

enum server_task_cmpl_type {
    SERVER_TASK_CMPL_TYPE_NORMAL,
    SERVER_TASK_CMPL_TYPE_EMBEDDING,
    SERVER_TASK_CMPL_TYPE_RERANK,
    SERVER_TASK_CMPL_TYPE_INFILL,
};

struct server_task {
    int id = -1; // to be filled by server_queue
    int id_target = -1;

    json data;
    server_task_type type;

    server_task_cmpl_type cmpl_type = SERVER_TASK_CMPL_TYPE_NORMAL;

    int tps = 0;

    // utility function
    static std::unordered_set<int> get_list_id(const std::vector<server_task> &tasks) {
        std::unordered_set<int> ids(tasks.size());
        for (size_t i = 0; i < tasks.size(); i++) {
            ids.insert(tasks[i].id);
        }
        return ids;
    }
};

struct server_task_result {
    int id = -1;

    json data;

    bool stop;
    bool error;
};

struct slot_params {
    bool stream = true;
    bool cache_prompt = false; // remember the prompt to avoid reprocessing all prompt

    int32_t n_keep = 0;     // number of tokens to keep from initial prompt
    int32_t n_discard = 0;  // number of tokens after n_keep that may be discarded when shifting
                            // context, 0 defaults to half
    int32_t n_predict = -1; // new tokens to predict

    std::vector<std::string> antiprompt;

    json input_prefix;
    json input_suffix;
};

struct server_slot {
    int id;
    int id_task = -1;

    // the index relative to completion multi-task request
    size_t index = 0;

    struct slot_params params;

    slot_state state = SLOT_STATE_IDLE;

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx = 0; // context size per slot
    int32_t n_past = 0;
    int32_t n_decoded = 0;
    int32_t n_remaining = -1;
    int32_t i_batch = -1;
    int32_t n_predict = -1; // TODO: disambiguate from params.n_predict

    int32_t n_prompt_tokens = 0;
    int32_t n_prompt_tokens_processed = 0;

    // can be either a string, array of strings or array of token ids
    json prompt;
    // when a task is submitted, we first tokenize the prompt and store it here
    std::vector<llama_token> prompt_tokens;

    std::string generated_text;
    std::vector<llama_token> cache_tokens;
    std::vector<completion_token_output> generated_token_probs;

    server_task_cmpl_type cmpl_type = SERVER_TASK_CMPL_TYPE_NORMAL;

    bool has_next_token = true;
    bool truncated = false;
    bool stopped_eos = false;
    bool stopped_word = false;
    bool stopped_limit = false;

    std::string stopping_word;

    bool oaicompat_completion = false;
    bool oaicompat_completion_chat = false;
    bool oaicompat_completion_chat_vision = false;

    // sampling
    std::vector<llama_token> sampled;

    json json_schema;

    struct common_sampler_params sparams;
    struct common_sampler *smpl = nullptr;

    // stats
    size_t n_sent_text = 0; // number of sent text character
    size_t n_sent_token_probs = 0;

    int64_t t_start_process_prompt;
    int64_t t_start_generation;

    double t_prompt_processing; // ms
    double t_token_generation;  // ms

    std::function<void(int)> callback_on_release;

    token_bucket *token_bkt = nullptr; // bucket for tokens per second

    /* speculative decoding */
    int32_t n_drafted = 0;
    int32_t n_drafted_accepted = 0;
    std::vector<llama_token> sampled_draft;
    // draft-model speculative decoding
    struct common_sampler *smpl_draft = nullptr;
    // model-free speculative decoding
    int32_t lookup_ngram_min = 0;
    common_ngram_cache ctx_ngram_cache;

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        n_prompt_tokens = 0;
        generated_text = "";
        truncated = false;
        stopped_eos = false;
        stopped_word = false;
        stopped_limit = false;
        stopping_word = "";
        n_past = 0;
        n_sent_text = 0;
        n_sent_token_probs = 0;
        cmpl_type = SERVER_TASK_CMPL_TYPE_NORMAL;

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

        n_drafted = 0;
        n_drafted_accepted = 0;
        sampled_draft.clear();

        if (smpl_draft != nullptr) {
            common_sampler_free(smpl_draft);
            smpl_draft = nullptr;
        }

        lookup_ngram_min = 0;
    }

    bool has_budget(common_params &global_params) {
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

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
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
            SLT_INF(*this, "stop processing: n_past = %d, truncated = %d\n", n_past, truncated);

            t_token_generation = double(ggml_time_us() - t_start_generation) / 1e3;
            state = SLOT_STATE_IDLE;
            callback_on_release(id);
        }
    }

    json get_formated_timings() const {
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
            ret["predicted_per_token_ms"] = t_token_generation / n_decoded_with_drafted;
            ret["predicted_per_second"] = 1e3 / t_token_generation * n_decoded_with_drafted;
            ret["drafted_n"] = n_drafted;
            ret["drafted_accepted_n"] = n_drafted_accepted;
            ret["drafted_accepted_p"] = float(n_drafted_accepted) / float(n_drafted);
        }

        return ret;
    }

    size_t find_stopping_strings(const std::string &text, const size_t last_token_size, const stop_type type) {
        size_t stop_pos = std::string::npos;

        for (const std::string &word : params.antiprompt) {
            size_t pos;

            if (type == STOP_TYPE_FULL) {
                const size_t tmp = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                pos = find_partial_stop_string(word, text);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (type == STOP_TYPE_FULL) {
                    stopped_word = true;
                    stopping_word = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void push_token_into_result(llama_token tok, completion_token_output &result) {
        if (lookup_ngram_min > 0) {
            prompt_tokens.push_back(tok);
            common_ngram_cache_update(ctx_ngram_cache, lookup_ngram_min, LLAMA_NGRAM_MAX, prompt_tokens, 1, false);
        }

        result.toks.push_back(tok);

        if (sparams.n_probs > 0) {
            result.probss.emplace_back();
            const auto last_idx = int32_t(result.probss.size() - 1);
            const llama_token_data_array *cur_p = common_sampler_get_candidates(smpl);
            for (size_t i = 0; i < (size_t)sparams.n_probs; ++i) {
                result.probss[last_idx].push_back({
                    cur_p->data[i].id,
                    i >= cur_p->size ? 0.0f : cur_p->data[i].p,
                });
            }
        }
    }
};

struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total = 0;
    uint64_t n_tokens_predicted_total = 0;
    uint64_t t_tokens_generation_total = 0;
    uint64_t n_tokens_drafted_total = 0;
    uint64_t n_tokens_drafted_accepted_total = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing = 0;

    uint64_t n_tokens_predicted = 0;
    uint64_t t_tokens_generation = 0;
    uint64_t n_tokens_drafted = 0;
    uint64_t n_tokens_drafted_accepted = 0;

    uint64_t n_decode_total = 0;
    uint64_t n_busy_slots_total = 0;

    void init() {
        t_start = ggml_time_us();
    }

    void on_prompt_eval(const server_slot &slot) {
        n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
        n_prompt_tokens_processed += slot.n_prompt_tokens_processed;
        t_prompt_processing += uint64_t(slot.t_prompt_processing);
        t_prompt_processing_total += uint64_t(slot.t_prompt_processing);
    }

    void on_prediction(const server_slot &slot) {
        n_tokens_predicted_total += slot.n_decoded;
        n_tokens_predicted += slot.n_decoded;
        t_tokens_generation += uint64_t(slot.t_token_generation);
        t_tokens_generation_total += uint64_t(slot.t_token_generation);
        n_tokens_drafted += slot.n_drafted;
        n_tokens_drafted_total += slot.n_drafted;
        n_tokens_drafted_accepted += slot.n_drafted_accepted;
        n_tokens_drafted_accepted_total += slot.n_drafted_accepted;
    }

    void on_decoded(const std::vector<server_slot> &slots) {
        n_decode_total++;
        for (const auto &slot : slots) {
            if (slot.is_processing()) {
                n_busy_slots_total++;
            }
        }
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        t_prompt_processing = 0;
        n_tokens_predicted = 0;
        t_tokens_generation = 0;
        n_tokens_drafted = 0;
        n_tokens_drafted_accepted = 0;
    }
};

struct server_queue {
    int id = 0;
    bool running;

    // queues
    std::deque<server_task> queue_tasks;
    std::deque<server_task> queue_tasks_deferred;

    std::mutex mutex_tasks;
    std::condition_variable condition_tasks;

    // callback functions
    std::function<void(server_task &)> callback_new_task;
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
    int post(std::vector<server_task> &tasks, bool front = false) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
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
        }
        condition_tasks.notify_one();
        return 0;
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
    void on_new_task(std::function<void(server_task &)> callback) {
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
                callback_new_task(task);
            }

            // all tasks in the current loop is processed, slots data is now
            // ready
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

struct server_response {
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
    void add_waiting_tasks(const std::vector<server_task> &tasks) {
        std::unique_lock<std::mutex> lock(mutex_results);
        for (const auto &task : tasks) {
            SRV_DBG("add task %d to waiting list. current waiting = %d (before add)\n", task.id, (int)waiting_task_ids.size());
            waiting_task_ids.insert(task.id);
        }
    }

    // when the request is finished, we can remove task associated with it
    void remove_waiting_task_id(int id_task) {
        SRV_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int)waiting_task_ids.size());
        std::unique_lock<std::mutex> lock(mutex_results);
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
        SRV_DBG("sending result for task id = %d\n", result.id);
        for (const auto &id_task : waiting_task_ids) {
            if (result.id == id_task) {
                SRV_DBG("task id = %d moved to result queue\n", result.id);
                queue_results.push_back(std::move(result));
                condition_results.notify_all();
                return;
            }
        }
    }
};

struct server_context {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    clip_ctx *ctx_clip = nullptr;
    std::vector<common_lora_adapter_container> lora_adapters;

    common_params params;

    llama_batch batch = {};

    bool add_bos_token = true;
    bool add_eos_token = false;

    int32_t n_ctx;            // total context for all clients / slots
    int32_t n_tps;            // max tokens per second
    int32_t lookup_ngram_min; // min ngram for lookup cache

    // slots / clients
    std::vector<server_slot> slots;
    json default_generation_settings_for_props;

    server_queue queue_tasks;
    server_response queue_results;

    server_metrics metrics;

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    // draft-model speculative decoding
    llama_batch batch_draft;
    llama_model *model_draft = nullptr;
    llama_context *ctx_draft = nullptr;
    // model-free speculative decoding
    common_ngram_cache ngram_cache_static;
    common_ngram_cache ngram_cache_dynamic;

    ~server_context() {
        if (ctx_clip != nullptr) {
            clip_free(ctx_clip);
            ctx_clip = nullptr;
        }

        if (ctx != nullptr) {
            llama_free(ctx);
            ctx = nullptr;
        }

        if (model != nullptr) {
            llama_free_model(model);
            model = nullptr;
        }

        // Clear any sampling context
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

        if (ctx_draft != nullptr) {
            llama_free(ctx_draft);
            ctx_draft = nullptr;
        }
        if (model_draft != nullptr) {
            llama_free_model(model_draft);
            model_draft = nullptr;
        }
        llama_batch_free(batch_draft);

        ngram_cache_static.clear();
        ngram_cache_dynamic.clear();
    }

    bool load_model(const llama_box_params &bparams) {
        params = bparams.gparams;

        // load multimodal projection model
        if (!params.mmproj.empty()) {
            if (params.n_ctx < 2048) {
                SRV_WRN("%s", "n_ctx is too small for multimodal projection, setting to 2048\n");
                params.n_ctx = 2048;
            }
            ctx_clip = clip_model_load(params.mmproj.c_str(), /* verbosity */ 1);
            if (ctx_clip == nullptr) {
                SRV_ERR("failed to load multimodal project model, '%s'\n", params.mmproj.c_str());
                return false;
            }
        }

        // load the draft model if needed
        if (!params.model_draft.empty() && params.n_draft > 0) {
            common_params params_draft = params;
            params_draft.model = params.model_draft;
            params_draft.n_gpu_layers = params.n_gpu_layers_draft;
            params_draft.cpuparams = params.draft_cpuparams;
            params_draft.cpuparams_batch = params.draft_cpuparams_batch;
            params_draft.warmup = false;
            common_init_result ir = common_init_from_params(params_draft);
            model_draft = ir.model;
            ctx_draft = ir.context;
            if (model_draft == nullptr) {
                SRV_ERR("failed to load draft model, '%s'\n", params.model_draft.c_str());
                return false;
            }
        }

        // load the ngram cache if needed
        if (bparams.lookup_ngram_min > 0) {
            if (!params.lookup_cache_static.empty()) {
                try {
                    ngram_cache_static = common_ngram_cache_load(params.lookup_cache_static);
                } catch (std::ifstream::failure const &) {
                    SRV_ERR("failed to load static ngram cache, '%s'\n", params.lookup_cache_static.c_str());
                    return false;
                }
            }
            if (!params.lookup_cache_dynamic.empty()) {
                try {
                    ngram_cache_dynamic = common_ngram_cache_load(params.lookup_cache_dynamic);
                } catch (std::ifstream::failure const &) {
                    // NOP
                }
            }
        }

        // reserve one extra sequence (seq_id == 0) for extra features
        params.n_parallel += 1;
        common_init_result ir = common_init_from_params(params);
        model = ir.model;
        ctx = ir.context;
        lora_adapters = ir.lora_adapters;
        params.n_parallel -= 1; // but be sneaky about it
        if (model == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params.model.c_str());
            return false;
        }

        // check multimodal projection model compatibility
        if (ctx_clip != nullptr) {
            const int n_embd_clip = clip_n_mmproj_embd(ctx_clip);
            const int n_embd = llama_n_embd(model);
            if (n_embd_clip != n_embd) {
                SRV_ERR("multimodal projector embedding length is not equal to the model, "
                        "n_embd_clip = %d, n_embd = %d\n",
                        n_embd_clip, n_embd);
                return false;
            }
        }

        // check draft model compatibility if needed
        if (ctx_draft != nullptr) {
            const bool vocab_type_draft = llama_vocab_type(model_draft);
            const bool vocab_type = llama_vocab_type(model);
            if (vocab_type_draft != vocab_type) {
                SRV_ERR("draft model vocabulary type is not equal to the model, "
                        "vocab_type_draft = %d, vocab_type = %d\n",
                        vocab_type_draft, vocab_type);
                return false;
            }

            if (llama_add_bos_token(model_draft) != llama_add_bos_token(model) || llama_add_eos_token(model_draft) != llama_add_eos_token(model) ||
                llama_token_bos(model_draft) != llama_token_bos(model) || llama_token_eos(model_draft) != llama_token_eos(model)) {
                SRV_ERR("%s", "draft model special tokens are not equal to the model\n");
                return false;
            }
        }

        n_ctx = int32_t(llama_n_ctx(ctx));
        n_tps = bparams.n_tps;
        lookup_ngram_min = bparams.lookup_ngram_min;

        add_bos_token = llama_add_bos_token(model);
        add_eos_token = llama_add_eos_token(model);

        // sample tokens per second
        if (n_tps < 0) {
            SRV_INF("%s", "sampling tokens per second, this will take some time...\n");
            const int32_t n_check = std::min(n_ctx, params.n_ubatch);
            std::vector<llama_token> check_prompt_tokens = {llama_token_bos(model)};
            common_sampler *check_smpl = common_sampler_init(model, params.sparams);
            int64_t t_start_decoding = ggml_time_us();
            int32_t n_check_decoded = 0;
            while (true) {
                auto i = int32_t(check_prompt_tokens.size());
                if (i >= n_check) {
                    break;
                }
                if (llama_decode(ctx, llama_batch_get_one(&check_prompt_tokens[i - 1], 1, 0, 0))) {
                    break;
                }
                n_check_decoded++;
                const int32_t id = common_sampler_sample(check_smpl, ctx, 0);
                if (llama_token_is_eog(model, id)) {
                    break;
                }
                common_sampler_accept(check_smpl, id, false);
                check_prompt_tokens.push_back(id);
            }
            n_tps = ceil(1e3 / (double(ggml_time_us() - t_start_decoding) / 1e3) * n_check_decoded);
            common_sampler_free(check_smpl);
            llama_kv_cache_clear(ctx);
            llama_synchronize(ctx);
            llama_perf_context_reset(ctx);
            SRV_INF("sampled tokens per second, tps = %d\n", n_tps);
        }

        return true;
    }

    std::string load_chat_template() const {
        std::string tkey = "tokenizer.chat_template";
        int32_t tlen = llama_model_meta_val_str(model, tkey.c_str(), nullptr, 0);
        if (tlen > 0) {
            std::vector<char> tval(tlen + 1, 0);
            if (llama_model_meta_val_str(model, tkey.c_str(), tval.data(), tlen + 1) == tlen) {
                return {tval.data(), (unsigned long)tlen};
            }
        }
        return "chatml"; // see llama_chat_apply_template_internal
    }

    bool init() {
        SRV_INF("initializing slots, n_slots = %d\n", params.n_parallel);

        const int32_t n_ctx_slot = n_ctx / params.n_parallel;
        for (int i = 0; i < params.n_parallel; i++) {
            server_slot slot;

            slot.id = i;
            slot.n_ctx = n_ctx_slot;
            slot.n_predict = params.n_predict;

            SLT_INF(slot, "new slot n_ctx_slot = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int) { queue_tasks.pop_deferred_task(); };

            slot.reset();

            slots.push_back(slot);
        }

        default_generation_settings_for_props = get_formated_generation(slots.front());
        default_generation_settings_for_props["seed"] = -1;

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel
        // tokens note that n_batch can be > n_ctx (e.g. for non-causal
        // attention models such as BERT where the KV cache is not used)
        const auto n_batch = int32_t(llama_n_batch(ctx));
        batch = llama_batch_init(std::max(n_batch, params.n_parallel), 0, 1);
        if (ctx_draft != nullptr) {
            batch_draft = llama_batch_init(std::max(n_batch, params.n_parallel), 0, 1);
        }

        metrics.init();
        return true;
    }

    void clean(httplib::Server &svr) {
        svr.stop();
        if (lookup_ngram_min > 0 && !params.lookup_cache_dynamic.empty()) {
            common_ngram_cache_save(ngram_cache_dynamic, params.lookup_cache_dynamic);
        }
        llama_backend_free();
    }

    std::vector<llama_token> tokenize(const json &prompt, bool add_special, bool parse_special) const {
        // If `add_bos` is true, we only add BOS, when json_prompt is a string,
        // or the first element of the json_prompt array is a string.
        std::vector<llama_token> prompt_tokens;

        if (prompt.is_array()) {
            bool first = true;
            for (const json &jp : prompt) {
                if (jp.is_string()) {
                    std::string s = jp.get<std::string>();
                    std::vector<llama_token> p;
                    if (first) {
                        p = common_tokenize(ctx, s, add_special, parse_special);
                        first = false;
                    } else {
                        p = common_tokenize(ctx, s, false, parse_special);
                    }
                    prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
                } else if (jp.is_object() && jp.contains("text")) {
                    std::string s = json_value(jp, "text", std::string());
                    std::vector<llama_token> p;
                    if (first) {
                        p = common_tokenize(ctx, s, add_special, parse_special);
                        first = false;
                    } else {
                        p = common_tokenize(ctx, s, false, parse_special);
                    }
                    prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
                } else if (jp.is_number()) {
                    prompt_tokens.push_back(jp.get<llama_token>());
                }
            }
        } else {
            std::string s = prompt.get<std::string>();
            prompt_tokens = common_tokenize(ctx, s, add_special, parse_special);
        }

        return prompt_tokens;
    }

    server_slot *get_slot_by_id(int id) {
        for (server_slot &slot : slots) {
            if (slot.id == id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot *get_available_slot(const std::string &prompt) {
        server_slot *ret = nullptr;

        // find the slot that has at least n% prompt similarity
        if (ret == nullptr && slot_prompt_similarity != 0.0f && !prompt.empty()) {
            int max_lcp_len = 0;
            float similarity = 0;

            for (server_slot &slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // skip the slot if it does not contains prompt
                if (!slot.prompt.is_string()) {
                    continue;
                }

                // current slot's prompt
                std::string slot_prompt = slot.prompt.get<std::string>();

                // length of the current slot's prompt
                int slot_prompt_len = int(slot_prompt.size());

                // length of the Longest Common Prefix between the current
                // slot's prompt and the input prompt
                int lcp_len = int(common_part(slot_prompt, prompt));

                // fraction of the common substring length compared to the
                // current slot's prompt length
                similarity = float(lcp_len) / float(slot_prompt_len);

                // select the current slot if the criteria match
                if (lcp_len > max_lcp_len && similarity > slot_prompt_similarity) {
                    max_lcp_len = lcp_len;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_DBG(*ret, "selected slot by lcp similarity, max_lcp_len = %d, similarity = %f\n", max_lcp_len, similarity);
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
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_DBG(*ret, "selected slot by lru, t_last = %" PRId64 "\n", t_last);
            }
        }

        return ret;
    }

    bool launch_slot_with_task(server_slot &slot, const server_task &task) {
        common_sampler_params sparams = params.sparams;
        const json &data = task.data;

        slot.oaicompat_completion = json_value(data, "__oaicompat_completion", false);
        slot.oaicompat_completion_chat = json_value(data, "__oaicompat_completion_chat", false);
        slot.oaicompat_completion_chat_vision = json_value(data, "__oaicompat_completion_chat_vision", false) && ctx_clip != nullptr;

        slot.params.stream = json_value(data, "stream", false);
        slot.params.cache_prompt = json_value(data, "cache_prompt", false);
        slot.params.n_keep = json_value(data, "n_keep", params.n_keep);
        slot.params.n_predict = json_value(data, "n_predict", params.n_predict);
        slot.params.n_discard = json_value(data, "n_discard", 0);
        slot.params.input_prefix = json_value(data, "input_prefix", params.input_prefix);
        slot.params.input_suffix = json_value(data, "input_suffix", params.input_suffix);

        slot.sparams.top_k = json_value(data, "top_k", sparams.top_k);
        slot.sparams.top_p = json_value(data, "top_p", sparams.top_p);
        slot.sparams.min_p = json_value(data, "min_p", sparams.min_p);
        slot.sparams.tfs_z = json_value(data, "tfs_z", sparams.tfs_z);
        slot.sparams.typ_p = json_value(data, "typical_p", sparams.typ_p);
        slot.sparams.temp = json_value(data, "temperature", sparams.temp);
        slot.sparams.dynatemp_range = json_value(data, "dynatemp_range", sparams.dynatemp_range);
        slot.sparams.dynatemp_exponent = json_value(data, "dynatemp_exponent", sparams.dynatemp_exponent);
        slot.sparams.penalty_last_n = json_value(data, "repeat_last_n", sparams.penalty_last_n);
        slot.sparams.penalty_repeat = json_value(data, "repeat_penalty", sparams.penalty_repeat);
        slot.sparams.penalty_freq = json_value(data, "frequency_penalty", sparams.penalty_freq);
        slot.sparams.penalty_present = json_value(data, "presence_penalty", sparams.penalty_present);
        slot.sparams.mirostat = json_value(data, "mirostat", sparams.mirostat);
        slot.sparams.mirostat_tau = json_value(data, "mirostat_tau", sparams.mirostat_tau);
        slot.sparams.mirostat_eta = json_value(data, "mirostat_eta", sparams.mirostat_eta);
        slot.sparams.penalize_nl = json_value(data, "penalize_nl", sparams.penalize_nl);
        slot.sparams.seed = json_value(data, "seed", sparams.seed);
        slot.sparams.n_probs = json_value(data, "n_probs", sparams.n_probs);
        slot.sparams.min_keep = json_value(data, "min_keep", sparams.min_keep);

        // process "json_schema" and "grammar"
        if (data.contains("json_schema") && !data.at("json_schema").is_null() && data.contains("grammar") && !data.at("grammar").is_null()) {
            send_error(task,
                       "Either \"json_schema\" or \"grammar\" can be "
                       "specified, but not both",
                       ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        if (data.contains("json_schema") && !data.contains("grammar")) {
            try {
                auto schema = json_value(data, "json_schema", json::object());
                slot.sparams.grammar = json_schema_to_grammar(schema);
            } catch (const std::exception &e) {
                send_error(task, std::string("\"json_schema\": ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
        } else {
            slot.sparams.grammar = json_value(data, "grammar", sparams.grammar);
        }

        if (slot.n_predict > 0 && slot.params.n_predict > slot.n_predict) {
            // Might be better to reject the request with a 400 ?
            slot.params.n_predict = slot.n_predict;
            SLT_WRN(slot, "n_predict = %d exceeds server configuration, setting to %d", slot.n_predict, slot.n_predict);
        }

        // get prompt
        if (task.cmpl_type != SERVER_TASK_CMPL_TYPE_INFILL) {
            const auto &prompt = data.find("prompt");
            if (prompt == data.end()) {
                send_error(task, "\"prompt\" must be provided", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            if ((prompt->is_string()) || (prompt->is_array() && !prompt->empty())) {
                slot.prompt = *prompt;
            } else {
                send_error(task, "\"prompt\" must be a string or an array", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            if (slot.oaicompat_completion_chat_vision) {
                slot.params.cache_prompt = false;
            }
        }

        {
            slot.sparams.logit_bias.clear();

            if (json_value(data, "ignore_eos", false)) {
                slot.sparams.logit_bias.push_back({llama_token_eos(model), -INFINITY});
            }

            const auto &logit_bias = data.find("logit_bias");
            if (logit_bias != data.end() && logit_bias->is_array()) {
                const int n_vocab = llama_n_vocab(model);
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
                                slot.sparams.logit_bias.push_back({tok, bias});
                            }
                        } else if (el[0].is_string()) {
                            auto toks = common_tokenize(model, el[0].get<std::string>(), false);
                            for (auto tok : toks) {
                                slot.sparams.logit_bias.push_back({tok, bias});
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
            if (samplers != data.end() && samplers->is_array()) {
                std::vector<std::string> sampler_names;
                for (const auto &name : *samplers) {
                    if (name.is_string()) {
                        sampler_names.emplace_back(name);
                    }
                }
                slot.sparams.samplers = common_sampler_types_from_names(sampler_names, false);
            } else {
                slot.sparams.samplers = sparams.samplers;
            }
        }

        {
            if (slot.smpl != nullptr) {
                common_sampler_free(slot.smpl);
            }

            slot.smpl = common_sampler_init(model, slot.sparams);
            if (slot.smpl == nullptr) {
                // for now, the only error that may happen here is invalid
                // grammar
                send_error(task, "Failed to parse grammar", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
        }

        {
            if (slot.token_bkt != nullptr) {
                delete slot.token_bkt;
                slot.token_bkt = nullptr;
            }
            int tps = task.tps;
#ifndef NDEBUG
            // allow overriding tps for debugging
            tps = json_value(data, "tps", task.tps);
            if (tps > n_tps) {
                tps = n_tps;
            }
#endif
            if (tps > 0) {
                slot.token_bkt = new token_bucket(tps, tps);
                if (slot.token_bkt == nullptr) {
                    send_error(task, "Failed to create token bucket", ERROR_TYPE_SERVER);
                    return false;
                }
            }
        }

        if (lookup_ngram_min > 0) {
            slot.lookup_ngram_min = lookup_ngram_min;
            if (!slot.ctx_ngram_cache.empty()) {
                common_ngram_cache_merge(ngram_cache_dynamic, slot.ctx_ngram_cache);
                slot.ctx_ngram_cache.clear();
                const auto sz_ngram_cache = int32_t(ngram_cache_dynamic.size());
                if (sz_ngram_cache >= 100000) {
                    if (!params.lookup_cache_dynamic.empty()) {
                        common_ngram_cache_save(ngram_cache_dynamic, params.lookup_cache_dynamic);
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

        slot.state = SLOT_STATE_PROCESSING_PROMPT;
        slot.prompt_tokens.clear();

        SLT_INF(slot, "processing task, max_tps = %s\n", slot.token_bkt ? std::to_string(slot.token_bkt->capacity).c_str() : "N/A");

        return true;
    }

    bool process_token(completion_token_output &result, server_slot &slot) {
        // remember which tokens were sampled - used for repetition penalties
        // during sampling
        slot.sampled.clear();
        std::string token_str;
        for (const llama_token &tok : result.toks) {
            token_str += common_token_to_piece(ctx, tok, params.special);
            slot.sampled.push_back(tok);
        }

        // search stop word and delete it
        slot.generated_text += token_str;
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = false;
        for (unsigned i = 1; i < 5 && i <= slot.generated_text.size(); ++i) {
            unsigned char c = slot.generated_text[slot.generated_text.size() - i];
            if ((c & 0xC0) == 0x80) {
                // continuation byte: 10xxxxxx
                continue;
            }
            if ((c & 0xE0) == 0xC0) {
                // 2-byte character: 110xxxxx ...
                incomplete = i < 2;
            } else if ((c & 0xF0) == 0xE0) {
                // 3-byte character: 1110xxxx ...
                incomplete = i < 3;
            } else if ((c & 0xF8) == 0xF0) {
                // 4-byte character: 11110xxx ...
                incomplete = i < 4;
            }
            // else 1-byte character or invalid byte
            break;
        }

        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool is_stop_full = false;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), STOP_TYPE_FULL);
            if (stop_pos != std::string::npos) {
                is_stop_full = true;
                slot.generated_text.erase(slot.generated_text.begin() + long(pos) + long(stop_pos), slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else {
                is_stop_full = false;
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), STOP_TYPE_PARTIAL);
            }

            // check if there is any token to predict
            if (stop_pos == std::string::npos || (!is_stop_full && stop_pos > 0)) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, stop_pos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            }

            slot.add_token(result);
            if (slot.params.stream) {
                send_partial_response(slot, result);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // check the limits
        if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(params)) {
            slot.stopped_limit = true;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_decoded = %d, n_predict = %d\n", slot.n_decoded, slot.params.n_predict);
        }

        // if context shift is disabled, we stop when it reaches the context limit
        if (!params.ctx_shift && slot.n_past >= slot.n_ctx) {
            slot.truncated = true;
            slot.stopped_limit = true;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, n_past = %d, n_prompt_tokens = %d, n_decoded = %d, n_ctx = %d\n",
                    slot.n_past, slot.n_prompt_tokens, slot.n_decoded, slot.n_ctx);
        }

        // check the EOT
        for (const llama_token &tok : result.toks) {
            if (llama_token_is_eog(model, tok)) {
                slot.stopped_eos = true;
                slot.has_next_token = false;

                SLT_DBG(slot, "%s", "stopped by EOS\n");
                break;
            }
        }

        int32_t n_ctx_train = llama_n_ctx_train(model);
        if (slot.params.n_predict < 1 && slot.n_predict < 1 && slot.n_prompt_tokens + slot.n_decoded >= n_ctx_train) {
            SLT_WRN(slot,
                    "n_predict (%d) is set for infinite generation, "
                    "limiting generated tokens to n_ctx_train (%d) to avoid EOS-less generation infinite loop\n",
                    slot.params.n_predict, n_ctx_train);
            slot.truncated = true;
            slot.stopped_limit = true;
            slot.has_next_token = false; // stop prediction
        }

        return slot.has_next_token; // continue
    }

    json get_formated_generation(const server_slot &slot) const {
        std::vector<std::string> samplers;
        samplers.reserve(slot.sparams.samplers.size());
        for (const auto &sampler : slot.sparams.samplers) {
            samplers.emplace_back(common_sampler_type_to_str(sampler));
        }

        return json{{"n_ctx", slot.n_ctx},
                    {"n_predict", slot.n_predict}, // Server configured n_predict
                    {"model", params.model_alias},
                    {"seed", slot.sparams.seed},
                    {"seed_cur", slot.smpl ? common_sampler_get_seed(slot.smpl) : 0},
                    {"temperature", slot.sparams.temp},
                    {"dynatemp_range", slot.sparams.dynatemp_range},
                    {"dynatemp_exponent", slot.sparams.dynatemp_exponent},
                    {"top_k", slot.sparams.top_k},
                    {"top_p", slot.sparams.top_p},
                    {"min_p", slot.sparams.min_p},
                    {"tfs_z", slot.sparams.tfs_z},
                    {"typical_p", slot.sparams.typ_p},
                    {"repeat_last_n", slot.sparams.penalty_last_n},
                    {"repeat_penalty", slot.sparams.penalty_repeat},
                    {"presence_penalty", slot.sparams.penalty_present},
                    {"frequency_penalty", slot.sparams.penalty_freq},
                    {"mirostat", slot.sparams.mirostat},
                    {"mirostat_tau", slot.sparams.mirostat_tau},
                    {"mirostat_eta", slot.sparams.mirostat_eta},
                    {"penalize_nl", slot.sparams.penalize_nl},
                    {"stop", slot.params.antiprompt},
                    {"max_tokens", slot.params.n_predict}, // User configured n_predict
                    {"n_keep", slot.params.n_keep},
                    {"n_discard", slot.params.n_discard},
                    {"ignore_eos", slot.sparams.ignore_eos},
                    {"stream", slot.params.stream},
                    {"n_probs", slot.sparams.n_probs},
                    {"min_keep", slot.sparams.min_keep},
                    {"grammar", slot.sparams.grammar},
                    {"samplers", samplers}};
    }

    void send_error(const server_task &task, const std::string &error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot &slot, const std::string &error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.id_task, error, type);
    }

    void send_error(const int id_task, const std::string &error, const enum error_type type = ERROR_TYPE_SERVER) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        server_task_result res;
        res.id = id_task;
        res.stop = false;
        res.error = true;
        res.data = format_error_response(error, type);

        queue_results.send(res);
    }

    void send_partial_response(server_slot &slot, completion_token_output tkn) {
        server_task_result res;
        res.id = slot.id_task;
        res.error = false;
        res.stop = false;
        res.data = json{{"content", tkn.text_to_send}, {"stop", false}, {"id_slot", slot.id}, {"multimodal", false}};

        if (slot.sparams.n_probs > 0) {
            const std::vector<llama_token> to_send_toks = common_tokenize(ctx, tkn.text_to_send, false);
            const size_t probs_pos = std::min(slot.n_sent_token_probs, slot.generated_token_probs.size());
            const size_t probs_stop_pos = std::min(slot.n_sent_token_probs + to_send_toks.size(), slot.generated_token_probs.size());

            std::vector<completion_token_output> probs_output;
            if (probs_pos < probs_stop_pos) {
                probs_output = std::vector<completion_token_output>(slot.generated_token_probs.begin() + long(probs_pos),
                                                                    slot.generated_token_probs.begin() + long(probs_stop_pos));
            }
            slot.n_sent_token_probs = probs_stop_pos;

            res.data["completion_probabilities"] = probs_vector_to_json(ctx, probs_output, slot.oaicompat_completion, slot.oaicompat_completion_chat);
        }

        queue_results.send(res);
    }

    void send_final_response(const server_slot &slot) {
        server_task_result res;
        res.id = slot.id_task;
        res.error = false;
        res.stop = true;
        res.data = json{{"content", !slot.params.stream ? slot.generated_text : ""},
                        {"id_slot", slot.id},
                        {"stop", true},
                        {"model", params.model_alias},
                        {"tokens_predicted", slot.n_decoded},
                        {"tokens_evaluated", slot.n_prompt_tokens},
                        {"tokens_cached", slot.n_past},
                        {"generation_settings", get_formated_generation(slot)},
                        {"truncated", slot.truncated},
                        {"stopped_eos", slot.stopped_eos},
                        {"stopped_word", slot.stopped_word},
                        {"stopped_limit", slot.stopped_limit},
                        {"stopping_word", slot.stopping_word},
                        {"timings", slot.get_formated_timings()},
                        {"index", slot.index}};

        if (slot.sparams.n_probs > 0) {
            std::vector<completion_token_output> probs;
            if (!slot.params.stream && slot.stopped_word) {
                const std::vector<llama_token> stop_word_toks = common_tokenize(ctx, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                probs = std::vector<completion_token_output>(slot.generated_token_probs.begin(), slot.generated_token_probs.end() - int(safe_offset));
            } else {
                probs = std::vector<completion_token_output>(slot.generated_token_probs.begin(), slot.generated_token_probs.end());
            }

            res.data["completion_probabilities"] = probs_vector_to_json(ctx, probs, slot.oaicompat_completion, slot.oaicompat_completion_chat);
        }

        queue_results.send(res);
    }

    void send_embedding(const server_slot &slot, const llama_batch &batch_view) {
        server_task_result res;
        res.id = slot.id_task;
        res.error = false;
        res.stop = true;

        const int n_embd = llama_n_embd(model);

        std::vector<float> embd_res(n_embd, 0.0f);

        for (int i = 0; i < batch_view.n_tokens; ++i) {
            if (!batch_view.logits[i] || batch_view.seq_id[i][0] != slot.id + 1) {
                continue;
            }

            const float *embd = llama_get_embeddings_seq(ctx, batch_view.seq_id[i][0]);
            if (embd == nullptr) {
                embd = llama_get_embeddings_ith(ctx, i);
            }

            if (embd == nullptr) {
                SLT_ERR(slot,
                        "failed to get embeddings, "
                        "token = %d, seq_id = %d\n",
                        batch.token[i], batch.seq_id[i][0]);

                res.data = json{
                    {"embedding", std::vector<float>(n_embd, 0.0f)},
                    {"index", slot.index},
                };

                continue;
            }

            common_embd_normalize(embd, embd_res.data(), n_embd);

            res.data = json{
                {"embedding", embd_res},
                {"index", slot.index},
            };
        }

        res.data["tokens_evaluated"] = slot.n_prompt_tokens;
        queue_results.send(res);
    }

    void send_rerank(const server_slot &slot, const llama_batch &batch) {
        server_task_result res;
        res.id = slot.id_task;
        res.error = false;
        res.stop = true;

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id + 1) {
                continue;
            }

            const float *embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            if (embd == nullptr) {
                embd = llama_get_embeddings_ith(ctx, i);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res.data = json{
                    {"index", slot.index},
                    {"score", -1e6},
                };

                continue;
            }

            res.data = json{
                {"index", slot.index},
                {"score", embd[0]},
            };
        }

        res.data["tokens_evaluated"] = slot.n_prompt_tokens;
        queue_results.send(res);
    }

    //
    // Functions to create new task(s) and receive result(s)
    //

    std::vector<server_task> create_tasks_cmpl(json data, server_task_cmpl_type cmpl_type, int tps = 0) {
        std::vector<server_task> tasks;
        auto create_task = [&](json &task_data, bool replace_prompt, json prompt, int tps = 0) {
            server_task task;
            task.id = queue_tasks.get_new_id();
            task.cmpl_type = cmpl_type;
            task.type = SERVER_TASK_TYPE_COMPLETION;
            if (replace_prompt) {
                task.data = task_data;
                task.data["prompt"] = prompt;
            } else {
                task.data = std::move(task_data);
            }
            task.tps = tps;
            tasks.push_back(std::move(task));
        };

        bool chat_vision = json_value(data, "__oaicompat_completion_chat_vision", false);
        if (!chat_vision) {
            static constexpr const char *error_msg = "\"prompt\" must be a string, an array of token ids or an array of prompts";
            if (!data.contains("prompt")) {
                throw std::runtime_error(error_msg);
            }

            json prompt = data.at("prompt");

            // if the prompt is a singleton (i.e. a string or a list of tokens), we only need to
            // create single task
            if (prompt.is_string() || json_is_array_of_numbers(prompt)) {
                data["index"] = 0;
                create_task(data, false, nullptr, tps);
            } else if (prompt.is_array()) {
                // otherwise, it's a multiple-prompt task, we break it into smaller tasks
                std::vector<json> prompts = prompt;
                if (cmpl_type == SERVER_TASK_CMPL_TYPE_RERANK) {
                    // prompts[0] is the question
                    // the rest are the answers/documents
                    SRV_DBG("creating rerank tasks, n_prompts = %d\n", (int)prompts.size() - 1);
                    for (size_t i = 1; i < prompts.size(); i++) {
                        json qd;
                        qd.push_back(prompts[0]);
                        qd.push_back(prompts[i]);
                        data["index"] = i - 1;
                        create_task(data, true, qd);
                    }
                } else {
                    SRV_DBG("creating multi-prompt tasks, n_prompts = %d\n", (int)prompts.size());
                    for (size_t i = 0; i < prompts.size(); i++) {
                        const auto &e = prompts[i];
                        if (e.is_string() || json_is_array_of_numbers(e)) {
                            data["index"] = i;
                            create_task(data, true, e);
                        } else {
                            throw std::runtime_error(error_msg);
                        }
                    }
                }
            } else {
                // invalid case
                throw std::runtime_error(error_msg);
            }
        } else {
            create_task(data, false, nullptr, tps);
        }

        return tasks;
    }

    void cancel_tasks(const std::unordered_set<int> &id_tasks) {
        std::vector<server_task> cancel_tasks;
        cancel_tasks.reserve(id_tasks.size());
        for (const auto &id_task : id_tasks) {
            server_task task;
            task.type = SERVER_TASK_TYPE_CANCEL;
            task.id_target = id_task;
            cancel_tasks.push_back(task);
        }
        // push to beginning of the queue, so it has highest priority
        queue_tasks.post(cancel_tasks, true);
        queue_results.remove_waiting_task_ids(id_tasks);
    }

    // receive the results from task(s) created by create_tasks_cmpl
    void receive_cmpl_results(const std::unordered_set<int> &id_tasks, std::function<void(std::vector<server_task_result> &)> result_handler,
                              std::function<void(json)> error_handler) {
        // TODO: currently, there is no way to detect the client has cancelled the request
        std::vector<server_task_result> results(id_tasks.size());
        for (size_t i = 0; i < id_tasks.size(); i++) {
            server_task_result result = queue_results.recv(id_tasks);

            if (result.error) {
                error_handler(result.data);
                return;
            }

            const size_t idx = result.data["index"];
            GGML_ASSERT(idx < results.size() && "index out of range");
            results[idx] = result;
        }
        result_handler(results);
    }

    // receive the results from task(s) created by create_tasks_cmpl, in stream mode
    void receive_cmpl_results_stream(const std::unordered_set<int> &id_tasks, std::function<bool(server_task_result &)> result_handler,
                                     std::function<void(json)> error_handler) {
        size_t n_finished = 0;
        while (true) {
            server_task_result result = queue_results.recv(id_tasks);
            if (!result_handler(result)) {
                break;
            }

            if (result.error) {
                error_handler(result.data);
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

    void process_single_task(const server_task &task) {
        switch (task.type) {
        case SERVER_TASK_TYPE_COMPLETION: {
            const int id_slot = json_value(task.data, "id_slot", -1);

            server_slot *slot;

            if (id_slot != -1) {
                slot = get_slot_by_id(id_slot);
            } else {
                std::string prompt;
                if (task.data.contains("prompt") && task.data.at("prompt").is_string()) {
                    prompt = json_value(task.data, "prompt", std::string());
                }

                slot = get_available_slot(prompt);
            }

            if (slot == nullptr) {
                // if no slot is available, we defer this task for
                // processing later
                queue_tasks.defer(task);
                break;
            }
            if (slot->is_processing()) {
                // if requested slot is unavailable, we defer this task for
                // processing later
                queue_tasks.defer(task);
                break;
            }

            slot->reset();

            slot->id_task = task.id;
            slot->cmpl_type = task.cmpl_type;
            slot->index = json_value(task.data, "index", 0);

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

            int n_idle_slots = 0;
            int n_processing_slots = 0;

            for (server_slot &slot : slots) {
                json slot_data = get_formated_generation(slot);
                slot_data["id"] = slot.id;
                slot_data["id_task"] = slot.id_task;
                slot_data["state"] = slot.state;
                slot_data["next_token"] = {
                    {"has_next_token", slot.has_next_token}, {"n_remain", slot.n_remaining},      {"n_decoded", slot.n_decoded},
                    {"stopped_eos", slot.stopped_eos},       {"stopped_word", slot.stopped_word}, {"stopped_limit", slot.stopped_limit},
                    {"stopping_word", slot.stopping_word},
                };

                if (slot_data["state"] == SLOT_STATE_IDLE) {
                    n_idle_slots++;
                } else {
                    n_processing_slots++;
                }

                slots_data.push_back(slot_data);
            }
            SRV_DBG("n_idle_slots = %d, n_processing_slots = %d\n", n_idle_slots, n_processing_slots);

            server_task_result res;
            res.id = task.id;
            res.stop = true;
            res.error = false;
            res.data = {
                {"idle", n_idle_slots},
                {"processing", n_processing_slots},
                {"deferred", queue_tasks.queue_tasks_deferred.size()},
                {"t_start", metrics.t_start},

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

                {"kv_cache_tokens_count", llama_get_kv_cache_token_count(ctx)},
                {"kv_cache_used_cells", llama_get_kv_cache_used_cells(ctx)},

                {"slots", slots_data},
            };

            if (json_value(task.data, "reset_bucket", false)) {
                metrics.reset_bucket();
            }
            queue_results.send(res);
        } break;
        case SERVER_TASK_TYPE_SLOT_SAVE: {
            int id_slot = task.data.at("id_slot");
            server_slot *slot = get_slot_by_id(id_slot);
            if (slot == nullptr) {
                send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                break;
            }
            if (slot->is_processing()) {
                // if requested slot is unavailable, we defer this task for
                // processing later
                queue_tasks.defer(task);
                break;
            }

            const size_t token_count = slot->cache_tokens.size();
            const int64_t t_start = ggml_time_us();

            std::string filename = task.data.at("filename");
            std::string filepath = task.data.at("filepath");

            const size_t nwrite = llama_state_seq_save_file(ctx, filepath.c_str(), slot->id + 1, slot->cache_tokens.data(), token_count);

            const int64_t t_end = ggml_time_us();
            const double t_save_ms = double(t_end - t_start) / 1000.0;

            server_task_result result;
            result.id = task.id;
            result.stop = true;
            result.error = false;
            result.data = json{{"id_slot", id_slot},
                               {"filename", filename},
                               {"n_saved", token_count}, // tokens saved
                               {"n_written", nwrite},    // bytes written
                               {"timings", {{"save_ms", t_save_ms}}}};
            queue_results.send(result);
        } break;
        case SERVER_TASK_TYPE_SLOT_RESTORE: {
            int id_slot = task.data.at("id_slot");
            server_slot *slot = get_slot_by_id(id_slot);
            if (slot == nullptr) {
                send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                break;
            }
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
                llama_state_seq_load_file(ctx, filepath.c_str(), slot->id + 1, slot->cache_tokens.data(), slot->cache_tokens.size(), &token_count);
            if (nread == 0) {
                slot->cache_tokens.resize(0);
                send_error(task,
                           "Unable to restore slot, no available space in KV "
                           "cache or invalid slot "
                           "save file",
                           ERROR_TYPE_INVALID_REQUEST);
                break;
            }
            slot->cache_tokens.resize(token_count);

            // TODO: maybe detokenize the slot->cache_tokens instead?
            slot->prompt = string_format("[restored %d tokens from file]", (int)token_count);

            const int64_t t_end = ggml_time_us();
            const double t_restore_ms = double(t_end - t_start) / 1000.0;

            server_task_result result;
            result.id = task.id;
            result.stop = true;
            result.error = false;
            result.data = json{{"id_slot", id_slot},
                               {"filename", filename},
                               {"n_restored", token_count}, // tokens restored
                               {"n_read", nread},           // bytes read
                               {"timings", {{"restore_ms", t_restore_ms}}}};
            queue_results.send(result);
        } break;
        case SERVER_TASK_TYPE_SLOT_ERASE: {
            int id_slot = task.data.at("id_slot");
            server_slot *slot = get_slot_by_id(id_slot);
            if (slot == nullptr) {
                send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                break;
            }
            if (slot->is_processing()) {
                // if requested slot is unavailable, we defer this task for
                // processing later
                queue_tasks.defer(task);
                break;
            }

            // Erase token cache
            const size_t n_erased = slot->cache_tokens.size();
            llama_kv_cache_seq_rm(ctx, slot->id + 1, -1, -1);
            if (ctx_draft != nullptr) {
                llama_kv_cache_seq_rm(ctx_draft, slot->id + 1, -1, -1);
            }
            slot->cache_tokens.clear();

            server_task_result result;
            result.id = task.id;
            result.stop = true;
            result.error = false;
            result.data = json{{"id_slot", id_slot}, {"n_erased", n_erased}};
            queue_results.send(result);
        } break;
        case SERVER_TASK_TYPE_SET_LORA: {
            common_lora_adapters_apply(ctx, lora_adapters);

            server_task_result result;
            result.id = task.id;
            result.stop = true;
            result.error = false;
            result.data = json{{"success", true}};
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
                llama_kv_cache_clear(ctx);
                if (ctx_draft != nullptr) {
                    llama_kv_cache_clear(ctx_draft);
                }
                return;
            }
        }

        {
            server_task task;
            task.type = SERVER_TASK_TYPE_NEXT_RESPONSE;
            task.id_target = -1;

            queue_tasks.post(task);
        }

        // apply context-shift if needed
        // TODO: simplify and improve
        for (server_slot &slot : slots) {
            if (slot.is_processing() && slot.n_past + 1 >= slot.n_ctx) {
                if (!params.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in
                    // process_token()
                    slot.release();
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    continue;
                }

                // Shift context
                const int n_keep = slot.params.n_keep + add_bos_token;
                const int n_left = slot.n_past - n_keep;
                const int n_discard = slot.params.n_discard ? slot.params.n_discard : (n_left / 2);

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                llama_kv_cache_seq_rm(ctx, slot.id + 1, n_keep, n_keep + n_discard);
                llama_kv_cache_seq_add(ctx, slot.id + 1, n_keep + n_discard, slot.n_past, -n_discard);
                if (ctx_draft != nullptr) {
                    llama_kv_cache_seq_rm(ctx_draft, slot.id + 1, n_keep, n_keep + n_discard);
                    llama_kv_cache_seq_add(ctx_draft, slot.id + 1, n_keep + n_discard, slot.n_past, -n_discard);
                }

                if (slot.params.cache_prompt) {
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
        if (ctx_draft) {
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

            int32_t slot_npast = slot.n_past;
            slot_npast += slot.n_drafted_accepted;

            common_batch_add(batch, slot.sampled[slot.sampled.size() - 1], slot_npast, {slot.id + 1}, true);
            if (!slot.sampled_draft.empty()) {
                for (const llama_token &tok : slot.sampled_draft) {
                    common_batch_add(batch, tok, slot_npast + 1, {slot.id + 1}, true);
                    slot_npast += 1;
                }
            }
            slot.n_past += 1;

            if (slot.params.cache_prompt) {
                for (const llama_token &tok : slot.sampled) {
                    slot.cache_tokens.push_back(tok);
                }
            }

            SLT_DBG(slot,
                    "slot decode token, "
                    "n_ctx = %d, n_past = %d, "
                    "n_cache_tokens = %d, truncated = %d\n",
                    slot.n_ctx, slot.n_past, (int)slot.cache_tokens.size(), slot.truncated);
        }

        // process in chunks of params.n_batch
        auto n_batch = int32_t(llama_n_batch(ctx));
        auto n_ubatch = int32_t(llama_n_ubatch(ctx));

        // track if this is an embedding or non-embedding batch
        // if we've added sampled tokens above, we are in non-embedding mode
        // -1: none, 0: non-embedding, 1: embedding
        // TODO: make enum
        int32_t batch_type = batch.n_tokens > 0 ? 0 : -1;

        // next, batch any pending prompts without exceeding n_batch
        if (params.cont_batching || batch.n_tokens == 0) {
            for (auto &slot : slots) {
                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT) {
                    auto &prompt_tokens = slot.prompt_tokens;

                    // we haven't tokenized the prompt yet - do it now:
                    if (prompt_tokens.empty()) {
                        SLT_INF(slot,
                                "tokenizing prompt, "
                                "len = %d\n",
                                (int)slot.prompt.size());

                        slot.t_start_process_prompt = ggml_time_us();
                        slot.t_start_generation = 0;

                        switch (slot.cmpl_type) {
                        case SERVER_TASK_CMPL_TYPE_NORMAL:
                        case SERVER_TASK_CMPL_TYPE_EMBEDDING: {
                            if (!slot.oaicompat_completion_chat_vision) {
                                // add BOS if there isn't system prompt
                                prompt_tokens = tokenize(slot.prompt, llama_add_bos_token(model), true);
                            }
                        } break;
                        case SERVER_TASK_CMPL_TYPE_RERANK: {
                            // require slot.prompt to be array of 2 strings
                            if (!slot.prompt.is_array() || slot.prompt.size() != 2) {
                                SLT_ERR(slot, "%s", "invalid prompt for rerank task\n");
                                slot.release();
                                send_error(slot, "invalid prompt for rerank task", ERROR_TYPE_INVALID_REQUEST);
                                continue;
                            }

                            // prompt: [BOS]query[EOS][SEP]doc[EOS]
                            prompt_tokens.clear();
                            prompt_tokens.push_back(llama_token_bos(model));
                            {
                                const auto part = tokenize(slot.prompt[0], false, false);
                                prompt_tokens.insert(prompt_tokens.end(), part.begin(), part.end());
                            }
                            prompt_tokens.push_back(llama_token_eos(model));
                            prompt_tokens.push_back(llama_token_sep(model));
                            {
                                const auto part = tokenize(slot.prompt[1], false, false);
                                prompt_tokens.insert(prompt_tokens.end(), part.begin(), part.end());
                            }
                            prompt_tokens.push_back(llama_token_eos(model));
                        } break;
                        case SERVER_TASK_CMPL_TYPE_INFILL: {
                            auto prefix_tokens = tokenize(slot.params.input_prefix, false, false);
                            auto suffix_tokens = tokenize(slot.params.input_suffix, false, false);

                            prefix_tokens.insert(prefix_tokens.begin(), llama_token_fim_pre(model));
                            suffix_tokens.insert(suffix_tokens.begin(), llama_token_fim_suf(model));

                            auto embd_inp = params.spm_infill ? suffix_tokens : prefix_tokens;
                            auto embd_end = params.spm_infill ? prefix_tokens : suffix_tokens;

                            if (llama_add_bos_token(model)) {
                                embd_inp.insert(embd_inp.begin(), llama_token_bos(model));
                            }

                            embd_inp.insert(embd_inp.end(), embd_end.begin(), embd_end.end());
                            embd_inp.push_back(llama_token_fim_mid(model));

                            prompt_tokens = std::move(embd_inp);
                        } break;
                        }

                        slot.n_past = 0;
                        slot.n_prompt_tokens = int32_t(prompt_tokens.size());

                        // empty prompt passed -> release the slot and send
                        // empty response
                        if (!slot.oaicompat_completion_chat_vision && prompt_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.release();
                            send_final_response(slot);
                            continue;
                        }

                        if (slot.cmpl_type == SERVER_TASK_CMPL_TYPE_EMBEDDING || slot.cmpl_type == SERVER_TASK_CMPL_TYPE_RERANK) {
                            // this prompt is too large to process - discard it
                            if (slot.n_prompt_tokens > n_ubatch) {
                                slot.release();
                                send_error(slot,
                                           "input is too large to process, "
                                           "please increase the physical batch size",
                                           ERROR_TYPE_SERVER);
                                continue;
                            }
                        } else {
                            if (!params.ctx_shift) {
                                // if context shift is disabled, we make sure prompt size is smaller than KV size
                                // TODO: there should be a separate parameter that control prompt truncation
                                //       context shift should be applied only during the generation phase
                                if (slot.n_prompt_tokens >= slot.n_ctx) {
                                    slot.release();
                                    send_error(slot,
                                               "the request exceeds the available context size. try "
                                               "increasing the context size or enable context shift",
                                               ERROR_TYPE_INVALID_REQUEST);
                                    continue;
                                }
                            }

                            if (slot.params.n_keep < 0) {
                                slot.params.n_keep = slot.n_prompt_tokens;
                            }
                            slot.params.n_keep = std::min(slot.n_ctx - 4, slot.params.n_keep);

                            // if input prompt is too big, truncate it (if group attention self-extend is disabled)
                            if (slot.n_prompt_tokens >= slot.n_ctx) {
                                const int n_left = slot.n_ctx - slot.params.n_keep;

                                const int n_block_size = n_left / 2;
                                const int erased_blocks = (slot.n_prompt_tokens - slot.params.n_keep - n_block_size) / n_block_size;

                                std::vector<llama_token> new_tokens(prompt_tokens.begin(), prompt_tokens.begin() + slot.params.n_keep);

                                new_tokens.insert(new_tokens.end(), prompt_tokens.begin() + slot.params.n_keep + erased_blocks * n_block_size,
                                                  prompt_tokens.end());

                                prompt_tokens = std::move(new_tokens);

                                slot.truncated = true;
                                slot.n_prompt_tokens = int32_t(prompt_tokens.size());

                                SLT_WRN(slot,
                                        "input truncated, "
                                        "n_ctx = %d, n_keep = %d, n_left = %d, n_prompt_tokens = %d\n",
                                        slot.n_ctx, slot.params.n_keep, n_left, slot.n_prompt_tokens);

                                GGML_ASSERT(slot.n_prompt_tokens < slot.n_ctx);
                            }

                            common_sampler_reset(slot.smpl);

                            if (slot.params.cache_prompt) {
                                // reuse any previously computed tokens that are
                                // common with the new prompt
                                slot.n_past = int32_t(common_part(slot.cache_tokens, prompt_tokens));

                                // push the prompt into the sampling context (do
                                // not apply grammar)
                                for (int i = 0; i < slot.n_past; ++i) {
                                    common_sampler_accept(slot.smpl, slot.cache_tokens[i], false);
                                }
                            }
                        }

                        if (slot.n_past == slot.n_prompt_tokens && slot.n_past > 0) {
                            // we have to evaluate at least 1 token to generate logits.
                            SLT_WRN(slot,
                                    "need to evaluate at least 1 token to generate logits, "
                                    "n_past = %d, n_prompt_tokens = %d\n",
                                    slot.n_past, slot.n_prompt_tokens);

                            slot.n_past--;
                        }

                        slot.n_prompt_tokens_processed = 0;
                    }

                    if (slot.cmpl_type == SERVER_TASK_CMPL_TYPE_EMBEDDING) {
                        // cannot fit the prompt in the current batch - will try
                        // next iter
                        if (batch.n_tokens + slot.n_prompt_tokens > n_batch) {
                            continue;
                        }
                    }

                    // check that we are in the right batch_type, if not defer the slot
                    // clang-format off
                    const int32_t slot_type =
                        slot.cmpl_type == SERVER_TASK_CMPL_TYPE_EMBEDDING ||
                        slot.cmpl_type == SERVER_TASK_CMPL_TYPE_RERANK     ? 1 : 0;
                    // clang-format on
                    if (batch_type == -1) {
                        batch_type = slot_type;
                    } else if (batch_type != slot_type) {
                        continue;
                    } else if (batch_type == 1 && llama_pooling_type(ctx) > 0) {
                        continue;
                    }

                    // keep only the common part
                    int32_t slot_npast = slot.n_past if (!llama_kv_cache_seq_rm(ctx, slot.id + 1, slot_npast, -1)) {
                        // could not partially delete (likely using a on-Transformer model)
                        llama_kv_cache_seq_rm(ctx, slot.id + 1, -1, -1);

                        // there is no common part left
                        slot.n_past = 0;

                        common_sampler_reset(slot.smpl);
                    }
                    if (ctx_draft != nullptr) {
                        if (!llama_kv_cache_seq_rm(ctx_draft, slot.id + 1, slot_npast, -1)) {
                            llama_kv_cache_seq_rm(ctx_draft, slot.id + 1, -1, -1);
                            if (slot_npast != 0) {
                                llama_kv_cache_seq_cp(ctx_draft, 0, slot.id + 1, -1, -1);
                            }
                            common_sampler_reset(slot.smpl_draft);
                        }
                    }

                    SLT_INF(slot, "kv cache rm [%d, end)\n", slot.n_past);

                    // remove the non-common part from the cache
                    slot.cache_tokens.resize(slot.n_past);

                    // add prompt tokens for processing in the current batch
                    while (slot.n_past < slot.n_prompt_tokens && batch.n_tokens < n_batch) {
                        common_batch_add(batch, prompt_tokens[slot.n_past], slot.n_past, {slot.id + 1}, false);
                        if (ctx_draft != nullptr) {
                            common_batch_add(batch_draft, prompt_tokens[slot.n_past], slot.n_past, {slot.id + 1}, false);
                        }

                        if (slot.params.cache_prompt) {
                            slot.cache_tokens.push_back(prompt_tokens[slot.n_past]);
                        }

                        slot.n_prompt_tokens_processed++;
                        slot.n_past++;
                    }

                    if (slot.oaicompat_completion_chat_vision && !process_vision_prompt(slot, n_batch)) {
                        slot.release();
                        continue;
                    }

                    // entire prompt has been processed - start decoding new
                    // tokens
                    if (slot.n_past == slot.n_prompt_tokens) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.n_tokens > 0);

                        // extract the logits only for the last token
                        batch.logits[batch.n_tokens - 1] = true;

                        slot.n_decoded = 0;
                        slot.i_batch = batch.n_tokens - 1;
                    }
                }

                if (batch.n_tokens >= n_batch) {
                    break;
                }
            }
        }

        if (batch.n_tokens == 0) {
            return;
        }

        // make sure we're in the right embedding mode
        llama_set_embeddings(ctx, batch_type == 1);

        // process the created batch of tokens
        for (int32_t i = 0; i < batch.n_tokens; i += n_batch) {
            const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

            // clang-format off
            llama_batch batch_view = {
                n_tokens,
                batch.token + i,
                nullptr,
                batch.pos + i,
                batch.n_seq_id + i,
                batch.seq_id + i,
                batch.logits + i,
                0, 0, 0, // unused
            };
            // clang-format on

            const int ret = llama_decode(ctx, batch_view);
            metrics.on_decoded(slots);
            if (ret != 0) {
                if (n_batch == 1 || ret < 0) {
                    // if you get here, it means the KV cache is full - try
                    // increasing it via the context size
                    SRV_ERR("failed to decode the batch: "
                            "KV cache is full - "
                            "try increasing it via the context size, "
                            "i = %d, n_batch = %d, ret = %d\n",
                            i, n_batch, ret);
                    for (auto &slot : slots) {
                        slot.release();
                        send_error(slot, "input prompt is too big compared to KV size. "
                                         "please increase KV size");
                    }
                    break; // break loop of n_batch
                }

                // retry with half the batch size to try to find a free slot in
                // the KV cache
                n_batch /= 2;
                i -= n_batch;

                SRV_WRN("failed to find free space in the KV cache, "
                        "retrying with smaller batch size - "
                        "try increasing it via the context size or enable defragmentation, "
                        "i = %d, n_batch = %d, ret = %d\n",
                        i, n_batch, ret);

                continue; // continue loop of n_batch
            }
            if (ctx_draft != nullptr && batch_draft.n_tokens > 0) {
                const int32_t n_draft_tokens = std::min(n_batch, batch_draft.n_tokens - i);

                // clang-format off
                llama_batch batch_draft_view = {
                    n_draft_tokens,
                    batch_draft.token + i,
                    nullptr,
                    batch_draft.pos + i,
                    batch_draft.n_seq_id + i,
                    batch_draft.seq_id + i,
                    batch_draft.logits + i,
                    0, 0, 0, // unused
                };
                // clang-format on

                const int ret_draft = llama_decode(ctx_draft, batch_draft_view);
                GGML_ASSERT(ret_draft == 0);
            }

            for (auto &slot : slots) {
                if (slot.i_batch < (int)i || slot.i_batch >= (int)(i + n_tokens)) {
                    continue; // continue loop of slots
                }

                if (slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.cmpl_type == SERVER_TASK_CMPL_TYPE_EMBEDDING) {
                        // prompt evaluated for embedding
                        send_embedding(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    if (slot.cmpl_type == SERVER_TASK_CMPL_TYPE_RERANK) {
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
                    llama_token tok;
                    auto sz_draft = int32_t(slot.sampled_draft.size());
                    // +1 to allow for the last token to be generated
                    for (int32_t j = 0; j < sz_draft + 1; ++j) {
                        // greedy verification only
                        bool accept = false;
                        tok = common_sampler_sample(slot.smpl, ctx, slot.i_batch - i + j);
                        common_sampler_accept(slot.smpl, tok, true);
                        slot.push_token_into_result(tok, result);
                        if (j < sz_draft && tok == slot.sampled_draft[j]) {
                            accept = true;
                        }
                        slot.n_decoded += 1;
                        if (!accept) {
                            break;
                        }
                        slot.n_drafted_accepted += 1;
                    }
                } else {
                    llama_token tok = common_sampler_sample(slot.smpl, ctx, slot.i_batch - i);
                    common_sampler_accept(slot.smpl, tok, true);
                    slot.push_token_into_result(tok, result);
                    slot.n_decoded += 1;
                }

                if (slot.n_decoded == 1) {
                    slot.t_start_generation = ggml_time_us();
                    slot.t_prompt_processing = double(slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                    metrics.on_prompt_eval(slot);
                }

                if (ctx_draft != nullptr) {
                    llama_pos pos = slot.n_past + slot.n_drafted_accepted;
                    llama_kv_cache_seq_rm(ctx, slot.id + 1, pos, -1);
                    llama_kv_cache_seq_rm(ctx_draft, slot.id + 1, pos, -1);

                    slot.sampled_draft.clear();
                    if (slot.smpl_draft != nullptr) {
                        common_sampler_free(slot.smpl_draft);
                    }
                    slot.smpl_draft = common_sampler_clone(slot.smpl);

                    common_batch_clear(batch_draft);
                    common_batch_add(batch_draft, result.toks[result.toks.size() - 1], pos, {slot.id + 1}, true);
                    if (llama_decode(ctx_draft, batch_draft)) {
                        slot.release();
                        send_error(slot, "Failed to draft decode", ERROR_TYPE_SERVER);
                        continue; // continue loop of slots
                    }
                    slot.n_drafted += 1;

                    for (int32_t j = 0; j < params.n_draft; ++j) {
                        llama_token tok = common_sampler_sample(slot.smpl_draft, ctx_draft, 0);
                        slot.sampled_draft.push_back(tok);
                        common_sampler_accept(slot.smpl_draft, tok, true);
                        if (llama_token_is_eog(model_draft, tok)) {
                            break;
                        }
                        common_batch_clear(batch_draft);
                        common_batch_add(batch_draft, tok, pos + 1 + j, {slot.id + 1}, true);
                        if (llama_decode(ctx_draft, batch_draft)) {
                            break;
                        }
                        slot.n_drafted += 1;
                    }
                } else if (lookup_ngram_min > 0) {
                    llama_pos pos = slot.n_past + slot.n_drafted_accepted;
                    llama_kv_cache_seq_rm(ctx, slot.id + 1, pos, -1);

                    slot.sampled_draft.clear();

                    slot.sampled_draft.push_back(result.toks[result.toks.size() - 1]);
                    common_ngram_cache_draft(slot.prompt_tokens, slot.sampled_draft, params.n_draft, lookup_ngram_min, LLAMA_NGRAM_MAX,
                                             slot.ctx_ngram_cache, ngram_cache_dynamic, ngram_cache_static);
                    slot.n_drafted += int32_t(slot.sampled_draft.size()) - 1;

                    slot.sampled_draft.erase(slot.sampled_draft.begin());
                }

                if (!process_token(result, slot)) {
                    // release slot because of stop condition
                    slot.release();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                }

                slot.i_batch = -1;
            }
        }
    }

    bool process_vision_prompt(server_slot &slot, int n_batch) {
        const int n_embd = llama_n_embd(model);
        const auto sz_i = int32_t(slot.prompt.size());
        for (int32_t i = 0; i < sz_i; ++i) {
            const json &jp = slot.prompt.at(i);
            if (!jp.contains("type")) {
                continue;
            }
            const std::string type = jp.at("type");
            if (type == "text" && jp.contains("text")) {
                // process text prompt
                std::vector<llama_token> tokens = tokenize(jp.at("text"), false, true);
                const auto sz_j = int32_t(tokens.size());
                for (int32_t j = 0; j < sz_j; ++j) {
                    common_batch_add(batch, tokens[j], slot.n_past, {slot.id + 1}, false);
                    slot.n_past += 1;
                    slot.n_prompt_tokens += 1;
                    slot.n_prompt_tokens_processed += 1;
                }
                if (i + 1 >= sz_i) {
                    continue;
                }
                for (int32_t j = 0; j < batch.n_tokens; j += n_batch) {
                    const int32_t n_tokens = std::min(n_batch, batch.n_tokens - j);
                    // clang-format off
                    llama_batch batch_view = {
                        n_tokens,
                        batch.token    + j,
                        nullptr,
                        batch.pos      + j,
                        batch.n_seq_id + j,
                        batch.seq_id   + j,
                        batch.logits   + j,
                        0, 0, 0,
                    };
                    // clang-format on
                    if (llama_decode(ctx, batch_view)) {
                        send_error(slot, "Failed to decode text", ERROR_TYPE_SERVER);
                        return false;
                    }
                }
                common_batch_clear(batch);
            } else if (type == "image_url" && jp.contains("image_url")) {
                // process image prompt
                std::string img = json_value(jp.at("image_url"), "url", std::string());
                if (img.find("data:image/") != 0) {
                    send_error(slot, "Failed to load image: illegal prefix", ERROR_TYPE_INVALID_REQUEST);
                    return false;
                }
                const std::string split = "base64,";
                const size_t idx = img.find(split);
                if (idx <= 0) {
                    send_error(slot, "Failed to load image: illegal format", ERROR_TYPE_INVALID_REQUEST);
                    return false;
                }
                img = img.substr(idx + split.length());
                if (img.empty()) {
                    send_error(slot, "Failed to load image: empty data", ERROR_TYPE_INVALID_REQUEST);
                    return false;
                }
                const std::vector<uint8_t> buff = base64_decode(img);
                llava_image_embed *img_embd = llava_image_embed_make_with_bytes(ctx_clip, params.cpuparams.n_threads, buff.data(), int(buff.size()));
                if (!img_embd) {
                    send_error(slot, "Failed to embed image", ERROR_TYPE_INVALID_REQUEST);
                    return false;
                }
                for (int32_t j = 0; j < img_embd->n_image_pos; j += n_embd) {
                    const int32_t n_tokens = std::min(n_embd, img_embd->n_image_pos - j);
                    // clang-format off
                    llama_batch batch_img = {
                        n_tokens,
                        nullptr,
                        (img_embd -> embed + j),
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        slot.n_past, 1, slot.id + 1,
                    };
                    // clang-format on
                    if (llama_decode(ctx, batch_img)) {
                        send_error(slot, "Failed to decode image", ERROR_TYPE_SERVER);
                        return false;
                    }
                    slot.n_past += n_tokens;
                    slot.n_prompt_tokens += n_tokens;
                    slot.n_prompt_tokens_processed += n_tokens;
                }
                llava_image_embed_free(img_embd);
            }
        }

        std::vector<llama_token> tokens = tokenize(slot.params.input_suffix, false, true);
        if (tokens.empty()) {
            std::string suffix;
            if (batch.n_tokens == 0) {
                suffix = "\n### Assistant:\n";
            } else {
                suffix = "\nAnswer the questions.\n### Assistant:\n";
            }
            tokens = common_tokenize(ctx, suffix, false, true);
        }
        const auto sz_j = int32_t(tokens.size());
        for (int32_t j = 0; j < sz_j; ++j) {
            common_batch_add(batch, tokens[j], slot.n_past, {slot.id + 1}, false);
            slot.n_past += 1;
            slot.n_prompt_tokens += 1;
            slot.n_prompt_tokens_processed += 1;
        }

        return true;
    }

    json model_meta() const {
        return json{
            {"vocab_type", llama_vocab_type(model)}, {"n_vocab", llama_n_vocab(model)},         {"n_ctx_train", llama_n_ctx_train(model)},
            {"n_embd", llama_n_embd(model)},         {"n_params", llama_model_n_params(model)}, {"size", llama_model_size(model)},
        };
    }
};

static void log_server_request(const httplib::Request &req, const httplib::Response &res) {
    if (req.path == "/v1/health") {
        return;
    }

    SRV_INF("request %d: "
            "%s %s %s:%d\n",
            res.status, req.method.c_str(), req.path.c_str(), req.remote_addr.c_str(), req.remote_port);
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

    llama_box_params bparams;
    if (!llama_box_params_parse(argc, argv, bparams)) {
        llama_box_params_print_usage(argc, argv, bparams);
        return 1;
    }
    common_params &params = bparams.gparams;

    llama_numa_init(params.numa);

    if (bparams.rparams.port > 0) {
        rpcserver_params &rparams = bparams.rparams;
        return rpcserver_start(rparams);
    }

    server_context ctx_server;

    if (params.model_alias == "unknown") {
        params.model_alias = params.model;
    }

    llama_backend_init();

    LOG_INF("build: %s (%s) by %s with llama.cpp %d (%s)\n", LLAMA_BOX_GIT_VERSION, LLAMA_BOX_GIT_COMMIT, LLAMA_COMPILER, LLAMA_BUILD_NUMBER,
            LLAMA_COMMIT);
    LOG_INF("system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n", params.cpuparams.n_threads, params.cpuparams_batch.n_threads,
            std::thread::hardware_concurrency());
    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("\n");

    httplib::Server svr;
    std::atomic<server_state> state{SERVER_STATE_LOADING_MODEL};

    // default headers
    svr.set_default_headers({{"Server", "llama-box"}});

    // CORS preflight
    svr.Options(R"(.*)", [](const httplib::Request &, httplib::Response &res) {
        // Access-Control-Allow-Origin is already set by middleware
        res.set_header("Access-Control-Allow-Methods", "POST");
        res.set_header("Access-Control-Allow-Headers", "*");
        return res.set_content("", "text/html"); // blank response, no data
    });

    // logger
    svr.set_logger(log_server_request);

    // error handlers
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

    svr.set_exception_handler([&res_error](const httplib::Request &, httplib::Response &res, const std::exception_ptr &ep) {
        error_type err_type = ERROR_TYPE_SERVER;
        std::string message;
        try {
            std::rethrow_exception(ep);
        } catch (std::runtime_error &e) {
            err_type = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (std::exception &e) {
            message = e.what();
        } catch (...) {
            message = "Unknown Exception";
        }

        json formatted_error = format_error_response(message, err_type);
        res_error(res, formatted_error);
    });
    svr.set_error_handler([&res_error](const httplib::Request &, httplib::Response &res) {
        if (res.status == 404) {
            res_error(res, format_error_response("Not Found", ERROR_TYPE_NOT_FOUND));
        }
        // for other error codes, we skip processing here because it's
        // already done by res_error()
    });

    // configure and bind
    svr.set_read_timeout(params.timeout_read);
    svr.set_write_timeout(params.timeout_write);
    svr.set_payload_max_length(1024 * 1024 * 10);
    svr.set_idle_interval(bparams.conn_idle);
    svr.set_keep_alive_timeout(bparams.conn_keepalive);

    std::unordered_map<std::string, std::string> log_data;
    log_data["hostname"] = params.hostname;
    log_data["port"] = std::to_string(params.port);

    // necessary similarity of prompt for slot selection
    ctx_server.slot_prompt_similarity = params.slot_prompt_similarity;

    // pre routing
    svr.set_pre_routing_handler([&res_error, &state](const httplib::Request &req, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));

        // Server state loading.
        server_state current_state = state.load();
        if (current_state == SERVER_STATE_LOADING_MODEL) {
            res_error(res, format_error_response("Loading model", ERROR_TYPE_UNAVAILABLE));
            return httplib::Server::HandlerResponse::Handled;
        }

        return httplib::Server::HandlerResponse::Unhandled;
    });

    //
    // Handlers
    //

    const auto handle_health = [&](const httplib::Request &, httplib::Response &res) {
        // error and loading states are handled by middleware
        json health = {{"status", "ok"}};
        res_ok(res, health);
    };

    const auto handle_metrics = [&](const httplib::Request &, httplib::Response &res) {
        // request slots data using task queue
        server_task task;
        task.id_target = -1;
        task.type = SERVER_TASK_TYPE_METRICS;
        task.data.push_back({{"reset_bucket", true}});

        // post the task
        task.id = ctx_server.queue_tasks.post(task, true); // high-priority task
        ctx_server.queue_results.add_waiting_task_id(task.id);

        // get the result
        server_task_result result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        std::stringstream metrics;
        {
            json data = result.data;
            uint64_t n_prompt_tokens_processed_total = data.at("n_prompt_tokens_processed_total");
            uint64_t t_prompt_processing_total = data.at("t_prompt_processing_total");
            uint64_t n_tokens_predicted_total = data.at("n_tokens_predicted_total");
            uint64_t t_tokens_generation_total = data.at("t_tokens_generation_total");
            uint64_t n_tokens_drafted_total = data.at("n_tokens_drafted_total");
            uint64_t n_tokens_drafted_accepted_total = data.at("n_tokens_drafted_accepted_total");
            uint64_t n_decode_total = data.at("n_decode_total");
            uint64_t n_busy_slots_total = data.at("n_busy_slots_total");

            uint64_t n_prompt_tokens_processed = data.at("n_prompt_tokens_processed");
            uint64_t t_prompt_processing = data.at("t_prompt_processing");
            uint64_t n_tokens_predicted = data.at("n_tokens_predicted");
            uint64_t t_tokens_generation = data.at("t_tokens_generation");
            int32_t kv_cache_used_cells = data.at("kv_cache_used_cells");
            uint64_t kv_cache_tokens_count = data.at("kv_cache_tokens_count");
            uint64_t processing = data.at("processing");
            uint64_t deferred = data.at("deferred");

            // metrics definition:
            // https://prometheus.io/docs/practices/naming/#metric-names
            json all_metrics_def = json{
                {"counter",
                 {{{"name", "prompt_tokens_total"}, {"help", "Number of prompt tokens processed."}, {"value", n_prompt_tokens_processed_total}},
                  {{"name", "prompt_seconds_total"}, {"help", "Prompt process time"}, {"value", double(t_prompt_processing_total) / 1.e3}},
                  {{"name", "tokens_predicted_total"}, {"help", "Number of generation tokens processed."}, {"value", n_tokens_predicted_total}},
                  {{"name", "tokens_predicted_seconds_total"}, {"help", "Predict process time"}, {"value", double(t_tokens_generation_total) / 1.e3}},
                  {{"name", "tokens_drafted_total"}, {"help", "Number of speculative decoding tokens processed."}, {"value", n_tokens_drafted_total}},
                  {{"name", "tokens_drafted_accepted_total"},
                   {"help", "Number of speculative decoding tokens to be accepted."},
                   {"value", n_tokens_drafted_accepted_total}},
                  {{"name", "n_decode_total"}, {"help", "Total number of llama_decode() calls"}, {"value", n_decode_total}},
                  {{"name", "n_busy_slots_per_decode"},
                   {"help", "Average number of busy slots per llama_decode() call"},
                   {"value", (float)n_busy_slots_total / (float)n_decode_total}}}},
                {"gauge",
                 {{{"name", "prompt_tokens_seconds"},
                   {"help", "Average prompt throughput in tokens/s."},
                   {"value", n_prompt_tokens_processed ? 1.e3 / double(t_prompt_processing * n_prompt_tokens_processed) : 0.}},
                  {{"name", "predicted_tokens_seconds"},
                   {"help", "Average generation throughput in tokens/s."},
                   {"value", n_tokens_predicted ? 1.e3 / double(t_tokens_generation * n_tokens_predicted) : 0.}},
                  {{"name", "kv_cache_usage_ratio"},
                   {"help", "KV-cache usage. 1 means 100 percent usage."},
                   {"value", 1. * kv_cache_used_cells / params.n_ctx}},
                  {{"name", "kv_cache_tokens"}, {"help", "KV-cache tokens."}, {"value", kv_cache_tokens_count}},
                  {{"name", "requests_processing"}, {"help", "Number of request processing."}, {"value", processing}},
                  {{"name", "requests_deferred"}, {"help", "Number of request deferred."}, {"value", deferred}}}}};

            for (const auto &el : all_metrics_def.items()) {
                const auto &type = el.key();
                const auto &metrics_def = el.value();

                for (const auto &metric_def : metrics_def) {
                    const std::string name = metric_def.at("name");
                    const std::string help = metric_def.at("help");

                    auto value = json_value(metric_def, "value", 0.);
                    metrics << "# HELP llamacpp:" << name << " " << help << "\n"
                            << "# TYPE llamacpp:" << name << " " << type << "\n"
                            << "llamacpp:" << name << " " << value << "\n";
                }
            }
        }
        res.set_content(metrics.str(), "text/plain; version=0.0.4");
    };

    const auto handle_props = [&ctx_server, &params, &res_ok](const httplib::Request &, httplib::Response &res) {
        json props = {{"total_slots", params.n_parallel},
                      {"chat_template", params.chat_template.c_str()},
                      {"default_generation_settings", ctx_server.default_generation_settings_for_props}};

        res_ok(res, props);
    };

    const auto handle_tokenize = [&ctx_server, &res_error, &res_ok](const httplib::Request &req, httplib::Response &res) {
        const json request = json::parse(req.body);

        if (!request.contains("content")) {
            res_error(res, format_error_response("\"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        json tokens_json = json::array();

        const bool add_special = json_value(request, "add_special", false);
        const bool with_pieces = json_value(request, "with_pieces", false);
        const std::vector<llama_token> tokens = ctx_server.tokenize(request.at("content"), add_special, true);
        if (with_pieces) {
            for (const llama_token &token : tokens) {
                json piece_json;

                std::string piece = common_token_to_piece(ctx_server.ctx, token);
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

        const json response = json{{"tokens", tokens_json}};
        res_ok(res, response);
    };

    const auto handle_detokenize = [&ctx_server, &res_error, &res_ok](const httplib::Request &req, httplib::Response &res) {
        const json request = json::parse(req.body);

        if (!request.contains("tokens")) {
            res_error(res, format_error_response("\"tokens\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        const std::string content = common_detokenize(ctx_server.ctx, request.at("tokens"), false);

        const json response = json{{"content", content}};
        res_ok(res, response);
    };

    const auto handle_slots = [&](const httplib::Request &req, httplib::Response &res) {
        // request slots data using task queue
        server_task task;
        task.id_target = -1;
        task.type = SERVER_TASK_TYPE_METRICS;

        // post the task
        task.id = ctx_server.queue_tasks.post(task, true); // high-priority task
        ctx_server.queue_results.add_waiting_task_id(task.id);

        // get the result
        server_task_result result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        const int n_idle_slots = result.data.at("idle");
        if (req.has_param("fail_on_no_slot")) {
            if (n_idle_slots == 0) {
                res_error(res, format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return;
            }
        }

        res.set_content(result.data.at("slots").dump(), "application/json");
    };

    const auto handle_slots_save = [&ctx_server, &res_error, &res_ok, &params](const httplib::Request &req, httplib::Response &res, int id_slot) {
        json request = json::parse(req.body);

        std::string filename = request.at("filename");
        if (!fs_validate_filename(filename)) {
            res_error(res, format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        std::string filepath = params.slot_save_path + filename;

        server_task task;
        task.type = SERVER_TASK_TYPE_SLOT_SAVE;
        task.data = {{"id_slot", id_slot}, {"filename", filename}, {"filepath", filepath}};

        // post the task
        task.id = ctx_server.queue_tasks.post(task);
        ctx_server.queue_results.add_waiting_task_id(task.id);

        // get the result
        server_task_result result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        if (result.error) {
            res_error(res, result.data);
            return;
        }

        res_ok(res, result.data);
    };

    const auto handle_slots_restore = [&ctx_server, &res_error, &res_ok, &params](const httplib::Request &req, httplib::Response &res, int id_slot) {
        json request = json::parse(req.body);

        std::string filename = request.at("filename");
        if (!fs_validate_filename(filename)) {
            res_error(res, format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        std::string filepath = params.slot_save_path + filename;

        server_task task;
        task.type = SERVER_TASK_TYPE_SLOT_RESTORE;
        task.data = {{"id_slot", id_slot}, {"filename", filename}, {"filepath", filepath}};

        // post the task
        task.id = ctx_server.queue_tasks.post(task);
        ctx_server.queue_results.add_waiting_task_id(task.id);

        // get the result
        server_task_result result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        if (result.error) {
            res_error(res, result.data);
            return;
        }

        res_ok(res, result.data);
    };

    const auto handle_slots_erase = [&ctx_server, &res_error, &res_ok](const httplib::Request &, httplib::Response &res, int id_slot) {
        server_task task;
        task.type = SERVER_TASK_TYPE_SLOT_ERASE;
        task.data = {{"id_slot", id_slot}};

        // post the task
        task.id = ctx_server.queue_tasks.post(task);
        ctx_server.queue_results.add_waiting_task_id(task.id);

        // get the result
        server_task_result result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        if (result.error) {
            res_error(res, result.data);
            return;
        }

        res_ok(res, result.data);
    };

    const auto handle_slots_action = [&res_error, &handle_slots_save, &handle_slots_restore, &handle_slots_erase](const httplib::Request &req,
                                                                                                                  httplib::Response &res) {
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
            if (id_slot < 0) {
                res_error(res, format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
                return;
            }
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

    const auto handle_lora_adapters = [&ctx_server, &res_ok](const httplib::Request &, httplib::Response &res) {
        json result = json::array();
        for (size_t i = 0; i < ctx_server.lora_adapters.size(); ++i) {
            auto &la = ctx_server.lora_adapters[i];
            result.push_back({
                {"id", i},
                {"path", la.path},
                {"scale", la.scale},
            });
        }

        res_ok(res, result);
    };

    const auto handle_lora_adapters_apply = [&ctx_server, &res_ok](const httplib::Request &req, httplib::Response &res) {
        const std::vector<json> request = json::parse(req.body);
        auto max_idx = int32_t(ctx_server.lora_adapters.size());

        for (common_lora_adapter_container &la : ctx_server.lora_adapters) {
            la.scale = 0.0f;
        }
        for (json part : request) {
            int id = part.at("id");
            float scale = part.at("scale");
            if (0 <= id && id < max_idx) {
                ctx_server.lora_adapters[id].scale = scale;
                continue;
            }
            throw std::runtime_error("invalid adapter id");
        }

        // post the task
        server_task task;
        task.type = SERVER_TASK_TYPE_SET_LORA;
        task.id = ctx_server.queue_tasks.post(task);
        ctx_server.queue_results.add_waiting_task_id(task.id);

        // get the result
        server_task_result result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        res_ok(res, result.data);
    };

    const auto handle_models = [&ctx_server, &params, &res_ok](const httplib::Request &, httplib::Response &res) {
        json models = {{"object", "list"},
                       {"data",
                        {
                            {{"id", params.model_alias},
                             {"object", "model"},
                             {"created", std::time(nullptr)},
                             {"owned_by", "llama-box"},
                             {"meta", ctx_server.model_meta()}},
                        }}};

        res_ok(res, models);
    };

    const auto handle_infill = [&ctx_server, &res_error, &res_ok](const httplib::Request &req, httplib::Response &res) {
        // llama_supports_embedding_only is a patch.
        if (llama_supports_embedding_only(ctx_server.ctx)) {
            res.status = httplib::StatusCode::Forbidden_403;
            res.set_content("You are not allowed to sample from this model", "text/plain; charset=utf-8");
            return;
        }

        std::string err;
        if (llama_token_fim_pre(ctx_server.model) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_token_fim_suf(ctx_server.model) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_token_fim_mid(ctx_server.model) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            json formatted_error =
                format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED);
            res_error(res, formatted_error);
            return;
        }

        int tps = 0;
        {
            const std::string tps_s = req.get_header_value("X-Request-Tokens-Per-Second");
            if (!tps_s.empty()) {
                try {
                    tps = std::stoi(tps_s);
                } catch (const std::exception &) {
                    tps = ctx_server.n_tps;
                }
            }
            if (tps > ctx_server.n_tps) {
                // if the request exceeds the maximum tokens per second, return
                // 410 Gone
                if (ctx_server.n_tps > 0) {
                    res.status = httplib::StatusCode::Gone_410;
                    res.set_content("This request exceeds the maximum tokens per second", "text/plain; charset=utf-8");
                    return;
                }
                // if the server is not limited by tokens per second, set tps to
                // 0
                tps = 0;
            }
        }

        const json request = json::parse(req.body);

        // post tasks
        std::vector<server_task> tasks = ctx_server.create_tasks_cmpl(request, SERVER_TASK_CMPL_TYPE_INFILL, tps);
        ctx_server.queue_results.add_waiting_tasks(tasks);
        ctx_server.queue_tasks.post(tasks);

        std::unordered_set<int> task_ids = server_task::get_list_id(tasks);

        // process non-streaming requests
        if (!json_value(request, "stream", false)) {
            ctx_server.receive_cmpl_results(
                task_ids,
                [&](const std::vector<server_task_result> &results) {
                    json infills_json;
                    std::string pps;
                    if (results.size() == 1) {
                        infills_json = results[0].data;
                        pps = std::to_string(json_value(results[0].data.at("timings"), "predicted_per_second", double(tps)));
                    } else {
                        infills_json = json::array();
                        for (const auto &result : results) {
                            infills_json.push_back(result.data);
                        }
                        pps = std::to_string(json_value(results[0].data.at("timings"), "predicted_per_second", double(tps)));
                    }
                    res.set_header("X-Response-Tokens-Per-Second", pps);
                    res_ok(res, infills_json);
                },
                [&](const json &error_data) { res_error(res, error_data); });

            ctx_server.cancel_tasks(task_ids);
            return;
        }

        // process streaming requests
        const auto on_chunk = [task_ids, &ctx_server, tps](size_t, httplib::DataSink &sink) {
            ctx_server.receive_cmpl_results_stream(
                task_ids,
                [&](const server_task_result &result) -> bool {
                    json infills_json = result.data;
                    if (!server_sent_event(sink, "data", infills_json)) {
                        sink.done();
                        return false;
                    }
                    if (result.stop) {
                        std::string pps = std::to_string(json_value(result.data.at("timings"), "predicted_per_second", double(tps)));
                        sink.done_with_trailer({{"X-Response-Tokens-Per-Second", pps}});
                    }
                    return true;
                },
                [&](const json &error_data) {
                    server_sent_event(sink, "error", error_data);
                    sink.done();
                });

            return false;
        };
        const auto on_complete = [task_ids, &ctx_server](bool) { ctx_server.cancel_tasks(task_ids); };

        res.set_header("Trailer", "X-Response-Tokens-Per-Second");
        res.set_chunked_content_provider("text/event-stream", on_chunk, on_complete);
    };

    const auto handle_completions = [&ctx_server, &res_error, &res_ok](const httplib::Request &req, httplib::Response &res) {
        // llama_supports_embedding_only is a patch.
        if (llama_supports_embedding_only(ctx_server.ctx)) {
            res.status = httplib::StatusCode::Forbidden_403;
            res.set_content("You are not allowed to sample from this model", "text/plain; charset=utf-8");
            return;
        }

        int tps = 0;
        {
            const std::string tps_s = req.get_header_value("X-Request-Tokens-Per-Second");
            if (!tps_s.empty()) {
                try {
                    tps = std::stoi(tps_s);
                } catch (const std::exception &) {
                    tps = ctx_server.n_tps;
                }
            }
            if (tps > ctx_server.n_tps) {
                // if the request exceeds the maximum tokens per second, return
                // 410 Gone
                if (ctx_server.n_tps > 0) {
                    res.status = httplib::StatusCode::Gone_410;
                    res.set_content("This request exceeds the maximum tokens per second", "text/plain; charset=utf-8");
                    return;
                }
                // if the server is not limited by tokens per second, set tps to
                // 0
                tps = 0;
            }
        }

        bool oaicompat = req.path == "/v1/completions";
        json request = json::parse(req.body);
        if (!request.contains("prompt")) {
            res_error(res, format_error_response("\"prompt\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        if (oaicompat) {
            request = oaicompat_completion_request(ctx_server.model, request, std::string());
        }

        // post tasks
        std::vector<server_task> tasks = ctx_server.create_tasks_cmpl(request, SERVER_TASK_CMPL_TYPE_NORMAL, tps);
        ctx_server.queue_results.add_waiting_tasks(tasks);
        ctx_server.queue_tasks.post(tasks);

        std::unordered_set<int> task_ids = server_task::get_list_id(tasks);

        const std::string completion_id = gen_cmplid();

        // process non-streaming requests
        if (!json_value(request, "stream", false)) {
            ctx_server.receive_cmpl_results(
                task_ids,
                [&](std::vector<server_task_result> &results) {
                    json completions_json;
                    std::string pps;
                    if (results.size() == 1) {
                        completions_json = results[0].data;
                        if (oaicompat) {
                            completions_json = oaicompat_completion_response(request, completions_json, completion_id);
                        }
                        pps = std::to_string(json_value(results[0].data.at("timings"), "predicted_per_second", double(tps)));
                    } else {
                        completions_json = json::array();
                        for (const auto &result : results) {
                            auto tmp = result.data;
                            if (oaicompat) {
                                tmp = oaicompat_completion_response(request, tmp, completion_id);
                            }
                            completions_json.push_back(tmp);
                        }
                        pps = std::to_string(json_value(results[0].data.at("timings"), "predicted_per_second", double(tps)));
                    }
                    res.set_header("X-Response-Tokens-Per-Second", pps);
                    res_ok(res, completions_json);
                },
                [&](const json &error_data) { res_error(res, error_data); });

            ctx_server.cancel_tasks(task_ids);
            return;
        }

        // process streaming requests
        const auto on_chunk = [task_ids, &ctx_server, completion_id, oaicompat, request, tps](size_t, httplib::DataSink &sink) {
            ctx_server.receive_cmpl_results_stream(
                task_ids,
                [&](const server_task_result &result) -> bool {
                    json completions_json = result.data;
                    if (!server_sent_event(sink, "data", completions_json)) {
                        sink.done();
                        return false;
                    }
                    if (result.stop) {
                        if (oaicompat) {
                            const std::string done = "data: [DONE] \n\n";
                            if (!sink.write(done.c_str(), done.size())) {
                                sink.done();
                                return false;
                            }
                        }
                        std::string pps = std::to_string(json_value(result.data.at("timings"), "predicted_per_second", double(tps)));
                        sink.done_with_trailer({{"X-Response-Tokens-Per-Second", pps}});
                    }
                    return true;
                },
                [&](const json &error_data) {
                    server_sent_event(sink, "error", error_data);
                    sink.done();
                });

            return false;
        };
        const auto on_complete = [task_ids, &ctx_server](bool) { ctx_server.cancel_tasks(task_ids); };

        res.set_header("Trailer", "X-Response-Tokens-Per-Second");
        res.set_chunked_content_provider("text/event-stream", on_chunk, on_complete);
    };

    const auto handle_chat_completions = [&ctx_server, &params, &res_error, &res_ok](const httplib::Request &req, httplib::Response &res) {
        // llama_supports_embedding_only is a patch.
        if (llama_supports_embedding_only(ctx_server.ctx)) {
            res.status = httplib::StatusCode::Forbidden_403;
            res.set_content("You are not allowed to sample from this model", "text/plain; charset=utf-8");
            return;
        }

        int tps = 0;
        {
            const std::string tps_s = req.get_header_value("X-Request-Tokens-Per-Second");
            if (!tps_s.empty()) {
                try {
                    tps = std::stoi(tps_s);
                } catch (const std::exception &) {
                    tps = ctx_server.n_tps;
                }
            }
            if (tps > ctx_server.n_tps) {
                // if the request exceeds the maximum tokens per second, return
                // 410 Gone
                if (ctx_server.n_tps > 0) {
                    res.status = httplib::StatusCode::Gone_410;
                    res.set_content("This request exceeds the maximum tokens per second", "text/plain; charset=utf-8");
                    return;
                }
                // if the server is not limited by tokens per second, set tps to
                // 0
                tps = 0;
            }
        }

        json request = json::parse(req.body);
        if (!request.contains("messages") || !request.at("messages").is_array()) {
            res_error(res, format_error_response("\"messages\" must be provided and must be an array", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        request = oaicompat_completion_request(ctx_server.model, request, params.chat_template);

        // post the task
        std::vector<server_task> tasks = ctx_server.create_tasks_cmpl(request, SERVER_TASK_CMPL_TYPE_NORMAL, tps);
        ctx_server.queue_results.add_waiting_tasks(tasks);
        ctx_server.queue_tasks.post(tasks);

        std::unordered_set<int> task_ids = server_task::get_list_id(tasks);

        const std::string completion_id = gen_chatcmplid();

        // process non-streaming requests
        if (!json_value(request, "stream", false)) {
            ctx_server.receive_cmpl_results(
                task_ids,
                [&](std::vector<server_task_result> &results) {
                    json completions_json;
                    std::string pps;
                    if (results.size() == 1) {
                        completions_json = oaicompat_completion_response(request, results[0].data, completion_id);
                        pps = std::to_string(json_value(results[0].data.at("timings"), "predicted_per_second", double(tps)));
                    } else {
                        completions_json = json::array();
                        for (const auto &result : results) {
                            auto tmp = oaicompat_completion_response(request, result.data, completion_id);
                            completions_json.push_back(tmp);
                        }
                        pps = std::to_string(json_value(results[0].data.at("timings"), "predicted_per_second", double(tps)));
                    }
                    res.set_header("X-Response-Tokens-Per-Second", pps);
                    res_ok(res, completions_json);
                },
                [&](const json &error_data) { res_error(res, error_data); });

            ctx_server.cancel_tasks(task_ids);
            return;
        }

        // process streaming requests
        const auto on_chunk = [task_ids, &ctx_server, completion_id, request, tps](size_t, httplib::DataSink &sink) {
            bool first = true;
            ctx_server.receive_cmpl_results_stream(
                task_ids,
                [&](const server_task_result &result) -> bool {
                    if (first) {
                        first = false;
                        json completions_json = oaicompat_completion_response(request, result.data, completion_id, true, true);
                        if (!server_sent_event(sink, "data", completions_json)) {
                            sink.done();
                            return false;
                        }
                    }

                    json completions_json = oaicompat_completion_response(request, result.data, completion_id, true);
                    if (!server_sent_event(sink, "data", completions_json)) {
                        sink.done();
                        return false;
                    }

                    if (result.stop) {
                        const std::string done = "data: [DONE] \n\n";
                        if (!sink.write(done.c_str(), done.size())) {
                            sink.done();
                            return false;
                        }

                        std::string pps = std::to_string(json_value(result.data.at("timings"), "predicted_per_second", double(tps)));
                        sink.done_with_trailer({{"X-Response-Tokens-Per-Second", pps}});
                    }
                    return true;
                },
                [&](const json &error_data) {
                    server_sent_event(sink, "error", error_data);
                    sink.done();
                });

            return false;
        };
        const auto on_complete = [task_ids, &ctx_server](bool) { ctx_server.cancel_tasks(task_ids); };

        res.set_header("Trailer", "X-Response-Tokens-Per-Second");
        res.set_chunked_content_provider("text/event-stream", on_chunk, on_complete);
    };

    const auto handle_embeddings = [&ctx_server, &res_error, &res_ok](const httplib::Request &req, httplib::Response &res) {
        json request = json::parse(req.body);
        if (!request.contains("input")) {
            res_error(res, format_error_response("\"input\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        request = oaicompat_embedding_request(ctx_server.params, request);

        // post tasks
        std::vector<server_task> tasks = ctx_server.create_tasks_cmpl(request, SERVER_TASK_CMPL_TYPE_EMBEDDING);
        ctx_server.queue_results.add_waiting_tasks(tasks);
        ctx_server.queue_tasks.post(tasks);

        std::unordered_set<int> task_ids = server_task::get_list_id(tasks);

        // process non-streaming requests
        ctx_server.receive_cmpl_results(
            task_ids,
            [&](std::vector<server_task_result> &results) {
                const json embeddings_json = oaicompat_embedding_response(request, results[0].data);
                res_ok(res, embeddings_json);
            },
            [&](const json &error_data) { res_error(res, error_data); });

        ctx_server.cancel_tasks(task_ids);
    };

    const auto handle_rerank = [&ctx_server, &res_error, &res_ok](const httplib::Request &req, httplib::Response &res) {
        json request = json::parse(req.body);
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
        request = jinaaicompat_rerank_request(ctx_server.params, request);

        // post tasks
        std::vector<server_task> tasks = ctx_server.create_tasks_cmpl(request, SERVER_TASK_CMPL_TYPE_RERANK);
        ctx_server.queue_results.add_waiting_tasks(tasks);
        ctx_server.queue_tasks.post(tasks);

        std::unordered_set<int> task_ids = server_task::get_list_id(tasks);

        // process non-streaming requests
        ctx_server.receive_cmpl_results(
            task_ids,
            [&](std::vector<server_task_result> &results) {
                json responses = json::array();
                for (const server_task_result &ret : results) {
                    responses.push_back(ret.data);
                }
                json rerank_json = jinaicompat_rerank_response(request, responses);
                res_ok(res, rerank_json);
            },
            [&](const json &error_data) { res_error(res, error_data); });
    };

    //
    // Router
    //

    svr.Get("/health", handle_health);
    if (params.endpoint_metrics) {
        svr.Get("/metrics", handle_metrics);
    }
    svr.Get("/props", handle_props);
    svr.Post("/tokenize", handle_tokenize);
    svr.Post("/detokenize", handle_detokenize);
    if (params.endpoint_slots) {
        svr.Get("/slots", handle_slots);
        if (!params.slot_save_path.empty()) {
            // only enable slot operate endpoint if slot_save_path is set
            svr.Post("/slots/:id_slot", handle_slots_action);
        }
    }
    if (!params.lora_adapters.empty()) {
        svr.Get("/lora-adapters", handle_lora_adapters);
        if (params.lora_init_without_apply) {
            // only enable lora adapters apply endpoint if lora_init_without_apply is set
            svr.Post("/lora-adapters", handle_lora_adapters_apply);
        }
    }
    svr.Get("/v1/models", handle_models);
    if (bparams.endpoint_infill) {
        svr.Post("/infill", handle_infill);
    }
    svr.Post("/completion", handle_completions);
    svr.Post("/v1/completions", handle_completions);
    svr.Post("/v1/chat/completions", handle_chat_completions);
    if (params.embedding) {
        svr.Post("/v1/embeddings", handle_embeddings);
    }
    if (params.reranking) {
        svr.Post("/v1/rerank", handle_rerank);
    }

    //
    // Start
    //

    if (params.n_threads_http < 1) {
        // +2 threads for monitoring endpoints: /metrics and /slots
        params.n_threads_http = std::max(params.n_parallel + 2, (int32_t)std::thread::hardware_concurrency() - 1);
    }
    log_data["n_threads_http"] = std::to_string(params.n_threads_http);
    svr.new_task_queue = [&params] { return new httplib::ThreadPool(params.n_threads_http); };

    // bind HTTP listen port, run the HTTP server in a thread
    SRV_INF("listening, "
            "hostname = %s, port = %d\n",
            params.hostname.c_str(), params.port);
    if (!svr.bind_to_port(params.hostname, params.port)) {
        SRV_ERR("existing due to listening error, "
                "hostname = %s, port = %d\n",
                params.hostname.c_str(), params.port);
        ctx_server.clean(svr);
        return 1;
    }
    std::thread t([&]() { svr.listen_after_bind(); });
    svr.wait_until_ready();

    // load the model
    SRV_INF("%s", "loading model\n");
    if (!ctx_server.load_model(bparams)) {
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

    if (params.enable_chat_template) {
        // if a custom chat template is not supplied, we will use the one that comes
        // with the model (if any)
        bool built_in_chat_template = false;
        if (params.chat_template.empty()) {
            params.chat_template = ctx_server.load_chat_template();
            built_in_chat_template = true;
        }
        if (params.chat_template.size() <= 20) {
            for (char &c : params.chat_template) {
                c = char(std::tolower(c));
            }
        }
        SRV_INF("chat template, "
                "built_in: %d, chat_example:\n%s",
                built_in_chat_template, common_chat_format_example(ctx_server.model, params.chat_template).c_str());
    }

    SRV_INF("%s", "starting server\n");
    ctx_server.queue_tasks.on_new_task(std::bind(&server_context::process_single_task, &ctx_server, std::placeholders::_1));
    ctx_server.queue_tasks.on_update_slots(std::bind(&server_context::update_slots, &ctx_server));

    shutdown_handler = [&](int) { ctx_server.queue_tasks.terminate(); };
    ctx_server.queue_tasks.start_loop();

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    struct sigaction sigint_action{};

    sigint_action.sa_handler = signal_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, nullptr);
    sigaction(SIGTERM, &sigint_action, nullptr);
#elif defined(_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL { return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false; };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    SRV_INF("%s", "stopping server\n");
    ctx_server.clean(svr);
    t.join();

    return 0;
}
