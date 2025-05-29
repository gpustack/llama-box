#pragma once

// heads

#include <atomic>
#include <csignal>
#include <memory>
#include <unordered_map>
#include <utility>
#include <variant>

#include "llama.cpp/tools/mtmd/clip-impl.h"
#include "llama.cpp/tools/mtmd/clip.h"
#include "llama.cpp/tools/mtmd/mtmd-audio.h"

// types

struct llama_multimodal_tokens {
    llama_token        dummy_token = LLAMA_TOKEN_NULL;
    int32_t            n_tokens    = 0;
    int32_t            n_pos       = 0;
    bool               is_audio    = false;
    std::vector<float> embed;
    clip_image_size    size;
    clip_image_size    grid_size;
};

// implementations

struct llama_multimodal_embed_batch_wrapper {
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t>         logits;
    llama_batch                 temp = {};

    llama_multimodal_embed_batch_wrapper() = default;

    llama_multimodal_embed_batch_wrapper(float * embd, int32_t n_tokens, llama_pos * pos, llama_seq_id seq_id) {
        n_seq_id.resize(n_tokens);
        seq_ids.resize(n_tokens + 1);
        logits.resize(n_tokens);
        seq_id_0.resize(1);
        seq_id_0[0]       = seq_id;
        seq_ids[n_tokens] = nullptr;
        temp              = {
            /*n_tokens       =*/n_tokens,
            /*tokens         =*/nullptr,
            /*embd           =*/embd,
            /*pos            =*/pos,
            /*n_seq_id       =*/n_seq_id.data(),
            /*seq_id         =*/seq_ids.data(),
            /*logits         =*/logits.data(),
        };
        for (int i = 0; i < n_tokens; i++) {
            temp.n_seq_id[i] = 1;
            temp.seq_id[i]   = seq_id_0.data();
            temp.logits[i]   = false;
        }
    }

    std::vector<llama_pos> pos;

    llama_multimodal_embed_batch_wrapper(float * embd, int32_t n_tokens, llama_pos pos_0, llama_seq_id seq_id) {
        pos.resize(n_tokens);
        n_seq_id.resize(n_tokens);
        seq_ids.resize(n_tokens + 1);
        logits.resize(n_tokens);
        seq_id_0.resize(1);
        seq_id_0[0]       = seq_id;
        seq_ids[n_tokens] = nullptr;
        temp              = {
            /*n_tokens       =*/n_tokens,
            /*tokens         =*/nullptr,
            /*embd           =*/embd,
            /*pos            =*/pos.data(),
            /*n_seq_id       =*/n_seq_id.data(),
            /*seq_id         =*/seq_ids.data(),
            /*logits         =*/logits.data(),
        };
        for (int i = 0; i < n_tokens; i++) {
            temp.pos[i]      = pos_0 + i;
            temp.n_seq_id[i] = 1;
            temp.seq_id[i]   = seq_id_0.data();
            temp.logits[i]   = false;
        }
    }
};

static std::atomic<llama_token> multimodal_dummy_token_generator{ LLAMA_TOKEN_NULL };

// tokenize_image is not thread-safe, must be called from mutex-protected context.
static inline std::vector<llama_multimodal_tokens> tokenize_image(clip_ctx * ctx_clip, const int n_threads,
                                                                  const clip_image_u8 * img) {
    clip_image_f32_batch batch;
    if (!clip_image_preprocess(ctx_clip, img, &batch)) {
        LOG_ERR("%s", "unable to preprocess image\n");
        return {};
    }

    const int32_t n_mmproj_embd = clip_n_mmproj_embd(ctx_clip);

    std::vector<llama_multimodal_tokens> result;

    // minicpmv, slicing
    if (clip_is_minicpmv(ctx_clip)) {
        result.resize(batch.entries.size());

        const auto & entries   = batch.entries;
        const size_t n_entries = entries.size();
        for (size_t i = 0; i < n_entries; i++) {
            // init
            result[i].n_tokens  = clip_n_output_tokens(ctx_clip, entries[i].get());
            result[i].n_pos     = result[i].n_tokens;
            result[i].size      = clip_image_size{ entries[i]->nx, entries[i]->ny };
            result[i].grid_size = clip_image_size{ batch.grid_x, batch.grid_y };
            result[i].embed.resize(result[i].n_tokens * n_mmproj_embd);
            // encode
            const int64_t t_start = ggml_time_us();
            bool          encoded = clip_image_encode(ctx_clip, n_threads, entries[i].get(), result[i].embed.data());
            if (!encoded) {
                LOG_ERR("failed to encode image %2zu/%zu\n", i + 1, n_entries);
                return {};
            }
            LOG_INFV(3, "encoded image %2zu/%zu within %8.2f ms, n_tokens = %d\n", i + 1, n_entries,
                     (ggml_time_us() - t_start) / 1000.0, result[i].n_tokens);
            result[i].dummy_token = multimodal_dummy_token_generator--;
        }
    }
    // llava / glm, non-batching
    else if (clip_is_llava(ctx_clip) || clip_is_glm(ctx_clip)) {
        result.resize(1);

        int32_t n_tokens = 0;
        for (const auto & entry : batch.entries) {
            n_tokens += clip_n_output_tokens(ctx_clip, entry.get());
        }

        // init
        result[0].n_tokens  = n_tokens;
        result[0].n_pos     = result[0].n_tokens;
        result[0].size      = clip_image_size{ batch.entries[0]->nx, batch.entries[0]->ny };
        result[0].grid_size = clip_image_size{ batch.grid_x, batch.grid_y };
        result[0].embed.resize(result[0].n_tokens * n_mmproj_embd);
        // encode
        const auto & entries   = batch.entries;
        const size_t n_entries = entries.size();
        for (size_t i = 0; i < n_entries; i++) {
            int32_t       n_entry_tokens = clip_n_output_tokens(ctx_clip, entries[i].get());
            const int64_t t_start        = ggml_time_us();
            bool          encoded        = clip_image_encode(ctx_clip, n_threads, entries[i].get(),
                                                             result[0].embed.data() + i * n_entry_tokens * n_mmproj_embd);
            if (!encoded) {
                LOG_ERR("failed to encode image %2zu/%zu\n", i + 1, n_entries);
                return {};
            }
            LOG_INFV(3, "encoded image %2zu/%zu within %8.2f ms, n_tokens = %d\n", i + 1, n_entries,
                     (ggml_time_us() - t_start) / 1000.0, n_entry_tokens);
        }
        result[0].dummy_token = multimodal_dummy_token_generator--;
    }
    // others, batching
    else {
        result.resize(1);

        // init
        result[0].n_tokens  = clip_n_output_tokens(ctx_clip, batch.entries[0].get());
        result[0].n_pos     = result[0].n_tokens;
        result[0].size      = clip_image_size{ batch.entries[0]->nx, batch.entries[0]->ny };
        result[0].grid_size = clip_image_size{ batch.grid_x, batch.grid_y };
        result[0].embed.resize(result[0].n_tokens * n_mmproj_embd);
        // encode
        const int64_t t_start = ggml_time_us();
        bool          encoded = clip_image_batch_encode(ctx_clip, n_threads, &batch, result[0].embed.data());
        if (!encoded) {
            LOG_ERR("%s", "failed to encode image in batch\n");
            return {};
        }
        LOG_INFV(3, "encoded image in batch within %8.2f ms, n_tokens = %d\n", (ggml_time_us() - t_start) / 1000.0,
                 result[0].n_tokens);
        if (clip_is_qwen2vl(ctx_clip)) {
            const int32_t ps = clip_get_patch_size(ctx_clip) * 2;
            const int32_t ph = batch.entries[0]->ny / ps + (batch.entries[0]->ny % ps > 0);
            result[0].n_pos  = ph;
        }
        result[0].dummy_token = multimodal_dummy_token_generator--;
    }

    return result;
}

static inline std::vector<llama_multimodal_tokens> tokenize_audio(clip_ctx * ctx_clip, const int n_threads,
                                                                  const clip_image_u8 * aud) {
    whisper_preprocessor::whisper_filters          filters = whisper_precalc_filters::get_128_bins();
    std::vector<whisper_preprocessor::whisper_mel> entries;
    if (!whisper_preprocessor::preprocess_audio((const float *) aud->buf.data(), aud->nx, filters, entries)) {
        LOG_ERR("%s", "unable to preprocess audio\n");
        return {};
    }
    if (entries.empty()) {
        LOG_ERR("%s", "no audio chunks after preprocessing\n");
        return {};
    }

    const int32_t n_mmproj_embd = clip_n_mmproj_embd(ctx_clip);

    std::vector<llama_multimodal_tokens> result;
    result.resize(entries.size());

    const size_t n_entries = entries.size();
    for (size_t i = 0; i < n_entries; i++) {
        clip_image_f32_ptr mel_f32(clip_image_f32_init());
        mel_f32->nx         = entries[i].n_len;
        mel_f32->ny         = entries[i].n_mel;
        mel_f32->buf        = std::move(entries[i].data);
        // init
        result[i].n_tokens  = clip_n_output_tokens(ctx_clip, mel_f32.get());
        result[i].n_pos     = result[i].n_tokens;
        result[i].is_audio  = true;
        result[i].size      = clip_image_size{ entries[i].n_len, entries[i].n_mel };
        result[i].grid_size = clip_image_size{ 1, 1 };
        result[i].embed.resize(result[i].n_tokens * n_mmproj_embd);
        // encode
        clip_image_f32_batch batch_f32;
        batch_f32.is_audio = true;
        batch_f32.entries.push_back(std::move(mel_f32));
        const int64_t t_start = ggml_time_us();
        bool          encoded = clip_image_batch_encode(ctx_clip, n_threads, &batch_f32, result[i].embed.data());
        if (!encoded) {
            LOG_ERR("failed to encode audio %2zu/%zu\n", i + 1, n_entries);
            return {};
        }
        LOG_INFV(3, "encoded audio %2zu/%zu within %8.2f ms, n_tokens = %d\n", i + 1, n_entries,
                 (ggml_time_us() - t_start) / 1000.0, result[i].n_tokens);
        result[i].dummy_token = multimodal_dummy_token_generator--;
    }

    return result;
}
