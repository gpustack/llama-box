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

// types

struct llama_image_tokens {
    int32_t            n_tokens = 0;
    int32_t            n_pos    = 0;
    std::vector<float> embed;
    clip_image_size    size;
};

// implementations

struct llama_image_embed_batch_wrapper {
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t>         logits;
    llama_batch                 batch;

    llama_image_embed_batch_wrapper(float * embd, int32_t n_tokens, llama_pos * pos, llama_seq_id seq_id) {
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

    std::vector<llama_pos> pos;

    llama_image_embed_batch_wrapper(float * embd, int32_t n_tokens, llama_pos pos_0, llama_seq_id seq_id) {
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

static inline std::vector<llama_image_tokens> tokenize_image(clip_ctx * ctx_clip, const int n_threads,
                                                             const clip_image_u8 * img) {
    clip_image_f32_batch batch;
    if (!clip_image_preprocess(ctx_clip, img, &batch)) {
        LOG_ERR("%s", "unable to preprocess image\n");
        return {};
    }

    const int32_t n_mmproj_embd = clip_n_mmproj_embd(ctx_clip);

    std::vector<llama_image_tokens> result;
    result.resize(batch.entries.size());

    if (clip_is_llava(ctx_clip) || clip_is_minicpmv(ctx_clip)) {
        int32_t n_tokens = 0;
        for (const auto & entry : batch.entries) {
            n_tokens += clip_n_output_tokens(ctx_clip, entry.get());
        }
        std::vector<float> embed;
        embed.resize(n_tokens * n_mmproj_embd);

        // init
        const auto & entries   = batch.entries;
        const size_t n_entries = entries.size();
        for (size_t i = 0; i < n_entries; i++) {
            size_t n_embed     = clip_embd_nbytes_by_img(ctx_clip, entries[i]->nx, entries[i]->ny);
            result[i].n_tokens = clip_n_output_tokens(ctx_clip, entries[i].get());
            result[i].n_pos    = result[i].n_tokens;
            result[i].size     = clip_image_size{ entries[i]->nx, entries[i]->ny };
            result[i].embed.resize(n_embed);
            // encode
            clip_add_load_image_size(ctx_clip, &result[i].size);
            const int64_t t_start = ggml_time_us();
            bool          encoded = clip_image_encode(ctx_clip, n_threads, entries[i].get(),
                                                      embed.data() + i * n_mmproj_embd * result[i].n_tokens);
            if (!encoded) {
                LOG_ERR("failed to encode image %2zu/%zu\n", i + 1, n_entries);
                return {};
            }
            std::memcpy(result[i].embed.data(), embed.data() + i * n_mmproj_embd * result[i].n_tokens, n_embed);
            LOG_INF("encoded image %2zu/%zu in %8.2f ms, n_tokens = %d\n", i + 1, n_entries,
                    (ggml_time_us() - t_start) / 1000.0, result[i].n_tokens);
        }
    } else {
        // init
        result[0].n_tokens = clip_n_output_tokens(ctx_clip, batch.entries[0].get());
        result[0].n_pos    = result[0].n_tokens;
        result[0].size     = clip_image_size{ batch.entries[0]->nx, batch.entries[0]->ny };
        result[0].embed.resize(result[0].n_tokens * n_mmproj_embd);
        // encode
        clip_add_load_image_size(ctx_clip, &result[0].size);
        const int64_t t_start = ggml_time_us();
        bool          encoded = clip_image_batch_encode(ctx_clip, n_threads, &batch, result[0].embed.data());
        if (!encoded) {
            LOG_ERR("%s", "failed to encode image 1/1\n");
            return {};
        }
        LOG_INF("encoded image 1/1 in %8.2f ms, n_tokens = %d\n", (ggml_time_us() - t_start) / 1000.0,
                result[0].n_tokens);
        if (clip_is_qwen2vl(ctx_clip)) {
            const int32_t ps = clip_get_patch_size(ctx_clip) * 2;
            const int32_t ph = batch.entries[0]->ny / ps + (batch.entries[0]->ny % ps > 0);
            result[0].n_pos  = ph;
        }
    }

    return result;
}
