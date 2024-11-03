#pragma once

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stable-diffusion.cpp/thirdparty/stb_image.h"
#define STB_IMAGE_RESIZE_STATIC
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stable-diffusion.cpp/thirdparty/stb_image_resize.h"
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stable-diffusion.cpp/thirdparty/stb_image_write.h"

#include "stable-diffusion.cpp/stable-diffusion.h"

// Names of the sampler method, same order as enum sample_method in stable-diffusion.h
const char *sample_method_str[] = {
    "euler_a", "euler", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2", "ipndm", "ipndm_v", "lcm",
};

sample_method_t common_sd_str_to_sampler_type(const char *sample_method) {
    for (int m = 0; m < N_SAMPLE_METHODS; m++) {
        if (!strcmp(sample_method, sample_method_str[m])) {
            return (sample_method_t)m;
        }
    }
    return EULER_A;
}

std::string common_sd_sampler_type_to_str(sample_method_t sample_method) {
    return std::string(sample_method_str[sample_method]);
}

// Names of the sigma schedule overrides, same order as sample_schedule in stable-diffusion.h
const char *schedule_str[] = {
    "default", "discrete", "karras", "exponential", "ays", "gits",
};

schedule_t common_sd_str_to_schedule(const char *schedule) {
    for (int d = 0; d < N_SCHEDULES; d++) {
        if (!strcmp(schedule, schedule_str[d])) {
            return (schedule_t)d;
        }
    }
    return DEFAULT;
}

std::string common_sd_schedule_to_str(schedule_t schedule) {
    return std::string(schedule_str[schedule]);
}

ggml_log_level common_sd_log_level_to_ggml_log_level(sd_log_level_t src) {
    switch (src) {
    case SD_LOG_DEBUG:
        return GGML_LOG_LEVEL_DEBUG;
    case SD_LOG_INFO:
        return GGML_LOG_LEVEL_INFO;
    case SD_LOG_WARN:
        return GGML_LOG_LEVEL_WARN;
    case SD_LOG_ERROR:
        return GGML_LOG_LEVEL_ERROR;
    }
}

void sd_log_set(sd_log_cb_t sd_log_cb, void *data) {
    sd_set_log_callback(sd_log_cb, data);
}

struct stablediffusion_params {
    int width = 1024;
    int height = 1024;
    int batch_count = 1;
    float min_cfg = 1.0f;
    float cfg_scale = 7.0f;
    float guidance = 3.5f;
    float style_ratio = 20.f;
    int clip_skip = -1; // <= 0 represents unspecified

    sample_method_t sampler = EULER_A;
    int sample_steps = 20;
    schedule_t schedule = DEFAULT;
    float strength = 0.75f;
    float control_strength = 0.9f;
    std::string input_id_images_path;
    bool normalize_input = false;

    std::string lora_dir;

    // inherited from common_params
    std::string model;
    std::string model_alias;
    int n_threads;
};

struct stablediffusion_sampler_params {
    uint32_t seed = LLAMA_DEFAULT_SEED; // the seed used to initialize llama_sampler
    int width = 1024;
    int height = 1024;
    int batch_count = 1;
    float min_cfg = 1.0f;
    float cfg_scale = 7.0f;
    float guidance = 3.5f;
    float style_ratio = 20.f;
    int clip_skip = -1; // <= 0 represents unspecified
    sample_method_t sampler = EULER_A;
    int sample_steps = 20;
};

struct stablediffusion_generated_image {
    int size;
    unsigned char *data;
};

class stablediffusion_context {
  public:
    stablediffusion_context(sd_ctx_t *sd_ctx, stablediffusion_params params)
        : sd_ctx(sd_ctx), params(params) {
    }

    ~stablediffusion_context();

    void free();

    stablediffusion_generated_image *generate(sd_image_t *init_img, const char *prompt, const stablediffusion_sampler_params sparams);

  private:
    sd_ctx_t *sd_ctx = nullptr;
    stablediffusion_params params;
};

stablediffusion_context::~stablediffusion_context() {
    free_sd_ctx(sd_ctx);
}

void stablediffusion_context::free() {
    free_sd_ctx(sd_ctx);
}

stablediffusion_generated_image *stablediffusion_context::generate(sd_image_t *img, const char *prompt,
                                                                   const stablediffusion_sampler_params sparams) {
    std::string negative_prompt;
    sd_image_t *control_image = nullptr;
    std::string input_id_images_path;

    sd_image_t *imgs = nullptr;
    if (img != nullptr) {
        // clang-format off
        imgs = img2img(
                sd_ctx,
                *img,
                prompt,
                negative_prompt.c_str(),
                params.clip_skip,
                params.cfg_scale,
                params.guidance,
                sparams.width,
                sparams.height,
                sparams.sampler,
                params.sample_steps,
                params.strength,
                sparams.seed,
                sparams.batch_count,
                control_image,
                params.control_strength,
                params.style_ratio,
                params.normalize_input,
                input_id_images_path.c_str());
        // clang-format on
    } else {
        // clang-format off
        imgs = txt2img(
                sd_ctx,
                prompt,
                negative_prompt.c_str(),
                params.clip_skip,
                params.cfg_scale,
                params.guidance,
                sparams.width,
                sparams.height,
                sparams.sampler,
                params.sample_steps,
                sparams.seed,
                sparams.batch_count,
                control_image,
                params.control_strength,
                params.style_ratio,
                params.normalize_input,
                input_id_images_path.c_str());
        // clang-format on
    }

    // TODO upscaler

    auto *pngs = new stablediffusion_generated_image[sparams.batch_count];
    for (int i = 0; i < sparams.batch_count; i++) {
        if (imgs[i].data == nullptr) {
            delete[] pngs;
            return nullptr;
        }
        int size = 0;
        unsigned char *data = stbi_write_png_to_mem((const unsigned char *)imgs[i].data, 0, (int)imgs[i].width, (int)imgs[i].height,
                                                    (int)imgs[i].channel, &size, nullptr);
        if (data == nullptr || size <= 0) {
            delete[] pngs;
            return nullptr;
        }
        pngs[i].size = size;
        pngs[i].data = data;
    }

    return pngs;
}

stablediffusion_context *common_sd_init_from_params(stablediffusion_params params) {
    std::string clip_l_path;
    std::string clip_g_path;
    std::string t5xxl_path;
    std::string diffusion_model_path;
    std::string vae_path;
    std::string taesd_path;
    std::string control_net_path_c_str;
    std::string embed_dir_c_str;
    std::string stacked_id_embed_dir_c_str;
    bool vae_decode_only = true;
    bool vae_tiling = false;
    bool free_params_immediately = false;
    rng_type_t rng_type = CUDA_RNG;
    bool keep_clip_on_cpu = false;
    bool keep_control_net_cpu = false;
    bool keep_vae_on_cpu = false;

    // clang-format off
    sd_ctx_t* sd_ctx = new_sd_ctx(
        params.model.c_str(),
        clip_l_path.c_str(),
        clip_g_path.c_str(),
        t5xxl_path.c_str(),
        diffusion_model_path.c_str(),
        vae_path.c_str(),
        taesd_path.c_str(),
        control_net_path_c_str.c_str(),
        params.lora_dir.c_str(),
        embed_dir_c_str.c_str(),
        stacked_id_embed_dir_c_str.c_str(),
        vae_decode_only,
        vae_tiling,
        free_params_immediately,
        params.n_threads,
        SD_TYPE_F32,
        rng_type,
        params.schedule,
        keep_clip_on_cpu,
        keep_control_net_cpu,
        keep_vae_on_cpu);
    if (sd_ctx == nullptr) {
        return nullptr;
    }
    // clang-format on

    return new stablediffusion_context(sd_ctx, params);
}
