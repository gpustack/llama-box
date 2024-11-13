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

struct stablediffusion_params {
    int height                      = 512;
    int width                       = 512;
    float guidance                  = 3.5f;
    float strength                  = 0.75f;
    sample_method_t sampler         = N_SAMPLE_METHODS;
    int sample_steps                = 0;
    float cfg_scale                 = 0.0f;
    schedule_t schedule             = DEFAULT;
    bool text_encoder_model_offload = true;
    std::string clip_l_model;
    std::string clip_g_model;
    std::string t5xxl_model;
    bool vae_model_offload = true;
    std::string vae_model;
    bool vae_tiling;
    std::string taesd_model;
    std::string lora_model_dir;
    std::string upscale_model;
    int upscale_repeats        = 1;
    bool control_model_offload = true;
    std::string control_net_model;
    float control_strength = 0.9f;
    bool control_canny     = false;

    // inherited from common_params
    std::string model;
    std::string model_alias;
    int n_threads;
    int main_gpu;
};

struct stablediffusion_sampler_params {
    uint32_t seed               = LLAMA_DEFAULT_SEED;
    int batch_count             = 1;
    int height                  = 1024;
    int width                   = 1024;
    sample_method_t sampler     = EULER_A;
    float cfg_scale             = 9.0f;
    int sample_steps            = 20;
    std::string negative_prompt = "";
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
    sample_method_t get_default_sample_method();
    int get_default_sample_steps();
    float get_default_cfg_scale();
    stablediffusion_generated_image *generate(sd_image_t *init_img, const char *prompt, stablediffusion_sampler_params sparams);

  private:
    sd_ctx_t *sd_ctx = nullptr;
    stablediffusion_params params;
};

stablediffusion_context::~stablediffusion_context() {
    sd_ctx_free(sd_ctx);
}

void stablediffusion_context::free() {
    sd_ctx_free(sd_ctx);
}

sample_method_t stablediffusion_context::get_default_sample_method() {
    return sd_get_default_sample_method(sd_ctx);
}

int stablediffusion_context::get_default_sample_steps() {
    return sd_get_default_sample_steps(sd_ctx);
}

float stablediffusion_context::get_default_cfg_scale() {
    return sd_get_default_cfg_scale(sd_ctx);
}

stablediffusion_generated_image *stablediffusion_context::generate(sd_image_t *img, const char *prompt, stablediffusion_sampler_params sparams) {
    int clip_skip               = -1;
    sd_image_t *control_image   = nullptr;
    std::string input_id_images_path;
    float style_ratio    = 20.f;
    bool normalize_input = false;

    sd_image_t *imgs = nullptr;
    if (img != nullptr) {
        // clang-format off
        imgs = img2img(
                sd_ctx,
                *img,
                prompt,
                sparams.negative_prompt.c_str(),
                clip_skip,
                sparams.cfg_scale,
                params.guidance,
                sparams.width,
                sparams.height,
                sparams.sampler,
                sparams.sample_steps,
                params.strength,
                sparams.seed,
                sparams.batch_count,
                control_image,
                params.control_strength,
                style_ratio,
                normalize_input,
                input_id_images_path.c_str());
        // clang-format on
    } else {
        // clang-format off
        imgs = txt2img(
                sd_ctx,
                prompt,
                sparams.negative_prompt.c_str(),
                clip_skip,
                sparams.cfg_scale,
                params.guidance,
                sparams.width,
                sparams.height,
                sparams.sampler,
                sparams.sample_steps,
                sparams.seed,
                sparams.batch_count,
                control_image,
                params.control_strength,
                style_ratio,
                normalize_input,
                input_id_images_path.c_str());
        // clang-format on
    }

    // TODO upscaler

    std::string img_params_str = "Sampler: " + std::string(sample_methods_argument_str[sparams.sampler]);
    if (params.schedule == KARRAS) {
        img_params_str += " karras";
    }
    img_params_str += ", ";
    img_params_str += "CFG Scale: " + std::to_string(sparams.cfg_scale) + ", ";
    img_params_str += "Steps: " + std::to_string(sparams.sample_steps) + ", ";
    img_params_str += "Guidance: " + std::to_string(params.guidance) + ", ";
    img_params_str += "Seed: " + std::to_string(sparams.seed) + ", ";
    img_params_str += "Size: " + std::to_string(sparams.width) + "x" + std::to_string(sparams.height) + ", ";
    img_params_str += "Model: " + params.model_alias + ", ";
    img_params_str += "Generator: llama-box";

    auto *pngs = new stablediffusion_generated_image[sparams.batch_count];
    for (int i = 0; i < sparams.batch_count; i++) {
        if (imgs[i].data == nullptr) {
            delete[] pngs;
            return nullptr;
        }
        int size            = 0;
        unsigned char *data = stbi_write_png_to_mem(
            (const unsigned char *)imgs[i].data,
            0,
            (int)imgs[i].width,
            (int)imgs[i].height,
            (int)imgs[i].channel,
            &size,
            img_params_str.c_str());
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
    std::string diffusion_model;
    std::string embed_dir;
    std::string stacked_id_embed_dir;
    bool vae_decode_only         = false;
    bool free_params_immediately = false;

    // clang-format off
    sd_ctx_t* sd_ctx = new_sd_ctx(
        params.model.c_str(),
        params.clip_l_model.c_str(),
        params.clip_g_model.c_str(),
        params.t5xxl_model.c_str(),
        diffusion_model.c_str(),
        params.vae_model.c_str(),
        params.taesd_model.c_str(),
        params.control_net_model.c_str(),
        params.lora_model_dir.c_str(),
        embed_dir.c_str(),
        stacked_id_embed_dir.c_str(),
        vae_decode_only,
        params.vae_tiling,
        free_params_immediately,
        params.n_threads,
        GGML_TYPE_COUNT,
        CUDA_RNG,
        params.schedule,
        !params.text_encoder_model_offload,
        !params.control_model_offload,
        !params.vae_model_offload,
        params.main_gpu);
    if (sd_ctx == nullptr) {
        return nullptr;
    }
    // clang-format on

    return new stablediffusion_context(sd_ctx, params);
}
