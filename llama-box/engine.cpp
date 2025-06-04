// heads

#include <sstream>

#include "llama.cpp/common/common.h"
#include "llama.cpp/common/log.h"
#include "llama.cpp/ggml/include/ggml.h"
#include "llama.cpp/include/llama.h"

#define SELF_PACKAGE 0
#include "engine_param.hpp"

// implementations

int main(int argc, char ** argv) {
#if __linux__ && defined(GGML_USE_HIP)
    // NB(thxCode): this is a workaround for the issue that the ROCm runtime occupies the CPU 100% utilization,
    // see https://github.com/gpustack/gpustack/issues/844.
    setenv("GPU_MAX_HW_QUEUES", "1", 1);
#endif

    // init log
    common_log_set_prefix(common_log_main(), true);
    common_log_set_timestamps(common_log_main(), true);
    llama_log_set(
        [](ggml_log_level level, const char * text, void * /*user_data*/) {
            if (level == GGML_LOG_LEVEL_DEBUG && common_log_verbosity_thold < 6) {
                return;
            }
            common_log_add(common_log_main(), level, "%s", text);
        },
        nullptr);
    // NB(thxCode): clip_log_set is a patch.
    clip_log_set(
        [](ggml_log_level level, const char * text, void * /*user_data*/) {
            if (level == GGML_LOG_LEVEL_DEBUG && common_log_verbosity_thold < 6) {
                return;
            }
            common_log_add(common_log_main(), level, "%s", text);
        },
        nullptr);
    sd_log_set(
        [](sd_log_level_t level, const char * text, void * /*user_data*/) {
            if (level == SD_LOG_DEBUG && common_log_verbosity_thold < 6) {
                return;
            }
            common_log_add(common_log_main(), sd_log_level_to_ggml_log_level(level), "%s", text);
        },
        nullptr);
    sd_progress_set(
        [](int /*step*/, int /*steps*/, float /*time*/, void * /*user_data*/) {
            // nothing to do
        },
        nullptr);

    // parse arguments
    llama_box_params params{};
    if (!llama_box_params_parse(argc, argv, params)) {
        llama_box_params_print_usage(argc, argv, params);
        return 1;
    }

    // tell backends zero-offloading or not
    // NB(thxCode): ggml_backend_register_metadata_set is a patch.
    ggml_backend_register_metadata_set(params.hs_params.llm_params.n_gpu_layers);

    // print arguments
    {
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
        LOG_INF(
            "vendor     : llama.cpp %s (%d), stable-diffusion.cpp %s (%d), concurrentqueue %s (%d), readerwriterqueue "
            "%s (%d)\n",
            LLAMA_CPP_COMMIT, LLAMA_CPP_BUILD_NUMBER, STABLE_DIFFUSION_CPP_COMMIT, STABLE_DIFFUSION_CPP_BUILD_NUMBER,
            CONCURRENT_QUEUE_COMMIT, CONCURRENT_QUEUE_BUILD_NUMBER, READER_WRITER_QUEUE_COMMIT,
            READER_WRITER_QUEUE_BUILD_NUMBER);
        LOG_INF("%s\n", common_params_get_system_info(params.hs_params.llm_params).c_str());
        LOG_INF("\n");
    }

    if (params.rs_params.port > 0) {
#if defined(GGML_USE_METAL)
        // NB(thxCode): disable residency set for Metal backend to avoid memory leak.
        setenv("GGML_METAL_NO_RESIDENCY", "1", 1);
#endif
        return start_rpcserver(params.rs_params);
    }

    return start_httpserver(params.hs_params);
}
