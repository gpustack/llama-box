// heads

// defines

// types

struct v2_rpcserver_params {
    std::string hostname  = "0.0.0.0";
    int port              = 0;
    int32_t main_gpu      = 0;
    size_t reserve_memory = 0;
    bool use_cache        = false;
    std::string cache_dir = fs_get_cache_directory() + +"rpc/";

    // inherited from common_params
    ggml_numa_strategy numa = GGML_NUMA_STRATEGY_DISABLED;
};

// implementations

struct RpcHandler {
    v2_rpcserver_params params;

    explicit RpcHandler(v2_rpcserver_params &params)
        : params(params) {
        llama_numa_init(params.numa);
        llama_backend_init();
    }

    ~RpcHandler() {
        llama_backend_free();
    }
};

static int start_rpcserver(v2_rpcserver_params &params) {
    return 0;
}