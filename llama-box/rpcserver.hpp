#pragma once

#include <cinttypes>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winsock2.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include "llama.cpp/ggml/include/ggml-alloc.h"
#include "llama.cpp/ggml/include/ggml-backend.h"
#include "llama.cpp/ggml/include/ggml-rpc.h"
#ifdef GGML_USE_CUDA
#include "llama.cpp/ggml/include/ggml-cuda.h"
#elif GGML_USE_METAL
#include "llama.cpp/ggml/include/ggml-metal.h"
#elif GGML_USE_CANN
#include "llama.cpp/ggml/include/ggml-cann.h"
#elif GGML_USE_SYCL
#include "llama.cpp/ggml/include/ggml-sycl.h"
#endif

#include "utils.hpp"

#ifdef _WIN32
typedef SOCKET sockfd_t;
using ssize_t = __int64;
#else
typedef int sockfd_t;
#endif

struct rpc_server_params {
    std::string hostname = "0.0.0.0";
    int port = 0;
    int32_t main_gpu = 0;
    size_t reserve_memory = 0;
};

static ggml_backend_t rpc_server_create_backend(int32_t gpu) {
    ggml_backend_t backend = NULL;

#ifdef GGML_USE_CUDA
    LOG_INFO("using CUDA backend", {{"gpu", gpu}});
    backend = ggml_backend_cuda_init(gpu);
    if (!backend) {
        LOG_ERROR("ggml_backend_cuda_init() failed", {{"gpu", gpu}});
    }
#elif GGML_USE_METAL
    LOG_INFO("using METAL backend", {});
    backend = ggml_backend_metal_init();
    if (!backend) {
        LOG_ERROR("ggml_backend_metal_init() failed", {});
    }
#elif GGML_USE_CANN
    LOG_INFO("using CANN backend", {{"gpu", gpu}});
    backend = ggml_backend_cann_init(gpu);
    if (!backend) {
        LOG_ERROR("ggml_backend_cann_init() failed", {{"gpu", gpu}});
    }
#elif GGML_USE_SYCL
    LOG_INFO("using SYCL backend", {{"gpu", gpu}});
    backend = ggml_backend_sycl_init(gpu);
    if (!backend) {
        LOG_ERROR("ggml_backend_sycl_init() failed", {{"gpu", gpu}});
    }
#endif

    // if there aren't GPU Backends fallback to CPU backend
    if (!backend) {
        LOG_INFO("fallback, using CPU backend", {});
        backend = ggml_backend_cpu_init();
    }
    return backend;
}

static void rpc_server_get_backend_memory(int32_t gpu, size_t *free_mem, size_t *total_mem) {
    if (gpu < 0) {
#ifdef _WIN32
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        *total_mem = status.ullTotalPhys;
        *free_mem = status.ullAvailPhys;
#else
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        *total_mem = pages * page_size;
        *free_mem = *total_mem;
#endif
        return;
    }
#ifdef GGML_USE_CUDA
    ggml_backend_cuda_get_device_memory(gpu, free_mem, total_mem);
#elif GGML_USE_METAL
    ggml_backend_metal_get_device_memory(free_mem, total_mem);
#elif GGML_USE_CANN
    ggml_backend_cann_get_device_memory(gpu, free_mem, total_mem);
#elif GGML_USE_SYCL
    ggml_backend_sycl_get_device_memory(gpu, free_mem, total_mem);
#else
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    *total_mem = status.ullTotalPhys;
    *free_mem = status.ullAvailPhys;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total_mem = pages * page_size;
    *free_mem = *total_mem;
#endif
#endif
}

static int rpc_server_start(const rpc_server_params &params) {
    LOG_INFO("starting rpc server", {{"hostname", params.hostname}, {"port", params.port}});

    ggml_backend_t backend;
    if (params.main_gpu < 0) {
        LOG_INFO("using CPU backend", {});
        backend = ggml_backend_cpu_init();
    } else {
        backend = rpc_server_create_backend(params.main_gpu);
    }
    if (!backend) {
        LOG_ERROR("failed to create backend", {});
        return 1;
    }

    size_t free_mem, total_mem;
    rpc_server_get_backend_memory(params.main_gpu, &free_mem, &total_mem);
    if (free_mem - params.reserve_memory <= 0) {
        LOG_ERROR("not enough memory", {{"free", free_mem >> 20}, {"total", total_mem >> 20}});
        return 1;
    }
    free_mem -= params.reserve_memory;
    total_mem -= params.reserve_memory;
    LOG_INFO("backend memory", {{"free", free_mem >> 20}, {"total", total_mem >> 20}});

    LOG_INFO("starting rpc server", {{"hostname", params.hostname}, {"port", params.port}});
    std::string endpoint = params.hostname + ":" + std::to_string(params.port);
    start_rpc_server(backend, endpoint.c_str(), free_mem, total_mem);

    LOG_INFO("stopped rpc server", {});
    ggml_backend_free(backend);
    return 0;
}