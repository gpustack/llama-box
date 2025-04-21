#pragma once

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstring>
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
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif
#include <filesystem>
#include <fstream>

#include "llama.cpp/common/log.h"
#include "llama.cpp/ggml/include/ggml-alloc.h"
#include "llama.cpp/ggml/include/ggml-backend.h"
#include "llama.cpp/ggml/include/ggml-cpp.h"
#include "llama.cpp/ggml/include/ggml-rpc.h"
#include "llama.cpp/ggml/src/ggml-backend-impl.h"
#include "llama.cpp/ggml/src/ggml-impl.h"
#ifdef GGML_USE_CUDA
#include "llama.cpp/ggml/include/ggml-cuda.h"
#endif
#ifdef GGML_USE_METAL
#include "llama.cpp/ggml/include/ggml-metal.h"
#endif
#ifdef GGML_USE_CANN
#include "llama.cpp/ggml/include/ggml-cann.h"
#endif
#ifdef GGML_USE_VULKAN
#include "llama.cpp/ggml/include/ggml-vulkan.h"
#endif
#ifdef GGML_USE_SYCL
#include "llama.cpp/ggml/include/ggml-sycl.h"
#endif

#include "utils.hpp"

namespace fs = std::filesystem;

#ifdef _WIN32
typedef SOCKET rpc_sockfd_t;
using ssize_t = __int64;
#else
typedef int rpc_sockfd_t;
#endif

// cross-platform socket
struct rpc_socket_t {
    rpc_sockfd_t fd;

    rpc_socket_t(rpc_sockfd_t fd)
        : fd(fd) {
    }

    ~rpc_socket_t() {
#ifdef _WIN32
        closesocket(this->fd);
#else
        close(this->fd);
#endif
    }
};

// ggml_tensor is serialized into rpc_tensor
#pragma pack(push, 1)

struct rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char name[GGML_MAX_NAME];

    char padding[4];
};

#pragma pack(pop)

static_assert(sizeof(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
enum rpc_cmd {
    RPC_CMD_ALLOC_BUFFER = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_SIZE,
    RPC_CMD_BUFFER_GET_BASE,
    RPC_CMD_FREE_BUFFER,
    RPC_CMD_BUFFER_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_SET_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_SIZE,
    RPC_CMD_HELLO,
    RPC_CMD_SUPPORT_OP,
    RPC_CMD_COUNT,
};

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
const size_t HASH_THRESHOLD = 10 * 1024 * 1024;

// RPC helper

// Computes FNV-1a hash of the data
static uint64_t fnv_hash(const uint8_t *data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash            = 0xcbf29ce484222325ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return hash;
}

static std::shared_ptr<rpc_socket_t> rpc_socket_create(rpc_sockfd_t fd) {
#ifdef _WIN32
    if (fd == INVALID_SOCKET) {
        return nullptr;
    }
#else
    if (fd < 0) {
        return nullptr;
    }
#endif
    return std::make_shared<rpc_socket_t>(fd);
}

const static size_t MAX_CHUNK = 1 << 23; // 8MiB

static bool rpc_send_data(rpc_sockfd_t sockfd, const void *data, size_t size) {
    if (size != 0) {
        SRV_DBG("sending data: data range = [%p, %p), bytes_target = %zu\n", data, (char *)data + size, size);
    }
    size_t bytes_sent = 0;
    while (bytes_sent < size) {
        size_t bytes_chunk = MIN(size - bytes_sent, MAX_CHUNK);
        ssize_t n          = send(sockfd, (const char *)data + bytes_sent, bytes_chunk, 0);
        if (n < 0) {
            int err = errno;
            if (err == EINTR || err == EAGAIN || err == EWOULDBLOCK) {
                SRV_WRN("interrupted: data range = [%p, %p), bytes_sent = %zu, bytes_target = %zu, errno = %d, errmsg = %s, retrying...\n", data, (char *)data + size, bytes_sent, size, err, strerror(err));
                continue; // try again
            }
            if (err != 0) {
                SRV_ERR("failed to send data: data range = [%p, %p), bytes_sent = %zu, bytes_target = %zu, errno = %d, errmsg = %s\n", data, (char *)data + size, bytes_sent, size, err, strerror(err));
                int serr           = 0;
                socklen_t serr_len = sizeof(serr);
                int ret            = getsockopt(sockfd, SOL_SOCKET, SO_ERROR, (char *)&serr, &serr_len);
                if (ret < 0) {
                    err = errno;
                    SRV_ERR("failed to get peer socket error: errno = %d, errmsg = %s\n", err, strerror(err));
                } else if (serr != 0) {
                    SRV_ERR("peer socket error: errno = %d, errmsg = %s\n", serr, strerror(serr));
                } else {
                    struct sockaddr_in sin{};
                    socklen_t addr_len = sizeof(sin);
                    ret                = getpeername(sockfd, (struct sockaddr *)&sin, &addr_len);
                    if (ret < 0) {
                        err = errno;
                        SRV_ERR("peer may have been disconnected: errno = %d, errmsg = %s\n", err, strerror(err));
                    }
                }
            }
            return false;
        }
        bytes_sent += n;
    }
    return true;
}

static bool rpc_recv_data(rpc_sockfd_t sockfd, void *data, size_t size) {
    if (size != 0) {
        SRV_DBG("receiving data: data range = [%p, %p), bytes_target = %zu\n", data, (char *)data + size, size);
    }
    size_t bytes_recv = 0;
    while (bytes_recv < size) {
        size_t bytes_chunk = MIN(size - bytes_recv, MAX_CHUNK);
        ssize_t n          = recv(sockfd, (char *)data + bytes_recv, bytes_chunk, 0);
        if (n <= 0) {
            int err = errno;
            if (err == EINTR || err == EAGAIN || err == EWOULDBLOCK) {
                SRV_WRN("interrupted: data range = [%p, %p), bytes_recv = %zu, bytes_target = %zu, errno = %d, errmsg = %s, retrying...\n", data, (char *)data + size, bytes_recv, size, err, strerror(errno));
                continue; // try again
            }
            if (err != 0 && err != ESRCH) {
                SRV_ERR("failed to recv data: data range = [%p, %p), bytes_recv = %zu, bytes_target = %zu, errno = %d, errmsg = %s\n", data, (char *)data + size, bytes_recv, size, err, strerror(errno));
                int serr           = 0;
                socklen_t serr_len = sizeof(serr);
                int ret            = getsockopt(sockfd, SOL_SOCKET, SO_ERROR, (char *)&serr, &serr_len);
                if (ret < 0) {
                    err = errno;
                    SRV_ERR("failed to get peer socket error: errno = %d, errmsg = %s\n", err, strerror(err));
                } else if (serr != 0) {
                    SRV_ERR("peer socket error: errno = %d, errmsg = %s\n", serr, strerror(serr));
                } else {
                    struct sockaddr_in sin{};
                    socklen_t addr_len = sizeof(sin);
                    ret                = getpeername(sockfd, (struct sockaddr *)&sin, &addr_len);
                    if (ret < 0) {
                        err = errno;
                        SRV_ERR("peer may have been disconnected: errno = %d, errmsg = %s\n", err, strerror(err));
                    }
                }
            }
            return false;
        }
        bytes_recv += n;
    }
    return true;
}

static ggml_backend_t rpcserver_create_backend(int32_t &gpu) {
    ggml_backend_t backend = nullptr;

#ifdef GGML_USE_CUDA
    SRV_INF("using CUDA backend, gpu: %d\n", gpu);
    backend = ggml_backend_cuda_init(gpu);
    if (!backend) {
        SRV_ERR("ggml_backend_cuda_init(%d) failed\n", gpu);
    }
#elif GGML_USE_METAL
    SRV_INF("using METAL backend, gpu: %d\n", gpu);
    backend = ggml_backend_metal_init();
    if (!backend) {
        SRV_ERR("%s", "ggml_backend_metal_init() failed\n");
    }
#elif GGML_USE_CANN
    SRV_INF("using CANN backend, gpu: %d\n", gpu);
    backend = ggml_backend_cann_init(gpu);
    if (!backend) {
        SRV_ERR("ggml_backend_cann_init(%d) failed\n", gpu);
    }
#elif GGML_USE_VULKAN
    SRV_INF("using VULKAN backend, gpu: %d\n", gpu);
    backend = ggml_backend_vk_init(gpu);
    if (!backend) {
        SRV_ERR("ggml_backend_vk_init(%d) failed\n", gpu);
    }
#elif GGML_USE_SYCL
    SRV_INF("using SYCL backend, gpu: %d\n", gpu);
    backend = ggml_backend_sycl_init(gpu);
    if (!backend) {
        SRV_ERR("ggml_backend_sycl_init(%d) failed\n", gpu);
    }
#endif

    // if there aren't GPU Backends fallback to CPU backend
    if (!backend) {
        SRV_INF("%s", "fallback, using CPU backend\n");
        backend = ggml_backend_cpu_init();
        gpu     = -1;
    }
    return backend;
}

static void rpcserver_get_backend_memory(ggml_backend_t backend, int32_t gpu, size_t *free_mem, size_t *total_mem) {
    if (gpu < 0) {
#ifdef _WIN32
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        *total_mem = status.ullTotalPhys;
        *free_mem  = status.ullAvailPhys;
#else
        long pages     = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        *total_mem     = pages * page_size;
        *free_mem      = *total_mem;
#endif
        return;
    }
#ifdef GGML_USE_CUDA
    ggml_backend_cuda_get_device_memory(gpu, free_mem, total_mem);
#elif GGML_USE_METAL
    ggml_backend_metal_get_device_memory(backend, free_mem, total_mem);
#elif GGML_USE_CANN
    ggml_backend_cann_get_device_memory(gpu, free_mem, total_mem);
#elif GGML_USE_VULKAN
    ggml_backend_vk_get_device_memory(gpu, free_mem, total_mem);
#elif GGML_USE_SYCL
    ggml_backend_sycl_get_device_memory(gpu, free_mem, total_mem);
#else
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    *total_mem = status.ullTotalPhys;
    *free_mem  = status.ullAvailPhys;
#else
    long pages     = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total_mem     = pages * page_size;
    *free_mem      = *total_mem;
#endif
#endif
}

// RPC server implementation

class rpcserver {
  public:
    rpcserver(ggml_backend_t backend, const char *cache_dir, int32_t index = 0, size_t capacity = 0)
        : backend(backend), cache_dir(cache_dir), index(index), capacity(capacity) {
    }

    ~rpcserver();

    bool alloc_buffer(const std::vector<uint8_t> &input, std::vector<uint8_t> &output);
    bool get_alignment(std::vector<uint8_t> &output);
    bool get_max_size(std::vector<uint8_t> &output);
    bool buffer_get_base(const std::vector<uint8_t> &input, std::vector<uint8_t> &output);
    bool free_buffer(const std::vector<uint8_t> &input);
    bool buffer_clear(const std::vector<uint8_t> &input);
    bool set_tensor(const std::vector<uint8_t> &input);
    bool set_tensor_hash(const std::vector<uint8_t> &input, std::vector<uint8_t> &output);
    bool get_tensor(const std::vector<uint8_t> &input, std::vector<uint8_t> &output);
    bool copy_tensor(const std::vector<uint8_t> &input, std::vector<uint8_t> &output);
    bool graph_compute(const std::vector<uint8_t> &input, std::vector<uint8_t> &output);
    bool get_device_memory(std::vector<uint8_t> &output);
    bool init_tensor(const std::vector<uint8_t> &input);
    bool get_alloc_size(const std::vector<uint8_t> &input, std::vector<uint8_t> &output);
    bool say_hello(std::vector<uint8_t> &output);
    bool support_op(const std::vector<uint8_t> &input, std::vector<uint8_t> &output);

  private:
    bool get_cached_file(uint64_t hash, std::vector<uint8_t> &data);
    size_t get_free_memory() const;
    ggml_tensor *deserialize_tensor(struct ggml_context *ctx, const rpc_tensor *tensor);
    ggml_tensor *create_node(uint64_t id, struct ggml_context *ctx, const std::unordered_map<uint64_t, const rpc_tensor *> &tensor_ptrs, std::unordered_map<uint64_t, struct ggml_tensor *> &tensor_map);

    ggml_backend_t backend;
    const char *cache_dir                             = nullptr;
    int32_t index                                     = 0;
    size_t capacity                                   = 0;
    std::unordered_set<ggml_backend_buffer_t> buffers = {};
};

rpcserver::~rpcserver() {
    for (ggml_backend_buffer *buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
    buffers.clear();
}

bool rpcserver::alloc_buffer(const std::vector<uint8_t> &input, std::vector<uint8_t> &output) {
    // input serialization format: | size (8 bytes) |
    if (input.size() != sizeof(uint64_t)) {
        SRV_ERR("%s", "failed: input size invalid\n");
        return false;
    }
    uint64_t size;
    memcpy(&size, input.data(), sizeof(size));
    size_t free_mem = get_free_memory();
    if (size > free_mem) {
        SRV_ERR("failed: out of memory, "
                "request_mib = %lu, free_mib = %zu\n",
                size >> 20, free_mem >> 20);
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_t buffer    = ggml_backend_buft_alloc_buffer(buft, size);
    uint64_t remote_ptr             = 0;
    uint64_t remote_size            = 0;
    if (buffer != nullptr) {
        remote_ptr  = reinterpret_cast<uint64_t>(buffer);
        remote_size = buffer->size;
        buffers.insert(buffer);
    } else {
        SRV_ERR("failed: buffer allocation failed, "
                "request_mib = %lu, free_mib = %zu\n",
                size >> 20, free_mem >> 20);
    }
    // output serialization format: | remote_ptr (8 bytes) | remote_size (8 bytes) |
    output.resize(2 * sizeof(uint64_t), 0);
    memcpy(output.data(), &remote_ptr, sizeof(remote_ptr));
    memcpy(output.data() + sizeof(uint64_t), &remote_size, sizeof(remote_size));
    SRV_DBG("remote_ptr = %lu, request_mib = %lu\n", remote_ptr, size >> 20);
    return true;
}

bool rpcserver::get_alignment(std::vector<uint8_t> &output) {
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    size_t alignment                = ggml_backend_buft_get_alignment(buft);
    // output serialization format: | alignment (8 bytes) |
    output.resize(sizeof(uint64_t), 0);
    memcpy(output.data(), &alignment, sizeof(alignment));
    SRV_DBG("alignment = %zu\n", alignment);
    return true;
}

bool rpcserver::get_max_size(std::vector<uint8_t> &output) {
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    size_t max_size                 = ggml_backend_buft_get_max_size(buft);
    // output serialization format: | max_size (8 bytes) |
    output.resize(sizeof(uint64_t), 0);
    memcpy(output.data(), &max_size, sizeof(max_size));
    SRV_DBG("max_size = %zu\n", max_size);
    return true;
}

bool rpcserver::buffer_get_base(const std::vector<uint8_t> &input, std::vector<uint8_t> &output) {
    // input serialization format: | remote_ptr (8 bytes) |
    if (input.size() != sizeof(uint64_t)) {
        return false;
    }
    uint64_t remote_ptr;
    memcpy(&remote_ptr, input.data(), sizeof(remote_ptr));
    auto buffer = reinterpret_cast<ggml_backend_buffer_t>(remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        SRV_ERR("failed: remote_ptr = %lu\n", remote_ptr);
        return false;
    }
    void *base = ggml_backend_buffer_get_base(buffer);
    // output serialization format: | base_ptr (8 bytes) |
    auto base_ptr = reinterpret_cast<uint64_t>(base);
    output.resize(sizeof(uint64_t), 0);
    memcpy(output.data(), &base_ptr, sizeof(base_ptr));
    SRV_DBG("remote_ptr = %lu, base_ptr = %lu\n", remote_ptr, base_ptr);
    return true;
}

bool rpcserver::free_buffer(const std::vector<uint8_t> &input) {
    // input serialization format: | remote_ptr (8 bytes) |
    if (input.size() != sizeof(uint64_t)) {
        SRV_ERR("%s", "failed: input size invalid\n");
        return false;
    }
    uint64_t remote_ptr;
    memcpy(&remote_ptr, input.data(), sizeof(remote_ptr));
    auto buffer = reinterpret_cast<ggml_backend_buffer_t>(remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        SRV_WRN("failed: not found remote_ptr = %lu\n", remote_ptr);
        return true;
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    SRV_DBG("remote_ptr = %lu\n", remote_ptr);
    return true;
}

bool rpcserver::buffer_clear(const std::vector<uint8_t> &input) {
    // input serialization format: | remote_ptr (8 bytes) | value (1 byte) |
    if (input.size() != sizeof(uint64_t) + sizeof(uint8_t)) {
        SRV_ERR("%s", "failed: input size invalid\n");
        return false;
    }
    uint64_t remote_ptr;
    memcpy(&remote_ptr, input.data(), sizeof(remote_ptr));
    uint8_t value;
    memcpy(&value, input.data() + sizeof(uint64_t), sizeof(value));
    auto buffer = reinterpret_cast<ggml_backend_buffer_t>(remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        SRV_WRN("failed: not found remote_ptr = %lu\n", remote_ptr);
        return true;
    }
    ggml_backend_buffer_clear(buffer, value);
    SRV_DBG("remote_ptr = %lu\n", remote_ptr);
    return true;
}

bool rpcserver::set_tensor(const std::vector<uint8_t> &input) {
    // serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    if (input.size() < sizeof(rpc_tensor) + sizeof(uint64_t)) {
        SRV_ERR("%s", "failed: input size invalid\n");
        return false;
    }
    const auto *in_tensor = (const rpc_tensor *)input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    const size_t size = input.size() - sizeof(rpc_tensor) - sizeof(offset);

    struct ggml_init_params params{
        /*.mem_size   =*/ggml_tensor_overhead(),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };

    ggml_context_ptr ctx_ptr{ggml_init(params)};
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context *ctx   = ctx_ptr.get();
    ggml_tensor *tensor = deserialize_tensor(ctx, in_tensor);

    // sanitize tensor->data
    {
        const auto p0   = (size_t)ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            SRV_ERR("%s", "failed: tensor->data out of bounds\n");
            delete tensor;
            return false;
        }
    }

    const void *data   = input.data() + sizeof(rpc_tensor) + sizeof(offset);
    int caching_status = -1; // -1 = no cache, 0 = cached ok, 1 = cache failed
    if (cache_dir && size > HASH_THRESHOLD) {
        try {
            uint64_t hash = fnv_hash((const uint8_t *)data, size);
            char hash_str[17];
            snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
            fs::path cache_file = fs::path(cache_dir) / hash_str;
            std::ofstream ofs(cache_file, std::ios::binary);
            ofs.write((const char *)data, size);
            caching_status = 0;
        } catch (std::exception &e) {
            SRV_WRN("cannot cache tensor id = %lu: %s\n", in_tensor->id, e.what());
            caching_status = 1;
        }
    }
    ggml_backend_tensor_set(tensor, data, offset, size);
    SRV_DBG("id = %lu, size = %zu, name = %s, type = %s, caching = %d\n", in_tensor->id, size, in_tensor->name, ggml_type_name((ggml_type)in_tensor->type), caching_status);
    return true;
}

bool rpcserver::set_tensor_hash(const std::vector<uint8_t> &input, std::vector<uint8_t> &output) {
    // output serialization format: | result (1 byte) |
    output.resize(1, 0);

    // serialization format: | rpc_tensor | offset (8 bytes) | hash (8 bytes) |
    if (input.size() != sizeof(rpc_tensor) + 16) {
        return false;
    }
    const rpc_tensor *in_tensor = (const rpc_tensor *)input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    const uint64_t *hash = (const uint64_t *)(input.data() + sizeof(rpc_tensor) + sizeof(offset));

    // get cached
    std::vector<uint8_t> cached_file;
    if (!get_cached_file(*hash, cached_file)) {
        output[0] = 0;
        return true;
    }
    size_t size = cached_file.size();

    struct ggml_init_params params{
        /*.mem_size   =*/ggml_tensor_overhead(),
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };

    ggml_context_ptr ctx_ptr{ggml_init(params)};
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context *ctx   = ctx_ptr.get();
    ggml_tensor *tensor = deserialize_tensor(ctx, in_tensor);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t)ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            SRV_ERR("failed to set tensor with hash: tensor->data out of bounds: id = %lu, size = %lu, data = %p, offset = %llu, hash = %" PRIx64 "\n", in_tensor->id, size, tensor->data, offset, *hash);
            delete tensor;
            return false;
        }
    }

    ggml_backend_tensor_set(tensor, cached_file.data(), offset, size);
    output[0] = 1;
    SRV_DBG("id = %lu, size = %zu, name = %s, type = %s\n", in_tensor->id, size, in_tensor->name, ggml_type_name((ggml_type)in_tensor->type));
    return true;
}

bool rpcserver::get_tensor(const std::vector<uint8_t> &input, std::vector<uint8_t> &output) {
    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) |
    if (input.size() != sizeof(rpc_tensor) + 2 * sizeof(uint64_t)) {
        SRV_ERR("%s", "failed: input size invalid\n");
        return false;
    }
    const auto *in_tensor = (const rpc_tensor *)input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    uint64_t size;
    memcpy(&size, input.data() + sizeof(rpc_tensor) + sizeof(offset), sizeof(size));

    struct ggml_init_params params{
        /*.mem_size   =*/ggml_tensor_overhead(),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };

    ggml_context_ptr ctx_ptr{ggml_init(params)};
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context *ctx   = ctx_ptr.get();
    ggml_tensor *tensor = deserialize_tensor(ctx, in_tensor);

    // sanitize tensor->data
    {
        const auto p0   = (size_t)ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            SRV_ERR("%s", "failed: tensor->data out of bounds\n");
            delete tensor;
            return false;
        }
    }

    // output serialization format: | data (size bytes) |
    output.resize(size, 0);
    ggml_backend_tensor_get(tensor, output.data(), offset, size);
    SRV_DBG("id = %lu, size = %lu, name = %s, type = %s\n", in_tensor->id, size, in_tensor->name, ggml_type_name((ggml_type)in_tensor->type));
    return true;
}

bool rpcserver::copy_tensor(const std::vector<uint8_t> &input, std::vector<uint8_t> &output) {
    // serialization format: | rpc_tensor src | rpc_tensor dst |
    if (input.size() != 2 * sizeof(rpc_tensor)) {
        SRV_ERR("%s", "failed: input size invalid\n");
        return false;
    }
    const auto *rpc_src = (const rpc_tensor *)input.data();
    const auto *rpc_dst = (const rpc_tensor *)(input.data() + sizeof(rpc_src));

    struct ggml_init_params params{
        /*.mem_size   =*/2 * ggml_tensor_overhead(),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };

    ggml_context_ptr ctx_ptr{ggml_init(params)};
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context *ctx = ctx_ptr.get();
    ggml_tensor *src  = deserialize_tensor(ctx, rpc_src);
    ggml_tensor *dst  = deserialize_tensor(ctx, rpc_dst);

    auto src_size   = (uint64_t)ggml_nbytes(src);
    auto dst_data   = (uint64_t)dst->data;
    auto dst_base   = (uint64_t)ggml_backend_buffer_get_base(dst->buffer);
    auto dst_buf_sz = (uint64_t)ggml_backend_buffer_get_size(dst->buffer);
    if (dst_data + src_size > dst_base + dst_buf_sz) {
        SRV_ERR("failed: out-of-bounds write, src_size = %lu, dst_base = %lu, dst_buf_sz = %lu\n", src_size, dst_base, dst_buf_sz);
        return false;
    }

    bool result = ggml_backend_buffer_copy_tensor(src, dst);
    // output serialization format: | result (1 byte) |
    output.resize(1, 0);
    output[0] = result;
    SRV_DBG("src_id = %lu, dst_id = %lu, name = %s, type = %s\n", rpc_src->id, rpc_dst->id, rpc_src->name, ggml_type_name((ggml_type)rpc_dst->type));
    return true;
}

bool rpcserver::graph_compute(const std::vector<uint8_t> &input, std::vector<uint8_t> &output) {
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors
    // (n_tensors * sizeof(rpc_tensor)) |
    if (input.size() < sizeof(uint32_t)) {
        SRV_ERR("%s", "failed: input size too short\n");
        return false;
    }
    uint32_t n_nodes;
    memcpy(&n_nodes, input.data(), sizeof(n_nodes));
    if (input.size() < sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t)) {
        SRV_ERR("%s", "failed: input size too short\n");
        return false;
    }
    const auto *nodes = (const uint64_t *)(input.data() + sizeof(n_nodes));
    uint32_t n_tensors;
    memcpy(&n_tensors, input.data() + sizeof(n_nodes) + n_nodes * sizeof(uint64_t), sizeof(n_tensors));
    if (input.size() < sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor)) {
        SRV_ERR("%s", "failed: input size too short\n");
        return false;
    }
    const auto *tensors = (const rpc_tensor *)(input.data() + sizeof(n_nodes) + n_nodes * sizeof(uint64_t) + sizeof(n_tensors));

    size_t buf_size                = ggml_tensor_overhead() * (n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    struct ggml_init_params params = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context_ptr ctx_ptr{ggml_init(params)};
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context *ctx         = ctx_ptr.get();
    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes            = int(n_nodes);
    std::unordered_map<uint64_t, const rpc_tensor *> tensor_ptrs;
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs[tensors[i].id] = &tensors[i];
    }
    std::unordered_map<uint64_t, ggml_tensor *> tensor_map;
    for (uint32_t i = 0; i < n_nodes; i++) {
        uint64_t id;
        memcpy(&id, &nodes[i], sizeof(id));
        ggml_tensor *node = create_node(id, ctx, tensor_ptrs, tensor_map);
        if (node == nullptr) {
            SRV_ERR("%s", "failed: error creating node\n");
            return false;
        }
        graph->nodes[i] = node;
    }
    ggml_status status = ggml_backend_graph_compute(backend, graph);
    // output serialization format: | status (1 byte) |
    output.resize(1, 0);
    output[0] = status;
    SRV_DBG("status = %d\n", status);
    return true;
}

bool rpcserver::get_device_memory(std::vector<uint8_t> &output) {
    // output serialization format: | free (8 bytes) | total (8 bytes) |
    size_t free_mem  = get_free_memory();
    size_t total_mem = capacity;
    output.resize(2 * sizeof(uint64_t), 0);
    memcpy(output.data(), &free_mem, sizeof(free_mem));
    memcpy(output.data() + sizeof(uint64_t), &total_mem, sizeof(total_mem));
    SRV_DBG("free = %zu, total = %zu\n", free_mem, total_mem);
    return true;
}

bool rpcserver::init_tensor(const std::vector<uint8_t> &input) {
    // serialization format: | rpc_tensor |
    if (input.size() != sizeof(rpc_tensor)) {
        SRV_ERR("%s", "failed: input size invalid\n");
        return false;
    }
    const auto *in_tensor = (const rpc_tensor *)input.data();

    struct ggml_init_params params{
        /*.mem_size   =*/ggml_tensor_overhead(),
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };

    ggml_context_ptr ctx_ptr{ggml_init(params)};
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context *ctx   = ctx_ptr.get();
    ggml_tensor *tensor = deserialize_tensor(ctx, in_tensor);

    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } else {
        SRV_WRN("%s", "failed: null buffer for tensor passed to init_tensor function\n");
    }
    if (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        SRV_ERR("%s", "failed: tensor->extra populated by the backend, this is currently unsupported.\n");
        return false;
    }
    return true;
}

bool rpcserver::get_alloc_size(const std::vector<uint8_t> &input, std::vector<uint8_t> &output) {
    // serialization format: | rpc_tensor |
    if (input.size() != sizeof(rpc_tensor)) {
        SRV_ERR("%s", "failed: input size invalid\n");
        return false;
    }
    const auto *in_tensor = (const rpc_tensor *)input.data();
    ggml_backend_buffer_type_t buft;

    struct ggml_init_params params{
        /*.mem_size   =*/ggml_tensor_overhead(),
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };

    ggml_context_ptr ctx_ptr{ggml_init(params)};
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context *ctx   = ctx_ptr.get();
    ggml_tensor *tensor = deserialize_tensor(ctx, in_tensor);

    if (tensor->buffer == nullptr) {
        // No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backend);
    } else {
        buft = tensor->buffer->buft;
    }
    size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);
    // output serialization format: | alloc_size (8 bytes) |
    output.resize(sizeof(uint64_t), 0);
    memcpy(output.data(), &alloc_size, sizeof(alloc_size));
    SRV_DBG("alloc_size = %zu\n", alloc_size);
    return true;
}

bool rpcserver::say_hello(std::vector<uint8_t> &output) {
    // output serialization format: | major (8 bytes) | minor (8 bytes) | patch (8 bytes)
    output.resize(3, 0);
    output[0] = RPC_PROTO_MAJOR_VERSION;
    output[1] = RPC_PROTO_MINOR_VERSION;
    output[2] = RPC_PROTO_PATCH_VERSION;
    return true;
}

bool rpcserver::support_op(const std::vector<uint8_t> &input, std::vector<uint8_t> &output) {
    // serialization format: | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    if (input.size() < sizeof(uint32_t)) {
        SRV_ERR("%s", "invalid input size\n");
        return false;
    }
    uint32_t n_tensors;
    memcpy(&n_tensors, input.data(), sizeof(n_tensors));
    if (input.size() < sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor)) {
        SRV_ERR("%s", "invalid input size\n");
        return false;
    }
    const auto *tensors = (const rpc_tensor *)(input.data() + sizeof(uint32_t));

    size_t buf_size = ggml_tensor_overhead() * n_tensors;

    struct ggml_init_params params{
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };

    ggml_context_ptr ctx_ptr{ggml_init(params)};
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context *ctx   = ctx_ptr.get();
    ggml_tensor *tensor = deserialize_tensor(ctx, &tensors[n_tensors - 1]);
    for (uint32_t i = 0; i < n_tensors - 1; i++) {
        ggml_tensor *src = deserialize_tensor(ctx, &tensors[i]);
        tensor->src[i]   = src;
    }
    bool result = true;
    if (backend->device->iface.supports_op) {
        result = backend->device->iface.supports_op(backend->device, tensor);
    }

    // output serialization format: | result (8 bytes) |
    output.resize(1, 0);
    output[0] = result;
    SRV_DBG("result = %d\n", result);
    return true;
}

bool rpcserver::get_cached_file(uint64_t hash, std::vector<uint8_t> &data) {
    if (cache_dir == nullptr) {
        return false;
    }
    char hash_str[17];
    snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    if (!fs::exists(cache_file)) {
        return false;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(size);
    ifs.read((char *)data.data(), size);
    return true;
}

size_t rpcserver::get_free_memory() const {
    size_t free_mem, total_mem;
    rpcserver_get_backend_memory(backend, index, &free_mem, &total_mem);
    if (free_mem >= capacity) {
        free_mem = capacity;
    }
    SRV_DBG("allocatable_mib = %zu, capacity_mib = %zu\n", free_mem >> 20, total_mem >> 20);
    return free_mem;
}

ggml_tensor *rpcserver::deserialize_tensor(struct ggml_context *ctx, const rpc_tensor *tensor) {
    ggml_tensor *result = ggml_new_tensor_4d(ctx, (ggml_type)tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    if (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        if (common_log_verbosity_thold > 5) {
            SRV_WRN("failed: buffer not found, name = %s\n", tensor->name);
        }
        result->buffer = nullptr;
    }

    if (result->buffer) {
        // require that the tensor data does not go beyond the buffer end
        auto tensor_size  = (uint64_t)ggml_nbytes(result);
        auto buffer_start = (uint64_t)ggml_backend_buffer_get_base(result->buffer);
        auto buffer_size  = (uint64_t)ggml_backend_buffer_get_size(result->buffer);
        GGML_ASSERT(tensor->data + tensor_size >= tensor->data); // check for overflow
        GGML_ASSERT(tensor->data >= buffer_start && tensor->data + tensor_size <= buffer_start + buffer_size);
    }

    result->op = (ggml_op)tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data  = reinterpret_cast<void *>(tensor->data);
    ggml_set_name(result, tensor->name);
    return result;
}

ggml_tensor *rpcserver::create_node(uint64_t id, struct ggml_context *ctx, const std::unordered_map<uint64_t, const rpc_tensor *> &tensor_ptrs,
                                    std::unordered_map<uint64_t, struct ggml_tensor *> &tensor_map) {
    if (id == 0) {
        return nullptr;
    }
    if (tensor_map.find(id) != tensor_map.end()) {
        return tensor_map[id];
    }
    const rpc_tensor *tensor   = tensor_ptrs.at(id);
    struct ggml_tensor *result = deserialize_tensor(ctx, tensor);

    tensor_map[id] = result;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] == 0) {
            break;
        }
        result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
    }
    if (tensor->view_src != 0) {
        result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
    }
    result->view_offs = tensor->view_offs;

    SRV_DBG("id = %lu, name = %s, type = %s, op = %s\n", id, tensor->name, ggml_type_name(static_cast<ggml_type>(tensor->type)),
            ggml_op_name(static_cast<ggml_op>(tensor->op)));
    return result;
}

static void rpcserver_serve(ggml_backend_t bkd, const char *cached, int32_t idx, size_t cap, rpc_sockfd_t sockfd) {
    rpcserver server(bkd, cached, idx, cap);

    std::vector<uint8_t> input;
    std::vector<uint8_t> output;

    // wait for new command
    uint8_t cmd;
    if (!rpc_recv_data(sockfd, &cmd, 1)) {
        return;
    }
    if (cmd >= RPC_CMD_COUNT) {
        SRV_ERR("unknown command: %d\n", cmd);
        return;
    }
    // receive input size
    uint64_t input_size;
    if (!rpc_recv_data(sockfd, &input_size, sizeof(input_size))) {
        return;
    }
    try {
        input.resize(input_size);
    } catch (const std::bad_alloc &e) {
        SRV_ERR("cmd %d: failed to allocate input buffer: "
                "request_b = %lu\n",
                cmd, input_size);
        return;
    }

    // NB(thxCode): compatible with llama-box v0.0.134+.
    if (cmd == RPC_CMD_HELLO && input_size == 0) {
        server.say_hello(output);
        uint64_t output_size = output.size();
        if (!rpc_send_data(sockfd, &output_size, sizeof(output_size))) {
            SRV_ERR("cmd %d: failed to send output size, "
                    "b = %lu\n",
                    cmd, output_size);
            return;
        }
        if (!rpc_send_data(sockfd, output.data(), output_size)) {
            SRV_ERR("cmd %d: failed to send output data, "
                    "b = %lu\n",
                    cmd, output_size);
            return;
        }
        output.clear();

        // wait for new command
        if (!rpc_recv_data(sockfd, &cmd, 1)) {
            return;
        }
        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            SRV_ERR("unknown command: %d\n", cmd);
            return;
        }
        // receive input size
        if (!rpc_recv_data(sockfd, &input_size, sizeof(input_size))) {
            return;
        }
        try {
            input.resize(input_size);
        } catch (const std::bad_alloc &e) {
            SRV_ERR("cmd %d: failed to allocate input buffer: "
                    "request_b = %lu\n",
                    cmd, input_size);
            return;
        }
    }

    // process other command
    while (true) {
        // receive input
        if (!rpc_recv_data(sockfd, input.data(), input_size)) {
            SRV_ERR("cmd %d: failed to receive input data: "
                    "request_b = %lu\n",
                    cmd, input_size);
            return;
        }

        bool ok = false;
        try {
            switch (cmd) {
                case RPC_CMD_ALLOC_BUFFER: {
                    ok = server.alloc_buffer(input, output);
                    break;
                }
                case RPC_CMD_GET_ALIGNMENT: {
                    ok = server.get_alignment(output);
                    break;
                }
                case RPC_CMD_GET_MAX_SIZE: {
                    ok = server.get_max_size(output);
                    break;
                }
                case RPC_CMD_BUFFER_GET_BASE: {
                    ok = server.buffer_get_base(input, output);
                    break;
                }
                case RPC_CMD_FREE_BUFFER: {
                    ok = server.free_buffer(input);
                    break;
                }
                case RPC_CMD_BUFFER_CLEAR: {
                    ok = server.buffer_clear(input);
                    break;
                }
                case RPC_CMD_SET_TENSOR: {
                    ok = server.set_tensor(input);
                    break;
                }
                case RPC_CMD_SET_TENSOR_HASH: {
                    ok = server.set_tensor_hash(input, output);
                    break;
                }
                case RPC_CMD_GET_TENSOR: {
                    ok = server.get_tensor(input, output);
                    break;
                }
                case RPC_CMD_COPY_TENSOR: {
                    ok = server.copy_tensor(input, output);
                    break;
                }
                case RPC_CMD_GRAPH_COMPUTE: {
                    ok = server.graph_compute(input, output);
                    break;
                }
                case RPC_CMD_GET_DEVICE_MEMORY: {
                    ok = server.get_device_memory(output);
                    break;
                }
                case RPC_CMD_INIT_TENSOR: {
                    ok = server.init_tensor(input);
                    break;
                }
                case RPC_CMD_GET_ALLOC_SIZE: {
                    ok = server.get_alloc_size(input, output);
                    break;
                }
                case RPC_CMD_HELLO: {
                    // NB(thxCode): compatible with llama-box v0.0.116 - v0.0.133.
                    output.resize(sizeof(uint8_t), 0);
                    output[0] = true;
                    ok        = true;
                    break;
                }
                case RPC_CMD_SUPPORT_OP: {
                    ok = server.support_op(input, output);
                    break;
                }
                default: {
                    SRV_ERR("unknown command: %d\n", cmd);
                    break;
                }
            }
        } catch (const std::exception &e) {
            SRV_ERR("cmd %d: exception: %s\n", cmd, e.what());
        }
        if (!ok) {
            break;
        }

        // send output
        uint64_t output_size = output.size();
        if (!rpc_send_data(sockfd, &output_size, sizeof(output_size))) {
            SRV_ERR("cmd %d: failed to send output size, "
                    "b = %lu\n",
                    cmd, output_size);
            break;
        }
        if (!rpc_send_data(sockfd, output.data(), output_size)) {
            SRV_ERR("cmd %d: failed to send output data, "
                    "b = %lu\n",
                    cmd, output_size);
            break;
        }
        output.clear();

        // wait for new command
        if (!rpc_recv_data(sockfd, &cmd, 1)) {
            break;
        }
        if (cmd >= RPC_CMD_COUNT) {
            SRV_ERR("unknown command: %d\n", cmd);
            break;
        }
        // receive input size
        if (!rpc_recv_data(sockfd, &input_size, sizeof(input_size))) {
            SRV_ERR("cmd %d: failed to receive input size\n", cmd);
            break;
        }
        try {
            input.resize(input_size);
        } catch (const std::bad_alloc &e) {
            SRV_ERR("cmd %d: failed to allocate input buffer: "
                    "request_b = %lu\n",
                    cmd, input_size);
            break;
        }
    }

    output.clear();
    input.clear();
}

// RPC server entry point

static std::shared_ptr<rpc_socket_t> rpcserver_socket_create(const char *host, int port) {
    if (inet_addr(host) == INADDR_NONE) {
        SRV_ERR("failed to create server socket, host = %s\n", host);
        return nullptr;
    }

    int sockfd                               = socket(AF_INET, SOCK_STREAM, 0);
    std::shared_ptr<rpc_socket_t> srv_socket = rpc_socket_create(sockfd);
    if (srv_socket == nullptr) {
        SRV_ERR("%s", "failed to create server socket\n");
        return nullptr;
    }

    int reuse = 1;
    int ret   = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (char *)&reuse, sizeof(int));
    if (ret != 0) {
        SRV_ERR("failed to set server socket SO_REUSEADDR, errno = %d\n", ret);
        return nullptr;
    }

    struct sockaddr_in serv_addr{};

    serv_addr.sin_family      = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port        = htons(port);
    if (bind(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        SRV_ERR("failed to bind server socket, host = %s, port = %d\n", host, port);
        return nullptr;
    }
    if (listen(sockfd, 1) < 0) {
        SRV_ERR("%s", "failed to listen on server socket\n");
        return nullptr;
    }

    return srv_socket;
}

struct rpcserver_params {
    std::string hostname  = "0.0.0.0";
    int port              = 0;
    int32_t main_gpu      = 0;
    size_t reserve_memory = 0;
    bool use_cache        = false;
    std::string cache_dir;
};

static int rpcserver_start(rpcserver_params &params) {
#if defined(GGML_USE_METAL)
    // NB(thxCode): disable residency set for Metal backend to avoid memory leak.
    setenv("GGML_METAL_NO_RESIDENCY", "1", 1);
#endif
    ggml_backend_t backend;
    if (params.main_gpu < 0) {
        SRV_INF("%s", "using CPU backend\n");
        backend = ggml_backend_cpu_init();
    } else {
        backend = rpcserver_create_backend(params.main_gpu);
    }
    if (!backend) {
        SRV_ERR("%s", "failed to create backend\n");
        return 1;
    }
    const char *cache_dir = nullptr;
    if (params.use_cache) {
        if (params.cache_dir.empty()) {
            params.cache_dir = fs_get_cache_directory() + +"rpc/";
        }
        cache_dir = params.cache_dir.c_str();
        if (!fs_create_directory_with_parents(params.cache_dir)) {
            SRV_ERR("failed to create cache directory: %s\n", cache_dir);
            return 1;
        }
        SRV_INF("using cache directory: %s\n", cache_dir);
    }

    int32_t main_gpu = params.main_gpu;

    size_t free_mem, total_mem;
    rpcserver_get_backend_memory(backend, main_gpu, &free_mem, &total_mem);
    if (total_mem < params.reserve_memory) {
        SRV_ERR("not enough memory, "
                "free_mib = %zu, total_mib = %zu, reserve_mib = %zu\n",
                free_mem >> 20, total_mem >> 20, params.reserve_memory >> 20);
        return 1;
    }
    total_mem -= params.reserve_memory;

    SRV_INF("proto v%d.%d.%d, listening "
            "hostname = %s, port = %d, allocatable_mib = %zu, capacity_mib = %zu\n",
            RPC_PROTO_MAJOR_VERSION, RPC_PROTO_MINOR_VERSION, RPC_PROTO_PATCH_VERSION,
            params.hostname.c_str(), params.port, free_mem >> 20, total_mem >> 20);
#ifdef _WIN32
    {
        WSADATA wsaData;
        int ret = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (ret != 0) {
            SRV_ERR("WSAStartup failed, errno = %d\n", ret);
            return 1;
        }
    }
#endif
    std::shared_ptr<rpc_socket_t> server_socket = rpcserver_socket_create(params.hostname.c_str(), params.port);
    if (server_socket == nullptr) {
        SRV_ERR("%s", "failed to create server socket\n");
        return 1;
    }
    while (true) {
        struct sockaddr_in cli_addr{};

        socklen_t cli_addr_len                   = sizeof(cli_addr);
        int cli_socketfd                         = accept(server_socket->fd, (struct sockaddr *)&cli_addr, &cli_addr_len);
        std::shared_ptr<rpc_socket_t> cli_socket = rpc_socket_create(cli_socketfd);
        if (cli_socket == nullptr) {
            continue;
        }

        int nodelay = 1;
        int ret     = setsockopt(cli_socketfd, IPPROTO_TCP, TCP_NODELAY, (char *)&nodelay, sizeof(int));
        if (ret != 0) {
            SRV_WRN("failed to set client socket TCP_NODELAY, errno = %d\n", ret);
        }

        char cli_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &cli_addr.sin_addr, cli_ip, sizeof(cli_ip));
        unsigned short cli_port = ntohs(cli_addr.sin_port);

        try {
            std::thread([backend, cache_dir, main_gpu, total_mem, cli_socket, cli_ip, cli_port]() {
                LOG_INF("cli %25s: %s:%d\n", "accepted", cli_ip, cli_port);
                rpcserver_serve(backend, cache_dir, main_gpu, total_mem, cli_socket->fd);
                LOG_INF("cli %25s: %s:%d\n", "closed", cli_ip, cli_port);
            }).detach();
        } catch (const std::exception &e) {
            SRV_ERR("failed to process %s %d, error = %s\n", cli_ip, cli_port, e.what());
        }
    }
#ifdef _WIN32
    WSACleanup();
#endif

    SRV_INF("%s", "stopped rpc server\n");
    ggml_backend_free(backend);
    return 0;
}