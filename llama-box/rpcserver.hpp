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

#include "llama.cpp/common/log.h"
#include "llama.cpp/ggml/include/ggml-alloc.h"
#include "llama.cpp/ggml/include/ggml-backend.h"
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
#ifdef GGML_USE_SYCL
#include "llama.cpp/ggml/include/ggml-sycl.h"
#endif

#include "utils.hpp"

#define UNUSED GGML_UNUSED

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
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_SIZE,
    RPC_CMD_COUNT,
};

// RPC helper

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

static bool rpc_send_data(rpc_sockfd_t sockfd, const void *data, size_t size) {
    size_t bytes_sent = 0;
    while (bytes_sent < size) {
        ssize_t n = send(sockfd, (const char *)data + bytes_sent, size - bytes_sent, 0);
        if (n < 0) {
            return false;
        }
        bytes_sent += n;
    }
    return true;
}

static bool rpc_recv_data(rpc_sockfd_t sockfd, void *data, size_t size) {
    size_t bytes_recv = 0;
    while (bytes_recv < size) {
        ssize_t n = recv(sockfd, (char *)data + bytes_recv, size - bytes_recv, 0);
        if (n <= 0) {
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
    rpcserver(ggml_backend_t backend, int32_t index = 0, size_t capacity = 0)
        : backend(backend), index(index), capacity(capacity) {
    }

    ~rpcserver();

    bool alloc_buffer(const std::vector<uint8_t> &input, std::vector<uint8_t> &output);
    bool get_alignment(std::vector<uint8_t> &output);
    bool get_max_size(std::vector<uint8_t> &output);
    bool buffer_get_base(const std::vector<uint8_t> &input, std::vector<uint8_t> &output);
    bool free_buffer(const std::vector<uint8_t> &input);
    bool buffer_clear(const std::vector<uint8_t> &input);
    bool set_tensor(const std::vector<uint8_t> &input);
    bool get_tensor(const std::vector<uint8_t> &input, std::vector<uint8_t> &output);
    bool copy_tensor(const std::vector<uint8_t> &input, std::vector<uint8_t> &output);
    bool graph_compute(const std::vector<uint8_t> &input, std::vector<uint8_t> &output);
    bool get_device_memory(std::vector<uint8_t> &output);
    bool init_tensor(const std::vector<uint8_t> &input);
    bool get_alloc_size(const std::vector<uint8_t> &input, std::vector<uint8_t> &output);

  private:
    size_t get_free_memory() const;
    ggml_tensor *deserialize_tensor(struct ggml_context *ctx, const rpc_tensor *tensor);
    ggml_tensor *create_node(uint64_t id, struct ggml_context *ctx, const std::unordered_map<uint64_t, const rpc_tensor *> &tensor_ptrs, std::unordered_map<uint64_t, struct ggml_tensor *> &tensor_map);

    ggml_backend_t backend;
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
        SRV_ERR("failed: not found remote_ptr = %lu\n", remote_ptr);
        return false;
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
        SRV_ERR("failed: not found remote_ptr = %lu\n", remote_ptr);
        return false;
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
    struct ggml_context *ctx = ggml_init(params);
    ggml_tensor *tensor      = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr) {
        SRV_ERR("%s", "failed: error deserializing tensor\n");
        ggml_free(ctx);
        return false;
    }

    // sanitize tensor->data
    {
        const auto p0   = (size_t)ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            SRV_ERR("%s", "failed: tensor->data out of bounds\n");
            delete tensor;
            ggml_free(ctx);
            return false;
        }
    }

    const void *data = input.data() + sizeof(rpc_tensor) + sizeof(offset);
    ggml_backend_tensor_set(tensor, data, offset, size);
    ggml_free(ctx);
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
    struct ggml_context *ctx = ggml_init(params);
    ggml_tensor *tensor      = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr) {
        SRV_ERR("%s", "failed: error deserializing tensor\n");
        ggml_free(ctx);
        return false;
    }

    // sanitize tensor->data
    {
        const auto p0   = (size_t)ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            SRV_ERR("%s", "failed: tensor->data out of bounds\n");
            delete tensor;
            ggml_free(ctx);
            return false;
        }
    }

    // output serialization format: | data (size bytes) |
    output.resize(size, 0);
    ggml_backend_tensor_get(tensor, output.data(), offset, size);
    ggml_free(ctx);
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
    struct ggml_context *ctx = ggml_init(params);
    ggml_tensor *src         = deserialize_tensor(ctx, rpc_src);
    ggml_tensor *dst         = deserialize_tensor(ctx, rpc_dst);
    if (src == nullptr || dst == nullptr) {
        SRV_ERR("%s", "failed: error deserializing tensor\n");
        ggml_free(ctx);
        return false;
    }

    bool result = ggml_backend_buffer_copy_tensor(src, dst);
    // output serialization format: | result (1 byte) |
    output.resize(1, 0);
    output[0] = result;
    ggml_free(ctx);
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

    static size_t buf_size         = ggml_tensor_overhead() * (n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    struct ggml_init_params params = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    struct ggml_context *ctx  = ggml_init(params);
    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes            = int(n_nodes);
    std::unordered_map<uint64_t, const rpc_tensor *> tensor_ptrs;
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs[tensors[i].id] = &tensors[i];
    }
    std::unordered_map<uint64_t, ggml_tensor *> tensor_map;
    for (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], sizeof(id));
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);
    }
    ggml_status status = ggml_backend_graph_compute(backend, graph);
    // output serialization format: | status (1 byte) |
    output.resize(1, 0);
    output[0] = status;
    ggml_free(ctx);
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
    struct ggml_context *ctx = ggml_init(params);
    ggml_tensor *tensor      = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr) {
        SRV_ERR("%s", "failed: null tensor pointer passed to server init_tensor function\n");
        ggml_free(ctx);
        return false;
    }
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } else {
        SRV_ERR("%s", "failed: null buffer for tensor passed to init_tensor function\n");
        ggml_free(ctx);
        return false;
    }
    if (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        SRV_ERR("%s", "failed: tensor->extra populated by the backend, this is currently unsupported.\n");
        ggml_free(ctx);
        return false;
    }
    ggml_free(ctx);
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
    struct ggml_context *ctx = ggml_init(params);
    ggml_tensor *tensor      = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr) {
        SRV_ERR("%s", "failed: null tensor pointer passed to server get_alloc_size function\n");
        ggml_free(ctx);
        return false;
    }
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
    ggml_free(ctx);
    SRV_DBG("alloc_size = %zu\n", alloc_size);
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
        SRV_ERR("%s", "failed: buffer not found\n");
        return nullptr;
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
    if (result == nullptr) {
        SRV_ERR("failed: error deserializing tensor, id = %lu\n", id);
        return nullptr;
    }
    tensor_map[id] = result;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
    }
    result->view_src  = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
    result->view_offs = tensor->view_offs;

    SRV_DBG("id = %lu, name = %s, type = %s, op = %s\n", id, tensor->name, ggml_type_name(static_cast<ggml_type>(tensor->type)),
            ggml_op_name(static_cast<ggml_op>(tensor->op)));
    return result;
}

static void rpcserver_serve(ggml_backend_t bkd, int32_t idx, size_t cap, rpc_sockfd_t sockfd) {
    rpcserver server(bkd, idx, cap);
    while (true) {
        uint8_t cmd;
        if (!rpc_recv_data(sockfd, &cmd, 1)) {
            break;
        }
        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            SRV_ERR("unknown command: %d\n", cmd);
            break;
        }
        std::vector<uint8_t> input;
        std::vector<uint8_t> output;
        uint64_t input_size;
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
        if (!rpc_recv_data(sockfd, input.data(), input_size)) {
            SRV_ERR("cmd %d: failed to receive input data: "
                    "request_b = %lu\n",
                    cmd, input_size);
            break;
        }
        bool ok = false;
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
            default: {
                SRV_ERR("unknown command: %d\n", cmd);
                break;
            }
        }
        if (!ok) {
            break;
        }
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
        input.clear();
    }
}

// RPC server entry point

static std::shared_ptr<rpc_socket_t> rpcserver_socket_create(const char *host, int port) {
    int sockfd                               = socket(AF_INET, SOCK_STREAM, 0);
    std::shared_ptr<rpc_socket_t> srv_socket = rpc_socket_create(sockfd);
    if (srv_socket == nullptr) {
        SRV_ERR("%s", "failed to create server socket\n");
        return nullptr;
    }

    int flag = 1;
    int ret  = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (char *)&flag, sizeof(int));
    if (ret != 0) {
        SRV_ERR("failed to set server socket SO_REUSEADDR, errno = %d\n", ret);
        return nullptr;
    }
    if (inet_addr(host) == INADDR_NONE) {
        SRV_ERR("failed to create server socket, host = %s\n", host);
        return nullptr;
    }

    struct sockaddr_in serv_addr{};

    serv_addr.sin_family      = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port        = htons(port);
    if (bind(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
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
};

static int rpcserver_start(rpcserver_params &params) {
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

    size_t free_mem, total_mem;
    rpcserver_get_backend_memory(backend, params.main_gpu, &free_mem, &total_mem);
    if (total_mem < params.reserve_memory) {
        SRV_ERR("not enough memory, "
                "free_mib = %zu, total_mib = %zu, reserve_mib = %zu\n",
                free_mem >> 20, total_mem >> 20, params.reserve_memory >> 20);
        return 1;
    }
    total_mem -= params.reserve_memory;

    SRV_INF("listening "
            "hostname = %s, port = %d, allocatable_mib = %zu, capacity_mib = %zu\n",
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
            SRV_ERR("%s", "failed to accept client connection\n");
            continue;
        }

        int flag = 1;
        int ret  = setsockopt(cli_socketfd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));
        if (ret != 0) {
            SRV_ERR("failed to set client socket TCP_NODELAY, errno = %d\n", ret);
            continue;
        }

        char cli_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &cli_addr.sin_addr, cli_ip, sizeof(cli_ip));
        unsigned short cli_port = ntohs(cli_addr.sin_port);

        try {
            int32_t main_gpu = params.main_gpu;
            std::thread([backend, main_gpu, total_mem, cli_socket, cli_ip, cli_port]() {
                LOG_INF("cli %25s: %s:%d\n", "accepted", cli_ip, cli_port);
                rpcserver_serve(backend, main_gpu, total_mem, cli_socket->fd);
                LOG_INF("cli %25s: %s:%d\n", "closed", cli_ip, cli_port);
            }).detach();
        } catch (const std::system_error &e) {
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