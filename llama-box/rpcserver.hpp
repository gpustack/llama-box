// heads

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <winsock2.h>
#    include <ws2tcpip.h>
#else
#    include <arpa/inet.h>
#    include <netdb.h>
#    include <netinet/in.h>
#    include <netinet/tcp.h>
#    include <sys/socket.h>
#    include <sys/types.h>
#    include <unistd.h>
#endif

#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH 10485760
#define CPPHTTPLIB_TCP_NODELAY                         true
#include "llama.cpp/common/log.h"
#include "llama.cpp/ggml/include/ggml-alloc.h"
#include "llama.cpp/ggml/include/ggml-backend.h"
#include "llama.cpp/ggml/include/ggml-cpp.h"
#include "llama.cpp/ggml/include/ggml-rpc.h"
#include "llama.cpp/ggml/src/ggml-backend-impl.h"
#include "llama.cpp/ggml/src/ggml-impl.h"
#include "llama.cpp/vendor/cpp-httplib/httplib.h"

#define SELF_PACKAGE 0
#include "z_utils.hpp"

// defines

namespace fs = std::filesystem;

// types

#if defined(_WIN32)
typedef SOCKET rpc_sockfd_t;
using ssize_t = __int64;
#else
typedef int32_t rpc_sockfd_t;
#endif

// cross-platform socket
struct rpc_socket_t {
    rpc_sockfd_t fd;

    explicit rpc_socket_t(rpc_sockfd_t fd) : fd(fd) {}

    ~rpc_socket_t() {
#if defined(_WIN32)
        closesocket(this->fd);
#else
        close(this->fd);
#endif
    }
};

// all RPC structures must be packed
#pragma pack(push, 1)

// ggml_tensor is serialized into rpc_tensor
struct rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char     name[GGML_MAX_NAME];

    char padding[4];
};

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

struct rpc_msg_get_alloc_size_req {
    rpc_tensor tensor;
};

struct rpc_msg_get_alloc_size_rsp {
    uint64_t alloc_size;
};

struct rpc_msg_init_tensor_req {
    rpc_tensor tensor;
};

struct rpc_msg_alloc_buffer_req {
    uint64_t size;
};

struct rpc_msg_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

struct rpc_msg_get_alignment_rsp {
    uint64_t alignment;
};

struct rpc_msg_get_max_size_rsp {
    uint64_t max_size;
};

struct rpc_msg_buffer_get_base_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_get_base_rsp {
    uint64_t base_ptr;
};

struct rpc_msg_free_buffer_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t  value;
};

struct rpc_msg_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t   offset;
    uint64_t   hash;
};

struct rpc_msg_set_tensor_hash_rsp {
    uint8_t result;
};

struct rpc_msg_get_tensor_req {
    rpc_tensor tensor;
    uint64_t   offset;
    uint64_t   size;
};

struct rpc_msg_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

struct rpc_msg_copy_tensor_rsp {
    uint8_t result;
};

struct rpc_msg_graph_compute_rsp {
    uint8_t result;
};

struct rpc_msg_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

struct rpc_msg_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    bool    enabled_cache;
};

struct rpc_msg_support_op_rsp {
    uint8_t result;
};

#pragma pack(pop)

struct rpcserver_params {
    std::string hostname       = "0.0.0.0";
    int         port           = 0;
    int32_t     main_gpu       = 0;
    size_t      reserve_memory = 0;
    int         n_threads      = -1;
    bool        use_cache      = false;
    std::string cache_dir;

    // inherited from common_params
    ggml_numa_strategy numa = GGML_NUMA_STRATEGY_DISABLED;
};

// implementations

const static size_t MAX_CHUNK = 1 << 23;  // 8MiB

static bool rpc_recv_data(rpc_sockfd_t sockfd, void * data, size_t size) {
    if (size != 0) {
        SRV_DBG("receiving data: data range = [%p, %p), bytes_target = %zu\n", data, (char *) data + size, size);
    }
    size_t bytes_recv = 0;
    while (bytes_recv < size) {
        size_t  bytes_chunk = MIN(size - bytes_recv, MAX_CHUNK);
        ssize_t n           = recv(sockfd, (char *) data + bytes_recv, bytes_chunk, 0);
        if (n <= 0) {
            int err = errno;
            if (err == EINTR || err == EAGAIN || err == EWOULDBLOCK) {
                SRV_WRN(
                    "interrupted: data range = [%p, %p), bytes_recv = %zu, bytes_target = %zu, errno = %d, errmsg = "
                    "%s, retrying...\n",
                    data, (char *) data + size, bytes_recv, size, err, strerror(errno));
                continue;  // try again
            }
            if (err != 0 && err != ESRCH) {
                SRV_ERR(
                    "failed to recv data: data range = [%p, %p), bytes_recv = %zu, bytes_target = %zu, errno = %d, "
                    "errmsg = %s\n",
                    data, (char *) data + size, bytes_recv, size, err, strerror(errno));
                int       serr     = 0;
                socklen_t serr_len = sizeof(serr);
                int       ret      = getsockopt(sockfd, SOL_SOCKET, SO_ERROR, (char *) &serr, &serr_len);
                if (ret < 0) {
                    err = errno;
                    SRV_ERR("failed to get peer socket error: errno = %d, errmsg = %s\n", err, strerror(err));
                } else if (serr != 0) {
                    SRV_ERR("peer socket error: errno = %d, errmsg = %s\n", serr, strerror(serr));
                } else {
                    struct sockaddr_in sin{};
                    socklen_t          addr_len = sizeof(sin);
                    ret                         = getpeername(sockfd, (struct sockaddr *) &sin, &addr_len);
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

static bool rpc_send_data(rpc_sockfd_t sockfd, const void * data, size_t size) {
    if (size != 0) {
        SRV_DBG("sending data: data range = [%p, %p), bytes_target = %zu\n", data, (char *) data + size, size);
    }
    size_t bytes_sent = 0;
    while (bytes_sent < size) {
        size_t  bytes_chunk = MIN(size - bytes_sent, MAX_CHUNK);
        ssize_t n           = send(sockfd, (const char *) data + bytes_sent, bytes_chunk, 0);
        if (n < 0) {
            int err = errno;
            if (err == EINTR || err == EAGAIN || err == EWOULDBLOCK) {
                SRV_WRN(
                    "interrupted: data range = [%p, %p), bytes_sent = %zu, bytes_target = %zu, errno = %d, errmsg = "
                    "%s, retrying...\n",
                    data, (char *) data + size, bytes_sent, size, err, strerror(err));
                continue;  // try again
            }
            if (err != 0) {
                SRV_ERR(
                    "failed to send data: data range = [%p, %p), bytes_sent = %zu, bytes_target = %zu, errno = %d, "
                    "errmsg = %s\n",
                    data, (char *) data + size, bytes_sent, size, err, strerror(err));
                int       serr     = 0;
                socklen_t serr_len = sizeof(serr);
                int       ret      = getsockopt(sockfd, SOL_SOCKET, SO_ERROR, (char *) &serr, &serr_len);
                if (ret < 0) {
                    err = errno;
                    SRV_ERR("failed to get peer socket error: errno = %d, errmsg = %s\n", err, strerror(err));
                } else if (serr != 0) {
                    SRV_ERR("peer socket error: errno = %d, errmsg = %s\n", serr, strerror(serr));
                } else {
                    struct sockaddr_in sin{};
                    socklen_t          addr_len = sizeof(sin);
                    ret                         = getpeername(sockfd, (struct sockaddr *) &sin, &addr_len);
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

static bool rpc_recv_msg(rpc_sockfd_t sockfd, void * msg, size_t msg_size) {
    uint64_t size;
    if (!rpc_recv_data(sockfd, &size, sizeof(size))) {
        SRV_ERR("%s", "failed to recv msg size\n");
        return false;
    }
    if (size != msg_size) {
        SRV_ERR("failed: msg size mismatch, expected %zu, got %llu\n", msg_size, size);
        return false;
    }
    bool ret = rpc_recv_data(sockfd, msg, msg_size);
    if (!ret) {
        SRV_ERR("%s", "failed to recv msg data\n");
    }
    return ret;
}

static bool rpc_recv_msg(rpc_sockfd_t sockfd, std::vector<uint8_t> & input) {
    uint64_t size;
    if (!rpc_recv_data(sockfd, &size, sizeof(size))) {
        SRV_ERR("%s", "failed to recv msg size\n");
        return false;
    }
    try {
        input.resize(size);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate input buffer of size %llu\n", size);
        return false;
    }
    bool ret = rpc_recv_data(sockfd, input.data(), size);
    if (!ret) {
        SRV_ERR("%s", "failed to recv msg data\n");
    }
    return ret;
}

static bool rpc_send_msg(rpc_sockfd_t sockfd, const void * msg, size_t msg_size) {
    if (!rpc_send_data(sockfd, &msg_size, sizeof(msg_size))) {
        SRV_ERR("%s", "failed to send msg size\n");
        return false;
    }
    bool ret = rpc_send_data(sockfd, msg, msg_size);
    if (!ret) {
        SRV_ERR("%s", "failed to send msg data\n");
    }
    return ret;
}

static ggml_backend_t rpcserver_create_backend(rpcserver_params & params) {
    ggml_backend_t backend = nullptr;
    int32_t        gpu     = params.main_gpu;

    if (gpu < 0) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        SRV_INF("%s", "using CPU backend\n");
    } else {
        while(true) {
            ggml_backend_device * dev = ggml_backend_dev_get(gpu);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                backend = ggml_backend_dev_init(dev, nullptr);
                SRV_INF("using GPU backend: %s\n", ggml_backend_dev_name(dev));
                break;
            }
            gpu++;
        }
        if (!backend) {
            SRV_INF("%s", "failed to create GPU backend, fallback using CPU backend\n");
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        }
    }

    // set the number of threads
    if (backend) {
        auto * dev = ggml_backend_get_device(backend);
        auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto ggml_backend_set_n_threads_fn =
                (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, params.n_threads);
            }
        }
    }

    return backend;
}

static std::unique_ptr<rpc_socket_t> rpcserver_socket_create(const char * host, int port) {
    if (inet_addr(host) == INADDR_NONE) {
        SRV_ERR("failed to create server socket, host = %s\n", host);
        return nullptr;
    }

    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == INVALID_SOCKET) {
        SRV_ERR("%s", "failed to create server socket\n");
        return nullptr;
    }

    int reuse = 1;
    int ret   = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (char *) &reuse, sizeof(int));
    if (ret != 0) {
        SRV_ERR("failed to set server socket SO_REUSEADDR, errno = %d, errmsg = %s\n", ret, strerror(ret));
        return nullptr;
    }

    struct sockaddr_in serv_addr{};

    serv_addr.sin_family      = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port        = htons(port);
    if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        SRV_ERR("failed to bind server socket, host = %s, port = %d\n", host, port);
        return nullptr;
    }

    if (listen(sockfd, 1) < 0) {
        SRV_ERR("%s", "failed to listen on server socket\n");
        return nullptr;
    }

    return std::make_unique<rpc_socket_t>(sockfd);
}

static void rpcserver_get_backend_memory(ggml_backend_t backend, int32_t gpu, size_t * free_mem, size_t * total_mem) {
    if (gpu < 0) {
#if defined(_WIN32)
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
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        SRV_ERR("failed to get backend device for GPU %d\n", gpu);
        *free_mem  = 0;
        *total_mem = 0;
        return;
    }
    ggml_backend_dev_memory(dev, free_mem, total_mem);
}

std::function<void(int)> rpcserver_shutdown_handler;
std::atomic_flag         rpcserver_is_terminating = ATOMIC_FLAG_INIT;

inline void rpcserver_signal_handler(int32_t signal) {
    if (rpcserver_is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C
        // twice this is for better developer experience, we can remove when the
        // server is stable enough
        SRV_WRN("%s", "received second interrupt, terminating immediately\n");
        exit(1);
    }
    rpcserver_shutdown_handler(signal);
}

struct rpcserver {
    explicit rpcserver(rpcserver_params & params) : params(params) { llama_numa_init(params.numa); }

    rpcserver(const rpcserver & src) :
        params(src.params),
        backend(src.backend),
        cache_dir(src.cache_dir),
        index(src.index),
        capacity(src.capacity) {
        // copy constructor
    }

    ~rpcserver() {
        for (ggml_backend_buffer * buffer : buffers) {
            ggml_backend_buffer_free(buffer);
        }
        buffers.clear();
    }

    bool load() {
        if (params.n_threads <= 0) {
            params.n_threads = cpu_get_num_math();
        }

        backend = rpcserver_create_backend(params);
        if (!backend) {
            SRV_ERR("%s", "failed to create backend\n");
            return false;
        }

        if (params.use_cache) {
            if (params.cache_dir.empty()) {
                params.cache_dir = fs_get_cache_directory() + +"rpc/";
            }
            cache_dir = params.cache_dir.c_str();
            if (!fs_create_directory_with_parents(params.cache_dir)) {
                SRV_ERR("failed to create cache directory: %s\n", cache_dir);
                return false;
            }
            SRV_INF("using cache directory: %s\n", cache_dir);
        }

        size_t free_mem, total_mem;
        rpcserver_get_backend_memory(backend, params.main_gpu, &free_mem, &total_mem);
        if (total_mem < params.reserve_memory) {
            SRV_ERR(
                "no enough memory, "
                "free_mib = %zu, total_mib = %zu, reserve_mib = %zu\n",
                free_mem >> 20, total_mem >> 20, params.reserve_memory >> 20);
            return false;
        }
        capacity = total_mem - params.reserve_memory;

        return true;
    }

    int32_t start() {
        SRV_INF("%s", "starting\n");

        std::shared_ptr<httplib::ThreadPool> thread_pool =
            std::make_shared<httplib::ThreadPool>(cpu_get_num_physical_cores(), 1024);
        std::unique_ptr<rpc_socket_t> svr_socket             = nullptr;
        bool                          svr_socket_established = false;
#ifdef _WIN32
#    define SHUTDOWN_SOCKET                    \
        if (svr_socket_established) {          \
            shutdown(svr_socket->fd, SD_BOTH); \
            closesocket(svr_socket->fd);       \
            svr_socket_established = false;    \
        }
#else
#    define SHUTDOWN_SOCKET                      \
        if (svr_socket_established) {            \
            shutdown(svr_socket->fd, SHUT_RDWR); \
            close(svr_socket->fd);               \
            svr_socket_established = false;      \
        }
#endif

        // register shutdown handler
        rpcserver_shutdown_handler = [&](int) {
            SRV_FUNC_INF("start", "%s", "server is stopping\n");
            thread_pool->shutdown();
            SHUTDOWN_SOCKET;
        };
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
        struct sigaction sigint_action{};

        sigint_action.sa_handler = rpcserver_signal_handler;
        sigemptyset(&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, nullptr);
        sigaction(SIGTERM, &sigint_action, nullptr);
#elif defined(_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (rpcserver_signal_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

        // listening on port
#if defined(_WIN32)
        {
            WSADATA wsaData;
            int     ret = WSAStartup(MAKEWORD(2, 2), &wsaData);
            if (ret != 0) {
                SRV_ERR("WSAStartup failed, errno = %d\n", ret);
                return 1;
            }
        }
#endif
        svr_socket = rpcserver_socket_create(params.hostname.c_str(), params.port);
        if (svr_socket == nullptr) {
            SRV_ERR("%s", "failed to create server socket\n");
            return 1;
        }
        svr_socket_established = true;
        SRV_INF(
            "proto v%d.%d.%d, "
            "listening host = %s, port = %d, capacity_mib = %zu\n",
            RPC_PROTO_MAJOR_VERSION, RPC_PROTO_MINOR_VERSION, RPC_PROTO_PATCH_VERSION, params.hostname.c_str(),
            params.port, capacity >> 20);

        while (true) {
            struct sockaddr_in cli_addr{};

            socklen_t cli_addr_len = sizeof(cli_addr);
            int       sockfd       = accept(svr_socket->fd, (struct sockaddr *) &cli_addr, &cli_addr_len);
            if (sockfd == INVALID_SOCKET) {
                if (errno == EMFILE) {
                    std::this_thread::sleep_for(std::chrono::microseconds{ 1 });
                    continue;
                } else if (errno == EINTR || errno == EAGAIN) {
                    continue;
                }
                break;
            }

            char cli_ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &cli_addr.sin_addr, cli_ip, sizeof(cli_ip));
            unsigned short cli_port = ntohs(cli_addr.sin_port);

            int nodelay = 1;
            int ret     = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *) &nodelay, sizeof(int));
            if (ret != 0) {
                SRV_FUNC_WRN("loop", "failed to set client socket TCP_NODELAY, errno = %d, errmsg = %s\n", ret,
                             strerror(ret));
            }

            thread_pool->enqueue([this, sockfd, cli_ip, cli_port]() {
                SRV_FUNC_INF("loop", "accepted %s:%d\n", cli_ip, cli_port);
                rpcserver conn(*this);
                conn.process(sockfd);
                SRV_FUNC_INF("loop", "closed %s:%d\n", cli_ip, cli_port);
            });
        }

#if defined(_WIN32)
        WSACleanup();
#endif

        SHUTDOWN_SOCKET;
        return 1;
    }

  private:
    //
    // Attributes
    //

    rpcserver_params                          params;
    ggml_backend_t                            backend;
    const char *                              cache_dir = nullptr;
    int32_t                                   index     = 0;
    size_t                                    capacity  = 0;
    std::unordered_set<ggml_backend_buffer_t> buffers   = {};

    bool get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
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
        ifs.read((char *) data.data(), size);
        return true;
    }

    inline size_t get_free_memory() const {
        size_t free_mem, total_mem;
        rpcserver_get_backend_memory(backend, index, &free_mem, &total_mem);
        if (free_mem >= capacity) {
            free_mem = capacity;
        }
        SRV_DBG("allocatable_mib = %zu, capacity_mib = %zu\n", free_mem >> 20, total_mem >> 20);
        return free_mem;
    }

    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor) {
        if (tensor->type >= GGML_TYPE_COUNT) {
            SRV_ERR("invalid tensor type received: %u\n", tensor->type);
            return nullptr;
        }

        ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type, tensor->ne[0], tensor->ne[1],
                                                  tensor->ne[2], tensor->ne[3]);
        if (result == nullptr) {
            SRV_ERR("ggml_new_tensor_4d failed for type %u\\n", tensor->type);
            return nullptr;
        }

        for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
            result->nb[i] = tensor->nb[i];
        }
        result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
        if (result->buffer && buffers.find(result->buffer) == buffers.end()) {
            if (common_log_verbosity_thold > 5) {
                SRV_WRN(
                    "buffer not found, "
                    "id = %llu, name = %s, buffer = %p\n",
                    tensor->id, tensor->name, result->buffer);
            }
            result->buffer = nullptr;
        }

        if (result->buffer) {
            // require that the tensor data does not go beyond the buffer end
            auto tensor_size  = (uint64_t) ggml_nbytes(result);
            auto buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
            auto buffer_size  = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
            GGML_ASSERT(tensor->data + tensor_size >= tensor->data);  // check for overflow
            GGML_ASSERT(tensor->data >= buffer_start && tensor->data + tensor_size <= buffer_start + buffer_size);
        }

        result->op = (ggml_op) tensor->op;
        for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
            result->op_params[i] = tensor->op_params[i];
        }
        result->flags = tensor->flags;
        result->data  = reinterpret_cast<void *>(tensor->data);
        ggml_set_name(result, tensor->name);
        return result;
    }

    ggml_tensor * create_node(uint64_t id, struct ggml_context * ctx,
                              const std::unordered_map<uint64_t, const rpc_tensor *> & in_tensor_ptrs,
                              std::unordered_map<uint64_t, struct ggml_tensor *> &     in_tensor_map) {
        if (in_tensor_map.find(id) != in_tensor_map.end()) {
            return in_tensor_map[id];
        }

        auto it_ptr = in_tensor_ptrs.find(id);
        if (it_ptr == in_tensor_ptrs.end()) {
            return nullptr;
        }
        const rpc_tensor * in_tensor = it_ptr->second;

        struct ggml_tensor * result = deserialize_tensor(ctx, in_tensor);
        if (result == nullptr) {
            SRV_ERR("failed to deserialize tensor: id = %llu, name = %s, type = %s\n", id, in_tensor->name,
                    ggml_type_name((ggml_type) in_tensor->type));
            return nullptr;
        }
        in_tensor_map[id] = result;

        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (in_tensor->src[i] == 0) {
                result->src[i] = nullptr;
            } else {
                result->src[i] = create_node(in_tensor->src[i], ctx, in_tensor_ptrs, in_tensor_map);
                if (result->src[i] == nullptr) {
                    SRV_ERR("failed to create src node %d, id = %llu, name = %s, type = %s\n", i, id, in_tensor->name,
                            ggml_type_name((ggml_type) in_tensor->type));
                    return nullptr;
                }
            }
        }
        if (in_tensor->view_src == 0) {
            result->view_src = nullptr;
        } else {
            result->view_src = create_node(in_tensor->view_src, ctx, in_tensor_ptrs, in_tensor_map);
            if (result->view_src == nullptr) {
                SRV_ERR("failed to create view_src node, id = %llu, name = %s, type = %s\n", id, in_tensor->name,
                        ggml_type_name((ggml_type) in_tensor->type));
                return nullptr;
            }
        }
        result->view_offs = in_tensor->view_offs;

        SRV_DBG("id = %llu, name = %s, type = %s, op = %s\n", id, in_tensor->name,
                ggml_type_name(static_cast<ggml_type>(in_tensor->type)),
                ggml_op_name(static_cast<ggml_op>(in_tensor->op)));
        return result;
    }

    //
    // Logics
    //

    void process(rpc_sockfd_t sockfd) {
        // wait for new command
        uint8_t cmd;
        if (!rpc_recv_data(sockfd, &cmd, 1)) {
            return;
        }
        if (cmd >= RPC_CMD_COUNT) {
            SRV_ERR("unknown command: %d\n", cmd);
            return;
        }
        if (cmd != RPC_CMD_HELLO) {
            // send something to trigger main server crash
            SRV_ERR("expected command %d, please update main server\n", RPC_CMD_HELLO);
            rpc_msg_hello_rsp hello{};
            say_hello(hello);
            rpc_send_msg(sockfd, &hello, sizeof(hello));
            return;
        }

        // say hello
        if (!rpc_recv_msg(sockfd, nullptr, 0)) {
            return;
        }
        rpc_msg_hello_rsp hello{};
        say_hello(hello);
        if (!rpc_send_msg(sockfd, &hello, sizeof(hello))) {
            return;
        }

        // process other command
        while (true) {
            // wait for new command
            if (!rpc_recv_data(sockfd, &cmd, 1)) {
                return;
            }
            if (cmd >= RPC_CMD_COUNT) {
                SRV_ERR("unknown command: %d\n", cmd);
                break;
            }

            switch (cmd) {
                case RPC_CMD_ALLOC_BUFFER:
                    {
                        rpc_msg_alloc_buffer_req request{};
                        if (!rpc_recv_msg(sockfd, &request, sizeof(request))) {
                            return;
                        }
                        rpc_msg_alloc_buffer_rsp response{};
                        if (!alloc_buffer(request, response)) {
                            return;
                        }
                        if (!rpc_send_msg(sockfd, &response, sizeof(response))) {
                            return;
                        }
                        break;
                    }
                case RPC_CMD_GET_ALIGNMENT:
                    {
                        if (!rpc_recv_msg(sockfd, nullptr, 0)) {
                            return;
                        }
                        rpc_msg_get_alignment_rsp response{};
                        get_alignment(response);
                        if (!rpc_send_msg(sockfd, &response, sizeof(response))) {
                            return;
                        }
                        break;
                    }
                case RPC_CMD_GET_MAX_SIZE:
                    {
                        if (!rpc_recv_msg(sockfd, nullptr, 0)) {
                            return;
                        }
                        rpc_msg_get_max_size_rsp response{};
                        get_max_size(response);
                        if (!rpc_send_msg(sockfd, &response, sizeof(response))) {
                            return;
                        }
                        break;
                    }
                case RPC_CMD_BUFFER_GET_BASE:
                    {
                        rpc_msg_buffer_get_base_req request{};
                        if (!rpc_recv_msg(sockfd, &request, sizeof(request))) {
                            return;
                        }
                        rpc_msg_buffer_get_base_rsp response{};
                        if (!buffer_get_base(request, response)) {
                            return;
                        }
                        if (!rpc_send_msg(sockfd, &response, sizeof(response))) {
                            return;
                        }
                        break;
                    }
                case RPC_CMD_FREE_BUFFER:
                    {
                        rpc_msg_free_buffer_req request{};
                        if (!rpc_recv_msg(sockfd, &request, sizeof(request))) {
                            return;
                        }
                        if (!free_buffer(request)) {
                            return;
                        }
                        if (!rpc_send_msg(sockfd, nullptr, 0)) {
                            return;
                        }
                        break;
                    }
                case RPC_CMD_BUFFER_CLEAR:
                    {
                        rpc_msg_buffer_clear_req request{};
                        if (!rpc_recv_msg(sockfd, &request, sizeof(request))) {
                            return;
                        }
                        if (!buffer_clear(request)) {
                            return;
                        }
                        if (!rpc_send_msg(sockfd, nullptr, 0)) {
                            return;
                        }
                        break;
                    }
                case RPC_CMD_SET_TENSOR:
                    {
                        std::vector<uint8_t> input;
                        if (!rpc_recv_msg(sockfd, input)) {
                            return;
                        }
                        if (!set_tensor(input)) {
                            return;
                        }
                        break;
                    }
                case RPC_CMD_SET_TENSOR_HASH:
                    {
                        rpc_msg_set_tensor_hash_req request{};
                        if (!rpc_recv_msg(sockfd, &request, sizeof(request))) {
                            return;
                        }
                        rpc_msg_set_tensor_hash_rsp response{};
                        if (!set_tensor_hash(request, response)) {
                            return;
                        }
                        if (!rpc_send_msg(sockfd, &response, sizeof(response))) {
                            return;
                        }
                        break;
                    }
                case RPC_CMD_GET_TENSOR:
                    {
                        rpc_msg_get_tensor_req request{};
                        if (!rpc_recv_msg(sockfd, &request, sizeof(request))) {
                            return;
                        }
                        std::vector<uint8_t> response;
                        if (!get_tensor(request, response)) {
                            return;
                        }
                        if (!rpc_send_msg(sockfd, response.data(), response.size())) {
                            return;
                        }
                        break;
                    }
                case RPC_CMD_COPY_TENSOR:
                    {
                        rpc_msg_copy_tensor_req request{};
                        if (!rpc_recv_msg(sockfd, &request, sizeof(request))) {
                            return;
                        }
                        rpc_msg_copy_tensor_rsp response{};
                        if (!copy_tensor(request, response)) {
                            return;
                        }
                        if (!rpc_send_msg(sockfd, &response, sizeof(response))) {
                            return;
                        }
                        break;
                    }
                case RPC_CMD_GRAPH_COMPUTE:
                    {
                        std::vector<uint8_t> input;
                        if (!rpc_recv_msg(sockfd, input)) {
                            return;
                        }
                        rpc_msg_graph_compute_rsp response{};
                        if (!graph_compute(input, response)) {
                            return;
                        }
                        if (!rpc_send_msg(sockfd, &response, sizeof(response))) {
                            return;
                        }
                        break;
                    }
                case RPC_CMD_GET_DEVICE_MEMORY:
                    {
                        if (!rpc_recv_msg(sockfd, nullptr, 0)) {
                            return;
                        }
                        rpc_msg_get_device_memory_rsp response{};
                        get_device_memory(response);
                        if (!rpc_send_msg(sockfd, &response, sizeof(response))) {
                            return;
                        }
                        break;
                    }
                case RPC_CMD_INIT_TENSOR:
                    {
                        rpc_msg_init_tensor_req request{};
                        if (!rpc_recv_msg(sockfd, &request, sizeof(request))) {
                            return;
                        }
                        if (!init_tensor(request)) {
                            return;
                        }
                        if (!rpc_send_msg(sockfd, nullptr, 0)) {
                            return;
                        }
                        break;
                    }
                case RPC_CMD_GET_ALLOC_SIZE:
                    {
                        rpc_msg_get_alloc_size_req request{};
                        if (!rpc_recv_msg(sockfd, &request, sizeof(request))) {
                            return;
                        }
                        rpc_msg_get_alloc_size_rsp response{};
                        if (!get_alloc_size(request, response)) {
                            return;
                        }
                        if (!rpc_send_msg(sockfd, &response, sizeof(response))) {
                            return;
                        }
                        break;
                    }
                case RPC_CMD_SUPPORT_OP:
                    {
                        std::vector<uint8_t> input;
                        if (!rpc_recv_msg(sockfd, input)) {
                            return;
                        }
                        rpc_msg_support_op_rsp response{};
                        if (!support_op(input, response)) {
                            return;
                        }
                        if (!rpc_send_msg(sockfd, &response, sizeof(response))) {
                            return;
                        }
                        break;
                    }
                default:
                    {
                        SRV_ERR("unknown command: %d\n", cmd);
                        return;
                    }
            }
        }
    }

    //
    // Routes
    //

    bool alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response) {
        size_t free_mem = get_free_memory();
        if (request.size > free_mem) {
            SRV_ERR("out of memory, request_mib = %llu, free_mib = %zu\n", request.size >> 20, free_mem >> 20);
            return false;
        }

        ggml_backend_buffer_type_t buft   = ggml_backend_get_default_buffer_type(backend);
        ggml_backend_buffer_t      buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
        response.remote_ptr               = 0;
        response.remote_size              = 0;
        if (buffer != nullptr) {
            response.remote_ptr  = reinterpret_cast<uint64_t>(buffer);
            response.remote_size = buffer->size;
            buffers.insert(buffer);
            SRV_DBG("remote_ptr = %llu, request_mib = %llu\n", response.remote_ptr, request.size >> 20);
            return true;
        }

        SRV_ERR("buffer allocated failed, request_mib = %llu, free_mib = %zu\n", request.size >> 20, free_mem >> 20);
        return true;
    }

    bool get_alignment(rpc_msg_get_alignment_rsp & response) {
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        response.alignment              = ggml_backend_buft_get_alignment(buft);
        SRV_DBG("result = %llu\n", response.alignment);
        return true;
    }

    bool get_max_size(rpc_msg_get_max_size_rsp & response) {
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        response.max_size               = ggml_backend_buft_get_max_size(buft);
        SRV_DBG("result = %llu\n", response.max_size);
        return true;
    }

    bool buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response) {
        auto buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
        if (buffers.find(buffer) == buffers.end()) {
            SRV_ERR("buffer not found, remote_ptr = %llu\n", request.remote_ptr);
            return false;
        }
        response.base_ptr = reinterpret_cast<uint64_t>(ggml_backend_buffer_get_base(buffer));
        SRV_DBG("remote_ptr = %llu, base_ptr = %llu\n", request.remote_ptr, response.base_ptr);
        return true;
    }

    bool free_buffer(const rpc_msg_free_buffer_req & request) {
        auto buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
        if (buffers.find(buffer) == buffers.end()) {
            SRV_ERR("buffer not found, remote_ptr = %llu\n", request.remote_ptr);
            return false;
        }
        ggml_backend_buffer_free(buffer);
        buffers.erase(buffer);
        SRV_DBG("remote_ptr = %llu\n", request.remote_ptr);
        return true;
    }

    bool buffer_clear(const rpc_msg_buffer_clear_req & request) {
        auto buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
        if (buffers.find(buffer) == buffers.end()) {
            SRV_ERR("buffer not found, remote_ptr = %llu\n", request.remote_ptr);
            return false;
        }
        ggml_backend_buffer_clear(buffer, request.value);
        SRV_DBG("remote_ptr = %llu\n", request.remote_ptr);
        return true;
    }

    bool set_tensor(const std::vector<uint8_t> & input) {
        // serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
        if (input.size() < sizeof(rpc_tensor) + sizeof(uint64_t)) {
            SRV_ERR("%s", "input size invalid\n");
            return false;
        }
        const auto * in_tensor = (const rpc_tensor *) input.data();
        uint64_t     offset;
        memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
        const size_t size = input.size() - sizeof(rpc_tensor) - sizeof(offset);

        struct ggml_init_params gparams{
            /*.mem_size   =*/ggml_tensor_overhead(),
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };

        ggml_context_ptr ctx_ptr{ ggml_init(gparams) };
        GGML_ASSERT(ctx_ptr != nullptr);
        ggml_context * ctx    = ctx_ptr.get();
        ggml_tensor *  tensor = deserialize_tensor(ctx, in_tensor);
        if (tensor == nullptr) {
            SRV_ERR("failed to deserialize tensor: id = %llu, name = %s, type = %s\n", in_tensor->id, in_tensor->name,
                    ggml_type_name((ggml_type) in_tensor->type));
            return false;
        }

        // sanitize tensor->data
        {
            const auto   p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
            const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

            if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 ||
                size > (p1 - in_tensor->data - offset)) {
                SRV_ERR("out of bound, id = %llu\n", in_tensor->id);
                delete tensor;
                return false;
            }
        }

        const void * data           = input.data() + sizeof(rpc_tensor) + sizeof(offset);
        int          caching_status = -1;  // -1 = no cache, 0 = cached ok, 1 = cache failed
        if (cache_dir && size > HASH_THRESHOLD) {
            try {
                std::string   hash       = hash_fnv((const uint8_t *) data, size);
                fs::path      cache_file = fs::path(cache_dir) / hash;
                std::ofstream ofs(cache_file, std::ios::binary);
                ofs.write((const char *) data, size);
                caching_status = 0;
            } catch (std::exception & e) {
                SRV_WRN("cache tensor, id = %llu: %s\n", in_tensor->id, e.what());
                caching_status = 1;
            }
        }
        ggml_backend_tensor_set(tensor, data, offset, size);
        SRV_DBG(
            "id = %llu, name = %s, type = %s, op = %s, "
            "size = %zu, caching = %d\n",
            in_tensor->id, in_tensor->name, ggml_type_name((ggml_type) in_tensor->type),
            ggml_op_name((ggml_op) in_tensor->op), size, caching_status);
        return true;
    }

    bool set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response) {
        // get cached
        response.result = 0;
        std::vector<uint8_t> cached_file;
        if (get_cached_file(request.hash, cached_file)) {
            size_t size = cached_file.size();

            struct ggml_init_params gparams{
                /*.mem_size   =*/ggml_tensor_overhead(),
                /*.mem_buffer =*/nullptr,
                /*.no_alloc   =*/true,
            };

            ggml_context_ptr ctx_ptr{ ggml_init(gparams) };
            GGML_ASSERT(ctx_ptr != nullptr);
            ggml_context * ctx    = ctx_ptr.get();
            ggml_tensor *  tensor = deserialize_tensor(ctx, &request.tensor);
            if (tensor == nullptr) {
                SRV_ERR("failed to deserialize tensor: id = %llu, name = %s, type = %s\n", request.tensor.id,
                        request.tensor.name, ggml_type_name((ggml_type) request.tensor.type));
                return false;
            }

            // sanitize tensor->data
            {
                const auto   p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
                const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

                if (request.tensor.data + request.offset < p0 || request.tensor.data + request.offset >= p1 ||
                    size > (p1 - request.tensor.data - request.offset)) {
                    SRV_ERR(
                        "out of bound, "
                        "id = %llu, name = %s, type = %s, op = %s, "
                        "size = %zu, data = %p, offset = %llu, hash = %" PRIx64 "\n",
                        request.tensor.id, request.tensor.name, ggml_type_name((ggml_type) request.tensor.type),
                        ggml_op_name((ggml_op) request.tensor.op), size, tensor->data, request.offset, request.hash);
                    delete tensor;
                    return false;
                }
            }

            ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
            response.result = 1;
        }

        SRV_DBG(
            "id = %llu, name = %s, type = %s, op = %s, "
            "size = %zu, result = %d\n",
            request.tensor.id, request.tensor.name, ggml_type_name((ggml_type) request.tensor.type),
            ggml_op_name((ggml_op) request.tensor.op), cached_file.size(), response.result);
        return true;
    }

    bool get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response) {
        struct ggml_init_params gparams{
            /*.mem_size   =*/ggml_tensor_overhead(),
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };

        ggml_context_ptr ctx_ptr{ ggml_init(gparams) };
        GGML_ASSERT(ctx_ptr != nullptr);
        ggml_context * ctx    = ctx_ptr.get();
        ggml_tensor *  tensor = deserialize_tensor(ctx, &request.tensor);
        if (tensor == nullptr) {
            SRV_ERR("failed to deserialize tensor: id = %llu, name = %s, type = %s\n", request.tensor.id,
                    request.tensor.name, ggml_type_name((ggml_type) request.tensor.type));
            return false;
        }

        // sanitize tensor->data
        {
            const auto   p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
            const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

            if (request.tensor.data + request.offset < p0 || request.tensor.data + request.offset >= p1 ||
                request.size > (p1 - request.tensor.data - request.offset)) {
                delete tensor;
                SRV_ERR(
                    "out of bound, "
                    "id = %llu, name = %s, type = %s, op = %s, "
                    "size = %llu\n",
                    request.tensor.id, request.tensor.name, ggml_type_name((ggml_type) request.tensor.type),
                    ggml_op_name((ggml_op) request.tensor.op), request.size);
                return false;
            }
        }

        response.resize(request.size, 0);
        ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
        SRV_DBG(
            "id = %llu, name = %s, type = %s, op = %s, "
            "size = %llu\n",
            request.tensor.id, request.tensor.name, ggml_type_name((ggml_type) request.tensor.type),
            ggml_op_name((ggml_op) request.tensor.op), request.size);
        return true;
    }

    bool copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response) {
        struct ggml_init_params gparams{
            /*.mem_size   =*/2 * ggml_tensor_overhead(),
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };

        ggml_context_ptr ctx_ptr{ ggml_init(gparams) };
        GGML_ASSERT(ctx_ptr != nullptr);
        ggml_context * ctx = ctx_ptr.get();
        ggml_tensor *  src = deserialize_tensor(ctx, &request.src);
        if (src == nullptr) {
            SRV_ERR("failed to deserialize src tensor: id = %llu, name = %s, type = %s\n", request.src.id,
                    request.src.name, ggml_type_name((ggml_type) request.src.type));
            return false;
        }
        ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
        if (dst == nullptr) {
            SRV_ERR("failed to deserialize dst tensor: id = %llu, name = %s, type = %s\n", request.dst.id,
                    request.dst.name, ggml_type_name((ggml_type) request.dst.type));
            delete src;
            return false;
        }

        auto src_size   = (uint64_t) ggml_nbytes(src);
        auto dst_data   = (uint64_t) dst->data;
        auto dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
        auto dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);
        if (dst_data + src_size > dst_base + dst_buf_sz) {
            SRV_ERR(
                "out of bound, "
                "id = %llu, name = %s, type = %s, op = %s, "
                "size = %llu, dst_id = %llu, dst_base = %llu, dst_buf_sz = %llu\n",
                request.src.id, request.src.name, ggml_type_name((ggml_type) request.src.type),
                ggml_op_name((ggml_op) request.src.op), src_size, request.dst.id, dst_base, dst_buf_sz);
            return false;
        }

        response.result = ggml_backend_buffer_copy_tensor(src, dst);
        SRV_DBG(
            "id = %llu, name = %s, type = %s, op = %s, "
            "size = %llu, dst_id = %llu, dst_base = %llu, dst_buf_sz = %llu, result = %d\n",
            request.src.id, request.src.name, ggml_type_name((ggml_type) request.src.type),
            ggml_op_name((ggml_op) request.src.op), src_size, request.dst.id, dst_base, dst_buf_sz, response.result);
        return true;
    }

    bool graph_compute(const std::vector<uint8_t> & input, rpc_msg_graph_compute_rsp & response) {
        // serialization format:
        // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
        if (input.size() < sizeof(uint32_t)) {
            SRV_ERR("%s", "input size invalid\n");
            return false;
        }
        uint32_t n_nodes;
        memcpy(&n_nodes, input.data(), sizeof(n_nodes));
        if (input.size() < sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t)) {
            SRV_ERR("%s", "input size invalid\n");
            return false;
        }
        const auto * nodes = (const uint64_t *) (input.data() + sizeof(n_nodes));
        uint32_t     n_tensors;
        memcpy(&n_tensors, input.data() + sizeof(n_nodes) + n_nodes * sizeof(uint64_t), sizeof(n_tensors));
        if (input.size() <
            sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor)) {
            SRV_ERR("%s", "input size invalid\n");
            return false;
        }
        const auto * in_tensors =
            (const rpc_tensor *) (input.data() + sizeof(n_nodes) + n_nodes * sizeof(uint64_t) + sizeof(n_tensors));

        size_t buf_size = ggml_tensor_overhead() * (n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);

        struct ggml_init_params gparams = {
            /*.mem_size   =*/buf_size,
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };

        ggml_context_ptr ctx_ptr{ ggml_init(gparams) };
        GGML_ASSERT(ctx_ptr != nullptr);
        ggml_context *       ctx   = ctx_ptr.get();
        struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
        graph->n_nodes             = int32_t(n_nodes);
        std::unordered_map<uint64_t, const rpc_tensor *> in_tensor_ptrs;
        for (uint32_t i = 0; i < n_tensors; i++) {
            in_tensor_ptrs[in_tensors[i].id] = &in_tensors[i];
        }
        std::unordered_map<uint64_t, ggml_tensor *> in_tensor_map;
        for (uint32_t i = 0; i < n_nodes; i++) {
            int64_t id;
            memcpy(&id, &nodes[i], sizeof(id));
            graph->nodes[i] = create_node(id, ctx, in_tensor_ptrs, in_tensor_map);
            if (graph->nodes[i] == nullptr && id != 0) {
                SRV_ERR("failed to create graph node %d, id = %llu\n", i, id);
                return false;
            }
        }
        response.result = ggml_backend_graph_compute(backend, graph);
        SRV_DBG("result = %d\n", response.result);
        return true;
    }

    bool get_device_memory(rpc_msg_get_device_memory_rsp & response) {
        response.total_mem = capacity;
        response.free_mem  = get_free_memory();
        SRV_DBG("free = %llu, total = %llu\n", response.free_mem, response.total_mem);
        return true;
    }

    bool init_tensor(const rpc_msg_init_tensor_req & request) {
        struct ggml_init_params gparams{
            /*.mem_size   =*/ggml_tensor_overhead(),
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };

        ggml_context_ptr ctx_ptr{ ggml_init(gparams) };
        GGML_ASSERT(ctx_ptr != nullptr);
        ggml_context * ctx    = ctx_ptr.get();
        ggml_tensor *  tensor = deserialize_tensor(ctx, &request.tensor);
        if (tensor == nullptr) {
            SRV_ERR("failed to deserialize tensor: id = %llu, name = %s, type = %s\n", request.tensor.id,
                    request.tensor.name, ggml_type_name((ggml_type) request.tensor.type));
            return false;
        }

        bool                  result = false;
        ggml_backend_buffer_t buffer = tensor->buffer;
        if (buffer && buffer->iface.init_tensor) {
            buffer->iface.init_tensor(buffer, tensor);
            result = true;
        } else {
            SRV_WRN("%s", "null buffer for tensor passed to init_tensor function\n");
        }

        if (tensor->extra != nullptr) {
            // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
            // Currently unimplemented.
            SRV_ERR("%s", "extra populated by the backend, this is currently unsupported.\n");
            return false;
        }

        SRV_DBG(
            "id = %llu, name = %s, type = %s, op = %s, "
            "result = %d\n",
            request.tensor.id, request.tensor.name, ggml_type_name((ggml_type) request.tensor.type),
            ggml_op_name((ggml_op) request.tensor.op), result);
        return true;
    }

    bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response) {
        ggml_backend_buffer_type_t buft;

        struct ggml_init_params gparams{
            /*.mem_size   =*/ggml_tensor_overhead(),
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };

        ggml_context_ptr ctx_ptr{ ggml_init(gparams) };
        GGML_ASSERT(ctx_ptr != nullptr);
        ggml_context * ctx    = ctx_ptr.get();
        ggml_tensor *  tensor = deserialize_tensor(ctx, &request.tensor);
        if (tensor == nullptr) {
            SRV_ERR("failed to deserialize tensor: id = %llu, name = %s, type = %s\n", request.tensor.id,
                    request.tensor.name, ggml_type_name((ggml_type) request.tensor.type));
            return false;
        }

        if (tensor->buffer == nullptr) {
            // No buffer allocated.
            buft = ggml_backend_get_default_buffer_type(backend);
        } else {
            buft = tensor->buffer->buft;
        }

        response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);
        SRV_DBG(
            "id = %llu, name = %s, type = %s, op = %s, "
            "result = %llu\n",
            request.tensor.id, request.tensor.name, ggml_type_name((ggml_type) request.tensor.type),
            ggml_op_name((ggml_op) request.tensor.op), response.alloc_size);
        return true;
    }

    bool say_hello(rpc_msg_hello_rsp & response) {
        response.major         = RPC_PROTO_MAJOR_VERSION;
        response.minor         = RPC_PROTO_MINOR_VERSION;
        response.patch         = RPC_PROTO_PATCH_VERSION;
        response.enabled_cache = cache_dir != nullptr;
        return true;
    }

    bool support_op(const std::vector<uint8_t> & input, rpc_msg_support_op_rsp & response) {
        // serialization format: | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
        if (input.size() < sizeof(uint32_t)) {
            SRV_ERR("%s", "input size invalid\n");
            return false;
        }

        uint32_t n_tensors;
        memcpy(&n_tensors, input.data(), sizeof(n_tensors));
        if (input.size() < sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor)) {
            SRV_ERR("%s", "input size invalid\n");
            return false;
        }
        const auto * tensors = (const rpc_tensor *) (input.data() + sizeof(uint32_t));

        size_t buf_size = ggml_tensor_overhead() * n_tensors;

        struct ggml_init_params gparams{
            /*.mem_size   =*/buf_size,
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };

        ggml_context_ptr ctx_ptr{ ggml_init(gparams) };
        GGML_ASSERT(ctx_ptr != nullptr);
        ggml_context * ctx    = ctx_ptr.get();
        ggml_tensor *  tensor = deserialize_tensor(ctx, &tensors[n_tensors - 1]);
        if (tensor == nullptr) {
            SRV_ERR("failed to deserialize tensor: id = %llu, name = %s, type = %s\n", tensors[n_tensors - 1].id,
                    tensors[n_tensors - 1].name, ggml_type_name((ggml_type) tensors[n_tensors - 1].type));
            return false;
        }

        for (uint32_t i = 0; i < n_tensors - 1; i++) {
            ggml_tensor * src = deserialize_tensor(ctx, &tensors[i]);
            if (src == nullptr) {
                SRV_ERR("failed to deserialize src tensor: id = %llu, name = %s, type = %s\n", tensors[i].id,
                        tensors[i].name, ggml_type_name((ggml_type) tensors[i].type));
                return false;
            }

            tensor->src[i] = src;
        }
        response.result = true;
        if (backend->device->iface.supports_op) {
            response.result = backend->device->iface.supports_op(backend->device, tensor);
        }

        SRV_DBG(
            "id = %llu, name = %s, type = %s, op = %s, "
            "result = %d\n",
            tensors[n_tensors - 1].id, tensors[n_tensors - 1].name,
            ggml_type_name((ggml_type) tensors[n_tensors - 1].type), ggml_op_name((ggml_op) tensors[n_tensors - 1].op),
            response.result);
        return true;
    }
};

static int32_t start_rpcserver(rpcserver_params & params) {
    rpcserver srv(params);

    if (!srv.load()) {
        SRV_ERR("%s", "failed to load\n");
        return -1;
    }

    return srv.start();
}
