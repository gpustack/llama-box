#
# llama-box-v2
#

#
# Options
#

set(BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WITH_HTTP_CLIENT OFF CACHE BOOL "" FORCE)
set(ENABLE_UDS OFF CACHE BOOL "" FORCE)

#
# llama-box-v2
#
set(TARGET llama-box-v2)
include_directories(${CMAKE_SOURCE_DIR})
if (GGML_SYCL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
endif ()
if (GGML_CANN)
    if ("cann${CANN_INSTALL_DIR}" STREQUAL "cann" AND DEFINED ENV{ASCEND_TOOLKIT_HOME})
        set(CANN_INSTALL_DIR $ENV{ASCEND_TOOLKIT_HOME})
        message(STATUS "CANN: updated CANN_INSTALL_DIR from ASCEND_TOOLKIT_HOME=$ENV{ASCEND_TOOLKIT_HOME}")
    endif ()
    if (CANN_INSTALL_DIR)
        message(STATUS "CANN: updated link directories to ${CANN_INSTALL_DIR}/lib64 and ${CANN_INSTALL_DIR}/runtime/lib64/stub")
        link_directories(
                ${CANN_INSTALL_DIR}/lib64
                ${CANN_INSTALL_DIR}/runtime/lib64/stub)
        include_directories(
                ${CANN_INSTALL_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
                ${CANN_INSTALL_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/aclnn
                ${CANN_INSTALL_DIR}/acllib/include)
    endif ()
endif ()
if (GGML_MUSA)
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_C_EXTENSIONS OFF)
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif ()
if (GGML_HIP)
    if (NOT EXISTS $ENV{ROCM_PATH})
        if (NOT EXISTS /opt/rocm)
            set(ROCM_PATH /usr)
        else ()
            set(ROCM_PATH /opt/rocm)
        endif ()
    else ()
        set(ROCM_PATH $ENV{ROCM_PATH})
    endif ()
    include_directories(
            ${ROCM_PATH}/include)
endif ()
add_library(${TARGET} engine.cpp param.hpp httpserver.hpp rpcserver.hpp stablediffusion.hpp)
target_link_libraries(${TARGET} PRIVATE version common llava stable-diffusion hv_static ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${TARGET} PUBLIC cxx_std_17)
if (LLAMA_OPENSSL)
    message(STATUS "Building with OpenSSL")
    set(OPENSSL_USE_STATIC_LIBS TRUE)
    if (DEFINED ENV{OPENSSL_ROOT_DIR})
        message(STATUS "Search OpenSSL from $ENV{OPENSSL_ROOT_DIR}")
        set(OPENSSL_ROOT_DIR $ENV{OPENSSL_ROOT_DIR})
    endif ()
    find_package(OpenSSL)
    if (OpenSSL_FOUND)
        message(STATUS "OpenSSL found: ${OPENSSL_LIBRARIES}")
        target_link_libraries(${TARGET} PRIVATE OpenSSL::SSL OpenSSL::Crypto)
        target_compile_definitions(${TARGET} PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
    else ()
        message(WARNING "OpenSSL not found")
    endif ()
endif ()
if (GGML_STATIC OR GGML_CUDA OR GGML_HIP OR GGML_CANN OR GGML_MUSA)
    message(STATUS "Building with static libraries")
    set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE "CXX")
    target_link_options(${TARGET} PRIVATE -static-libgcc -static-libstdc++)
endif ()
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Building in debug mode")
    target_compile_definitions(ggml PUBLIC GGML_DEBUG)
endif ()
