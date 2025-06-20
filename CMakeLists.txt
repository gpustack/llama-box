cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target directories.
project("llama-box" C CXX)
include(CheckIncludeFileCXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
cmake_policy(SET CMP0077 NEW)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
endif ()

#
# Options
#

# debug
option(LLAMA_ALL_WARNINGS "llama: enable all compiler warnings" ON)
option(LLAMA_PATCH_DEBUG "llama: enable debug patches" OFF)

# build
option(LLAMA_FATAL_WARNINGS "llama: enable -Werror flag" OFF)
option(LLAMA_OPENSSL "llama: use OpenSSL for HTTPS" ON)

# sanitizers
option(LLAMA_SANITIZE_THREAD "llama: enable thread sanitizer" OFF)
option(LLAMA_SANITIZE_ADDRESS "llama: enable address sanitizer" OFF)
option(LLAMA_SANITIZE_UNDEFINED "llama: enable undefined sanitizer" OFF)

# utils
set(LLAMA_BUILD_COMMON ON CACHE BOOL "llama: build common utils library" FORCE)

# extra artifacts
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "llama: build tests" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "llama: build examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "llama: build server example" FORCE)

# 3rd party libs
set(LLAMA_CURL OFF CACHE BOOL "llama: use libcurl to download model from an URL" FORCE)
set(LLAMA_LLGUIDANCE OFF CACHE BOOL "llama-common: include LLGuidance library for structured output in common utils" FORCE)

# change the default for these ggml options
if (NOT DEFINED GGML_LLAMAFILE)
    set(GGML_LLAMAFILE_DEFAULT ON)
endif ()
if (NOT DEFINED GGML_CUDA_GRAPHS)
    set(GGML_CUDA_GRAPHS_DEFAULT ON)
endif ()
if (NOT DEFINED GGML_NATIVE)
    set(GGML_NATIVE OFF)
endif ()
add_definitions(-DGGML_MAX_NAME=128)

# change the default for these sd options
if (BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "build shared libraries" FORCE)
    set(GGML_BACKEND_DL ON CACHE BOOL "build backends as dynamic libraries (requires BUILD_SHARED_LIBS)" FORCE)
    set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "build all variants of the CPU backend (requires GGML_BACKEND_DL)" FORCE)
    set(SD_BUILD_SHARED_LIBS ON CACHE BOOL "sd: build shared libs" FORCE)
else ()
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "build shared libraries" FORCE)
    set(GGML_BACKEND_DL OFF CACHE BOOL "build backends as dynamic libraries (requires BUILD_SHARED_LIBS)" FORCE)
    set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "build all variants of the CPU backend (requires GGML_BACKEND_DL)" FORCE)
    set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "sd: build shared libs" FORCE)
endif ()
if (GGML_CUDA)
    set(SD_CUDA ON CACHE BOOL "sd: cuda backend" FORCE)
elseif (GGML_HIP)
    set(SD_HIP ON CACHE BOOL "sd: rocm backend" FORCE)
elseif (GGML_VULKAN)
    set(SD_VULKAN ON CACHE BOOL "sd: vulkan backend" FORCE)
elseif (GGML_METAL)
    set(SD_METAL ON CACHE BOOL "sd: metal backend" FORCE)
elseif (GGML_SYCL)
    set(SD_SYCL ON CACHE BOOL "sd: sycl backend" FORCE)
elseif (GGML_CANN)
    set(SD_CANN ON CACHE BOOL "sd: cann backend" FORCE)
elseif (GGML_MUSA)
    set(SD_MUSA ON CACHE BOOL "sd: musa backend" FORCE)
endif ()

#
# Compile flags
#

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)
include(CheckCXXCompilerFlag)

# enable libstdc++ assertions for debug builds
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_compile_definitions($<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)
endif ()

if (NOT MSVC)
    if (LLAMA_SANITIZE_THREAD)
        add_compile_options(-fsanitize=thread)
        link_libraries(-fsanitize=thread)
    endif ()

    if (LLAMA_SANITIZE_ADDRESS)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries(-fsanitize=address)
    endif ()

    if (LLAMA_SANITIZE_UNDEFINED)
        add_compile_options(-fsanitize=undefined)
        link_libraries(-fsanitize=undefined)
    endif ()
endif ()

function(get_flags CCID CCVER)
    set(C_FLAGS "")
    set(CXX_FLAGS "")

    if (CCID MATCHES "Clang")
        set(C_FLAGS -Wunreachable-code-break -Wunreachable-code-return)
        set(CXX_FLAGS -Wunreachable-code-break -Wunreachable-code-return -Wmissing-prototypes -Wextra-semi)

        if ((CCID STREQUAL "Clang" AND CCVER VERSION_GREATER_EQUAL 3.8.0) OR (CCID STREQUAL "AppleClang" AND CCVER VERSION_GREATER_EQUAL 7.3.0))
            list(APPEND C_FLAGS -Wdouble-promotion)
        endif ()
    elseif (CCID STREQUAL "GNU")
        set(C_FLAGS -Wdouble-promotion)
        set(CXX_FLAGS -Wno-array-bounds)

        if (CCVER VERSION_GREATER_EQUAL 7.1.0)
            list(APPEND CXX_FLAGS -Wno-format-truncation)
        endif ()
        if (CCVER VERSION_GREATER_EQUAL 8.1.0)
            list(APPEND CXX_FLAGS -Wextra-semi)
        endif ()
    endif ()

    set(GF_C_FLAGS ${C_FLAGS} PARENT_SCOPE)
    set(GF_CXX_FLAGS ${CXX_FLAGS} PARENT_SCOPE)
endfunction()

if (LLAMA_FATAL_WARNINGS)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND C_FLAGS -Werror)
        list(APPEND CXX_FLAGS -Werror)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        add_compile_options(/WX)
    endif ()
endif ()

if (LLAMA_ALL_WARNINGS)
    if (NOT MSVC)
        list(APPEND WARNING_FLAGS -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function)
        list(APPEND C_FLAGS -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes
                -Werror=implicit-int -Werror=implicit-function-declaration)
        list(APPEND CXX_FLAGS -Wmissing-declarations -Wmissing-noreturn)

        list(APPEND C_FLAGS ${WARNING_FLAGS})
        list(APPEND CXX_FLAGS ${WARNING_FLAGS})

        get_flags(${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})

        add_compile_options("$<$<COMPILE_LANGUAGE:C>:${C_FLAGS};${GF_C_FLAGS}>"
                "$<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS};${GF_CXX_FLAGS}>")
    else ()
        # todo : msvc
        set(C_FLAGS "")
        set(CXX_FLAGS "")
    endif ()
endif ()

#
# Programs
#

add_subdirectory(llama.cpp)
add_subdirectory(llama.cpp/tools/mtmd)
add_subdirectory(stable-diffusion.cpp)
add_subdirectory(concurrentqueue)
add_subdirectory(readerwriterqueue)
add_subdirectory(llama-box)
