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

option(BOX_PATCH_CI "box: patch CI" OFF)
option(BOX_PATCH_DEBUG "box: enable debug patches" OFF)
option(BOX_OPENSSL "llama: use OpenSSL for HTTPS" ON)

# debug
option(LLAMA_ALL_WARNINGS "llama: enable all compiler warnings" ON)

# build
option(LLAMA_FATAL_WARNINGS "llama: enable -Werror flag" OFF)

# sanitizers
option(LLAMA_SANITIZE_THREAD "llama: enable thread sanitizer" OFF)
option(LLAMA_SANITIZE_ADDRESS "llama: enable address sanitizer" OFF)
option(LLAMA_SANITIZE_UNDEFINED "llama: enable undefined sanitizer" OFF)

# utils
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)

# extra artifacts
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

# 3rd party libs
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_LLGUIDANCE OFF CACHE BOOL "" FORCE)

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
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    if (NOT DEFINED GGML_BACKEND_DL)
        set(GGML_BACKEND_DL ON)
    endif ()
    if (NOT DEFINED GGML_CPU_ALL_VARIANTS)
        set(GGML_CPU_ALL_VARIANTS ON)
    endif ()
    set(GGML_BACKEND_DL ${GGML_BACKEND_DL} CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS ${GGML_CPU_ALL_VARIANTS} CACHE BOOL "" FORCE)
    set(SD_BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
else ()
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
    set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
endif ()
if (GGML_CUDA)
    set(SD_CUDA ON CACHE BOOL "" FORCE)
elseif (GGML_HIP)
    set(SD_HIP ON CACHE BOOL "" FORCE)
elseif (GGML_VULKAN)
    set(SD_VULKAN ON CACHE BOOL "" FORCE)
elseif (GGML_METAL)
    set(SD_METAL ON CACHE BOOL "" FORCE)
elseif (GGML_SYCL)
    set(SD_SYCL ON CACHE BOOL "" FORCE)
elseif (GGML_CANN)
    set(SD_CANN ON CACHE BOOL "" FORCE)
elseif (GGML_MUSA)
    set(SD_MUSA ON CACHE BOOL "" FORCE)
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
        set(LLAMA_SANITIZE_THREAD ON CACHE BOOL "" FORCE)
        add_compile_options(-fsanitize=thread)
        link_libraries(-fsanitize=thread)
    endif ()

    if (LLAMA_SANITIZE_ADDRESS)
        set(LLAMA_SANITIZE_ADDRESS ON CACHE BOOL "" FORCE)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries(-fsanitize=address)
    endif ()

    if (LLAMA_SANITIZE_UNDEFINED)
        set(LLAMA_SANITIZE_UNDEFINED ON CACHE BOOL "" FORCE)
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
    set(LLAMA_FATAL_WARNINGS ON CACHE BOOL "" FORCE)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND C_FLAGS -Werror)
        list(APPEND CXX_FLAGS -Werror)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        add_compile_options(/WX)
    endif ()
endif ()

if (LLAMA_ALL_WARNINGS)
    set(LLAMA_ALL_WARNINGS ON CACHE BOOL "" FORCE)
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
# Patch
#

if (BOX_PATCH_CI)
    execute_process(
            COMMAND ${CMAKE_COMMAND} -DMSVC=${MSVC} -DCMAKE_C_COMPILER_VERSION=${CMAKE_C_COMPILER_VERSION}
            -DCMAKE_C_COMPILER_ID=${CMAKE_C_COMPILER_ID} -DCMAKE_VS_PLATFORM_NAME=${CMAKE_VS_PLATFORM_NAME}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -P "${CMAKE_CURRENT_SOURCE_DIR}/llama-box/scripts/configure-patch.cmake"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/llama-box"
    )
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
