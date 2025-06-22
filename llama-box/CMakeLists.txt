# llama-box

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

if (MINGW)
    # fix: https://github.com/ggerganov/llama.cpp/actions/runs/9651004652/job/26617901362?pr=8006
    add_compile_definitions(_WIN32_WINNT=${GGML_WIN_VER})
endif ()

#
# version
#
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../.git")
    set(GIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.git")

    # Is git submodule
    if (NOT IS_DIRECTORY "${GIT_DIR}")
        file(READ ${GIT_DIR} REAL_GIT_DIR_LINK)
        string(REGEX REPLACE "gitdir: (.*)\n$" "\\1" REAL_GIT_DIR ${REAL_GIT_DIR_LINK})
        string(FIND "${REAL_GIT_DIR}" "." SLASH_POS)
        if (SLASH_POS EQUAL 0)
            set(GIT_DIR "${REAL_GIT_DIR}")
        else ()
            set(GIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../${REAL_GIT_DIR}")
        endif ()
    endif ()

    if (EXISTS "${GIT_DIR}/index")
        set(GIT_INDEX "${GIT_DIR}/index")
    else ()
        set(GIT_INDEX "")
    endif ()
else ()
    set(GIT_INDEX "")
endif ()
add_custom_command(
        OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/version.cpp"
        COMMENT "Generating build details from Git"
        COMMAND ${CMAKE_COMMAND} -DMSVC=${MSVC} -DCMAKE_C_COMPILER_VERSION=${CMAKE_C_COMPILER_VERSION}
        -DCMAKE_C_COMPILER_ID=${CMAKE_C_COMPILER_ID} -DCMAKE_VS_PLATFORM_NAME=${CMAKE_VS_PLATFORM_NAME}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -P "${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen-version-cpp.cmake"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/version.cpp.in" ${GIT_INDEX}
        VERBATIM
)
set(TARGET version)
add_library(${TARGET} OBJECT version.cpp)

#
# patch
#
add_custom_target(patch
        COMMENT "Applying patches"
        COMMAND ${CMAKE_COMMAND} -DMSVC=${MSVC} -DCMAKE_C_COMPILER_VERSION=${CMAKE_C_COMPILER_VERSION}
        -DCMAKE_C_COMPILER_ID=${CMAKE_C_COMPILER_ID} -DCMAKE_VS_PLATFORM_NAME=${CMAKE_VS_PLATFORM_NAME}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -P "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build-patch.cmake"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        VERBATIM
)
add_dependencies(version patch)
add_dependencies(build_info patch)
add_dependencies(version build_info patch)
add_dependencies(ggml patch)
add_dependencies(ggml-base patch)
add_dependencies(ggml ggml-base patch)
add_dependencies(llama ggml ggml-base patch)
add_dependencies(stable-diffusion ggml ggml-base patch)
add_dependencies(mtmd ggml ggml-base patch)
add_dependencies(common ggml ggml-base patch)

#
# llama-box
#
set(TARGET llama-box)
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
add_executable(${TARGET} engine.cpp engine_param.hpp httpserver.hpp rpcserver.hpp z_multimodal.hpp z_stablediffusion.hpp z_utils.hpp)
target_link_libraries(${TARGET} PRIVATE version common mtmd stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if (WIN32)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
endif ()
target_compile_features(${TARGET} PUBLIC cxx_std_17)
if (BOX_OPENSSL)
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
if (NOT GGML_BACKEND_DL)
    # GGML_SYCL is not supported statically libstdc++ linked.
    if (GGML_CUDA OR GGML_HIP OR GGML_CANN OR GGML_MUSA OR GGML_VULKAN)
        message(STATUS "Building with static libraries")
        set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE "CXX")
        target_link_options(${TARGET} PRIVATE -static-libgcc -static-libstdc++)
    endif ()
endif ()
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Building in debug mode")
    target_compile_definitions(ggml PUBLIC GGML_DEBUG)
endif ()

#
# clean patches
#
if (NOT BOX_PATCH_DEBUG)
    add_custom_command(
            TARGET ${TARGET}
            POST_BUILD
            COMMENT "Cleaning patches"
            COMMAND ${CMAKE_COMMAND} -DMSVC=${MSVC} -DCMAKE_C_COMPILER_VERSION=${CMAKE_C_COMPILER_VERSION}
            -DCMAKE_C_COMPILER_ID=${CMAKE_C_COMPILER_ID} -DCMAKE_VS_PLATFORM_NAME=${CMAKE_VS_PLATFORM_NAME}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -P "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build-patch-reset.cmake"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            VERBATIM
    )
endif ()
