# llama-box

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
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -P "${CMAKE_CURRENT_SOURCE_DIR}/scripts/patch.cmake"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        VERBATIM
)
add_dependencies(patch version build_info)
add_dependencies(ggml-base patch)
add_dependencies(ggml patch)
add_dependencies(common ggml ggml-base patch)
add_dependencies(llava common patch)
add_dependencies(llama common patch)
add_dependencies(stable-diffusion common patch)

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
    endif ()
endif ()
if (GGML_MUSA)
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_C_EXTENSIONS OFF)
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif ()
add_executable(${TARGET} server.cpp param.hpp ratelimiter.hpp rpcserver.hpp stablediffusion.hpp utils.hpp)
if (NOT APPLE AND NOT GGML_SYCL)
    target_link_options(${TARGET} PRIVATE -static-libgcc -static-libstdc++)
endif ()
target_link_libraries(${TARGET} PRIVATE version common llama llava stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if (WIN32)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
endif ()
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Building in debug mode")
    target_compile_definitions(ggml PUBLIC GGML_DEBUG)
endif ()
target_compile_features(${TARGET} PUBLIC cxx_std_17)

#
# clean patches
#
add_custom_command(
        TARGET ${TARGET}
        POST_BUILD
        COMMENT "Cleaning patches"
        COMMAND ${CMAKE_COMMAND} -DMSVC=${MSVC} -DCMAKE_C_COMPILER_VERSION=${CMAKE_C_COMPILER_VERSION}
        -DCMAKE_C_COMPILER_ID=${CMAKE_C_COMPILER_ID} -DCMAKE_VS_PLATFORM_NAME=${CMAKE_VS_PLATFORM_NAME}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -P "${CMAKE_CURRENT_SOURCE_DIR}/scripts/patch-reset.cmake"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        VERBATIM
)
